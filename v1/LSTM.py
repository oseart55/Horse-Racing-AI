import os
import sqlite3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
NUMERIC_COLS = [
    "age","daysOff","distance","averageSpeedLast3",
    "oddsRank","postPosition","weight",
    "speedPoints","primePower"
]

CAT_COLS = [
    "sex","track","surfaceLabel","trackConditionLabel",
    "priorRunningStyle","trainer","jockey"
]

EMBED_DIMS = {
    "sex": 2,
    "track": 4,
    "surfaceLabel": 3,
    "trackConditionLabel": 3,
    "priorRunningStyle": 3,
    "trainer": 12,
    "jockey": 10
}

HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.2
EPOCHS_PER_WINDOW = 2
MIN_TRAIN_RACES = 1
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 300  # fixed-size sliding window
ALPHA_PICK = 0.9 # weight for top-pick reward

os.makedirs("data", exist_ok=True)

# ============================================================
# UTILS
# ============================================================
def parse_fractional_odds(val):
    if val in (None, "", "NA", "N/A"):
        return np.nan
    val = str(val).upper()
    if val in ("EVEN", "EVENS"):
        return 2.0
    if "/" in val:
        try:
            n, d = val.split("/")
            return float(n)/float(d) + 1.0
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan

# ============================================================
# LSTM MODEL
# ============================================================
class HorseLSTM(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.cat_order = list(EMBED_DIMS.keys())
        self.embeddings = nn.ModuleDict({
            c: nn.Embedding(cat_cardinalities[c], EMBED_DIMS[c])
            for c in self.cat_order
        })
        input_dim = num_numeric + sum(EMBED_DIMS.values())
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_num, x_cat):
        batch_size, seq_len, _ = x_num.shape
        embs = []
        for i, c in enumerate(self.cat_order):
            e = self.embeddings[c](x_cat[:, :, i])
            embs.append(e)
        x = torch.cat([x_num] + embs, dim=2)
        lstm_out, _ = self.lstm(x)
        scores = self.head(lstm_out).squeeze(-1)
        return scores

# ============================================================
# LOSS
# ============================================================
def reward_aware_loss(logits, finish_pos, odds_prob, alpha=ALPHA_PICK):
    """
    Combines standard Plackett-Luce loss with top-pick reward-aware adjustment.
    """
    # --- Standard Plackett-Luce loss ---
    order = torch.argsort(finish_pos)
    logits_sorted = logits[order]
    odds_sorted = odds_prob[order]
    weight = 1.0 / odds_sorted.clamp(min=1e-3)
    weight = weight / weight.mean()

    pl_loss = 0.0
    for i in range(len(logits_sorted)):
        pl_loss += weight[i]*(torch.logsumexp(logits_sorted[i:], dim=0)-logits_sorted[i])
    pl_loss /= len(logits_sorted)

    # --- Top-pick reward adjustment ---
    top_idx = torch.argmax(logits)
    reward = 1.0 if finish_pos[top_idx] <= 3 else -1.0
    pick_loss = -reward * torch.log_softmax(logits, dim=0)[top_idx]

    # --- Combine ---
    total_loss = pl_loss + alpha * pick_loss
    return total_loss

# ============================================================
# LOAD DATA
# ============================================================
conn = sqlite3.connect("horses.db")
df = pd.read_sql_query(
    "SELECT * FROM horses WHERE scratched=0 AND finishPosition IS NOT NULL",
    conn
)
conn.close()

df["odds"] = df["odds"].apply(parse_fractional_odds)
df["odds_prob"] = 1.0 / df["odds"]
df["odds_prob"].fillna(df["odds_prob"].median(), inplace=True)
df["raceDate"] = pd.to_datetime(df["raceDate"])
df["race_id"] = df["track"] + "_" + df["raceDate"].dt.strftime("%Y-%m-%d") + "_" + df["raceNumber"].astype(str)

race_order = df[["race_id","raceDate"]].drop_duplicates().sort_values("raceDate")["race_id"].tolist()

# ============================================================
# PREPROCESSING
# ============================================================
scaler = StandardScaler()
df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

encoders = {}
cat_cardinalities = {}
for c in CAT_COLS:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str)) + 1
    encoders[c] = le
    cat_cardinalities[c] = len(le.classes_) + 1

# ============================================================
# PRECOMPUTE TENSORS (Strategy B)
# ============================================================
race_tensors = {}
for rid, race in df.groupby("race_id"):
    X_num = torch.tensor(race[NUMERIC_COLS + ["odds_prob"]].values, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    X_cat = torch.tensor(race[CAT_COLS].values, dtype=torch.long, device=DEVICE).unsqueeze(0)
    y_fin = torch.tensor(race["finishPosition"].values, dtype=torch.float32, device=DEVICE)
    y_odds = torch.tensor(race["odds_prob"].values, dtype=torch.float32, device=DEVICE)
    for j, c in enumerate(CAT_COLS):
        X_cat[:, :, j] = X_cat[:, :, j].clamp(0, cat_cardinalities[c]-1)
    race_tensors[rid] = (X_num, X_cat, y_fin, y_odds)

# ============================================================
# INIT MODEL
# ============================================================
model = HorseLSTM(len(NUMERIC_COLS)+1, cat_cardinalities).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

total_hits = 0
total_races = 0
running_hit_rates = []

plt.ion()
fig, ax = plt.subplots(figsize=(10,5))
line, = ax.plot([], [])
ax.set_ylim(0,1)
ax.set_xlabel("Race number")
ax.set_ylabel("Running hit rate")
ax.set_title("Live Running Hit Rate")
ax.grid(True)

# ============================================================
# WALK-FORWARD TRAINING (Strategies A + C + Reward)
# ============================================================
for i in range(MIN_TRAIN_RACES, len(race_order)):
    # fixed-size sliding window
    train_ids = race_order[max(0, i-WINDOW_SIZE):i]
    test_id = race_order[i]

    # --- BATCH TRAINING ---
    X_num_window = torch.cat([race_tensors[rid][0] for rid in train_ids], dim=1)
    X_cat_window = torch.cat([race_tensors[rid][1] for rid in train_ids], dim=1)
    y_fin_window = torch.cat([race_tensors[rid][2] for rid in train_ids], dim=0)
    y_odds_window = torch.cat([race_tensors[rid][3] for rid in train_ids], dim=0)

    # TRAIN
    model.train()
    for _ in range(EPOCHS_PER_WINDOW):
        optimizer.zero_grad()
        logits = model(X_num_window, X_cat_window).squeeze(0)
        loss = reward_aware_loss(logits, y_fin_window, y_odds_window, alpha=ALPHA_PICK)
        loss.backward()
        optimizer.step()

    # PREDICT
    Xn_test, Xc_test, y_fin_test, _ = race_tensors[test_id]
    model.eval()
    with torch.no_grad():
        scores = model(Xn_test, Xc_test).squeeze(0).cpu().numpy()

    test_df = df[df["race_id"]==test_id].copy()
    test_df["score"] = scores
    pick = test_df.loc[test_df["score"].idxmax()]

    hit = int(pick["finishPosition"] <= 3)
    total_hits += hit
    total_races += 1
    running_hit_rates.append(total_hits / total_races)

    print(f"Race {i} | Pick: {pick['name']} | Finish: {pick['finishPosition']} | Running hit: {running_hit_rates[-1]:.3f}")

    line.set_data(range(len(running_hit_rates)), running_hit_rates)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)

plt.ioff()
plt.show()
torch.save(model.state_dict(), "data/horse_model_lstm_reward.pt")
print(f"FINAL HIT RATE: {total_hits/total_races:.3f}")
