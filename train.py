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

EPOCHS_PER_WINDOW = 2
MIN_TRAIN_RACES = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 300           # number of past races to train on for speed
USE_MIXED_PRECISION = False # set True if GPU supports AMP

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
            return float(n) / float(d) + 1.0
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan

# ============================================================
# MODEL
# ============================================================
class HorseMLP(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities):
        super().__init__()
        self.cat_order = list(EMBED_DIMS.keys())
        self.embeddings = nn.ModuleDict({
            c: nn.Embedding(cat_cardinalities[c], EMBED_DIMS[c])
            for c in self.cat_order
        })
        emb_total = sum(EMBED_DIMS.values())
        self.net = nn.Sequential(
            nn.Linear(num_numeric + emb_total, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x_num, x_cat):
        embs = [
            self.embeddings[c](x_cat[:, i])
            for i, c in enumerate(self.cat_order)
        ]
        x = torch.cat([x_num] + embs, dim=1)
        return self.net(x).squeeze(-1)

# ============================================================
# LOSS (Plackett–Luce, odds-aware)
# ============================================================
def plackett_luce_loss(logits, finish_pos, odds_prob):
    order = torch.argsort(finish_pos)
    logits = logits[order]
    odds_prob = odds_prob[order]
    weight = 1.0 / odds_prob.clamp(min=1e-3)
    weight = weight / weight.mean()
    loss = 0.0
    for i in range(len(logits)):
        loss += weight[i] * (torch.logsumexp(logits[i:], dim=0) - logits[i])
    return loss / len(logits)

# ============================================================
# LOAD DATA
# ============================================================
conn = sqlite3.connect("horses.db")
df = pd.read_sql_query(
    "SELECT * FROM horses WHERE scratched = 0 AND finishPosition IS NOT NULL",
    conn
)
conn.close()

df["odds"] = df["odds"].apply(parse_fractional_odds)
df["odds_prob"] = 1.0 / df["odds"]
df["odds_prob"].fillna(df["odds_prob"].median(), inplace=True)
df["raceDate"] = pd.to_datetime(df["raceDate"])
df["race_id"] = (
    df["track"] + "_" +
    df["raceDate"].dt.strftime("%Y-%m-%d") + "_" +
    df["raceNumber"].astype(str)
)

race_order = df[["race_id","raceDate"]].drop_duplicates().sort_values("raceDate")["race_id"].tolist()

# ============================================================
# PREPROCESSING: fit once
# ============================================================
# Numeric scaler
scaler = StandardScaler()
df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

# Categorical encoders
encoders = {}
cat_cardinalities = {}
for c in CAT_COLS:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str)) + 1 # reserve 0 for unknown
    encoders[c] = le
    cat_cardinalities[c] = len(le.classes_) + 1

def encode_test_row(row):
    out = []
    for c in CAT_COLS:
        val = str(row[c])
        if val in encoders[c].classes_:
            out.append(encoders[c].transform([val])[0] + 1)
        else:
            out.append(0)
    return out

# ============================================================
# WALK-FORWARD TRAINING (FAST VERSION)
# ============================================================
model = HorseMLP(len(NUMERIC_COLS)+1, cat_cardinalities).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

total_hits = 0
total_races = 0
running_hit_rates = []

plt.ion()
fig, ax = plt.subplots(figsize=(10,5))
line, = ax.plot([], [])
ax.set_ylim(0, 1)
ax.set_xlabel("Race number")
ax.set_ylabel("Running hit rate")
ax.set_title("Live Running Hit Rate")
ax.grid(True)

for i in range(MIN_TRAIN_RACES, len(race_order)):
    # sliding window of last WINDOW_SIZE races
    train_ids = race_order[max(0, i - WINDOW_SIZE):i]
    test_id = race_order[i]

    train_df = df[df["race_id"].isin(train_ids)].copy()
    test_df = df[df["race_id"] == test_id].copy()

    # X, Y tensors
    X_num = torch.tensor(train_df[NUMERIC_COLS + ["odds_prob"]].values, dtype=torch.float32, device=DEVICE)
    X_cat = torch.tensor(train_df[CAT_COLS].values, dtype=torch.long, device=DEVICE)
    y_fin = torch.tensor(train_df["finishPosition"].values, dtype=torch.float32, device=DEVICE)
    y_odds = torch.tensor(train_df["odds_prob"].values, dtype=torch.float32, device=DEVICE)

    # safety clamp
    for j, c in enumerate(CAT_COLS):
        X_cat[:, j] = X_cat[:, j].clamp(0, model.embeddings[c].num_embeddings - 1)

    # --- TRAIN ---
    model.train()
    for _ in range(EPOCHS_PER_WINDOW):
        if USE_MIXED_PRECISION:
            scaler_amp = torch.cuda.amp.GradScaler()
            with torch.cuda.amp.autocast():
                logits = model(X_num, X_cat)
                loss = plackett_luce_loss(logits, y_fin, y_odds)
            optimizer.zero_grad()
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            optimizer.zero_grad()
            logits = model(X_num, X_cat)
            loss = plackett_luce_loss(logits, y_fin, y_odds)
            loss.backward()
            optimizer.step()

    # --- PREDICT ---
    model.eval()
    X_num_test = torch.tensor(test_df[NUMERIC_COLS + ["odds_prob"]].values, dtype=torch.float32, device=DEVICE)
    X_cat_test = torch.tensor(test_df[CAT_COLS].values, dtype=torch.long, device=DEVICE)

    for j, c in enumerate(CAT_COLS):
        X_cat_test[:, j] = X_cat_test[:, j].clamp(0, model.embeddings[c].num_embeddings - 1)

    with torch.no_grad():
        scores = model(X_num_test, X_cat_test).cpu().numpy()

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

torch.save(model.state_dict(), "data/horse_model.pt")
print(f"FINAL HIT RATE: {total_hits/total_races:.3f}")
