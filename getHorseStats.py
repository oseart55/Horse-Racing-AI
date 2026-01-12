import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle

# ----------------------------
# Ensure /data folder exists
# ----------------------------
os.makedirs("data", exist_ok=True)

# ----------------------------
# UTILS
# ----------------------------
def parse_fractional_odds(val):
    if val is None:
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).strip().upper()
    if val in ("", "NA", "N/A"):
        return np.nan
    if val in ("EVEN", "EVENS"):
        return 2.0
    if "/" in val:
        try:
            num, den = val.split("/")
            return float(num) / float(den) + 1.0
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan

# ----------------------------
# CONFIG
# ----------------------------
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

EPOCHS_PER_WINDOW = 3
MIN_TRAIN_RACES = 50
LR = 0.001

# ----------------------------
# LOAD DATA
# ----------------------------
conn = sqlite3.connect("horses.db")
df = pd.read_sql_query(
    "SELECT * FROM horses WHERE scratched = 0 AND finishPosition IS NOT NULL",
    conn
)
conn.close()

# ----------------------------
# PREPROCESS
# ----------------------------
df["odds"] = df["odds"].apply(parse_fractional_odds)
df["odds_prob"] = 1.0 / df["odds"]
df["odds_prob"] = df["odds_prob"].fillna(df["odds_prob"].median())
df["odds_prob_model"] = df["odds_prob"]

df["raceDate"] = pd.to_datetime(df["raceDate"])
df["top3_target"] = (df["finishPosition"] <= 3).astype(float)

df["race_id"] = (
    df["track"] + "_" +
    df["raceDate"].dt.strftime("%Y-%m-%d") + "_" +
    df["raceNumber"].astype(str)
)

# ----------------------------
# ENCODE CATEGORICALS
# ----------------------------
encoders = {}
cat_cardinalities = {}

for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    cat_cardinalities[col] = df[col].nunique()

# ----------------------------
# SCALE NUMERICS
# ----------------------------
scaler = StandardScaler()
df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

# ----------------------------
# MODEL DEFINITION
# ----------------------------
class HorseMLP(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, embed_dims):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(cat_cardinalities[col], embed_dims[col])
            for col in embed_dims
        })
        emb_dim_total = sum(embed_dims.values())
        self.net = nn.Sequential(
            nn.Linear(num_numeric + emb_dim_total, 64),
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
        embs = [self.embeddings[col](x_cat[:, i]) for i, col in enumerate(self.embeddings)]
        x = torch.cat([x_num] + embs, dim=1)
        return self.net(x).squeeze(-1)

# ----------------------------
# Plackett–Luce ranking loss
# ----------------------------
def plackett_luce_loss(logits, finish_pos):
    order = torch.argsort(finish_pos)
    logits = logits[order]
    loss = 0.0
    for i in range(len(logits)):
        loss += torch.logsumexp(logits[i:], dim=0) - logits[i]
    return loss

# ----------------------------
# DEVICE
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# LOAD EXISTING MODEL IF AVAILABLE
# ----------------------------
model_path = "data/horse_model.pt"
prep_path = "data/horse_preprocessing.pkl"

if os.path.exists(model_path) and os.path.exists(prep_path):
    print("Loading existing model and preprocessing...")
    with open(prep_path, "rb") as f:
        prep = pickle.load(f)
    scaler = prep["scaler"]
    encoders = prep["encoders"]
    NUMERIC_COLS = prep["NUMERIC_COLS"]
    CAT_COLS = prep["CAT_COLS"]
    EMBED_DIMS = prep["EMBED_DIMS"]

    model = HorseMLP(len(NUMERIC_COLS)+1, {col: len(encoders[col].classes_) for col in CAT_COLS}, EMBED_DIMS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded. Continuing training to improve it...")
else:
    print("No existing model found. Starting fresh...")
    model = HorseMLP(len(NUMERIC_COLS)+1, cat_cardinalities, EMBED_DIMS).to(device)

# ----------------------------
# WALK-FORWARD TRAINING
# ----------------------------
race_order = df[["race_id","raceDate"]].drop_duplicates().sort_values("raceDate")["race_id"].tolist()
all_results = []
running_hit_rates = []

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
total_hits = 0
total_races = 0

# --- Setup live plot ---
plt.ion()
fig, ax = plt.subplots(figsize=(10,5))
line, = ax.plot([], [], marker='o', linestyle='-', color='blue')
ax.set_xlim(0, len(race_order)-MIN_TRAIN_RACES)
ax.set_ylim(0, 1)
ax.set_xlabel("Race Number")
ax.set_ylabel("Running Hit Rate")
ax.set_title("Live Running Hit Rate of Recommended Horse")
ax.grid(True)

for i in range(MIN_TRAIN_RACES, len(race_order)):
    train_races = race_order[:i]
    test_race = race_order[i]

    train_df = df[df["race_id"].isin(train_races)]
    test_df = df[df["race_id"] == test_race]

    # -------- TRAIN ----------
    model.train()
    for epoch in range(EPOCHS_PER_WINDOW):
        for rid in train_df["race_id"].unique():
            race = train_df[train_df["race_id"] == rid]

            X_num = torch.tensor(race[NUMERIC_COLS + ["odds_prob_model"]].values, dtype=torch.float32, device=device)
            X_cat = torch.tensor(race[CAT_COLS].values, dtype=torch.long, device=device)
            finish = torch.tensor(race["finishPosition"].values, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            logits = model(X_num, X_cat)
            loss = plackett_luce_loss(logits, finish)
            loss.backward()
            optimizer.step()

    # -------- PREDICT ----------
    model.eval()
    with torch.no_grad():
        X_num = torch.tensor(test_df[NUMERIC_COLS + ["odds_prob_model"]].values, dtype=torch.float32, device=device)
        X_cat = torch.tensor(test_df[CAT_COLS].values, dtype=torch.long, device=device)
        scores = model(X_num, X_cat).cpu().numpy()

    test_df = test_df.copy()
    test_df["score"] = scores

    # --- Pick single horse ---
    candidate = test_df.loc[test_df["score"].idxmax()]

    if pd.isna(candidate["finishPosition"]):
        hit = None
    else:
        hit = int(candidate["finishPosition"] <= 3)
        total_hits += hit
        total_races += 1

    running_hit_rate = total_hits / total_races if total_races > 0 else 0
    running_hit_rates.append(running_hit_rate)

    # --- Print results ---
    print(f"Race {i} Recommended Horse: {candidate['name']} (Program {candidate['programNumber']})")
    print(f"Score: {candidate['score']:.3f} | Actual Finish: {candidate['finishPosition']} | Hit: {hit}")
    print(f"Running Hit Rate: {running_hit_rate:.3f}\n")

    # --- Update live plot ---
    line.set_data(range(len(running_hit_rates)), running_hit_rates)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)

    all_results.append(test_df)

plt.ioff()
plt.show()

# ----------------------------
# SAVE MODEL AND PREPROCESSING
# ----------------------------
torch.save(model.state_dict(), model_path)
with open(prep_path, "wb") as f:
    pickle.dump({
        "scaler": scaler,
        "encoders": encoders,
        "NUMERIC_COLS": NUMERIC_COLS,
        "CAT_COLS": CAT_COLS,
        "EMBED_DIMS": EMBED_DIMS
    }, f)

# ----------------------------
# FINAL
# ----------------------------
all_results_df = pd.concat(all_results, ignore_index=True)
print("Finished training and walk-forward predictions.")
print(f"Overall Hit Rate: {total_hits/total_races:.3f}")
print(f"Model saved to {model_path} and preprocessing to {prep_path}")
