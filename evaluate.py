import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
from datetime import datetime

# -----------------------------
# DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD PREPROCESSING + MODEL
# -----------------------------
with open("data/horse_preprocessing.pkl", "rb") as f:
    prep = pickle.load(f)

scaler = prep["scaler"]
encoders = prep["encoders"]
NUMERIC_COLS = prep["NUMERIC_COLS"]
CAT_COLS = prep["CAT_COLS"]
EMBED_DIMS = prep["EMBED_DIMS"]

# -----------------------------
# MODEL DEFINITION (MUST MATCH TRAINING)
# -----------------------------
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

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
num_numeric = len(NUMERIC_COLS) + 1  # + odds_prob_model
cat_cardinalities = {c: len(encoders[c].classes_) for c in CAT_COLS}

model = HorseMLP(num_numeric, cat_cardinalities, EMBED_DIMS).to(device)
model.load_state_dict(torch.load("data/horse_model.pt", map_location=device))
model.eval()

# -----------------------------
# LOG FILE SETUP
# -----------------------------
os.makedirs("logs", exist_ok=True)

run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = f"logs/eval_{run_ts}.txt"

log_f = open(log_path, "w", encoding="utf-8")

log_f.write(f"Evaluation run: {run_ts}\n")
log_f.write("=" * 60 + "\n\n")


# -----------------------------
# LOAD UPCOMING / UNFINISHED RACES
# -----------------------------
conn = sqlite3.connect("horses.db")
df = pd.read_sql_query(
    """
    SELECT *
    FROM horses
    WHERE scratched = 0
      AND finishPosition IS NULL
      AND brisId IS NOT -1
    """,
    conn
)
conn.close()

if df.empty:
    print("No upcoming races found.")
    exit()

# -----------------------------
# RECREATE odds_prob_model (EXACTLY like training)
# -----------------------------
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

df["odds"] = df["odds"].apply(parse_fractional_odds)
df["odds_prob"] = 1.0 / df["odds"]
df["odds_prob"] = df["odds_prob"].replace([np.inf, -np.inf], np.nan)
df["odds_prob"] = df["odds_prob"].fillna(df["odds_prob"].median())
df["odds_prob_model"] = df["odds_prob"]


# -----------------------------
# RECREATE race_id EXACTLY LIKE TRAINING
# -----------------------------
df["raceDate"] = pd.to_datetime(df["raceDate"])
df["race_id"] = (
    df["track"] + "_" +
    df["raceDate"].dt.strftime("%Y-%m-%d") + "_" +
    df["raceNumber"].astype(str)
)

# -----------------------------
# ENCODE CATEGORICALS (NO REFIT)
# -----------------------------
for col in CAT_COLS:
    le = encoders[col]
    df[col] = df[col].astype(str).map(
        lambda x: le.transform([x])[0] if x in le.classes_ else 0
    )

results = []

for race_id, race_df in df.groupby("race_id"):
    race_df = race_df.copy()

    # ---- Numeric features ----
    X_num_scaled = scaler.transform(race_df[NUMERIC_COLS])
    X_num = torch.tensor(
        np.hstack([X_num_scaled, race_df[["odds_prob_model"]].values]),
        dtype=torch.float32,
        device=device
    )

    # ---- Categorical features ----
    X_cat = torch.tensor(
        np.stack([race_df[col].values for col in CAT_COLS], axis=1),
        dtype=torch.long,
        device=device
    )

    # ---- Predict ----
    with torch.no_grad():
        scores = model(X_num, X_cat).cpu().numpy()

    race_df["score"] = scores
    race_df = race_df.sort_values("score", ascending=False)

    winner = race_df.iloc[0]

    # ---- Console summary ----
    print(f"{race_id} → {winner['name']} ({winner['score']:.4f})")

    # ---- Log file ----
    log_f.write("=" * 60 + "\n")
    log_f.write(f"RACE: {race_id}\n")
    log_f.write(f"Track: {winner['track']} | Race: {winner['raceNumber']}\n\n")

    for _, row in race_df.iterrows():
        marker = " <-- PICK" if row.name == winner.name else ""
        log_f.write(
            f"{row['programNumber']:>2} "
            f"{row['name']:<20} "
            f"Score: {row['score']:.4f}"
            f"{marker}\n"
        )

    log_f.write("\n")

    results.append({
        "race_id": race_id,
        "track": winner["track"],
        "raceNumber": winner["raceNumber"],
        "horse": winner["name"],
        "programNumber": winner["programNumber"],
        "score": float(winner["score"])
    })

log_f.close()
print(f"\nResults logged to: {log_path}")


# # -----------------------------
# # OUTPUT
# # -----------------------------
# results_df = pd.DataFrame(results).sort_values(["track", "raceNumber"])
# print("\nRecommended horses for upcoming races:\n")
# print(results_df)
