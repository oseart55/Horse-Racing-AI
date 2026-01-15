import os
import sqlite3
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================

DB_PATH = "horses.db"
TABLE_NAME = "horses"
MIN_HORSES_PER_RACE = 4
INCLUDE_ODDS = False

EPOCHS = 10
LR = 1e-3
HIDDEN_DIM = 128
TRAIN_FRAC = 0.8
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "data/ranking_mlp.pth"
SCALER_PATH = "data/scaler.pkl"
ENCODERS_PATH = "data/encoders.pkl"
FEATURE_COLS_PATH = "data/feature_cols.pkl"

# =========================================================
# DATA LOADING
# =========================================================

def load_and_split_races(db_path, table_name, min_horses, train_frac, seed):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()

    df = df[df["finishPosition"].notna()]
    df["top3"] = df["finishPosition"].isin([1, 2, 3]).astype(int)

    numeric_features = [
        "distance","age","weight","speedPoints","averagePaceE1",
        "averagePaceE2","averagePaceLP","averageSpeedLast3",
        "bestSpeedAtDistance","daysOff","averageClass","lastClass",
        "primePower"
    ]

    categorical_features = [
        "surfaceLabel","trackConditionLabel","equipment","priorRunningStyle"
    ]

    encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna("NA"))
        encoders[col] = le

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    feature_cols = numeric_features + categorical_features

    races = [
        r.reset_index(drop=True)
        for _, r in df.groupby(["track","raceDate","raceNumber"])
        if len(r) >= min_horses
    ]

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(races), generator=g).tolist()
    split = int(len(idx) * train_frac)

    return (
        [races[i] for i in idx[:split]],
        [races[i] for i in idx[split:]],
        feature_cols,
        scaler,
        encoders
    )

# =========================================================
# MODEL
# =========================================================

class RankingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# =========================================================
# DATASET
# =========================================================

class HorseRaceDataset(Dataset):
    def __init__(self, races, feature_cols):
        self.races = races
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.races)

    def __getitem__(self, idx):
        r = self.races[idx]
        X = torch.tensor(r[self.feature_cols].values, dtype=torch.float32)
        y = torch.tensor(r["top3"].values, dtype=torch.float32)
        return X, y

# =========================================================
# TRAINING
# =========================================================

def train():
    train_races, val_races, feature_cols, scaler, encoders = load_and_split_races(
        DB_PATH, TABLE_NAME, MIN_HORSES_PER_RACE, TRAIN_FRAC, SEED
    )

    train_loader = DataLoader(HorseRaceDataset(train_races, feature_cols), batch_size=1, shuffle=True)
    val_dataset = HorseRaceDataset(val_races, feature_cols)

    model = RankingMLP(len(feature_cols), HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            X, y = X.squeeze(0).to(DEVICE), y.squeeze(0).to(DEVICE)

            logits = model(X)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation: did best-prob horse place top-3?
        model.eval()
        hits = 0
        with torch.no_grad():
            for X, y in val_dataset:
                logits = model(X.to(DEVICE))
                best = torch.argmax(logits).item()
                hits += int(y[best].item() == 1)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss {total_loss:.4f} | Val Acc {hits/len(val_dataset):.2%}")

    os.makedirs("data", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    pickle.dump(scaler, open(SCALER_PATH,"wb"))
    pickle.dump(encoders, open(ENCODERS_PATH,"wb"))
    pickle.dump(feature_cols, open(FEATURE_COLS_PATH,"wb"))

# =========================================================
# EVALUATION
# =========================================================

def eval_race(track, race_num, race_date):
    model = RankingMLP(
        input_dim=len(pickle.load(open(FEATURE_COLS_PATH,"rb"))),
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    scaler = pickle.load(open(SCALER_PATH,"rb"))
    encoders = pickle.load(open(ENCODERS_PATH,"rb"))
    feature_cols = pickle.load(open(FEATURE_COLS_PATH,"rb"))

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()

    df = df[
        (df["track"].str.lower() == track.lower()) &
        (df["raceNumber"] == race_num) &
        (pd.to_datetime(df["raceDate"]).dt.date == datetime.strptime(race_date,"%Y-%m-%d").date()) &
        (df["scratched"] != 1)
    ]

    for col, le in encoders.items():
        df[col] = le.transform(df[col].fillna("NA"))

    numeric_cols = [c for c in feature_cols if c not in encoders]
    df[numeric_cols] = scaler.transform(df[numeric_cols].fillna(0))

    X = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        df["prob_top3"] = torch.sigmoid(model(X)).cpu().numpy()

    df = df.sort_values("prob_top3", ascending=False)
    print(df[["name","prob_top3"]])

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-t", action="store_true")
    p.add_argument("-e", nargs=3)
    args = p.parse_args()

    if args.t:
        train()
    elif args.e:
        eval_race(args.e[0], int(args.e[1]), args.e[2])
