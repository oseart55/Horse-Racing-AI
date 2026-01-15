import os
import sqlite3
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
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
# UTILS
# =========================================================

def parse_fractional_odds(val):
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str) and "/" in val:
        try:
            num, den = val.split("/")
            return float(num)/float(den) + 1.0
        except:
            return None
    return None

def load_and_split_races(db_path, table_name, min_horses, include_odds, train_frac, seed, training=True):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()

    # Basic preprocessing
    if training:
        df = df[df["finishPosition"].notna()]
        df["top3"] = df["finishPosition"].isin([1,2,3]).astype(int)
    else:
        # future races: top3 placeholder
        df["top3"] = 0

    if include_odds:
        df["odds"] = df["odds"].apply(parse_fractional_odds)

    numeric_features = [
        "distance","age","weight","speedPoints","averagePaceE1",
        "averagePaceE2","averagePaceLP","averageSpeedLast3",
        "bestSpeedAtDistance","daysOff","averageClass","lastClass",
        "primePower","odds",
        "horseLtTrackStartCount","horseLtTrackWinCount","horseLtTrackPlacesCount","horseLtTrackShowsCount",
        "horseLtTrackQHStartCount","horseLtTrackQHWinCount","horseLtTrackQHPlacesCount","horseLtTrackQHShowsCount",
        "horseLtMudsloppyStartCount","horseLtMudsloppyWinCount","horseLtMudsloppyPlacesCount","horseLtMudsloppyShowsCount"
    ]

    categorical_features = [
        "surfaceLabel", "trackConditionLabel","equipment","priorRunningStyle"
    ]

    # Encode categoricals
    encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = df[col].fillna("NA")
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Fill numeric NaNs and scale
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median() if training else 0)  # fill zeros in eval

    scaler = StandardScaler()
    if training:
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
    else:
        df[numeric_features] = scaler.transform(df[numeric_features])

    feature_cols = numeric_features + categorical_features

    # Group into races
    races = [
        race_df.reset_index(drop=True)
        for _, race_df in df.groupby(["track","raceDate","raceNumber"])
        if len(race_df) >= min_horses
    ]

    if not races:
        raise ValueError("No valid races found")

    if training:
        # Shuffle & split
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(races), generator=g).tolist()
        split_idx = int(len(races) * train_frac)
        train_races = [races[i] for i in indices[:split_idx]]
        val_races = [races[i] for i in indices[split_idx:]]
        return train_races, val_races, feature_cols, scaler, encoders, numeric_features
    else:
        return races, feature_cols, scaler, encoders, numeric_features

# =========================================================
# LOSS
# =========================================================

def top3_listwise_loss(scores, top3_mask):
    log_probs = torch.log_softmax(scores, dim=0)
    mask_sum = top3_mask.sum()
    if mask_sum == 0:
        return torch.zeros((), device=scores.device)
    return -(top3_mask * log_probs).sum() / mask_sum

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

class HorseRaceRankingDataset(Dataset):
    def __init__(self, races, feature_cols):
        self.races = races
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.races)

    def __getitem__(self, idx):
        race_df = self.races[idx]
        X = race_df[self.feature_cols].values
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(race_df["top3"].values, dtype=torch.float32)
        return X, y

# =========================================================
# EVALUATION
# =========================================================

def eval_race(track, race_num, race_date):
    # -----------------------------
    # Load trained scaler, encoders, and feature_cols
    # -----------------------------
    scaler = pickle.load(open(SCALER_PATH,"rb"))
    encoders = pickle.load(open(ENCODERS_PATH,"rb"))
    feature_cols = pickle.load(open(FEATURE_COLS_PATH,"rb"))

    # -----------------------------
    # Load race data
    # -----------------------------
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()

    # Convert types for safe filtering
    df["track"] = df["track"].str.lower()
    df["raceNumber"] = df["raceNumber"].astype(int)
    df["raceDate"] = pd.to_datetime(df["raceDate"]).dt.date
    race_date_dt = datetime.strptime(race_date, "%Y-%m-%d").date()

    # Filter for the requested race
    df = df[(df["track"] == track.lower()) &
            (df["raceNumber"] == race_num) &
            (df["raceDate"] == race_date_dt)]

    if df.empty:
        raise ValueError(f"No race found for {track} race {race_num} on {race_date}")

    # -----------------------------
    # Preprocess numeric features
    # -----------------------------
    numeric_features = [col for col in feature_cols if col not in encoders]

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df[numeric_features] = scaler.transform(df[numeric_features])

    # -----------------------------
    # Preprocess categorical features
    # -----------------------------
    for col in encoders:
        df[col] = df[col].fillna("NA")
        df[col] = encoders[col].transform(df[col])

    # -----------------------------
    # Load model and predict
    # -----------------------------
    model = RankingMLP(len(feature_cols), HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    X = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        df["score"] = model(X).cpu().numpy()

    # -----------------------------
    # Sort by predicted score
    # -----------------------------
    df = df.sort_values("score", ascending=False)

    print(f"\n🏇 {track.upper()} RACE {race_num} on {race_date}")
    print(df[["name","score"]].to_string(index=False))

    top_pick = df.iloc[0]
    print("\nTOP PICK:", top_pick["name"])

# =========================================================
# TRAINING
# =========================================================

def train():
    train_races, val_races, feature_cols, scaler, encoders, numeric_features = load_and_split_races(
        DB_PATH, TABLE_NAME, MIN_HORSES_PER_RACE, INCLUDE_ODDS, TRAIN_FRAC, SEED, training=True
    )

    train_dataset = HorseRaceRankingDataset(train_races, feature_cols)
    val_dataset = HorseRaceRankingDataset(val_races, feature_cols)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = RankingMLP(input_dim=len(feature_cols), hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Train races: {len(train_dataset)}, Validation races: {len(val_dataset)}, Device: {DEVICE}")

    epoch_losses = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for X, y in train_loader:
            X = X.squeeze(0).to(DEVICE)
            y = y.squeeze(0).to(DEVICE)

            scores = model(X)
            loss = top3_listwise_loss(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        # Validation
        model.eval()
        hits = 0
        with torch.no_grad():
            for i in range(len(val_dataset)):
                X, y = val_dataset[i]
                scores = model(X.to(DEVICE))
                best_idx = torch.argmax(scores).item()
                if y[best_idx].item() == 1:
                    hits += 1
        val_acc = hits / len(val_dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val Top-3 Acc: {val_acc*100:.2f}%")

    # Validation
    model.eval()
    hits = 0
    with torch.no_grad():
        for i in range(len(val_dataset)):
            X, y = val_dataset[i]
            scores = model(X.to(DEVICE))
            best_idx = torch.argmax(scores).item()
            if y[best_idx].item() == 1:
                hits += 1

    acc = hits / len(val_dataset)
    print(f"\nVALIDATION Top-3 accuracy: {acc*100:.2f}%")

    # Save model and preprocessors
    os.makedirs("./data", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(SCALER_PATH,"wb") as f:
        pickle.dump(scaler, f)
    with open(ENCODERS_PATH,"wb") as f:
        pickle.dump(encoders, f)
    with open(FEATURE_COLS_PATH,"wb") as f:
        pickle.dump(feature_cols, f)

    print("\nSaved model, scaler, encoders, and feature_cols.")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true", help="Train the model")
    parser.add_argument("-e", "--eval", nargs=3, metavar=("TRACK","RACE","DATE"), help="Evaluate a single race")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.eval:
        track, race, date = args.eval
        eval_race(track, int(race), date)
    else:
        print("Please pass -t to train or -e TRACK RACE DATE to evaluate")
