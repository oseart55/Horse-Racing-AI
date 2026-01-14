import requests
import pandas as pd
import torch
import torch.nn as nn
import pickle
from datetime import datetime

# ===============================
# CONFIG
# ===============================
DEVICE = "cpu"
DATA_DIR = "/data"

MODEL_PATH = f"{DATA_DIR}/ranking_mlp.pth"
SCALER_PATH = f"{DATA_DIR}/scaler.pkl"
ENCODERS_PATH = f"{DATA_DIR}/encoders.pkl"
FEATURE_COLS_PATH = f"{DATA_DIR}/feature_cols.pkl"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_2 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Mobile/15E148 Safari/604.1"
    )
}

# ===============================
# HELPERS
# ===============================
def parse_fractional_odds(val):
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str) and "/" in val:
        try:
            num, den = val.split("/")
            return float(num) / float(den) + 1.0
        except:
            return 0.0
    return 0.0

def to_furlongs(distance_str):
    if not isinstance(distance_str, str):
        return 0.0
    distance_str = distance_str.strip().upper()
    try:
        value, unit = distance_str.split()
        value = float(value)
        if unit == "Y":
            return value / 220
        elif unit == "M":
            return value * 8
        elif unit == "F":
            return value
        else:
            return 0.0
    except:
        return 0.0

# ===============================
# MODEL
# ===============================
class RankingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
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

# ===============================
# FETCHERS
# ===============================
def fetch_json(url):
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json() if r.status_code == 200 else {}

def fetch_lifetime_stats(bris_id):
    url = f"https://www.twinspires.com/api/bdsio/bds/getHorseStats?brisId={bris_id}"
    data = fetch_json(url)
    return data.get("lifetimeStats") or {}

# ===============================
# BUILD DATAFRAME
# ===============================
def build_dataframe(entries_url, race_url):
    entries = fetch_json(entries_url)
    race = fetch_json(race_url)

    if not entries or not race:
        return pd.DataFrame()  # return empty if no data

    distance = to_furlongs(race.get("distance"))
    surface_label = race.get("surfaceLabel")
    track_condition = race.get("surfaceCondition")
    current_year = datetime.now().year

    rows = []
    for entry in entries:
        if entry.get("scratched", True):
            continue  # skip scratched horses
        ltStats = fetch_lifetime_stats(entry.get("entryId"))
        row = {
            "programNumber": entry.get("programNumber"),
            "name": entry.get("name"),
            "distance": distance,
            "age": current_year - int(entry.get("yob")),
            "weight": entry.get("weight"),
            "speedPoints": entry.get("speedPoints"),
            "averagePaceE1": entry.get("averagePaceE1"),
            "averagePaceE2": entry.get("averagePaceE2"),
            "averagePaceLP": entry.get("averagePaceLP"),
            "averageSpeedLast3": entry.get("averageSpeedLast3"),
            "bestSpeedAtDistance": entry.get("bestSpeedAtDistance"),
            "averageClass": entry.get("averageClass"),
            "lastClass": entry.get("lastClass"),
            "primePower": entry.get("primePower"),
            "odds": parse_fractional_odds(entry.get("oddsTrend", {}).get("current", {}).get("oddsText")),
            "horseLtTrackStartCount": ltStats.get("horseLtTrackStartCount", 0),
            "horseLtTrackWinCount": ltStats.get("horseLtTrackWinCount", 0),
            "horseLtTrackPlacesCount": ltStats.get("horseLtTrackPlacesCount", 0),
            "horseLtTrackShowsCount": ltStats.get("horseLtTrackShowsCount", 0),
            "horseLtTrackQHStartCount": ltStats.get("horseLtTrackQHStartCount", 0),
            "horseLtTrackQHWinCount": ltStats.get("horseLtTrackQHWinCount", 0),
            "horseLtTrackQHPlacesCount": ltStats.get("horseLtTrackQHPlacesCount", 0),
            "horseLtTrackQHShowsCount": ltStats.get("horseLtTrackQHShowsCount", 0),
            "horseLtMudsloppyStartCount": ltStats.get("horseLtMudsloppyStartCount", 0),
            "horseLtMudsloppyWinCount": ltStats.get("horseLtMudsloppyWinCount", 0),
            "horseLtMudsloppyPlacesCount": ltStats.get("horseLtMudsloppyPlacesCount", 0),
            "horseLtMudsloppyShowsCount": ltStats.get("horseLtMudsloppyShowsCount", 0),
            "surfaceLabel": surface_label,
            "trackConditionLabel": track_condition,
            "equipment": entry.get("equipment", "NA"),
            "priorRunningStyle": entry.get("priorRunningStyle") or "NA"
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

# ===============================
# PREPROCESS
# ===============================
def preprocess_for_model(df, feature_cols, scaler, encoders):
    df = df.copy()

    # --- encode categoricals ---
    for col, encoder in encoders.items():
        if col not in df:
            df[col] = "NA"
        df[col] = df[col].fillna("NA")
        df[col] = encoder.transform(df[col])

    # --- ensure scaler columns exist ---
    for col in scaler.feature_names_in_:
        if col not in df:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- scale features ---
    df.loc[:, scaler.feature_names_in_] = scaler.transform(df[scaler.feature_names_in_])

    # --- select final features ---
    df = df[feature_cols]

    # --- convert to numeric tensor ---
    X = torch.tensor(df.apply(pd.to_numeric, errors='coerce').fillna(0).values, dtype=torch.float32)
    return X, df

# ===============================
# PREDICT
# ===============================
def predict_top3(entries_url, race_url):
    # load artifacts
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    with open(FEATURE_COLS_PATH, "rb") as f:
        feature_cols = pickle.load(f)

    model = RankingMLP(len(feature_cols))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    df = build_dataframe(entries_url, race_url)
    if df.empty:
        return pd.DataFrame(columns=["programNumber", "name"])

    # keep identifiers
    df_identifiers = df[["programNumber", "name"]].copy()

    X, _ = preprocess_for_model(df, feature_cols, scaler, encoders)

    with torch.no_grad():
        scores = model(X)
        top3_idx = torch.argsort(scores, descending=True)[:3]

    return df_identifiers.iloc[top3_idx]

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    entries_url = "https://www.twinspires.com/adw/todays-tracks/mvr/Thoroughbred/races/2/entries?affid=2800"
    race_url = "https://www.twinspires.com/adw/todays-tracks/mvr/Thoroughbred/races/2?affid=2800"

    print(predict_top3(entries_url, race_url))
