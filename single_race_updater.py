import requests, re
from fractions import Fraction
import sqlite3

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_2 like Mac OS X) "
                  "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Mobile/15E148 Safari/604.1"
}
DB_FILE = "horses.db"
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()

UPDATE_COLUMNS = [
    "name", "postPosition", "programNumber", "distance", "surfaceLabel", "trackConditionLabel",
    "odds", "oddsRank", "scratched", "finishPosition", "equipment", "medication",
    "weight", "jockey", "trainer", "owner",
    "priorRunningStyle", "speedPoints",
    "averagePaceE1", "averagePaceE2", "averagePaceLP",
    "averageSpeedLast3", "bestSpeedAtDistance", "daysOff",
    "averageClass", "lastClass", "primePower",
    "horseLtTrackStartCount", "horseLtTrackWinCount",
    "horseLtTrackPlacesCount", "horseLtTrackShowsCount",
    "horseLtTrackQHStartCount", "horseLtTrackQHWinCount",
    "horseLtTrackQHPlacesCount", "horseLtTrackQHShowsCount",
    "horseLtMudsloppyStartCount", "horseLtMudsloppyWinCount",
    "horseLtMudsloppyPlacesCount", "horseLtMudsloppyShowsCount",
    "finishPosition"
]

update_clause = ",\n    ".join(f"{col} = excluded.{col}" for col in UPDATE_COLUMNS)

def get_surface_label(pre_race_data):   
    return pre_race_data.get("surfaceLabel")

def get_track_condition_label(pre_race_data):   
    return pre_race_data.get("surfaceCondition")

def get_json(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return {}
        return r.json() or {}
    except Exception:
        return {}

def distance_str_to_units(distance: str) -> float:
    if not distance or not isinstance(distance, str):
        return None

    s = distance.upper().strip()
    furlongs = 0.0

    # --- Miles ---
    mile_match = re.search(r'([\d\s/]+)\s*M', s)
    if mile_match:
        miles = sum(Fraction(p) for p in mile_match.group(1).split())
        furlongs += float(miles * 8)

    # --- Furlongs ---
    furlong_match = re.search(r'([\d\s/]+)\s*F', s)
    if furlong_match:
        furlongs += float(sum(Fraction(p) for p in furlong_match.group(1).split()))

    # --- Yards ---
    yard_match = re.search(r'(\d+)\s*Y', s)
    if yard_match:
        furlongs += int(yard_match.group(1)) / 220.0

    # 🔥 Return "long form" units (furlongs × 100)
    return round(furlongs * 100, 2)

def updateTableForRace(trackCode: str, raceNumber: int):
    race_date = get_json(f"https://www.twinspires.com/adw/ami/wager/racedate_string?affid=2000").get("raceDate")
    pre_race_data = get_json(f"https://www.twinspires.com/adw/todays-tracks/{trackCode}/Thoroughbred/races/{raceNumber}?affid=2800")
    surfaceCondition = get_surface_label(pre_race_data)
    trackConditionLabel = get_track_condition_label(pre_race_data)
    distance = distance_str_to_units(pre_race_data.get("distance")) / 100

    url = f"https://www.twinspires.com/adw/todays-tracks/{trackCode}/Thoroughbred/races/{raceNumber}/entries?affid=2000"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        raise ValueError("Cannot get data")
    data = resp.json()
    race_year = int(race_date.split("-")[0])
    horses = []
    for h in data:
        print(h.get("name"))
        brisId = h.get("entryId")
        stats = get_json(f"https://www.twinspires.com/api/bdsio/bds/getHorseStats?brisId={brisId}")
        lt = stats.get("lifetimeStats") or {}
        raw_age = lt.get("age")
        age = race_year - int(h["yob"]) if raw_age is None and h.get("yob") else raw_age
        
        horses.append({
            "brisId": brisId,
            "name": h.get("name"),
            "postPosition": h.get("postPosition"),
            "programNumber": h.get("programNumber"),
            "distance": distance,
            "surfaceLabel": surfaceCondition,
            "trackConditionLabel": trackConditionLabel,
            "odds": h.get("liveOdds"),
            "oddsRank": h.get("oddsRank"),
            "scratched": int(h.get("scratched", False)),
            "finishPosition": None,
            "age": age,
            "yob": h.get("yob"),
            "sex": h.get("sex"),
            "equipment": h.get("equipment"),
            "medication": h.get("medication"),
            "weight": h.get("weight"),
            "jockey": h.get("jockeyName"),
            "trainer": h.get("trainerName"),
            "owner": h.get("ownerName"),
            "sire": h.get("sireName"),
            "dam": h.get("damName"),
            "priorRunningStyle": h.get("priorRunStyle") or None,
            "speedPoints": h.get("speedPoints"),
            "averagePaceE1": h.get("averagePaceE1"),
            "averagePaceE2": h.get("averagePaceE2"),
            "averagePaceLP": h.get("averagePaceLP"),
            "averageSpeedLast3": h.get("averageSpeedLast3"),
            "bestSpeedAtDistance": h.get("bestSpeedAtDistance"),
            "daysOff": h.get("daysOff"),
            "averageClass": h.get("averageClass"),
            "lastClass": h.get("lastClass"),
            "primePower": h.get("primePower"),
            "horseLtTrackStartCount": lt.get("horseLtTrackStartCount", 0),
            "horseLtTrackWinCount": lt.get("horseLtTrackWinCount", 0),
            "horseLtTrackPlacesCount": lt.get("horseLtTrackPlacesCount", 0),
            "horseLtTrackShowsCount": lt.get("horseLtTrackShowsCount", 0),
            "horseLtTrackQHStartCount": lt.get("horseLtTrackQHStartCount", 0),
            "horseLtTrackQHWinCount": lt.get("horseLtTrackQHWinCount", 0),
            "horseLtTrackQHPlacesCount": lt.get("horseLtTrackQHPlacesCount", 0),
            "horseLtTrackQHShowsCount": lt.get("horseLtTrackQHShowsCount", 0),
            "horseLtMudsloppyStartCount": lt.get("horseLtMudsloppyStartCount", 0),
            "horseLtMudsloppyWinCount": lt.get("horseLtMudsloppyWinCount", 0),
            "horseLtMudsloppyPlacesCount": lt.get("horseLtMudsloppyPlacesCount", 0),
            "horseLtMudsloppyShowsCount": lt.get("horseLtMudsloppyShowsCount", 0),
            "raceDate": race_date,
            "track": trackCode,
            "raceNumber": raceNumber,
        })
    if horses:
            UPSERT_SQL = f"""
            INSERT INTO horses (
                brisId, name, postPosition, programNumber, distance, surfaceLabel, trackConditionLabel, odds, oddsRank,
                scratched, finishPosition, age, yob, sex, equipment, medication, weight,
                jockey, trainer, owner, sire, dam,
                priorRunningStyle, speedPoints,
                averagePaceE1, averagePaceE2, averagePaceLP,
                averageSpeedLast3, bestSpeedAtDistance, daysOff,
                averageClass, lastClass, primePower,
                horseLtTrackStartCount, horseLtTrackWinCount,
                horseLtTrackPlacesCount, horseLtTrackShowsCount,
                horseLtTrackQHStartCount, horseLtTrackQHWinCount,
                horseLtTrackQHPlacesCount, horseLtTrackQHShowsCount,
                horseLtMudsloppyStartCount, horseLtMudsloppyWinCount,
                horseLtMudsloppyPlacesCount, horseLtMudsloppyShowsCount,
                raceDate, track, raceNumber
            )
            VALUES (
                :brisId, :name, :postPosition, :programNumber, :distance, :surfaceLabel, :trackConditionLabel, :odds, :oddsRank,
                :scratched, :finishPosition, :age, :yob, :sex, :equipment, :medication, :weight,
                :jockey, :trainer, :owner,
                :sire, :dam,
                :priorRunningStyle, :speedPoints,
                :averagePaceE1, :averagePaceE2, :averagePaceLP,
                :averageSpeedLast3, :bestSpeedAtDistance, :daysOff,
                :averageClass, :lastClass, :primePower,
                :horseLtTrackStartCount, :horseLtTrackWinCount,
                :horseLtTrackPlacesCount, :horseLtTrackShowsCount,
                :horseLtTrackQHStartCount, :horseLtTrackQHWinCount,
                :horseLtTrackQHPlacesCount, :horseLtTrackQHShowsCount,
                :horseLtMudsloppyStartCount, :horseLtMudsloppyWinCount,
                :horseLtMudsloppyPlacesCount, :horseLtMudsloppyShowsCount,
                :raceDate, :track, :raceNumber
            )
            ON CONFLICT (brisId, raceDate, track, raceNumber)
            DO UPDATE SET
                {update_clause},
                lastUpdated = CURRENT_TIMESTAMP
            """
            cur.executemany(UPSERT_SQL, horses)
            conn.commit()
    conn.close()