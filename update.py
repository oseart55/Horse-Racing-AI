import requests
import sqlite3
import time

# -----------------------------
# CONSTANTS
# -----------------------------
TRACKS = []  # list of tracks
AFFID = "2800"
DB_FILE = "horses.db"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_2 like Mac OS X) "
                  "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Mobile/15E148 Safari/604.1"
}

# -----------------------------
# GET ALL TRACKS RUNNING RACES TODAY
# -----------------------------
IGNORE_TRACKS = {
    # International / foreign
    "swl",   # Southwell (UK)
    # Equibase internal / synthetic / placeholder
    "eqk",
    "eqb",
    "eqz",
    "la",
    # Simulcast / composite / non-track feeds
    "ccp",
}
tracks = requests.get("https://www.twinspires.com/adw/todays-tracks?affid=2800", headers=HEADERS)
if tracks.status_code == 200:
    tracks = tracks.json()
    for track in tracks:
        if track.get("type") == "Thoroughbred" and track.get("hostCountry") == "USA" and track.get("brisCode") not in IGNORE_TRACKS:
            TRACKS.append(track.get("brisCode"))
# -----------------------------
# UPSERT UPDATE COLUMNS
# -----------------------------
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

# -----------------------------
# HELPERS
# -----------------------------
def get_json(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return {}
        return r.json() or {}
    except Exception:
        return {}

# -----------------------------
# PROCESS TRACK SEQUENTIALLY
# -----------------------------
horses_updated = 0
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()

# Create table if missing
cur.execute("""CREATE TABLE IF NOT EXISTS horses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brisId INTEGER,
    name TEXT,
    postPosition INTEGER,
    programNumber TEXT,
    distance INTEGER,
    surfaceLabel TEXT,
    trackConditionLabel TEXT,
    odds REAL,
    oddsRank INTEGER,
    scratched INTEGER,
    finishPosition INTEGER,
    age INTEGER,
    yob INTEGER,
    sex TEXT,
    equipment TEXT,
    medication TEXT,
    weight INTEGER,
    jockey TEXT,
    trainer TEXT,
    owner TEXT,
    sire TEXT,
    dam TEXT,
    priorRunningStyle TEXT,
    speedPoints INTEGER,
    averagePaceE1 REAL,
    averagePaceE2 REAL,
    averagePaceLP REAL,
    averageSpeedLast3 REAL,
    bestSpeedAtDistance REAL,
    daysOff INTEGER,
    averageClass REAL,
    lastClass REAL,
    primePower REAL,
    horseLtTrackStartCount INTEGER,
    horseLtTrackWinCount INTEGER,
    horseLtTrackPlacesCount INTEGER,
    horseLtTrackShowsCount INTEGER,
    horseLtTrackQHStartCount INTEGER,
    horseLtTrackQHWinCount INTEGER,
    horseLtTrackQHPlacesCount INTEGER,
    horseLtTrackQHShowsCount INTEGER,
    horseLtMudsloppyStartCount INTEGER,
    horseLtMudsloppyWinCount INTEGER,
    horseLtMudsloppyPlacesCount INTEGER,
    horseLtMudsloppyShowsCount INTEGER,
    raceDate TEXT,
    track TEXT,
    raceNumber INTEGER,
    lastUpdated TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (brisId, raceDate, track, raceNumber)
)""")
conn.commit()

race_date = get_json(f"https://www.twinspires.com/adw/ami/wager/racedate_string?affid={AFFID}").get("raceDate")
for track in TRACKS:
    print(track)
    if not race_date:
        print(f"No race date for {track}")
        continue

    races = get_json(f"https://www.twinspires.com/adw/todays-tracks/{track}/Thoroughbred/races?affid={AFFID}")
    race_count = len(races)
    if race_count == 0:
        continue

    race_year = int(race_date.split("-")[0])
    print(f"{track} | {race_date} | {race_count} races")

    for race_num in range(1, race_count+1):
        entries_url = f"https://www.twinspires.com/adw/todays-tracks/{track}/Thoroughbred/races/{race_num}/entries?affid={AFFID}"
        race_results_url = f"https://www.twinspires.com/api/raceresults/results/{race_date}/{track}/Thoroughbred/{race_num}"

        entries = get_json(entries_url)
        race_results = get_json(race_results_url)
        race_details = race_results.get("raceDetails") or {}
        distance = race_details.get("distance")
        if distance is not None:
            distance = distance / 100
        finish_map = {fin.get("brisId"): fin.get("finishPosition")
                      for fin in race_results.get("raceDetails", {}).get("finishOrder", [])
                      if fin.get("brisId") and fin.get("finishPosition")}

        horses = []
        for h in entries:
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
                "surfaceLabel": race_results.get("raceDetails", {}).get("surfaceLabel"),
                "trackConditionLabel": race_results.get("raceDetails", {}).get("trackConditionLabel"),
                "odds": h.get("liveOdds"),
                "oddsRank": h.get("oddsRank"),
                "scratched": int(h.get("scratched", False)),
                "finishPosition": finish_map.get(brisId),
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
                "priorRunningStyle": h.get("priorRunStyle") or "NA",
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
                "track": track,
                "raceNumber": race_num,
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
            horses_updated += len(horses)
            print(f"{track} Race {race_num}: {len(horses)} horses upserted")

conn.close()
print(f"Total horses updated: {horses_updated}")
