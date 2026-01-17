from datetime import date
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
import requests
import pandas as pd

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_2 like Mac OS X) "
                  "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 "
                  "Mobile/15E148 Safari/604.1"
}
def first_horse(pool_key):
    pools = race_details.get(pool_key)
    if isinstance(pools, list) and len(pools) > 0:
        return pools[0].get("horseName")
    return None

def getResults(brisCode: str, raceNumber: int):
    today_str = date.today().strftime("%Y-%m-%d")
    resultsUrl = f"https://www.twinspires.com/api/raceresults/results/{today_str}/{brisCode}/Thoroughbred/{raceNumber}"
    results = requests.get(resultsUrl, headers=HEADERS)
    if results.status_code == 200:
        return results.json()


df = pd.read_excel("predictions.xlsx")
for idx, row in df[~df['Correct'].isin(['Y', 'N'])].iterrows():
    raceResults = getResults(row["Track"], row['Race Number'])
    if not raceResults:
        continue
    
    race_details = raceResults.get("raceDetails") or {}

    winPool   = first_horse("winPools")
    placePool = first_horse("placePools")
    showPool  = first_horse("showPools")

    # Check if Prediction matches any of them
    if row["Prediction"] in [winPool, placePool, showPool]:
        df.at[idx, "Correct"] = "Y"
    else:
        df.at[idx, "Correct"] = "F"



