import requests
import xml.etree.ElementTree as ET
import html
import re
import pandas as pd
from datetime import datetime

# ======================================================
# FETCH RSS FROM URL
# ======================================================

def fetch_rss(url: str, timeout=10) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": "application/rss+xml,application/xml;q=0.9,*/*;q=0.8"
    }

    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    return resp.text


# ======================================================
# PARSE RSS
# ======================================================

def parse_equibase_rss(xml_text: str) -> list:
    root = ET.fromstring(xml_text)
    channel = root.find("channel")

    records = []

    for item in channel.findall("item"):
        pub_date_raw = item.findtext("pubDate")
        description = item.findtext("description")

        pub_date = datetime.strptime(
            pub_date_raw, "%a, %d %b %Y %H:%M:%S %Z"
        )

        description = html.unescape(description)

        lines = [l.strip() for l in description.split("<br/>") if l.strip()]

        for line in lines:
            records.append({
                "pub_date": pub_date,
                "raw_text": line
            })

    return records


# ======================================================
# NORMALIZE
# ======================================================

def normalize_change(raw: str) -> dict:
    data = {
        "race": None,
        "horse_number": None,
        "horse_name": None,
        "change_type": None,
        "detail": None
    }

    # ---------------------------
    # Race number
    # ---------------------------
    race_match = re.search(r"Race\s+(\d+):", raw)
    if race_match:
        data["race"] = int(race_match.group(1))

    # ---------------------------
    # SCRATCH
    # ---------------------------
    scratch_match = re.search(
        r"#\s*(\d+)\s+([^<]+).*Scratched.*-\s*(.+)",
        raw,
        re.IGNORECASE
    )
    if scratch_match:
        data.update({
            "horse_number": int(scratch_match.group(1)),
            "horse_name": scratch_match.group(2).strip(),
            "change_type": "SCRATCH",
            "detail": scratch_match.group(3).strip()
        })
        return data

    # ---------------------------
    # SCRATCH REASON UPDATE
    # ---------------------------
    reason_update = re.search(
        r"#\s*(\d+)\s+([^<]+).*Scratch Reason.*changed to\s*(.+)",
        raw,
        re.IGNORECASE
    )
    if reason_update:
        data.update({
            "horse_number": int(reason_update.group(1)),
            "horse_name": reason_update.group(2).strip(),
            "change_type": "SCRATCH_REASON_UPDATE",
            "detail": reason_update.group(3).strip()
        })
        return data

    # ---------------------------
    # TRACK CONDITION (Dirt / Turf / All Weather)
    # ---------------------------
    track_match = re.search(
        r"Current\s+(Dirt|Turf|All Weather)\s+Track\s+Condition\s+-\s*(.+)",
        raw,
        re.IGNORECASE
    )
    if track_match:
        data.update({
            "change_type": "TRACK_CONDITION",
            "detail": track_match.group(2).strip()
        })
        return data

    # ---------------------------
    # COURSE CHANGE
    # ---------------------------
    course_match = re.search(
        r"Course\s*-\s*(.+?)\s+changed to\s+(.+)",
        raw,
        re.IGNORECASE
    )
    if course_match:
        data.update({
            "change_type": "COURSE_CHANGE",
            "detail": f"{course_match.group(1)} → {course_match.group(2)}"
        })
        return data

    # ---------------------------
    # DISTANCE CHANGE
    # ---------------------------
    distance_match = re.search(
        r"Distance\s*-\s*(.+?)\s+changed to\s+(.+)",
        raw,
        re.IGNORECASE
    )
    if distance_match:
        data.update({
            "change_type": "DISTANCE_CHANGE",
            "detail": f"{distance_match.group(1)} → {distance_match.group(2)}"
        })
        return data

    # ---------------------------
    # TEMP RAIL DISTANCE
    # ---------------------------
    rail_match = re.search(
        r"Temp Rail Distance set at\s*(.+)",
        raw,
        re.IGNORECASE
    )
    if rail_match:
        data.update({
            "change_type": "TEMP_RAIL_DISTANCE",
            "detail": rail_match.group(1).strip()
        })
        return data

    # ---------------------------
    # FIRST START SINCE
    # ---------------------------
    layoff_match = re.search(
        r"#\s*(\d+)\s+([^<]+).*First Start Since\s*(.+)",
        raw,
        re.IGNORECASE
    )
    if layoff_match:
        data.update({
            "horse_number": int(layoff_match.group(1)),
            "horse_name": layoff_match.group(2).strip(),
            "change_type": "FIRST_START_SINCE",
            "detail": layoff_match.group(3).strip()
        })
        return data

    # ---------------------------
    # MEDICATION
    # ---------------------------
    medication_match = re.search(
        r"#\s*(\d+)\s+([^<]+).*Medication\s*-\s*(.+)",
        raw,
        re.IGNORECASE
    )
    if medication_match:
        data.update({
            "horse_number": int(medication_match.group(1)),
            "horse_name": medication_match.group(2).strip(),
            "change_type": "MEDICATION",
            "detail": medication_match.group(3).strip()
        })
        return data

    # ---------------------------
    # GLOBAL NOTE
    # ---------------------------
    if "NOTE:" in raw.upper():
        data.update({
            "change_type": "NOTE",
            "detail": re.sub(r"<.*?>", "", raw).strip()
        })
        return data

    # ---------------------------
    # FALLBACK
    # ---------------------------
    data["change_type"] = "UNKNOWN"
    data["detail"] = re.sub(r"<.*?>", "", raw).strip()
    return data

# ======================================================
# PUBLIC ENTRY
# ======================================================

def equibase_rss_url_to_dataframe(url: str) -> pd.DataFrame:
    xml_text = fetch_rss(url)
    parsed = parse_equibase_rss(xml_text)

    rows = []
    for entry in parsed:
        normalized = normalize_change(entry["raw_text"])
        normalized["pub_date"] = entry["pub_date"]
        rows.append(normalized)

    df = pd.DataFrame(rows)[
        [
            "pub_date",
            "race",
            "horse_number",
            "horse_name",
            "change_type",
            "detail",
        ]
    ]

    return df


# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":
    URL = "https://www.equibase.com/static/latechanges/rss/GP-USA.rss"

    df = equibase_rss_url_to_dataframe(URL)
    print(df)
