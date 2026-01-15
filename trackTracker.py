import os
import time
import requests
from datetime import datetime, timezone, date
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from plyer import notification  # pip install plyer

# =========================
# CONFIG
# =========================
API_URL = "https://www.twinspires.com/adw/todays-tracks?affid=2800"
REFRESH_INTERVAL = 30  # seconds
HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_2 like Mac OS X) "
                  "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 "
                  "Mobile/15E148 Safari/604.1"
}

console = Console()
alerted_races = set()

# =========================
# FETCH API
# =========================
def fetch_tracks():
    try:
        response = requests.get(API_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        console.print(f"[red]Error fetching data: {e}[/red]")
        return []

# =========================
# FILTER TRACKS
# =========================
def get_usa_thoroughbred_tracks(data):
    return [
        track for track in data
        if track.get("hostCountry") == "USA"
        and track.get("type") == "Thoroughbred"
        and "Simulcast" not in track.get("name", "")
        and "eq" not in track.get("brisCode", "").lower()
        and "Pick 5" not in track.get("name", "")
        and "Pick 6" not in track.get("name", "")
    ]

# =========================
# ALERT CHECK
# =========================
def check_for_alert(track_name, race_number, post_time):
    now = datetime.now(timezone.utc)
    delta = post_time - now
    seconds_until = delta.total_seconds()

    race_id = f"{track_name}|{race_number}|{post_time.isoformat()}"
    if 0 < seconds_until <= 5 * 60 and race_id not in alerted_races:
        alerted_races.add(race_id)
        notification.notify(
            title=f"Upcoming Race: {track_name}",
            message=f"Race {race_number} starts in {int(seconds_until // 60)}m {int(seconds_until % 60)}s",
            timeout=10
        )

# =========================
# BUILD LIVE TABLE
# =========================
def build_table(tracks):
    table = Table(title="USA Thoroughbred Tracks – Live Next Race", expand=True)
    table.add_column("Track", style="cyan", no_wrap=True)
    table.add_column("Next Race", style="green")
    table.add_column("Post Time (UTC)", style="magenta")
    table.add_column("Time Until Post", style="yellow")

    now = datetime.now(timezone.utc)

    for track in tracks:
        races = track.get("races", [])
        future_races = []

        for race in races:
            post_str = race.get("postTime")
            if not post_str:
                continue
            try:
                post_time = datetime.fromisoformat(post_str)
                if post_time.tzinfo is None:
                    post_time = post_time.replace(tzinfo=timezone.utc)
            except:
                continue

            if post_time > now:
                future_races.append((race, post_time))

        if future_races:
            next_race, post_time = min(future_races, key=lambda x: x[1])
            check_for_alert(track.get("name"), next_race.get("raceNumber"), post_time)

            delta = post_time - now
            total_seconds = int(delta.total_seconds())
            hours, rem = divmod(total_seconds, 3600)
            mins, secs = divmod(rem, 60)
            time_str = f"{hours}h {mins}m {secs}s"

            if total_seconds <= 5 * 60:
                time_str = f"[bold red]{time_str}[/bold red]"
                style = "bold red"
            else:
                style = None

            table.add_row(
                f"{track.get('name')} ({track.get('brisCode')})",
                f"Race {next_race.get('raceNumber')}",
                post_time.strftime("%Y-%m-%d %H:%M:%S"),
                time_str,
                style=style
            )
        else:
            table.add_row(
                f"{track.get('name')} ({track.get('brisCode')})",
                "All races completed",
                "N/A",
                "N/A"
            )
    return table

# =========================
# MAIN LOOP
# =========================
def main():
    os.system("cls" if os.name == "nt" else "clear")
    last_fetch = 0
    cached_tracks = []

    with Live(console=console, refresh_per_second=1, screen=False) as live:
        while True:
            now_ts = time.time()
            if now_ts - last_fetch >= REFRESH_INTERVAL:
                data = fetch_tracks()
                cached_tracks = get_usa_thoroughbred_tracks(data)
                last_fetch = now_ts

            table = build_table(cached_tracks)

            # Combine both table and results into a single renderable
            from rich.console import Group
            group = Group(
                table,
            )
            live.update(group, refresh=True)

            time.sleep(1)

if __name__ == "__main__":
    main()
