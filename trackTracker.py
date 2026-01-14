import os
import time
import requests
from datetime import datetime, timezone
from rich.console import Console
from rich.table import Table
from rich.live import Live

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_2 like Mac OS X) "
                  "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Mobile/15E148 Safari/604.1"
}

API_URL = "https://www.twinspires.com/adw/todays-tracks?affid=2800"
REFRESH_INTERVAL = 30  # seconds

console = Console()

def fetch_tracks():
    try:
        data = requests.get(API_URL, headers=HEADERS, timeout=10)
        data.raise_for_status()
        return data.json()
    except Exception as e:
        console.print(f"[red]Error fetching data: {e}[/red]")
        return []

def get_usa_thoroughbred_tracks(data):
    return [
        track for track in data
        if track.get("hostCountry") == "USA" 
           and track.get("type") == "Thoroughbred"
           and "Simulcast" not in track.get("name", "")
    ]

def build_table(tracks):
    table = Table(title="USA Thoroughbred Tracks Live Races", expand=True)
    table.add_column("Track", style="cyan", no_wrap=True)
    table.add_column("Current Race", style="green")
    table.add_column("Next Race Time", style="magenta")
    table.add_column("Time Until Next Race", style="yellow")

    now = datetime.now(timezone.utc)

    for track in tracks:
        current_race_number = track.get("currentRaceNumber", 0)
        # find next race
        next_race = None
        for race in track.get("races", []):
            if race["raceNumber"] == current_race_number + 1:
                next_race = race
                break

        if next_race:
            post_time = datetime.fromisoformat(next_race["postTime"])
            time_until = post_time - now
            if time_until.total_seconds() < 0:
                time_until_str = "Starting soon!"
            else:
                minutes, seconds = divmod(int(time_until.total_seconds()), 60)
                hours, minutes = divmod(minutes, 60)
                time_until_str = f"{hours}h {minutes}m {seconds}s"
            next_race_time_str = post_time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            next_race_time_str = "N/A"
            time_until_str = "N/A"

        table.add_row(
            f"{track['name']} ({track['brisCode']})",
            str(current_race_number),
            next_race_time_str,
            time_until_str
        )

    return table

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    with Live(console=console, refresh_per_second=1) as live:
        while True:
            data = fetch_tracks()
            tracks = get_usa_thoroughbred_tracks(data)
            table = build_table(tracks)
            live.update(table)
            time.sleep(REFRESH_INTERVAL)

if __name__ == "__main__":
    main()