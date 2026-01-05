import requests
import csv
from datetime import datetime, timedelta
import time
import math
import os

# --- 1. Configuration ---
API_TOKEN = "232034-RBGwKswgRWx3yZ" 
SPORT_ID = "92"
LEAGUE_ID = "22742"
# Set to 350 for the full run, or a smaller number for a quick test.
DAYS_TO_FETCH = 350 
OUTPUT_FILE = "historical_odds_v7.1.csv"

# Maps the market type from the API key (e.g., the '1' in '92_1') to a readable name
MARKET_TYPE_MAP = {
    '1': 'Match Winner',
    '2': 'Handicap Sets',
    '3': 'Total Points (Over/Under)'
}

# --- 2. Startup and CSV Check ---
existing_ids = set()
file_exists = os.path.exists(OUTPUT_FILE)

if file_exists:
    print(f"File '{OUTPUT_FILE}' found. Reading existing match IDs...")
    with open(OUTPUT_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            for row in reader:
                if row: existing_ids.add(row[0])
        except StopIteration:
            pass # File is empty
    print(f"Found odds for {len(existing_ids)} existing matches. Resuming download.")

# --- 3. Main Script Logic ---
try:
    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        if not file_exists or os.path.getsize(OUTPUT_FILE) == 0:
            csv_writer.writerow([
                'Match_ID', 'Market_Name', 'Odds_Timestamp', 'Live_Score',
                'P1_Odds', 'P2_Odds', 'Handicap'
            ])

        today = datetime.now()
        for i in range(DAYS_TO_FETCH):
            current_date = today - timedelta(days=i)
            date_str = current_date.strftime('%Y%m%d')
            print(f"--- Processing Day: {current_date.strftime('%Y-%m-%d')} ---")

            # Get list of all match IDs for the day
            event_ids_to_fetch = []
            page, total_pages = 1, 1
            while page <= total_pages:
                list_url = f"https://api.betsapi.com/v1/events/ended?token={API_TOKEN}&sport_id={SPORT_ID}&league_id={LEAGUE_ID}&day={date_str}&page={page}"
                try:
                    response = requests.get(list_url)
                    response.raise_for_status()
                    data = response.json()
                    if data and data.get('success') == 1:
                        for event in data.get('results', []):
                            if event['id'] not in existing_ids:
                                event_ids_to_fetch.append(event['id'])
                        
                        if page == 1 and data.get('pager'):
                            total_results = data['pager'].get('total', 0)
                            if total_results > 0: total_pages = math.ceil(total_results / data['pager'].get('per_page', 50))
                            else: break
                    else: break
                except requests.exceptions.RequestException as e: print(f"Network error getting event list: {e}"); break
                page += 1
                time.sleep(2)

            if not event_ids_to_fetch:
                print("No new matches to fetch odds for this day.")
                continue

            print(f"Found {len(event_ids_to_fetch)} new matches. Fetching historical odds...")

            # Get detailed odds for each match ID
            for count, event_id in enumerate(event_ids_to_fetch):
                print(f"Fetching Odds for Match {count + 1}/{len(event_ids_to_fetch)} (ID: {event_id})")
                odds_url = f"https://api.betsapi.com/v2/event/odds?token={API_TOKEN}&event_id={event_id}"
                
                try:
                    response = requests.get(odds_url)
                    response.raise_for_status()
                    data = response.json()
                    
                    if data and data.get('success') == 1 and 'results' in data and 'odds' in data['results']:
                        # --- NEW: Parsing the corrected odds structure ---
                        for market_key, odds_history in data['results']['odds'].items():
                            market_type = market_key.split('_')[-1]
                            market_name = MARKET_TYPE_MAP.get(market_type, f"Unknown Market {market_type}")

                            for odds_snapshot in odds_history:
                                odds_time = datetime.fromtimestamp(int(odds_snapshot['add_time'])).strftime('%Y-%m-%d %H:%M:%S')
                                live_score = odds_snapshot.get('ss', 'N/A')
                                # Use 'over_od' and 'under_od' for Total Points market
                                p1_odds = odds_snapshot.get('home_od') or odds_snapshot.get('over_od')
                                p2_odds = odds_snapshot.get('away_od') or odds_snapshot.get('under_od')
                                handicap = odds_snapshot.get('handicap')

                                csv_writer.writerow([
                                    event_id, market_name, odds_time, live_score,
                                    p1_odds, p2_odds, handicap
                                ])
                    
                    existing_ids.add(event_id)
                    time.sleep(2)
                    
                except requests.exceptions.RequestException as e:
                    print(f"Network error on event {event_id}: {e}")
                    continue
                
    print(f"\n✅ Odds script finished successfully. Data saved to {OUTPUT_FILE}")
except IOError:
    print(f"Error: Could not write to the file {OUTPUT_FILE}.")