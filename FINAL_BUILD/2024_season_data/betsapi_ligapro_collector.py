import requests
import csv
from datetime import datetime, timedelta
import time
import math
import os # NEW: Imported the 'os' module to check for the file's existence
import pytz 

# --- 1. Configuration ---
API_TOKEN = "232034-RBGwKswgRWx3yZ" 
SPORT_ID = "92"
LEAGUE_ID = "22742"
# Set to 350 for the full run, or a smaller number for testing.
DAYS_TO_FETCH = 350 
OUTPUT_FILE = "czech_liga_pro_advanced_stats_FIXED.csv"

# --- 2. Advanced Stats Analysis Function (Unchanged) ---
def analyze_timeline(timeline_data, scores_data):
    stats = {
        'p1_total_points': 0, 'p2_total_points': 0,
        'p1_pressure_points': 0, 'p2_pressure_points': 0,
        'p1_comebacks': 0, 'p2_comebacks': 0
    }
    if not scores_data: return stats
    for set_num, set_scores in scores_data.items():
        # --- FIX APPLIED HERE ---
        stats['p1_total_points'] += int(float(set_scores.get('home', 0)))
        stats['p2_total_points'] += int(float(set_scores.get('away', 0)))
    if not timeline_data: return stats
    sets = {}
    for point in timeline_data:
        set_num = point.get('gm')
        if set_num not in sets: sets[set_num] = []
        sets[set_num].append(point['ss'])
    for set_num, points in sets.items():
        p1_max_deficit, p2_max_deficit = 0, 0
        prev_p1_score, prev_p2_score = 0, 0
        for score_str in points:
            try:
                p1_score, p2_score = map(int, score_str.split('-'))
                p1_max_deficit = max(p1_max_deficit, p2_score - p1_score)
                p2_max_deficit = max(p2_max_deficit, p1_score - p2_score)
                if prev_p1_score >= 9 and prev_p2_score >= 9 and prev_p1_score == prev_p2_score:
                    if p1_score > prev_p1_score: stats['p1_pressure_points'] += 1
                    elif p2_score > prev_p2_score: stats['p2_pressure_points'] += 1
                prev_p1_score, prev_p2_score = p1_score, p2_score
            except (ValueError, IndexError): continue
        set_info = scores_data.get(str(set_num))
        if set_info:
            # --- FIX APPLIED HERE ---
            set_winner_p1 = int(float(set_info.get('home', 0))) > int(float(set_info.get('away', 0)))
            if set_winner_p1 and p1_max_deficit >= 4: stats['p1_comebacks'] += 1
            elif not set_winner_p1 and p2_max_deficit >= 4: stats['p2_comebacks'] += 1
    return stats

# --- 3. Startup and CSV Check ---
existing_ids = set()
file_exists = os.path.exists(OUTPUT_FILE)

if file_exists:
    print(f"File '{OUTPUT_FILE}' found. Reading existing match IDs...")
    with open(OUTPUT_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader) # Skip header
        for row in reader:
            if row: # Make sure row is not empty
                existing_ids.add(row[0])
    print(f"Found {len(existing_ids)} existing matches. Resuming download.")

# --- 4. Main Script Logic ---
try:
    # --- NEW: Open file in 'a' (append) mode ---
    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # --- NEW: Only write the header if the file is new ---
        if not file_exists:
            csv_writer.writerow([
                'Match ID', 'Date', 'Time', 'Player 1 ID', 'Player 1', 'Player 2 ID', 'Player 2', 
                'Final Score', 'Set Scores', 'Match Format', 'P1 Total Points', 'P2 Total Points',
                'P1 Pressure Points', 'P2 Pressure Points', 'P1 Set Comebacks', 'P2 Set Comebacks'
            ])

        today = datetime.now()
        for i in range(DAYS_TO_FETCH):
            current_date = today - timedelta(days=i+60)
            date_str = current_date.strftime('%Y%m%d')
            print(f"--- Processing Day: {current_date.strftime('%Y-%m-%d')} ---")

            event_ids = []
            page, total_pages = 1, 1
            while page <= total_pages:
                list_url = f"https://api.betsapi.com/v1/events/ended?token={API_TOKEN}&sport_id={SPORT_ID}&league_id={LEAGUE_ID}&day={date_str}&page={page}"
                try:
                    response = requests.get(list_url)
                    response.raise_for_status()
                    data = response.json()
                    if data and data.get('success') == 1:
                        # --- NEW: Check for duplicates before adding to the list to fetch ---
                        for event in data.get('results', []):
                            if event['id'] not in existing_ids:
                                event_ids.append(event['id'])
                        
                        if page == 1 and data.get('pager'):
                            total_results, per_page = (data['pager'].get('total', 0), data['pager'].get('per_page', 50))
                            if total_results > 0: total_pages = math.ceil(total_results / per_page)
                            else: break
                    else: break
                except requests.exceptions.RequestException as e: print(f"Network error getting event list: {e}"); break
                page += 1
                time.sleep(2)

            if not event_ids:
                print("No new matches to fetch for this day.")
                continue

            print(f"Found {len(event_ids)} new matches to fetch for the day...")

            for count, event_id in enumerate(event_ids):
                print(f"Fetching Match {count + 1}/{len(event_ids)} (ID: {event_id})")
                detail_url = f"https://api.betsapi.com/v1/event/view?token={API_TOKEN}&event_id={event_id}"
                try:
                    response = requests.get(detail_url)
                    response.raise_for_status()
                    data = response.json()
                    if data and data.get('success') == 1 and data.get('results'):
                        event = data['results'][0]
                        
                        # Extract all data points
                        match_id, player1_id, player2_id = (event.get('id'), event.get('home', {}).get('id'), event.get('away', {}).get('id'))
                        match_format = event.get('extra', {}).get('bestofsets')
                        match_time_unix = event.get('time')
                        match_time = datetime.fromtimestamp(int(match_time_unix)).strftime('%H:%M:%S') if match_time_unix else 'N/A'
                        player1, player2 = (event.get('home', {}).get('name'), event.get('away', {}).get('name'))
                        final_score = f'="{event.get("ss", "N/A")}"'
                        
                        scores_data = event.get('scores', {})
                        set_scores_list = []
                        if isinstance(scores_data, dict):
                            for set_num in sorted(scores_data.keys()):
                                set_scores_list.append(f"{scores_data[set_num].get('home', '?')}-{scores_data[set_num].get('away', '?')}")
                        set_scores_str = ", ".join(set_scores_list) if set_scores_list else "N/A"
                        
                        timeline_data = event.get('timeline', [])
                        advanced_stats = analyze_timeline(timeline_data, scores_data)

                        # Write the row
                        csv_writer.writerow([
                            match_id, current_date.strftime('%Y-%m-%d'), match_time, player1_id, player1, player2_id, player2,
                            final_score, set_scores_str, match_format, advanced_stats['p1_total_points'], advanced_stats['p2_total_points'],
                            advanced_stats['p1_pressure_points'], advanced_stats['p2_pressure_points'], 
                            advanced_stats['p1_comebacks'], advanced_stats['p2_comebacks']
                        ])
                        
                        # --- NEW: Add the new ID to our set to prevent duplicates in the same session ---
                        existing_ids.add(match_id)
                        
                    time.sleep(2)
                except requests.exceptions.RequestException as e: print(f"Network error on event {event_id}: {e}"); continue
                
    print(f"\n✅ Script finished successfully. Data saved to {OUTPUT_FILE}")
except IOError:
    print(f"Error: Could not write to the file {OUTPUT_FILE}.")