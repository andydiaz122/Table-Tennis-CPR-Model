import requests
import csv
from datetime import datetime, timedelta
import time
import math
import os
import pytz

# --- 1. Configuration ---
API_TOKEN = "232034-RBGwKswgRWx3yZ"
SPORT_ID = "92"
LEAGUE_ID = "22742"
DAYS_TO_FETCH = 1 # Set to 350 for the full run, or a smaller number for testing.
OUTPUT_FILE = "czech_liga_pro_advanced_stats_FIXED.csv"

# Define headers as a constant list for writing the CSV with DictWriter
HEADERS = [
    'Match ID', 'Date', 'Time', 'Player 1 ID', 'Player 1', 'Player 2 ID', 'Player 2',
    'Final Score', 'Set Scores', 'Match Format', 'P1 Total Points', 'P2 Total Points',
    'P1 Pressure Points', 'P2 Pressure Points', 'P1 Set Comebacks', 'P2 Set Comebacks'
]

# --- 2. Advanced Stats Analysis Function (Unchanged) ---
def analyze_timeline(timeline_data, scores_data):
    stats = {
        'p1_total_points': 0, 'p2_total_points': 0,
        'p1_pressure_points': 0, 'p2_pressure_points': 0,
        'p1_comebacks': 0, 'p2_comebacks': 0
    }
    if not scores_data: return stats
    for set_num, set_scores in scores_data.items():
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
            set_winner_p1 = int(float(set_info.get('home', 0))) > int(float(set_info.get('away', 0)))
            if set_winner_p1 and p1_max_deficit >= 4: stats['p1_comebacks'] += 1
            elif not set_winner_p1 and p2_max_deficit >= 4: stats['p2_comebacks'] += 1
    return stats

# --- 3. Startup and Data Loading ---
all_matches = {}
file_exists = os.path.exists(OUTPUT_FILE)

if file_exists:
    print(f"File '{OUTPUT_FILE}' found. Reading existing matches into memory...")
    try:
        with open(OUTPUT_FILE, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('Match ID'):
                    all_matches[row['Match ID']] = row
        print(f"Loaded {len(all_matches)} existing matches.")
    except (IOError, csv.Error) as e:
        print(f"Warning: Could not read '{OUTPUT_FILE}': {e}. Starting with an empty dataset.")

# --- 4. Main Script Logic ---
try:
    eastern_tz = pytz.timezone('America/New_York')
    today_utc = datetime.now(pytz.utc)

    # --- NEW: Define current time and 60-minute window ---
    current_utc_unix = int(today_utc.timestamp())
    SIXTY_MINUTES_IN_SECONDS = 60 * 60

    for i in range(DAYS_TO_FETCH):
        current_date_to_query = today_utc - timedelta(days=i)
        date_str = current_date_to_query.strftime('%Y%m%d')

        print(f"--- Processing Day: {current_date_to_query.strftime('%Y-%m-%d')} (UTC) ---")

        # --- MODIFIED: Store the whole event object, not just the ID ---
        events_from_api = []
        page, total_pages = 1, 1
        while page <= total_pages:
            list_url = f"https://api.betsapi.com/v1/events/ended?token={API_TOKEN}&sport_id={SPORT_ID}&league_id={LEAGUE_ID}&day={date_str}&page={page}"
            try:
                response = requests.get(list_url)
                response.raise_for_status()
                data = response.json()
                if data and data.get('success') == 1:
                    for event in data.get('results', []):
                        # Ensure the event has the necessary info before adding
                        if event.get('id') and event.get('time'):
                            events_from_api.append(event)

                    if page == 1 and data.get('pager'):
                        total_results, per_page = data['pager'].get('total', 0), data['pager'].get('per_page', 50)
                        if total_results > 0: total_pages = math.ceil(total_results / per_page)
                        else: break
                else: break
            except requests.exceptions.RequestException as e: print(f"Network error getting event list: {e}"); break
            page += 1
            time.sleep(2)

        if not events_from_api:
            print("No matches found for this day.")
            continue

        print(f"Found {len(events_from_api)} matches. Checking for new or recently finished entries...")

        for count, event in enumerate(events_from_api):
            event_id = event.get('id')
            match_time_unix = int(event.get('time'))

            # --- NEW CORE LOGIC ---
            # Determine if we need to fetch details. We fetch if:
            # 1. The match is not in our database at all.
            # OR
            # 2. The match started within the last 60 minutes (to get final data for recent games).
            time_since_match_start = current_utc_unix - match_time_unix
            should_fetch_details = (event_id not in all_matches) or (time_since_match_start <= SIXTY_MINUTES_IN_SECONDS)

            if should_fetch_details:
                print(f"Fetching Match {count + 1}/{len(events_from_api)} (ID: {event_id})")
                detail_url = f"https://api.betsapi.com/v1/event/view?token={API_TOKEN}&event_id={event_id}"
                try:
                    response = requests.get(detail_url)
                    response.raise_for_status()
                    data = response.json()
                    if data and data.get('success') == 1 and data.get('results'):
                        event_details = data['results'][0]
                        match_id = event_details.get('id')
                        if not match_id: continue

                        match_time_unix_detail = event_details.get('time')
                        correct_date_str, correct_time_str = 'N/A', 'N/A'
                        if match_time_unix_detail:
                            match_datetime_utc = datetime.fromtimestamp(int(match_time_unix_detail), tz=pytz.utc)
                            match_datetime_et = match_datetime_utc.astimezone(eastern_tz)
                            correct_date_str = match_datetime_et.strftime('%Y-%m-%d')
                            correct_time_str = match_datetime_et.strftime('%H:%M:%S')

                        scores_data = event_details.get('scores', {})
                        set_scores_list = [f"{s.get('home', '?')}-{s.get('away', '?')}" for k, s in sorted(scores_data.items())]
                        advanced_stats = analyze_timeline(event_details.get('timeline', []), scores_data)

                        match_data = {
                            'Match ID': match_id,
                            'Date': correct_date_str,
                            'Time': correct_time_str,
                            'Player 1 ID': event_details.get('home', {}).get('id'),
                            'Player 1': event_details.get('home', {}).get('name'),
                            'Player 2 ID': event_details.get('away', {}).get('id'),
                            'Player 2': event_details.get('away', {}).get('name'),
                            'Final Score': f'="{event_details.get("ss", "N/A")}"',
                            'Set Scores': ", ".join(set_scores_list) if set_scores_list else "N/A",
                            'Match Format': event_details.get('extra', {}).get('bestofsets'),
                            'P1 Total Points': advanced_stats['p1_total_points'],
                            'P2 Total Points': advanced_stats['p2_total_points'],
                            'P1 Pressure Points': advanced_stats['p1_pressure_points'],
                            'P2 Pressure Points': advanced_stats['p2_pressure_points'],
                            'P1 Set Comebacks': advanced_stats['p1_comebacks'],
                            'P2 Set Comebacks': advanced_stats['p2_comebacks']
                        }
                        all_matches[match_id] = match_data
                    time.sleep(2)
                except requests.exceptions.RequestException as e: print(f"Network error on event {event_id}: {e}"); continue

    # --- 5. Final Write to CSV ---
    print(f"\nProcessing complete. Writing {len(all_matches)} total matches to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=HEADERS)
        writer.writeheader()
        sorted_matches = sorted(all_matches.values(), key=lambda x: (x.get('Date', ''), x.get('Time', '')), reverse=True)
        writer.writerows(sorted_matches)

    print(f"✅ Script finished successfully.")

except IOError as e:
    print(f"Error: Could not write to the file '{OUTPUT_FILE}': {e}.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
