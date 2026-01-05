import requests
import csv
import time
from datetime import datetime
import pytz
import os

# --- Configuration ---
API_TOKEN = "232034-RBGwKswgRWx3yZ" 
INPUT_FILE = "czech_liga_pro_advanced_stats_FIXED.csv"
OUTPUT_FILE = "czech_liga_pro_advanced_stats_CORRECTED.csv"

# --- Main Script ---
print(f"Starting date correction process for '{INPUT_FILE}'...")

# --- RESUME LOGIC START ---
# Check if the output file already exists to see what we've already processed.
processed_ids = set()
try:
    with open(OUTPUT_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed_ids.add(row['Match ID'])
    print(f"Found {len(processed_ids)} matches already processed in '{OUTPUT_FILE}'. Resuming.")
except FileNotFoundError:
    print("Output file not found. Starting a new run.")
# --- RESUME LOGIC END ---

# 1. Read all existing data from the original input file
try:
    with open(INPUT_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        existing_data = list(reader)
        fieldnames = reader.fieldnames # Save the header
    print(f"Successfully read {len(existing_data)} total matches from the input file.")
except FileNotFoundError:
    print(f"Error: The input file '{INPUT_FILE}' was not found. Please make sure it's in the same directory.")
    exit()

prague_tz = pytz.timezone('Europe/Prague')
total_matches = len(existing_data)

# 2. Open the output file in APPEND mode
try:
    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write the header only if the file is new (i.e., no processed IDs were found)
        if not processed_ids:
            writer.writeheader()

        # 3. Loop through each match, check if we need to process it, and update
        for index, row in enumerate(existing_data):
            match_id = row.get('Match ID')
            if not match_id:
                continue

            # --- RESUME LOGIC CHECK ---
            if match_id in processed_ids:
                # No need to print for every single one, this will just confirm it's working at the start
                if index < 5 or index % 100 == 0:
                     print(f"Skipping match {index + 1}/{total_matches} (ID: {match_id}), already processed.")
                continue
            
            print(f"Processing match {index + 1}/{total_matches} (ID: {match_id})...")
            
            try:
                # Make the API call just for this one match
                detail_url = f"https://api.betsapi.com/v1/event/view?token={API_TOKEN}&event_id={match_id}"
                response = requests.get(detail_url)
                response.raise_for_status()
                data = response.json()

                if data and data.get('success') == 1 and data.get('results'):
                    event = data['results'][0]
                    match_time_unix = event.get('time')
                    
                    if match_time_unix:
                        match_datetime_utc = datetime.fromtimestamp(int(match_time_unix), tz=pytz.utc)
                        match_datetime_local = match_datetime_utc.astimezone(prague_tz)
                        row['Date'] = match_datetime_local.strftime('%Y-%m-%d')
                        row['Time'] = match_datetime_local.strftime('%H-%M-%S')
                    
                    writer.writerow(row) # Write the corrected row immediately
                else:
                    print(f"  -> Warning: Could not find API data for match ID {match_id}. Skipping.")

                time.sleep(2) # IMPORTANT: Respect the API rate limit

            except requests.exceptions.RequestException as e:
                print(f"  -> Network Error for match ID {match_id}: {e}. Will retry on next run.")
                # We don't write the row, so it will be retried automatically the next time you run the script.
                continue

    print("\n✅ Run complete. All matches have been processed.")
    print(f"Your fully corrected data is in '{OUTPUT_FILE}'.")

except IOError as e:
    print(f"Error writing to file: {e}")