import requests
import json

# --- 1. Configuration ---
API_TOKEN = "232034-RBGwKswgRWx3yZ" 
SPORT_ID = "92"
LEAGUE_ID = "22742"

# --- 2. Main Diagnostic Logic ---
try:
    # --- STEP 1: Find the next single upcoming event ---
    list_url = f"https://api.betsapi.com/v1/events/upcoming?token={API_TOKEN}&sport_id={SPORT_ID}&league_id={LEAGUE_ID}"
    
    print("--- Finding the next upcoming match... ---")
    response = requests.get(list_url)
    response.raise_for_status()
    data = response.json()

    if data and data.get('success') == 1 and data.get('results'):
        # Get the very first upcoming match from the list
        event_to_inspect = data['results'][0]
        event_id = event_to_inspect.get('id')
        print(f"Found upcoming match. ID: {event_id}. Fetching its odds data...")

        # --- STEP 2: Fetch all odds for that single event ---
        odds_url = f"https://api.betsapi.com/v2/event/odds?token={API_TOKEN}&event_id={event_id}"
        
        odds_response = requests.get(odds_url)
        odds_response.raise_for_status()
        odds_data = odds_response.json()
        
        # --- STEP 3: Print the entire raw JSON response ---
        print("\n--- RAW ODDS RESPONSE FROM API ---")
        print(json.dumps(odds_data, indent=2))
        print("--- END OF RESPONSE ---")

    else:
        print("Could not find any upcoming matches for this league at this time.")

except Exception as e:
    print(f"An error occurred: {e}")