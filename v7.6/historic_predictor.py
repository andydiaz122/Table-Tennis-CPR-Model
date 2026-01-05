import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# --- 1. Configuration for HISTORIC Paper Trading ---
# ---!!!--- MANUALLY EDIT THIS SECTION FOR HISTORIC MATCHES ---!!!---
# NOTE: Kickoff_Time is ESSENTIAL to prevent data leakage from future matches.
historic_matches_to_test = [
    {
    'Player 1 ID': 1022513, 'Player 1 Name': 'Milan Cetner',
    'Player 2 ID': 657454, 'Player 2 Name': 'Milan Smesny',
    'P1_ML': -170,
    'P2_ML': 120,
    'Kickoff_Time': '2025-09-25 10:00:00' # Example time
},
{
    'Player 1 ID': 654692, 'Player 1 Name': 'Pavel Sprynar',
    'Player 2 ID': 346257, 'Player 2 Name': 'Petr Sudek',
    'P1_ML': 135,
    'P2_ML': -190,
    'Kickoff_Time': '2025-09-25 11:30:00' # Example time
},
]
# ---!!!--- ---!!!--- ---!!!--- ---!!!--- ---!!!--- ---!!!---

# --- File Paths ---
HISTORICAL_DATA_FILE = "final_dataset_v7.4.csv"
GBM_MODEL_FILE = "cpr_v7.4_gbm_specialist.joblib"
# ... (rest of file paths are the same)
META_MODEL_FILE = "cpr_v7.4_meta_model.pkl"

# --- STRATEGIC FILTERS (Synchronized with Back-tester) ---
EDGE_THRESHOLD_MIN = 0.10
# ... (rest of filters are the same)
FORM_EQUALITY_BAND = 0.10
MIN_PLAYER_HISTORY = 10

# --- Model Parameters ---
SEQUENCE_LENGTH = 5
ROLLING_WINDOW = 10

# --- Helper function to convert odds ---
def moneyline_to_decimal(moneyline_odds):
    # ... (function is the same)
    pass

# --- 2. Load Models and Full Historical Data ---
try:
    print("--- Loading All Models for HISTORIC Prediction ---")
    df_full_history = pd.read_csv(HISTORICAL_DATA_FILE)
    df_full_history['Date'] = pd.to_datetime(df_full_history['Date'])

    gbm_model = joblib.load(GBM_MODEL_FILE)
    # ... (rest of model loading is the same)
    meta_model = joblib.load(META_MODEL_FILE)
    print("All models and filters loaded successfully.")

    # --- 3. Feature Engineering and Prediction for Each Historic Match ---
    print("\n--- Analyzing Historic Matches with Point-in-Time Logic ---")
    
    for match in historic_matches_to_test:
        # CRITICAL: Establish the exact moment of prediction to prevent data leakage
        prediction_timestamp = pd.to_datetime(match['Kickoff_Time'])
        point_in_time_history = df_full_history[df_full_history['Date'] < prediction_timestamp]

        p1_id, p2_id = match['Player 1 ID'], match['Player 2 ID']
        p1_name, p2_name = match['Player 1 Name'], match['Player 2 Name']
        
        # --- Point-in-Time Feature Engineering ---
        # Use the filtered 'point_in_time_history' dataframe for all calculations
        p1_games = point_in_time_history[(point_in_time_history['Player 1 ID'] == p1_id) | (point_in_time_history['Player 2 ID'] == p1_id)]
        p2_games = point_in_time_history[(point_in_time_history['Player 1 ID'] == p2_id) | (point_in_time_history['Player 2 ID'] == p2_id)]

        print("\n---------------------------------")
        print(f"Matchup: {p1_name} vs. {p2_name} (as of {prediction_timestamp.strftime('%Y-%m-%d %H:%M')})")

        if len(p1_games) < MIN_PLAYER_HISTORY or len(p2_games) < MIN_PLAYER_HISTORY:
            print("RECOMMENDATION: NO BET (Insufficient player history at time of match)")
            print("---------------------------------")
            continue

        # --- Calculate Rest Advantage based on Kickoff_Time ---
        p1_last_date = p1_games['Date'].max()
        p2_last_date = p2_games['Date'].max()
        p1_rest = (prediction_timestamp - p1_last_date).days if pd.notna(p1_last_date) else 30
        p2_rest = (prediction_timestamp - p2_last_date).days if pd.notna(p2_last_date) else 30
        rest_advantage = p1_rest - p2_rest
        
        # ... (The rest of the feature engineering and prediction logic is identical to the previous version)
        # ... (It will automatically use the correct, point-in-time dataframes like 'p1_games')
        
        print("---------------------------------")

except Exception as e:
    print(f"An error occurred: {e}")
    # ... (error handling is the same)