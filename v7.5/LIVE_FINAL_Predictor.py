import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime


# --- 1. Configuration ---
# ---!!!--- MANUALLY EDIT THIS SECTION FOR DAILY MATCHES ---!!!---
upcoming_matches = [
    # --- Matches for 2025-09-00 ---
{
    'Player 1 ID': 357334, 'Player 1 Name': 'Rotislav Hasmanda',
    'Player 2 ID': 677266, 'Player 2 Name': 'Marek Placek',
    'P1_ML': 237,
    'P2_ML': -325
},
{
    'Player 1 ID': 1099292, 'Player 1 Name': 'Jiri Dedek',
    'Player 2 ID': 548705, 'Player 2 Name': 'Jiri Plachy',
    'P1_ML': 109,
    'P2_ML': -140
},
{
    'Player 1 ID': 373963, 'Player 1 Name': 'Tomas Janata',
    'Player 2 ID': 689390, 'Player 2 Name': 'Marek Kostal',
    'P1_ML': -143,
    'P2_ML': 111
},
{
    'Player 1 ID': 891882, 'Player 1 Name': 'Jaroslav Bresky',
    'Player 2 ID': 548705, 'Player 2 Name': 'Jiri Plachy',
    'P1_ML': -115,
    'P2_ML': -115
},

]
# ---!!!--- ---!!!--- ---!!!--- ---!!!--- ---!!!--- ---!!!---

# --- File Paths (Using the stable v7.4 models) ---
HISTORICAL_DATA_FILE = "final_dataset_v7.4.csv"
GBM_MODEL_FILE = "cpr_v7.4_gbm_specialist.joblib"
GBM_PREPROCESSOR_FILE = "gbm_preprocessor_v7.4.joblib"
LSTM_MODEL_FILE = "cpr_v7.1_lstm_specialist.h5"
LSTM_SCALER_FILE = "lstm_scaler_v7.1.joblib"
META_MODEL_FILE = "cpr_v7.4_meta_model.pkl"

# --- FINAL STRATEGIC FILTERS v7.4 (Synchronized with Back-tester) ---
EDGE_THRESHOLD_MIN = 0.10
EDGE_THRESHOLD_MAX = 0.99
ODDS_THRESHOLD_MAX = 3.0
H2H_DISADVANTAGE_THRESHOLD = 0.40
FORM_EQUALITY_BAND = 0.10
MIN_PLAYER_HISTORY = 10 # Basic check to ensure player exists and has some data

# --- Model Parameters ---
SEQUENCE_LENGTH = 5
ROLLING_WINDOW = 10

# --- Helper function to convert odds ---
def moneyline_to_decimal(moneyline_odds):
    try:
        moneyline_odds = float(moneyline_odds)
        if moneyline_odds >= 100: return (moneyline_odds / 100) + 1
        elif moneyline_odds < 0: return (100 / abs(moneyline_odds)) + 1
        else: return np.nan
    except (ValueError, TypeError): return np.nan

# --- 2. Load All Models and Historical Data ---
try:
    print("--- Loading All Models and Historical Data for Prediction ---")
    df_history = pd.read_csv(HISTORICAL_DATA_FILE)
    df_history['Date'] = pd.to_datetime(df_history['Date'])

    gbm_model = joblib.load(GBM_MODEL_FILE)
    gbm_preprocessor = joblib.load(GBM_PREPROCESSOR_FILE)
    lstm_model = load_model(LSTM_MODEL_FILE)
    lstm_scaler = joblib.load(LSTM_SCALER_FILE)
    meta_model = joblib.load(META_MODEL_FILE)
    print("All v7.5 models and filters loaded successfully.")

    # --- 3. Feature Engineering and Prediction for Each Match ---
    print("\n--- Analyzing Upcoming Matches with Final v7.5 Strategy ---")
    
    col_group1 = ['P1_Rolling_Win_Rate_L10', 'P1_Rolling_Pressure_Points_L10', 'P1_Rest_Days']
    col_group2 = ['P2_Rolling_Win_Rate_L10', 'P2_Rolling_Pressure_Points_L10', 'P2_Rest_Days']


    for match in upcoming_matches:
        p1_id, p2_id = match['Player 1 ID'], match['Player 2 ID']
        p1_name, p2_name = match['Player 1 Name'], match['Player 2 Name']
        
        # --- Point-in-Time Feature Engineering (Synchronized with Back-tester) ---
        # --- MORE EFFICIENT VERSION ---
        p1_games = df_history[(df_history['Player 1 ID'] == p1_id) | (df_history['Player 2 ID'] == p1_id)]
        p2_games = df_history[(df_history['Player 1 ID'] == p2_id) | (df_history['Player 2 ID'] == p2_id)]

        print("\n---------------------------------")
        print(f"Matchup: {p1_name} vs. {p2_name}")

        if len(p1_games) < MIN_PLAYER_HISTORY or len(p2_games) < MIN_PLAYER_HISTORY:
            print("RECOMMENDATION: NO BET (Insufficient player history)")
            print("---------------------------------")
            continue

        p1_rolling_games = p1_games.tail(ROLLING_WINDOW)
        p2_rolling_games = p2_games.tail(ROLLING_WINDOW)
        
        # --- Calculate Rest Advantage ---
        today = datetime.now()
        p1_last_date = p1_games['Date'].max()
        p2_last_date = p2_games['Date'].max()
        p1_rest = (today - p1_last_date).days if pd.notna(p1_last_date) else 30
        p2_rest = (today - p2_last_date).days if pd.notna(p2_last_date) else 30
        rest_advantage = p1_rest - p2_rest
        
        # --- Calculate other features ---
        p1_win_rate = p1_rolling_games.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p1_rolling_games.empty else 0.5
        p2_win_rate = p2_rolling_games.apply(lambda r: 1 if (r['Player 1 ID'] == p2_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p2_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p2_rolling_games.empty else 0.5
        win_rate_advantage = p1_win_rate - p2_win_rate

        p1_pressure_points = p1_rolling_games.apply(lambda r: r['P1 Pressure Points'] if r['Player 1 ID'] == p1_id else r['P2 Pressure Points'], axis=1).mean() if not p1_rolling_games.empty else 0.0
        p2_pressure_points = p2_rolling_games.apply(lambda r: r['P1 Pressure Points'] if r['Player 1 ID'] == p2_id else r['P2 Pressure Points'], axis=1).mean() if not p2_rolling_games.empty else 0.0
        pressure_points_advantage = p1_pressure_points - p2_pressure_points
        
        h2h_df = df_history[((df_history['Player 1 ID'] == p1_id) & (df_history['Player 2 ID'] == p2_id)) | ((df_history['Player 1 ID'] == p2_id) & (df_history['Player 2 ID'] == p1_id))]
        p1_h2h_wins = h2h_df.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).sum()
        h2h_p1_win_rate = p1_h2h_wins / len(h2h_df) if len(h2h_df) > 0 else 0.5
        
        p1_seq_df, p2_seq_df = p1_games.tail(SEQUENCE_LENGTH), p2_games.tail(SEQUENCE_LENGTH)
        if len(p1_seq_df) < SEQUENCE_LENGTH or len(p2_seq_df) < SEQUENCE_LENGTH:
            print("RECOMMENDATION: NO BET (Insufficient data for LSTM sequences)")
            print("---------------------------------")
            continue

        p1_seq, p2_seq = [], []
        for _, row in p1_seq_df.iterrows():
            if row['Player 1 ID'] == p1_id: p1_seq.append(np.concatenate([row[col_group1].values, row[col_group2].values, [row['H2H_P1_Win_Rate']]]))
            else: p1_seq.append(np.concatenate([row[col_group2].values, row[col_group1].values, [1 - row['H2H_P1_Win_Rate']]]))
        for _, row in p2_seq_df.iterrows():
            if row['Player 1 ID'] == p2_id: p2_seq.append(np.concatenate([row[col_group1].values, row[col_group2].values, [row['H2H_P1_Win_Rate']]]))
            else: p2_seq.append(np.concatenate([row[col_group2].values, row[col_group1].values, [1 - row['H2H_P1_Win_Rate']]]))

        # --- Full Ensemble Prediction Logic ---
        gbm_features = pd.DataFrame([{
            'Win_Rate_Advantage': win_rate_advantage,
            'Pressure_Points_Advantage': pressure_points_advantage,
            'Player 1 ID': p1_id,
            'Player 2 ID': p2_id
        }])
        X_gbm_processed = gbm_preprocessor.transform(gbm_features)
        gbm_pred = gbm_model.predict_proba(X_gbm_processed)[0, 1]
        
        X_p1, X_p2 = np.array([p1_seq]), np.array([p2_seq])
        nsamples, nsteps, nfeatures = X_p1.shape
        X_p1_scaled = lstm_scaler.transform(X_p1.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
        X_p2_scaled = lstm_scaler.transform(X_p2.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
        lstm_pred = lstm_model.predict([X_p1_scaled, X_p2_scaled], verbose=0)[0][0]
        
        X_meta = np.array([[gbm_pred, lstm_pred]])
        model_prob_p1 = meta_model.predict_proba(X_meta)[0, 1]
        model_prob_p2 = 1 - model_prob_p1

        # --- Apply Filters and Display Results ---
        p1_market_odds = moneyline_to_decimal(match['P1_ML'])
        p2_market_odds = moneyline_to_decimal(match['P2_ML'])
        
        edge_p1 = model_prob_p1 * p1_market_odds - 1 if pd.notna(p1_market_odds) else -1
        edge_p2 = model_prob_p2 * p2_market_odds - 1 if pd.notna(p2_market_odds) else -1
        
        p1_form_advantage = win_rate_advantage
        p2_form_advantage = -win_rate_advantage
        h2h_p2_win_rate = 1 - h2h_p1_win_rate

        # --- REFACTORED: Filter conditions mirroring the back-tester ---
        p1_conditions = {
            "Edge Min": edge_p1 > EDGE_THRESHOLD_MIN,
            "Edge Max": edge_p1 < EDGE_THRESHOLD_MAX,
            "Max Odds": p1_market_odds <= ODDS_THRESHOLD_MAX if pd.notna(p1_market_odds) else False,
            "H2H Adv": h2h_p1_win_rate >= H2H_DISADVANTAGE_THRESHOLD,
            "Form Adv": abs(p1_form_advantage) > FORM_EQUALITY_BAND
        }

        p2_conditions = {
            "Edge Min": edge_p2 > EDGE_THRESHOLD_MIN,
            "Edge Max": edge_p2 < EDGE_THRESHOLD_MAX,
            "Max Odds": p2_market_odds <= ODDS_THRESHOLD_MAX if pd.notna(p2_market_odds) else False,
            "H2H Adv": h2h_p2_win_rate >= H2H_DISADVANTAGE_THRESHOLD,
            "Form Adv": abs(p2_form_advantage) > FORM_EQUALITY_BAND
        }
        
        p1_pass_all = all(p1_conditions.values())
        p2_pass_all = all(p2_conditions.values())
        
        print(f"Model Prediction: {p1_name} ({model_prob_p1:.2%}) vs. {p2_name} ({model_prob_p2:.2%})")
        
        print(f"\nAnalysis for {p1_name}:")
        print(f"  - Market Odds: {match['P1_ML']} ({p1_market_odds:.2f} dec) -> {'PASS' if p1_conditions['Max Odds'] else 'FAIL'}")
        print(f"  - Model Edge: {edge_p1:.2%} -> {'PASS' if p1_conditions['Edge Min'] and p1_conditions['Edge Max'] else 'FAIL'}")
        print(f"  - H2H Win Rate: {h2h_p1_win_rate:.2%} -> {'PASS' if p1_conditions['H2H Adv'] else 'FAIL'}")
        print(f"  - Form Adv: {p1_form_advantage:+.2f} -> {'PASS' if p1_conditions['Form Adv'] else 'FAIL'}")
        
        # ... Inside the loop, in the Analysis for Player 1 section ...
        if p1_pass_all:
            print(f"  RECOMMENDATION: BET on {p1_name}")
        else:
            print(f"  RECOMMENDATION: NO BET")
            
        print(f"\nAnalysis for {p2_name}:")
        print(f"  - Market Odds: {match['P2_ML']} ({p2_market_odds:.2f} dec) -> {'PASS' if p2_conditions['Max Odds'] else 'FAIL'}")
        print(f"  - Model Edge: {edge_p2:.2%} -> {'PASS' if p2_conditions['Edge Min'] and p2_conditions['Edge Max'] else 'FAIL'}")
        print(f"  - H2H Win Rate: {h2h_p2_win_rate:.2%} -> {'PASS' if p2_conditions['H2H Adv'] else 'FAIL'}")
        print(f"  - Form Adv: {p2_form_advantage:+.2f} -> {'PASS' if p2_conditions['Form Adv'] else 'FAIL'}")

        # ... Inside the loop, in the Analysis for Player 2 section ...
        if p2_pass_all:
            print(f"  RECOMMENDATION: BET on {p2_name}")
        else:
            print(f"  RECOMMENDATION: NO BET")

        print("---------------------------------")

except FileNotFoundError as e:
    print(f"Error: A required model or data file was not found. Ensure all trainers have been run and files are in the correct folder.")
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()