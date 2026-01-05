import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

# --- 1. Configuration ---
# ---!!!--- MANUALLY EDIT THIS SECTION FOR DAILY MATCHES ---!!!---
upcoming_matches = [
    # --- Matches for 2025-09-29 ---
    {
    'Player 1 ID': 685221, 'Player 1 Name': 'Jiri Nesnera',
    'Player 2 ID': 338943, 'Player 2 Name': 'Jan Kanera',
    'P1_ML': -120,
    'P2_ML': -120
},
{
    'Player 1 ID': 341885, 'Player 1 Name': 'Petr Macela',
    'Player 2 ID': 338649, 'Player 2 Name': 'Jan Zajicek',
    'P1_ML': -155,
    'P2_ML': 110
},
{
    'Player 1 ID': 358665, 'Player 1 Name': 'Marek Sedlak',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': -160,
    'P2_ML': 115
},
{
    'Player 1 ID': 359407, 'Player 1 Name': 'Jaroslav Prokupek',
    'Player 2 ID': 1098619, 'Player 2 Name': 'Jaroslav Bares',
    'P1_ML': 120,
    'P2_ML': -170
},

]
# ---!!!--- ---!!!--- ---!!!--- ---!!!--- ---!!!--- ---!!!---

# --- File Paths (Synchronized with Backtester) ---
HISTORICAL_DATA_FILE = "final_dataset_v7.4_no_duplicates.csv"
GBM_MODEL_FILE = "cpr_v7.4_gbm_specialist.joblib"
GBM_PREPROCESSOR_FILE = "gbm_preprocessor_v7.4.joblib"
LSTM_MODEL_FILE = "cpr_v7.1_lstm_specialist.h5"
LSTM_SCALER_FILE = "lstm_scaler_v7.1.joblib"
META_MODEL_FILE = "cpr_v7.4_meta_model.pkl"

# --- SYNC: Filters from Back-tester ---
KELLY_FRACTION = 0.035
EDGE_THRESHOLD_MIN = 0.10
EDGE_THRESHOLD_MAX = 0.99
MIN_ODDS_DENOMINATOR = 0.10
MIN_PLAYER_HISTORY_LSTM = 5
# SYNC: Added new filters from back-tester's betting logic
ODDS_THRESHOLD_MAX = 3
H2H_MIN_WIN_RATE = 0.5
# NEW: Define the odds ceiling for the contrarian filter
CONTRARIAN_ODDS_CEILING = 1.5

MIN_PLAYER_HISTORY = 20
MIN_H2H_HISTORY = 5

# --- Model Parameters ---
SEQUENCE_LENGTH = 5
ROLLING_WINDOW = 20

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
    df_history['Date'] = pd.to_datetime(df_history['Date'], format='mixed')
    
    gbm_model = joblib.load(GBM_MODEL_FILE)
    gbm_preprocessor = joblib.load(GBM_PREPROCESSOR_FILE)
    lstm_model = load_model(LSTM_MODEL_FILE)
    lstm_scaler = joblib.load(LSTM_SCALER_FILE)
    meta_model = joblib.load(META_MODEL_FILE)
    print("All models and filters loaded successfully.")

    # --- 3. Feature Engineering and Prediction for Each Match ---
    print("\n--- Analyzing Upcoming Matches with Synced Strategy ---")
    
    for match in upcoming_matches:
        p1_id, p2_id = match['Player 1 ID'], match['Player 2 ID']
        p1_name, p2_name = match['Player 1 Name'], match['Player 2 Name']
        
        print("\n---------------------------------")
        print(f"Matchup: {p1_name} vs. {p2_name}")

        p1_games = df_history[(df_history['Player 1 ID'] == p1_id) | (df_history['Player 2 ID'] == p1_id)]
        p2_games = df_history[(df_history['Player 1 ID'] == p2_id) | (df_history['Player 2 ID'] == p2_id)]

        # NEW RULE: Calculate total history for Warm-Up Period checks
        p1_games_total = df_history[(df_history['Player 1 ID'] == p1_id) | (df_history['Player 2 ID'] == p1_id)]
        p2_games_total = df_history[(df_history['Player 1 ID'] == p2_id) | (df_history['Player 2 ID'] == p2_id)]
        p1_history_count = len(p1_games_total)
        p2_history_count = len(p2_games_total)

        h2h_df_check = df_history[((df_history['Player 1 ID']==p1_id)&(df_history['Player 2 ID']==p2_id))|((df_history['Player 1 ID']==p2_id)&(df_history['Player 2 ID']==p1_id))]
        h2h_match_count = len(h2h_df_check)

        # NEW RULE: "Warm-Up" Period Hard Filter
        if (p1_history_count < MIN_PLAYER_HISTORY or
            p2_history_count < MIN_PLAYER_HISTORY or
            h2h_match_count < MIN_H2H_HISTORY):
            print("RECOMMENDATION: NO BET (Insufficient player history)")
            print("---------------------------------")
            continue # Skip bet if data is not mature enough

        if len(p1_games) < MIN_PLAYER_HISTORY_LSTM or len(p2_games) < MIN_PLAYER_HISTORY_LSTM:
            print("RECOMMENDATION: NO BET (Insufficient player history for LSTM)")
            print("---------------------------------")
            continue

        # --- SYNC: On-the-fly Feature Engineering (from back-tester) ---
        p1_games_gbm = p1_games.tail(ROLLING_WINDOW)
        p2_games_gbm = p2_games.tail(ROLLING_WINDOW)
        p1_win_rate = p1_games_gbm.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p1_games_gbm.empty else 0.5
        p2_win_rate = p2_games_gbm.apply(lambda r: 1 if (r['Player 1 ID'] == p2_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p2_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p2_games_gbm.empty else 0.5
        win_rate_advantage = p1_win_rate - p2_win_rate
        p1_pressure_points = p1_games_gbm.apply(lambda r: r['P1 Pressure Points'] if r['Player 1 ID'] == p1_id else r['P2 Pressure Points'], axis=1).mean() if not p1_games_gbm.empty else 0.0
        p2_pressure_points = p2_games_gbm.apply(lambda r: r['P1 Pressure Points'] if r['Player 1 ID'] == p2_id else r['P2 Pressure Points'], axis=1).mean() if not p2_games_gbm.empty else 0.0
        pressure_points_advantage = p1_pressure_points - p2_pressure_points

        p1_games_lstm = p1_games.tail(SEQUENCE_LENGTH)
        p2_games_lstm = p2_games.tail(SEQUENCE_LENGTH)
        
        # --- SYNC: Correct LSTM Sequence Building ---
        p1_seq, p2_seq = [], []
        for _, row in p1_games_lstm.iterrows():
            if row['Player 1 ID'] == p1_id:
                p1_perspective_vector = [row['P1_Win'], row['P1_Rest_Days'], 1.0 - row['P1_Win'], row['P2_Rest_Days'], row['H2H_P1_Win_Rate']]
            else:
                p1_perspective_vector = [1.0 - row['P1_Win'], row['P2_Rest_Days'], row['P1_Win'], row['P1_Rest_Days'], 1.0 - row['H2H_P1_Win_Rate']]
            p1_seq.append(p1_perspective_vector)

        for _, row in p2_games_lstm.iterrows():
            if row['Player 1 ID'] == p2_id:
                p2_perspective_vector = [row['P1_Win'], row['P1_Rest_Days'], 1.0 - row['P1_Win'], row['P2_Rest_Days'], row['H2H_P1_Win_Rate']]
            else:
                p2_perspective_vector = [1.0 - row['P1_Win'], row['P2_Rest_Days'], row['P1_Win'], row['P1_Rest_Days'], 1.0 - row['H2H_P1_Win_Rate']]
            p2_seq.append(p2_perspective_vector)

        # --- SYNC: Full Ensemble Prediction Logic (Symmetrical GBM Features) ---
        # NOTE: Rest_Advantage and Player IDs are removed to match the back-tester and ensure symmetry.
        gbm_features = pd.DataFrame([{'Win_Rate_Advantage': win_rate_advantage, 
                                      'Pressure_Points_Advantage': pressure_points_advantage}])
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

        print(f"Model Prediction: {p1_name} ({model_prob_p1:.2%}) vs. {p2_name} ({model_prob_p2:.2%})")

        # --- SYNC: Final Betting Logic and Recommendation from Back-tester ---
        p1_market_odds = moneyline_to_decimal(match['P1_ML'])
        p2_market_odds = moneyline_to_decimal(match['P2_ML'])
        edge_p1 = model_prob_p1 * p1_market_odds - 1 if pd.notna(p1_market_odds) else -1
        edge_p2 = model_prob_p2 * p2_market_odds - 1 if pd.notna(p2_market_odds) else -1
        
        # SYNC: Calculate H2H for filtering, just like in the back-tester
        h2h_df = df_history[((df_history['Player 1 ID']==p1_id)&(df_history['Player 2 ID']==p2_id))|((df_history['Player 1 ID']==p2_id)&(df_history['Player 2 ID']==p1_id))]
        p1_h2h_wins = len(h2h_df[((h2h_df['Player 1 ID']==p1_id)&(h2h_df['P1_Win']==1))|((h2h_df['Player 2 ID']==p1_id)&(h2h_df['P1_Win']==0))])
#        h2h_p1_win_rate = p1_h2h_wins/len(h2h_df) if len(h2h_df)>0 else 0.5
        h2h_p1_win_rate = p1_h2h_wins / len(h2h_df_check) if len(h2h_df_check) > 0 else 0.5
        
        # --- Create filter check dictionaries for detailed output ---
        p1_filters = {
            "Edge between Min/Max": EDGE_THRESHOLD_MIN < edge_p1 < EDGE_THRESHOLD_MAX,
            "Form Advantage != 0": win_rate_advantage != 0,
            "H2H Win Rate >= 50%": h2h_p1_win_rate >= H2H_MIN_WIN_RATE,
            "Odds < 3.0": p1_market_odds < ODDS_THRESHOLD_MAX,
            "Odds are Betable": pd.notna(p1_market_odds) and (p1_market_odds - 1) > MIN_ODDS_DENOMINATOR,
        }
        
        p2_filters = {
            "Edge between Min/Max": EDGE_THRESHOLD_MIN < edge_p2 < EDGE_THRESHOLD_MAX,
            "Form Advantage != 0": win_rate_advantage != 0,
            "H2H Win Rate >= 50%": (1 - h2h_p1_win_rate) >= H2H_MIN_WIN_RATE,
            "Odds < 3.0": p2_market_odds < ODDS_THRESHOLD_MAX,
            "Odds are Betable": pd.notna(p2_market_odds) and (p2_market_odds - 1) > MIN_ODDS_DENOMINATOR,
        }
        
        recommendation_found = False
        
        # --- Analysis for Player 1 ---
        print(f"\n--- Analysis for {p1_name} ---")
        print(f"Market Odds: {match['P1_ML']} ({p1_market_odds:.2f} dec) | H2H Win Rate: {h2h_p1_win_rate:.2%} | Form Adv: {win_rate_advantage:+.2f}")
        print("Filter Checklist:")
        for name, passed in p1_filters.items():
            print(f"  - {name}: {'PASS' if passed else 'FAIL'}")

        if all(p1_filters.values()):
            # APPLY V2 CONTRARIAN SUB-FILTER
            if p1_market_odds >= CONTRARIAN_ODDS_CEILING:
                kelly_fraction_rec = (edge_p1 / (p1_market_odds - 1)) * KELLY_FRACTION
                capped_fraction = min(kelly_fraction_rec, 0.05)
                if (h2h_p1_win_rate > 0.5):
                    capped_fraction = capped_fraction * 2
            # SYNC: The forensic rule (division by 4) is NOT in the new back-tester, so it's removed.
            
            print(f"\nRECOMMENDATION: BET on {p1_name}")
            print(f"  - Model Edge: {edge_p1:.2%}")
            print(f"  - Recommended Bet Size (Fraction of Bankroll): {capped_fraction:.2%}")
            recommendation_found = True
        else:
            print("\nRECOMMENDATION: NO BET")

        # --- Analysis for Player 2 ---
        print(f"\n--- Analysis for {p2_name} ---")
        print(f"Market Odds: {match['P2_ML']} ({p2_market_odds:.2f} dec) | H2H Win Rate: {1-h2h_p1_win_rate:.2%} | Form Adv: {-win_rate_advantage:+.2f}")
        print("Filter Checklist:")
        for name, passed in p2_filters.items():
            print(f"  - {name}: {'PASS' if passed else 'FAIL'}")

        if all(p2_filters.values()) and not recommendation_found:
            # APPLY V2 CONTRARIAN SUB-FILTER
            if p2_market_odds >= CONTRARIAN_ODDS_CEILING:
                kelly_fraction_rec = (edge_p2 / (p2_market_odds - 1)) * KELLY_FRACTION
                capped_fraction = min(kelly_fraction_rec, 0.05)
                if (h2h_p1_win_rate < 0.5):
                    capped_fraction = capped_fraction * 2
            # SYNC: The forensic rule (division by 4) is NOT in the new back-tester, so it's removed.
            
            print(f"\nRECOMMENDATION: BET on {p2_name}")
            print(f"  - Model Edge: {edge_p2:.2%}")
            print(f"  - Recommended Bet Size (Fraction of Bankroll): {capped_fraction:.2%}")
        elif not recommendation_found:
            print("\nRECOMMENDATION: NO BET")

        print("---------------------------------")

except FileNotFoundError as e:
    print(f"\n--- ERROR ---")
    print(f"A required model or data file was not found. Ensure all trainers have been run and files are in the correct folder.")
    print(f"Missing file: {e.filename}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()