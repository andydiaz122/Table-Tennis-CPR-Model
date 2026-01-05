import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

# --- 1. Configuration ---
# ---!!!--- MANUALLY EDIT THIS SECTION FOR DAILY MATCHES ---!!!---
upcoming_matches = [
    {
    'Player 1 ID': 689390, 'Player 1 Name': 'Marek Kostal',
    'Player 2 ID': 373963, 'Player 2 Name': 'Tomas Janata',
    'P1_ML': 120,
    'P2_ML': -170
},
{
    'Player 1 ID': 689390, 'Player 1 Name': 'Marek Kostal',
    'Player 1 ID': 373963, 'Player 1 Name': 'Tomas Janata',
    'Player 2 ID': 689390, 'Player 2 Name': 'Marek Kostal',
    'P1_ML': -170,
    'P2_ML': 120
},
{
    'Player 1 ID': 689390, 'Player 1 Name': 'Marek Kostal',
    'Player 2 ID': 373963, 'Player 2 Name': 'Tomas Janata',
    'P1_ML': 120,
    'P2_ML': -170
},
{
    'Player 1 ID': 342540, 'Player 1 Name': 'Matous Klimenta',
    'Player 2 ID': 339710, 'Player 2 Name': 'Ales Bayer',
    'P1_ML': -115,
    'P2_ML': -125
},
{
    'Player 1 ID': 339710, 'Player 1 Name': 'Ales Bayer',
    'Player 2 ID': 342540, 'Player 2 Name': 'Matous Klimenta',
    'P1_ML': -125,
    'P2_ML': -115
},
{
    'Player 1 ID': 339710, 'Player 1 Name': 'Ales Bayer',
    'Player 2 ID': 342540, 'Player 2 Name': 'Matous Klimenta',
    'P1_ML': -125,
    'P2_ML': -115
},
]
# ---!!!--- ---!!!--- ---!!!--- ---!!!--- ---!!!--- ---!!!---

# --- File Paths ---
HISTORICAL_DATA_FILE = "final_dataset_v7.1.csv"
GBM_MODEL_FILE = "cpr_v7.1_gbm_specialist.joblib"
GBM_PREPROCESSOR_FILE = "gbm_preprocessor_v7.1.joblib"
LSTM_MODEL_FILE = "cpr_v7.1_lstm_specialist.h5"
LSTM_SCALER_FILE = "lstm_scaler_v7.1.joblib"
META_MODEL_FILE = "cpr_v7.1_meta_model.pkl"

# --- v7.3 FINAL STRATEGIC FILTERS (Synchronized with Back-tester) ---
EDGE_THRESHOLD_MIN = 0.10
EDGE_THRESHOLD_MAX = 0.99
H2H_DISADVANTAGE_THRESHOLD = 0.4

# --- Model Parameters ---
SEQUENCE_LENGTH = 5
ROLLING_WINDOW = 10

# --- Helper function to convert odds ---
def moneyline_to_decimal(moneyline_odds):
    """Converts American Moneyline odds to decimal odds."""
    try:
        moneyline_odds = float(moneyline_odds)
        if moneyline_odds >= 100:
            return (moneyline_odds / 100) + 1
        elif moneyline_odds < 0:
            return (100 / abs(moneyline_odds)) + 1
        else:
            return np.nan # Invalid odds
    except (ValueError, TypeError):
        return np.nan

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
    print("All v7.3 models and filters loaded successfully.")

    # --- 3. Feature Engineering and Prediction for Each Match ---
    print("\n--- Analyzing Upcoming Matches with v7.3 Strategy ---")
    
    col_group1 = ['P1_Rolling_Win_Rate_L10', 'P1_Rolling_Pressure_Points_L10', 'P1_Rest_Days']
    col_group2 = ['P2_Rolling_Win_Rate_L10', 'P2_Rolling_Pressure_Points_L10', 'P2_Rest_Days']
    
    for match in upcoming_matches:
        p1_id, p2_id = match['Player 1 ID'], match['Player 2 ID']
        p1_name, p2_name = match['Player 1 Name'], match['Player 2 Name']
        
        # --- Point-in-Time Feature Engineering ---
        p1_last_date = df_history[(df_history['Player 1 ID'] == p1_id) | (df_history['Player 2 ID'] == p1_id)]['Date'].max()
        p2_last_date = df_history[(df_history['Player 1 ID'] == p2_id) | (df_history['Player 2 ID'] == p2_id)]['Date'].max()
        p1_rest = (datetime.now() - p1_last_date).days if pd.notna(p1_last_date) else 30
        p2_rest = (datetime.now() - p2_last_date).days if pd.notna(p2_last_date) else 30
        rest_advantage = p1_rest - p2_rest

        h2h_df = df_history[((df_history['Player 1 ID'] == p1_id) & (df_history['Player 2 ID'] == p2_id)) | ((df_history['Player 1 ID'] == p2_id) & (df_history['Player 2 ID'] == p1_id))]
        p1_h2h_wins = len(h2h_df[((h2h_df['Player 1 ID'] == p1_id) & (h2h_df['P1_Win'] == 1)) | ((h2h_df['Player 2 ID'] == p1_id) & (h2h_df['P1_Win'] == 0))])
        h2h_p1_win_rate = p1_h2h_wins / len(h2h_df) if len(h2h_df) > 0 else 0.5

        p1_games = df_history[(df_history['Player 1 ID'] == p1_id) | (df_history['Player 2 ID'] == p1_id)].tail(ROLLING_WINDOW)
        p2_games = df_history[(df_history['Player 1 ID'] == p2_id) | (df_history['Player 2 ID'] == p2_id)].tail(ROLLING_WINDOW)
        
        p1_win_rate = p1_games.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p1_games.empty else 0.5
        p2_win_rate = p2_games.apply(lambda r: 1 if (r['Player 1 ID'] == p2_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p2_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p2_games.empty else 0.5
        win_rate_advantage = p1_win_rate - p2_win_rate

        # --- Generate Model Prediction ---
        p1_seq_df = df_history[(df_history['Player 1 ID'] == p1_id) | (df_history['Player 2 ID'] == p1_id)].tail(SEQUENCE_LENGTH)
        p2_seq_df = df_history[(df_history['Player 1 ID'] == p2_id) | (df_history['Player 2 ID'] == p2_id)].tail(SEQUENCE_LENGTH)

        if len(p1_seq_df) < SEQUENCE_LENGTH or len(p2_seq_df) < SEQUENCE_LENGTH:
            print(f"\n--- Match: {p1_name} vs. {p2_name} ---")
            print("RECOMMENDATION: NO BET (Insufficient historical data for LSTM model)")
            continue

        gbm_features = pd.DataFrame([{'Win_Rate_Advantage': win_rate_advantage, 'Pressure_Points_Advantage': 0, 'Rest_Advantage': rest_advantage, 'H2H_P1_Win_Rate': h2h_p1_win_rate, 'Player 1 ID': p1_id, 'Player 2 ID': p2_id}])
        X_gbm_processed = gbm_preprocessor.transform(gbm_features)
        gbm_pred = gbm_model.predict_proba(X_gbm_processed)[0, 1]

        p1_seq = [np.concatenate([r[col_group1].values, r[col_group2].values, [r['H2H_P1_Win_Rate']]]) for _, r in p1_seq_df.iterrows()]
        p2_seq = [np.concatenate([r[col_group2].values, r[col_group1].values, [1 - r['H2H_P1_Win_Rate']]]) for _, r in p2_seq_df.iterrows()]

        X_p1, X_p2 = np.array([p1_seq]), np.array([p2_seq])
        nsamples, nsteps, nfeatures = X_p1.shape
        X_p1_scaled = lstm_scaler.transform(X_p1.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
        X_p2_scaled = lstm_scaler.transform(X_p2.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
        lstm_pred = lstm_model.predict([X_p1_scaled, X_p2_scaled], verbose=0)[0][0]
        
        X_meta = np.array([[gbm_pred, lstm_pred]])
        model_prob_p1 = meta_model.predict_proba(X_meta)[0, 1]
        model_prob_p2 = 1 - model_prob_p1

        # --- Apply v7.3 Filters and Display Results ---
        p1_market_odds = moneyline_to_decimal(match['P1_ML'])
        p2_market_odds = moneyline_to_decimal(match['P2_ML'])
        
        edge_p1 = model_prob_p1 * p1_market_odds - 1 if pd.notna(p1_market_odds) else None
        edge_p2 = model_prob_p2 * p2_market_odds - 1 if pd.notna(p2_market_odds) else None
        
        # Check filters for Player 1
        p1_passes_edge = (edge_p1 is not None) and (EDGE_THRESHOLD_MIN < edge_p1 < EDGE_THRESHOLD_MAX)
        p1_passes_h2h = h2h_p1_win_rate >= H2H_DISADVANTAGE_THRESHOLD
        
        # Check filters for Player 2
        h2h_p2_win_rate = 1 - h2h_p1_win_rate
        p2_passes_edge = (edge_p2 is not None) and (EDGE_THRESHOLD_MIN < edge_p2 < EDGE_THRESHOLD_MAX)
        p2_passes_h2h = h2h_p2_win_rate >= H2H_DISADVANTAGE_THRESHOLD

        print("\n---------------------------------")
        print(f"Matchup: {p1_name} vs. {p2_name}")
        print(f"Model Prediction: {p1_name} ({model_prob_p1:.2%}) vs. {p2_name} ({model_prob_p2:.2%})")
        
        # Player 1 Analysis
        print(f"\nAnalysis for {p1_name}:")
        print(f"  - Market Odds: {match['P1_ML']} ({p1_market_odds:.2f} dec)")
        print(f"  - Model Edge: {edge_p1:.2%}" if edge_p1 is not None else "  - Edge: N/A (Invalid Odds)")
        print(f"  - H2H Win Rate: {h2h_p1_win_rate:.2%}")
        if p1_passes_edge and p1_passes_h2h:
            print(f"  RECOMMENDATION: BET on {p1_name}")
        else:
            print(f"  RECOMMENDATION: NO BET (Edge Filter: {'PASS' if p1_passes_edge else 'FAIL'}, H2H Filter: {'PASS' if p1_passes_h2h else 'FAIL'})")
            
        # Player 2 Analysis
        print(f"\nAnalysis for {p2_name}:")
        print(f"  - Market Odds: {match['P2_ML']} ({p2_market_odds:.2f} dec)")
        print(f"  - Model Edge: {edge_p2:.2%}" if edge_p2 is not None else "  - Edge: N/A (Invalid Odds)")
        print(f"  - H2H Win Rate: {h2h_p2_win_rate:.2%}")
        if p2_passes_edge and p2_passes_h2h:
            print(f"  RECOMMENDATION: BET on {p2_name}")
        else:
            print(f"  RECOMMENDATION: NO BET (Edge Filter: {'PASS' if p2_passes_edge else 'FAIL'}, H2H Filter: {'PASS' if p2_passes_h2h else 'FAIL'})")
        print("---------------------------------")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
