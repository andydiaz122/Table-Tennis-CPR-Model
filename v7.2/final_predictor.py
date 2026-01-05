import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

# --- 1. Configuration ---
# ---!!!--- MANUALLY EDIT THIS SECTION FOR DAILY MATCHES ---!!!---
upcoming_matches = [
    {
        'Player 1 ID': 680338, 'Player 1 Name': 'Matej Pycha',
        'Player 2 ID': 708127, 'Player 2 Name': 'Leos Havrda',
        'P1_ML': 260, # Player 1 Moneyline Odds (e.g., +150)
        'P2_ML': -170 # Player 2 Moneyline Odds (e.g., -175)
    },
    {
        'Player 1 ID': 1088851, 'Player 1 Name': 'Oleksandr Kolisnyk',
        'Player 2 ID': 341888, 'Player 2 Name': 'Martin Huk',
        'P1_ML': -110,
        'P2_ML': -110
    }
]
# ---!!!--- ---!!!--- ---!!!--- ---!!!--- ---!!!--- ---!!!---

# --- File Paths ---
HISTORICAL_DATA_FILE = "final_dataset_v7.1.csv"
GBM_MODEL_FILE = "cpr_v7.1_gbm_specialist.joblib"
GBM_PREPROCESSOR_FILE = "gbm_preprocessor_v7.1.joblib"
LSTM_MODEL_FILE = "cpr_v7.1_lstm_specialist.h5"
LSTM_SCALER_FILE = "lstm_scaler_v7.1.joblib"
META_MODEL_FILE = "cpr_v7.1_meta_model.pkl"

# --- v7.2 Strategic Filter Parameters ---
EDGE_THRESHOLD_MIN = 0.02
EDGE_THRESHOLD_MAX = 0.99
ODDS_MAX_THRESHOLD = 3.0
H2H_ADVANTAGE_MAX_THRESHOLD = 0.667
WIN_RATE_ADVANTAGE_MAX_THRESHOLD = 0.233

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
    print("All v7.1 models and data loaded successfully.")

    # --- 3. Feature Engineering and Sequence Building ---
    print("\n--- Engineering Features & Building Sequences for New Matches ---")
    
    new_match_features = []
    X_p1_sequences, X_p2_sequences = [], []

    features_for_lstm = [
        'P1_Rolling_Win_Rate_L10', 'P1_Rolling_Pressure_Points_L10', 'P1_Rest_Days',
        'P2_Rolling_Win_Rate_L10', 'P2_Rolling_Pressure_Points_L10', 'P2_Rest_Days', 'H2H_P1_Win_Rate'
    ]
    
    for match in upcoming_matches:
        p1_id, p2_id = match['Player 1 ID'], match['Player 2 ID']
        
        # Engineer GBM Features
        p1_last_date = df_history[((df_history['Player 1 ID'] == p1_id) | (df_history['Player 2 ID'] == p1_id))]['Date'].max()
        p2_last_date = df_history[((df_history['Player 1 ID'] == p2_id) | (df_history['Player 2 ID'] == p2_id))]['Date'].max()
        p1_rest = (datetime.now() - p1_last_date).days if pd.notna(p1_last_date) else 30
        p2_rest = (datetime.now() - p2_last_date).days if pd.notna(p2_last_date) else 30

        h2h_df = df_history[((df_history['Player 1 ID'] == p1_id) & (df_history['Player 2 ID'] == p2_id)) | ((df_history['Player 1 ID'] == p2_id) & (df_history['Player 2 ID'] == p1_id))]
        p1_h2h_wins = len(h2h_df[((h2h_df['Player 1 ID'] == p1_id) & (h2h_df['P1_Win'] == 1)) | ((h2h_df['Player 2 ID'] == p1_id) & (h2h_df['P1_Win'] == 0))])
        h2h_p1_win_rate = p1_h2h_wins / len(h2h_df) if len(h2h_df) > 0 else 0.5

        p1_games = df_history[((df_history['Player 1 ID'] == p1_id) | (df_history['Player 2 ID'] == p1_id))].tail(ROLLING_WINDOW)
        p2_games = df_history[((df_history['Player 1 ID'] == p2_id) | (df_history['Player 2 ID'] == p2_id))].tail(ROLLING_WINDOW)
        
        p1_win_rate = p1_games.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).mean()
        p2_win_rate = p2_games.apply(lambda r: 1 if (r['Player 1 ID'] == p2_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p2_id and r['P1_Win'] == 0) else 0, axis=1).mean()
        p1_pressure = p1_games.apply(lambda r: r['P1 Pressure Points'] if r['Player 1 ID'] == p1_id else r['P2 Pressure Points'], axis=1).mean()
        p2_pressure = p2_games.apply(lambda r: r['P1 Pressure Points'] if r['Player 1 ID'] == p2_id else r['P2 Pressure Points'], axis=1).mean()

        match_data = match.copy()
        match_data.update({
            'Win_Rate_Advantage': p1_win_rate - p2_win_rate,
            'Pressure_Points_Advantage': p1_pressure - p2_pressure,
            'Rest_Advantage': p1_rest - p2_rest, 'H2H_P1_Win_Rate': h2h_p1_win_rate
        })
        new_match_features.append(match_data)
        
        # Build LSTM Sequences
        p1_seq_df = df_history[((df_history['Player 1 ID'] == p1_id) | (df_history['Player 2 ID'] == p1_id))].tail(SEQUENCE_LENGTH)
        p2_seq_df = df_history[((df_history['Player 1 ID'] == p2_id) | (df_history['Player 2 ID'] == p2_id))].tail(SEQUENCE_LENGTH)
        p1_seq, p2_seq = [], []
        if len(p1_seq_df) < SEQUENCE_LENGTH or len(p2_seq_df) < SEQUENCE_LENGTH:
            print(f"Warning: Not enough historical data for {match['Player 1 Name']} or {match['Player 2 Name']}. LSTM predictions may be less reliable.")
        for _, r in p1_seq_df.iterrows():
            p1_vec = np.concatenate([r[features_for_lstm[:3]].values, r[features_for_lstm[3:6]].values, [r['H2H_P1_Win_Rate']]])
            p1_seq.append(p1_vec)
        for _, r in p2_seq_df.iterrows():
            p2_vec = np.concatenate([r[features_for_lstm[3:6]].values, r[features_for_lstm[:3]].values, [1 - r['H2H_P1_Win_Rate']]])
            p2_seq.append(p2_vec)
        
        # Pad sequences if not long enough
        while len(p1_seq) < SEQUENCE_LENGTH: p1_seq.insert(0, np.zeros(len(features_for_lstm)))
        while len(p2_seq) < SEQUENCE_LENGTH: p2_seq.insert(0, np.zeros(len(features_for_lstm)))

        X_p1_sequences.append(p1_seq)
        X_p2_sequences.append(p2_seq)

    df_predict = pd.DataFrame(new_match_features)
    df_predict['P1_Market_Odds'] = df_predict['P1_ML'].apply(moneyline_to_decimal)
    df_predict['P2_Market_Odds'] = df_predict['P2_ML'].apply(moneyline_to_decimal)

    # --- 4. Generate Predictions ---
    print("\n--- Generating Final Predictions ---")
    X_gbm = df_predict[['Win_Rate_Advantage', 'Pressure_Points_Advantage', 'Rest_Advantage', 'H2H_P1_Win_Rate', 'Player 1 ID', 'Player 2 ID']]
    X_gbm_processed = gbm_preprocessor.transform(X_gbm)
    gbm_preds = gbm_model.predict_proba(X_gbm_processed)[:, 1]

    X_p1, X_p2 = np.array(X_p1_sequences), np.array(X_p2_sequences)
    nsamples, nsteps, nfeatures = X_p1.shape
    X_p1_scaled = lstm_scaler.transform(X_p1.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
    X_p2_scaled = lstm_scaler.transform(X_p2.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
    lstm_preds = lstm_model.predict([X_p1_scaled, X_p2_scaled], verbose=0).flatten()

    X_meta = np.vstack([gbm_preds, lstm_preds]).T
    final_probs = meta_model.predict_proba(X_meta)[:, 1]
    df_predict['P1_Win_Probability'] = final_probs

    # --- 5. Apply Strategic Filter and Display Results ---
    for index, row in df_predict.iterrows():
        p1_name, p2_name = row['Player 1 Name'], row['Player 2 Name']
        p1_prob = row['P1_Win_Probability']
        p2_prob = 1 - p1_prob
        p1_market_odds = row['P1_Market_Odds']
        p2_market_odds = row['P2_Market_Odds']

        edge_p1 = (p1_prob * p1_market_odds - 1) if p1_market_odds else None
        edge_p2 = (p2_prob * p2_market_odds - 1) if p2_market_odds else None

        # --- Filter for Player 1 ---
        p1_passes_filter = False
        if edge_p1 is not None:
            p1_passes_filter = all([
                edge_p1 > EDGE_THRESHOLD_MIN,
                edge_p1 < EDGE_THRESHOLD_MAX,
                p1_market_odds < ODDS_MAX_THRESHOLD,
                row['H2H_P1_Win_Rate'] < H2H_ADVANTAGE_MAX_THRESHOLD,
                row['Win_Rate_Advantage'] < WIN_RATE_ADVANTAGE_MAX_THRESHOLD
            ])

        # --- Filter for Player 2 (invert advantage metrics) ---
        p2_passes_filter = False
        if edge_p2 is not None:
            h2h_p2_win_rate = 1 - row['H2H_P1_Win_Rate']
            p2_win_rate_advantage = -row['Win_Rate_Advantage']
            p2_passes_filter = all([
                edge_p2 > EDGE_THRESHOLD_MIN,
                edge_p2 < EDGE_THRESHOLD_MAX,
                p2_market_odds < ODDS_MAX_THRESHOLD,
                h2h_p2_win_rate < H2H_ADVANTAGE_MAX_THRESHOLD,
                p2_win_rate_advantage < WIN_RATE_ADVANTAGE_MAX_THRESHOLD
            ])
            
        print("\n---------------------------------")
        print(f"Matchup: {p1_name} vs. {p2_name}")
        print(f"Predicted Win Probabilities:")
        print(f"  - {p1_name}: {p1_prob:.2%}")
        print(f"  - {p2_name}: {p2_prob:.2%}")
        
        print(f"\nAnalysis for {p1_name}:")
        print(f"  - Market Odds: {row['P1_ML']} ({p1_market_odds:.2f} dec)")
        print(f"  - Model's Perceived Edge: {edge_p1:.2%}" if edge_p1 is not None else "  - Edge: N/A (Invalid Odds)")
        
        print(f"\nAnalysis for {p2_name}:")
        print(f"  - Market Odds: {row['P2_ML']} ({p2_market_odds:.2f} dec)")
        print(f"  - Model's Perceived Edge: {edge_p2:.2%}" if edge_p2 is not None else "  - Edge: N/A (Invalid Odds)")

        print("\n--- Recommendation ---")
        if p1_passes_filter:
            print(f"✅ BET on {p1_name}")
        elif p2_passes_filter:
            print(f"✅ BET on {p2_name}")
        else:
            print("❌ NO BET")
        print("---------------------------------")

except FileNotFoundError as e:
    print(f"Error: A required file was not found. Please check file paths. Details: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
