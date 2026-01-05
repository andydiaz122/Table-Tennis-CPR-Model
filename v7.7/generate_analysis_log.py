import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from tqdm import tqdm

# --- 1. Configuration ---
FINAL_DATASET_FILE = "final_dataset_v7.1.csv"
GBM_MODEL_FILE = "cpr_v7.1_gbm_specialist.joblib"
GBM_PREPROCESSOR_FILE = "gbm_preprocessor_v7.1.joblib"
LSTM_MODEL_FILE = "cpr_v7.1_lstm_specialist.h5"
LSTM_SCALER_FILE = "lstm_scaler_v7.1.joblib"
META_MODEL_FILE = "cpr_v7.1_meta_model.pkl"
ANALYSIS_LOG_FILE = "backtest_analysis_log_v7.2_FIXED.csv" # New file name

# --- NOTE: These thresholds are from your old script. ---
# --- For a true analysis, these should match your v7.2 backtest script. ---
EDGE_THRESHOLD_MIN = 0.1
EDGE_THRESHOLD_MAX = 0.99

TRAIN_SPLIT_PERCENTAGE = 0.7
SEQUENCE_LENGTH = 5

# --- 2. Load and Prepare Data ---
try:
    print("--- Loading All Models and Data for Forensic Analysis ---")
    df = pd.read_csv(FINAL_DATASET_FILE)
    # --- FIX: Load and clean odds for BOTH players ---
    df.dropna(subset=['P1_Win', 'Kickoff_P1_Odds', 'Kickoff_P2_Odds'], inplace=True)
    df['Kickoff_P1_Odds'] = pd.to_numeric(df['Kickoff_P1_Odds'], errors='coerce')
    df['Kickoff_P2_Odds'] = pd.to_numeric(df['Kickoff_P2_Odds'], errors='coerce')
    df.dropna(subset=['Kickoff_P1_Odds', 'Kickoff_P2_Odds'], inplace=True)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    split_index = int(len(df) * TRAIN_SPLIT_PERCENTAGE)
    train_df = df.iloc[:split_index]
    backtest_df = df.iloc[split_index:].copy()

    gbm_model = joblib.load(GBM_MODEL_FILE)
    gbm_preprocessor = joblib.load(GBM_PREPROCESSOR_FILE)
    lstm_model = load_model(LSTM_MODEL_FILE)
    lstm_scaler = joblib.load(LSTM_SCALER_FILE)
    meta_model = joblib.load(META_MODEL_FILE)

    # --- 3. Feature Engineering & Prediction (Using the non-point-in-time method from original script) ---
    print("--- Generating Predictions for Log Creation ---")
    backtest_df['Win_Rate_Advantage'] = backtest_df['P1_Rolling_Win_Rate_L10'] - backtest_df['P2_Rolling_Win_Rate_L10']
    backtest_df['Pressure_Points_Advantage'] = backtest_df['P1_Rolling_Pressure_Points_L10'] - backtest_df['P2_Rolling_Pressure_Points_L10']
    backtest_df['Rest_Advantage'] = backtest_df['P1_Rest_Days'] - backtest_df['P2_Rest_Days']
    
    X_gbm = backtest_df[['Win_Rate_Advantage', 'Pressure_Points_Advantage', 'Rest_Advantage', 'H2H_P1_Win_Rate', 'Player 1 ID', 'Player 2 ID']]
    X_gbm_processed = gbm_preprocessor.transform(X_gbm)
    gbm_preds = gbm_model.predict_proba(X_gbm_processed)[:, 1]

    player_history = {}
    features_to_sequence = [
        'P1_Rolling_Win_Rate_L10', 'P1_Rolling_Pressure_Points_L10', 'P1_Rest_Days',
        'P2_Rolling_Win_Rate_L10', 'P2_Rolling_Pressure_Points_L10', 'P2_Rest_Days', 'H2H_P1_Win_Rate'
    ]
    aligned_indices, X_p1_sequences, X_p2_sequences = [], [], []

    for index, row in df.iterrows():
        p1_id, p2_id = row['Player 1 ID'], row['Player 2 ID']
        if index >= split_index:
            p1_hist, p2_hist = player_history.get(p1_id, []), player_history.get(p2_id, [])
            if len(p1_hist) >= SEQUENCE_LENGTH and len(p2_hist) >= SEQUENCE_LENGTH and index in backtest_df.index:
                aligned_indices.append(index)
                X_p1_sequences.append(p1_hist[-SEQUENCE_LENGTH:])
                X_p2_sequences.append(p2_hist[-SEQUENCE_LENGTH:])
        
        p1_vec = np.concatenate([row[features_to_sequence[:3]].values, row[features_to_sequence[3:6]].values, [row['H2H_P1_Win_Rate']]])
        if p1_id not in player_history: player_history[p1_id] = []
        player_history[p1_id].append(p1_vec)
        
        p2_vec = np.concatenate([row[features_to_sequence[3:6]].values, row[features_to_sequence[:3]].values, [1 - row['H2H_P1_Win_Rate']]])
        if p2_id not in player_history: player_history[p2_id] = []
        player_history[p2_id].append(p2_vec)
    
    final_backtest_df = backtest_df.loc[aligned_indices].copy()
    gbm_preds = pd.Series(gbm_preds, index=backtest_df.index).loc[aligned_indices].values
    
    X_p1, X_p2 = np.array(X_p1_sequences), np.array(X_p2_sequences)
    nsamples, nsteps, nfeatures = X_p1.shape
    X_p1_scaled = lstm_scaler.transform(X_p1.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
    X_p2_scaled = lstm_scaler.transform(X_p2.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
    lstm_preds = lstm_model.predict([X_p1_scaled, X_p2_scaled]).flatten()
    X_meta = np.vstack([gbm_preds, lstm_preds]).T
    final_probs = meta_model.predict_proba(X_meta)[:, 1]
    final_backtest_df['Model_P1_Win_Prob'] = final_probs
    
    # --- 4. Run Back-test and Create Corrected Log ---
    print("\n--- Generating Corrected Analysis Log ---")
    bet_log = []

    for index, row in tqdm(final_backtest_df.iterrows(), total=final_backtest_df.shape[0]):
        # --- FIX: Calculate probabilities and edges for both players ---
        model_prob_p1 = row['Model_P1_Win_Prob']
        model_prob_p2 = 1 - model_prob_p1
        p1_market_odds = row['Kickoff_P1_Odds']
        p2_market_odds = row['Kickoff_P2_Odds']
        
        edge_p1 = model_prob_p1 * p1_market_odds - 1
        edge_p2 = model_prob_p2 * p2_market_odds - 1
        
        actual_winner = row['P1_Win']
        
        # --- FIX: Symmetrical bet evaluation and logging ---
        bet_details = None
        
        if edge_p1 > EDGE_THRESHOLD_MIN and edge_p1 < EDGE_THRESHOLD_MAX:
            profit = row['Stake'] * (p1_market_odds - 1) if actual_winner == 1 else -row['Stake']
            bet_details = {
                'Bet_On_Player': row['Player 1'],
                'Market_Odds': p1_market_odds,
                'Model_Prob': model_prob_p1,
                'Edge': edge_p1,
                'Outcome': "Win" if actual_winner == 1 else "Loss",
                'Profit': profit
            }

        elif edge_p2 > EDGE_THRESHOLD_MIN and edge_p2 < EDGE_THRESHOLD_MAX:
            profit = row['Stake'] * (p2_market_odds - 1) if actual_winner == 0 else -row['Stake']
            bet_details = {
                'Bet_On_Player': row['Player 2'],
                'Market_Odds': p2_market_odds,
                'Model_Prob': model_prob_p2,
                'Edge': edge_p2,
                'Outcome': "Win" if actual_winner == 0 else "Loss",
                'Profit': profit
            }

        if bet_details:
            p1_hist_count = len(train_df[(train_df['Player 1 ID'] == row['Player 1 ID']) | (train_df['Player 2 ID'] == row['Player 1 ID'])])
            p2_hist_count = len(train_df[(train_df['Player 1 ID'] == row['Player 2 ID']) | (train_df['Player 2 ID'] == row['Player 2 ID'])])

            log_entry = {
                'Match_ID': row['Match ID'],
                'Date': row['Date'].strftime('%Y-%m-%d'),
                'Player_1': row['Player 1'],
                'Player_2': row['Player 2'],
                'P1_Historical_Matches': p1_hist_count,
                'P2_Historical_Matches': p2_hist_count,
                'H2H_Advantage': row['H2H_P1_Win_Rate'],
                'Win_Rate_Advantage': row['Win_Rate_Advantage'],
                'Stake': row['Stake'], # Assuming stake is available from a previous run or calculation
            }
            log_entry.update(bet_details)
            bet_log.append(log_entry)

    # --- 5. Save the Analysis Log ---
    if bet_log:
        log_df = pd.DataFrame(bet_log)
        # Reorder columns for clarity
        column_order = [
            'Match_ID', 'Date', 'Player_1', 'Player_2', 'Bet_On_Player', 'Outcome', 
            'Profit', 'Stake', 'Model_Prob', 'Market_Odds', 'Edge', 
            'P1_Historical_Matches', 'P2_Historical_Matches', 'H2H_Advantage', 'Win_Rate_Advantage'
        ]
        log_df = log_df[column_order]
        log_df.to_csv(ANALYSIS_LOG_FILE, index=False)
        print(f"\n✅ Corrected, detailed bet analysis log saved to '{ANALYSIS_LOG_FILE}'")
    else:
        print("\nNo bets met the criteria to be logged.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
