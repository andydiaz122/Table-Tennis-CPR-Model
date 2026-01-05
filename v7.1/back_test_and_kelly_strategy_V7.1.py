import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# --- 1. Configuration ---
FINAL_DATASET_FILE = "final_dataset_v7.1.csv"
GBM_MODEL_FILE = "cpr_v7.1_gbm_specialist.joblib"
GBM_PREPROCESSOR_FILE = "gbm_preprocessor_v7.1.joblib"
LSTM_MODEL_FILE = "cpr_v7.1_lstm_specialist.h5"
LSTM_SCALER_FILE = "lstm_scaler_v7.1.joblib"
META_MODEL_FILE = "cpr_v7.1_meta_model.pkl"
# --- NEW: Output file for our analysis ---
ANALYSIS_LOG_FILE = "backtest_analysis_log.csv"

INITIAL_BANKROLL = 1000
KELLY_FRACTION = 0.25
EDGE_THRESHOLD_MIN = 0.03   # Bet if edge is > 2%
EDGE_THRESHOLD_MAX = 0.99   # Bet if edge is < 10%
TRAIN_SPLIT_PERCENTAGE = 0.7    

# --- 2. Load and Prepare Data ---
try:
    print("--- Loading Final Dataset for Back-test ---")
    df = pd.read_csv(FINAL_DATASET_FILE)
    df.dropna(subset=['P1_Win', 'Kickoff_P1_Odds'], inplace=True)
    df['Kickoff_P1_Odds'] = pd.to_numeric(df['Kickoff_P1_Odds'], errors='coerce')
    df.dropna(subset=['Kickoff_P1_Odds'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    split_index = int(len(df) * TRAIN_SPLIT_PERCENTAGE)
    backtest_df = df.iloc[split_index:].copy()
    print(f"Loaded and prepared {len(backtest_df)} matches for the v7.0 back-test.")

    # Load Models
    gbm_model = joblib.load(GBM_MODEL_FILE)
    gbm_preprocessor = joblib.load(GBM_PREPROCESSOR_FILE)
    lstm_model = load_model(LSTM_MODEL_FILE)
    lstm_scaler = joblib.load(LSTM_SCALER_FILE)
    meta_model = joblib.load(META_MODEL_FILE)
    print("All v7.0 models loaded successfully.")

    # --- 3. Feature Engineering & Prediction ---
    backtest_df['Win_Rate_Advantage'] = backtest_df['P1_Rolling_Win_Rate_L10'] - backtest_df['P2_Rolling_Win_Rate_L10']
    backtest_df['Pressure_Points_Advantage'] = backtest_df['P1_Rolling_Pressure_Points_L10'] - backtest_df['P2_Rolling_Pressure_Points_L10']
    backtest_df['Rest_Advantage'] = backtest_df['P1_Rest_Days'] - backtest_df['P2_Rest_Days']
    
    print(f"\nGenerating final predictions for {len(backtest_df)} matches...")
    X_gbm = backtest_df[['Win_Rate_Advantage', 'Pressure_Points_Advantage', 'Rest_Advantage', 'H2H_P1_Win_Rate', 'Player 1 ID', 'Player 2 ID']]
    X_gbm_processed = gbm_preprocessor.transform(X_gbm)
    gbm_preds = gbm_model.predict_proba(X_gbm_processed)[:, 1]

    player_history = {}
    features_to_sequence = [
        'P1_Rolling_Win_Rate_L10', 'P1_Rolling_Pressure_Points_L10', 'P1_Rest_Days',
        'P2_Rolling_Win_Rate_L10', 'P2_Rolling_Pressure_Points_L10', 'P2_Rest_Days',
        'H2H_P1_Win_Rate'
    ]
    aligned_indices, X_p1_sequences, X_p2_sequences = [], [], []
    full_history_df = df.copy() 
    for index, row in full_history_df.iterrows():
        p1_id, p2_id = row['Player 1 ID'], row['Player 2 ID']
        is_test_row = index >= split_index
        p1_hist, p2_hist = player_history.get(p1_id, []), player_history.get(p2_id, [])
        if is_test_row and len(p1_hist) >= 5 and len(p2_hist) >= 5:
            if index in backtest_df.index:
                aligned_indices.append(index)
                X_p1_sequences.append(p1_hist[-5:])
                X_p2_sequences.append(p2_hist[-5:])
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
    
    # --- 4. Run Back-test and Create Log ---
    print("\n--- Running Final Back-test and Generating Analysis Log ---")
    bankroll, bet_count, total_staked = INITIAL_BANKROLL, 0, 0
    bet_log = []

    for index, row in final_backtest_df.iterrows():
        model_prob = row['Model_P1_Win_Prob']
        market_odds = row['Kickoff_P1_Odds']
        edge = model_prob * market_odds - 1
        
        if EDGE_THRESHOLD_MIN < edge < EDGE_THRESHOLD_MAX:
            bet_count += 1
            if (market_odds - 1) <= 0: continue
            kelly_stake_fraction = edge / (market_odds - 1)
            stake = bankroll * kelly_stake_fraction * KELLY_FRACTION
            stake = min(stake, bankroll)
            total_staked += stake
            
            actual_winner = row['P1_Win']
            profit = 0
            outcome = "Loss"
            if actual_winner == 1:
                profit = stake * (market_odds - 1)
                bankroll += profit
                outcome = "Win"
            else:
                profit = -stake
                bankroll -= stake

            # --- NEW: Log detailed information about the bet ---
            bet_log.append({
                'Match_ID': row['Match ID'],
                'Outcome': outcome,
                'Profit': profit,
                'Model_Prob': model_prob,
                'Market_Odds': market_odds,
                'Edge': edge,
                'Stake': stake,
                'H2H_Advantage': row['H2H_P1_Win_Rate'],
                'Win_Rate_Advantage': row['Win_Rate_Advantage'],
                'Rest_Advantage': row['Rest_Advantage']
            })

    # --- 5. Save the Analysis Log ---
    log_df = pd.DataFrame(bet_log)
    log_df.to_csv(ANALYSIS_LOG_FILE, index=False)
    print(f"\n✅ Detailed bet analysis log saved to '{ANALYSIS_LOG_FILE}'")

    # --- 6. Final Results ---
    print("\n--- Final v7.0 Back-test Complete ---")
    print(f"Initial Bankroll: ${INITIAL_BANKROLL:.2f}")
    print(f"Final Bankroll: ${bankroll:.2f}")
    final_profit = bankroll - INITIAL_BANKROLL
    roi = (final_profit / total_staked) * 100 if total_staked > 0 else 0
    print(f"Total Profit: ${final_profit:.2f}")
    print(f"Total Bets Placed: {bet_count}")
    print(f"Total Staked: ${total_staked:.2f}")
    print(f"Return on Investment (ROI): {roi:.2f}%")

except Exception as e:
    print(f"An error occurred: {e}")