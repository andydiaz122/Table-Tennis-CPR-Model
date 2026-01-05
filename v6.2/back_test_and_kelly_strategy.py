import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# --- 1. Configuration ---
BACKTEST_DATA_FILE = "backtest_data_with_odds.csv"
# Input Model Files
GBM_MODEL_FILE = "cpr_v6.2_gbm_specialist.joblib"
GBM_PREPROCESSOR_FILE = "gbm_preprocessor.joblib"
LSTM_MODEL_FILE = "cpr_v6.2_lstm_specialist.h5"
LSTM_SCALER_FILE = "lstm_scaler.joblib"
META_MODEL_FILE = "cpr_v6.2_meta_model.pkl"

# Back-test Configuration
INITIAL_BANKROLL = 1000
KELLY_FRACTION = 0.25 # Use a fraction of the Kelly stake to be less aggressive
# --- NEW: Only bet if the perceived edge is greater than this value ---
EDGE_THRESHOLD_lower = 0.001 # This is 2%
EDGE_THRESHOLD_upper = 0.084 

# --- 2. Load All Models and Data ---
try:
    print("--- Loading All Models and Data for Back-test ---")
    df = pd.read_csv(BACKTEST_DATA_FILE)
    df.dropna(inplace=True)

    # Load Models
    gbm_model = joblib.load(GBM_MODEL_FILE)
    gbm_preprocessor = joblib.load(GBM_PREPROCESSOR_FILE)
    lstm_model = load_model(LSTM_MODEL_FILE)
    lstm_scaler = joblib.load(LSTM_SCALER_FILE)
    meta_model = joblib.load(META_MODEL_FILE)
    print("All assets loaded successfully.")

    # --- 3. Prepare Data for Prediction ---
    # This process must be identical to the training process
    # A. Feature Engineering (creating delta features)
    df['Total_Points_Advantage'] = df['P1 Total Points'] - df['P2 Total Points']
    df['Pressure_Points_Advantage'] = df['P1 Pressure Points'] - df['P2 Pressure Points']
    df['Comeback_Advantage'] = df['P1 Set Comebacks'] - df['P2 Set Comebacks']
    
    # B. LSTM Sequence Creation (using a simplified history from the test set itself)
    # Note: A production system would use a larger, pre-compiled history.
    player_history = {}
    features_to_sequence = ['Total_Points_Advantage', 'Pressure_Points_Advantage', 'Comeback_Advantage']
    
    aligned_indices, X_p1_sequences, X_p2_sequences = [], [], []
    for index, row in df.iterrows():
        p1_id, p2_id = row['Player 1 ID'], row['Player 2 ID']
        p1_hist = player_history.get(p1_id, [])
        p2_hist = player_history.get(p2_id, [])
        if len(p1_hist) >= 5 and len(p2_hist) >= 5:
            aligned_indices.append(index)
            X_p1_sequences.append(p1_hist[-5:])
            X_p2_sequences.append(p2_hist[-5:])
        if p1_id not in player_history: player_history[p1_id] = []
        player_history[p1_id].append(row[features_to_sequence].values)
        if p2_id not in player_history: player_history[p2_id] = []
        player_history[p2_id].append(-row[features_to_sequence].values)
    
    # Filter the main dataframe to only include matches where we could create a sequence
    backtest_df = df.loc[aligned_indices].copy()

    # --- 4. Generate Predictions from the Full Model Ensemble ---
    print(f"\nGenerating final predictions for {len(backtest_df)} matches...")
    # GBM Predictions
    X_gbm = backtest_df[['Total_Points_Advantage', 'Pressure_Points_Advantage', 'Comeback_Advantage', 'Player 1 ID', 'Player 2 ID']]
    X_gbm_processed = gbm_preprocessor.transform(X_gbm)
    gbm_preds = gbm_model.predict_proba(X_gbm_processed)[:, 1]

    # LSTM Predictions
    X_p1 = np.array(X_p1_sequences)
    X_p2 = np.array(X_p2_sequences)
    nsamples, nsteps, nfeatures = X_p1.shape
    X_p1_scaled = lstm_scaler.transform(X_p1.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
    X_p2_scaled = lstm_scaler.transform(X_p2.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
    lstm_preds = lstm_model.predict([X_p1_scaled, X_p2_scaled]).flatten()

    # Meta-Model Final Prediction
    X_meta = np.vstack([gbm_preds, lstm_preds]).T
    final_probs = meta_model.predict_proba(X_meta)[:, 1]
    backtest_df['Model_P1_Win_Prob'] = final_probs
    
    # --- 5. Run the Kelly Criterion Back-test ---
    print("\n--- Running Back-test Simulation ---")
    bankroll = INITIAL_BANKROLL
    bet_count = 0
    total_staked = 0

    for index, row in backtest_df.iterrows():
        model_prob = row['Model_P1_Win_Prob']
        market_odds = row['P1_Simulated_Odds']
        
        # Kelly Criterion: stake_fraction = (probability * odds - 1) / (odds - 1)
        # We bet if the model finds an "edge" (model_prob * market_odds > 1)
        edge = model_prob * market_odds - 1
        
        # --- THIS IS THE MODIFIED LINE ---
        if EDGE_THRESHOLD_lower < edge < EDGE_THRESHOLD_upper:
            bet_count += 1
            kelly_stake_fraction = edge / (market_odds - 1)
            stake = bankroll * kelly_stake_fraction * KELLY_FRACTION
            stake = min(stake, bankroll)
            total_staked += stake
            
            p1_actual_score = int(str(row['Final Score']).strip('="').split('-')[0])
            p2_actual_score = int(str(row['Final Score']).strip('="').split('-')[1])
            actual_winner = 1 if p1_actual_score > p2_actual_score else 0

            if actual_winner == 1: # We won the bet
                profit = stake * (market_odds - 1)
                bankroll += profit
                print(f"Bet {bet_count}: WON +${profit:.2f} | New Bankroll: ${bankroll:.2f}")
            else: # We lost the bet
                bankroll -= stake
                print(f"Bet {bet_count}: LOST -${stake:.2f} | New Bankroll: ${bankroll:.2f}")

    # --- 6. Final Results ---
    print("\n--- Back-test Complete ---")
    print(f"Initial Bankroll: ${INITIAL_BANKROLL:.2f}")
    print(f"Final Bankroll: ${bankroll:.2f}")
    profit = bankroll - INITIAL_BANKROLL
    roi = (profit / total_staked) * 100 if total_staked > 0 else 0
    print(f"Total Profit: ${profit:.2f}")
    print(f"Total Bets Placed: {bet_count}")
    print(f"Total Staked: ${total_staked:.2f}")
    print(f"Return on Investment (ROI): {roi:.2f}%")

except Exception as e:
    print(f"An error occurred: {e}")