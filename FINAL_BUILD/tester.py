import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from tqdm import tqdm

# --- 1. Configuration ---
FINAL_DATASET_FILE = "final_dataset_v7.4.csv" # Using the new symmetrical dataset
GBM_MODEL_FILE = "cpr_v7.1_gbm_specialist.joblib" # Using the newly retrained models
GBM_PREPROCESSOR_FILE = "gbm_preprocessor_v7.1.joblib"
LSTM_MODEL_FILE = "cpr_v7.1_lstm_specialist.h5"
LSTM_SCALER_FILE = "lstm_scaler_v7.1.joblib"
META_MODEL_FILE = "cpr_v7.1_meta_model.pkl"

INITIAL_BANKROLL = 1000
KELLY_FRACTION = 0.25
TRAIN_SPLIT_PERCENTAGE = 0.7
ROLLING_WINDOW = 10
SEQUENCE_LENGTH = 5

# --- v7.5 FINAL STRATEGIC FILTERS ---
EDGE_THRESHOLD_MIN = 0.10
EDGE_THRESHOLD_MAX = 0.99
H2H_DISADVANTAGE_THRESHOLD = 0.4

# --- 2. Load Models and Prepare Data ---
try:
    print("--- Loading Data and Models for FINAL v7.5 Back-test ---")
    df = pd.read_csv(FINAL_DATASET_FILE)
    df.dropna(subset=['P1_Win', 'Kickoff_P1_Odds', 'Kickoff_P2_Odds'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['Kickoff_P1_Odds'] = pd.to_numeric(df['Kickoff_P1_Odds'], errors='coerce')
    df['Kickoff_P2_Odds'] = pd.to_numeric(df['Kickoff_P2_Odds'], errors='coerce')
    df.dropna(subset=['Kickoff_P1_Odds', 'Kickoff_P2_Odds'], inplace=True)

    split_index = int(len(df) * TRAIN_SPLIT_PERCENTAGE)
    backtest_df = df.iloc[split_index:].copy()
    print(f"Loaded {len(df)} total matches. Back-testing on {len(backtest_df)} matches.")

    gbm_model = joblib.load(GBM_MODEL_FILE)
    gbm_preprocessor = joblib.load(GBM_PREPROCESSOR_FILE)
    lstm_model = load_model(LSTM_MODEL_FILE)
    lstm_scaler = joblib.load(LSTM_SCALER_FILE)
    meta_model = joblib.load(META_MODEL_FILE)

    # --- 3. Main Back-testing Loop ---
    print(f"\n--- Running FINAL Back-test with v7.5 Intelligent Filters ---")

    col_group1 = ['P1_Rolling_Win_Rate_L10', 'P1_Rolling_Pressure_Points_L10', 'P1_Rest_Days']
    col_group2 = ['P2_Rolling_Win_Rate_L10', 'P2_Rolling_Pressure_Points_L10', 'P2_Rest_Days']
    
    bankroll = INITIAL_BANKROLL
    bet_count = 0
    total_staked = 0

    for index, match in tqdm(backtest_df.iterrows(), total=backtest_df.shape[0]):
        # Point-in-time features are already pre-calculated in the v7.4 dataset
        p1_id, p2_id = match['Player 1 ID'], match['Player 2 ID']
        p1_market_odds, p2_market_odds = match['Kickoff_P1_Odds'], match['Kickoff_P2_Odds']
        
        # LSTM sequences still need to be built point-in-time
        history_df = df.iloc[:index]
        p1_seq_df = history_df[(history_df['Player 1 ID'] == p1_id) | (history_df['Player 2 ID'] == p1_id)].tail(SEQUENCE_LENGTH)
        p2_seq_df = history_df[(history_df['Player 1 ID'] == p2_id) | (history_df['Player 2 ID'] == p2_id)].tail(SEQUENCE_LENGTH)
        if len(p1_seq_df) < SEQUENCE_LENGTH or len(p2_seq_df) < SEQUENCE_LENGTH: continue
        
        p1_seq, p2_seq = [], []
        for _, row in p1_seq_df.iterrows():
            if row['Player 1 ID'] == p1_id: p1_seq.append(np.concatenate([row[col_group1].values, row[col_group2].values, [row['H2H_P1_Win_Rate']]]))
            else: p1_seq.append(np.concatenate([row[col_group2].values, row[col_group1].values, [1 - row['H2H_P1_Win_Rate']]]))
        for _, row in p2_seq_df.iterrows():
            if row['Player 1 ID'] == p2_id: p2_seq.append(np.concatenate([row[col_group1].values, row[col_group2].values, [row['H2H_P1_Win_Rate']]]))
            else: p2_seq.append(np.concatenate([row[col_group2].values, row[col_group1].values, [1 - row['H2H_P1_Win_Rate']]]))

        # Prediction
        gbm_features = match[['P1_Rolling_Win_Rate_L10', 'P2_Rolling_Win_Rate_L10', 'P1_Rest_Days', 'P2_Rest_Days', 'H2H_P1_Win_Rate', 'Player 1 ID', 'Player 2 ID']].copy()
        gbm_features['Win_Rate_Advantage'] = gbm_features['P1_Rolling_Win_Rate_L10'] - gbm_features['P2_Rolling_Win_Rate_L10']
        gbm_features['Rest_Advantage'] = gbm_features['P1_Rest_Days'] - gbm_features['P2_Rest_Days']
        gbm_features['Pressure_Points_Advantage'] = 0 # Placeholder if not used
        
        X_gbm_processed = gbm_preprocessor.transform(gbm_features.to_frame().T)
        gbm_pred = gbm_model.predict_proba(X_gbm_processed)[0, 1]
        
        X_p1, X_p2 = np.array([p1_seq]), np.array([p2_seq])
        nsamples, nsteps, nfeatures = X_p1.shape
        X_p1_scaled = lstm_scaler.transform(X_p1.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
        X_p2_scaled = lstm_scaler.transform(X_p2.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
        lstm_pred = lstm_model.predict([X_p1_scaled, X_p2_scaled], verbose=0)[0][0]
        
        X_meta = np.array([[gbm_pred, lstm_pred]])
        model_prob_p1 = meta_model.predict_proba(X_meta)[0, 1]
        
        edge_p1 = model_prob_p1 * p1_market_odds - 1
        edge_p2 = (1 - model_prob_p1) * p2_market_odds - 1
        actual_winner = match['P1_Win']
        h2h_p1_win_rate = match['H2H_P1_Win_Rate']

        # Applying v7.5 Filters
        if edge_p1 > EDGE_THRESHOLD_MIN and edge_p1 < EDGE_THRESHOLD_MAX:
            if h2h_p1_win_rate >= H2H_DISADVANTAGE_THRESHOLD:
                bet_count += 1
                if (p1_market_odds - 1) > 0:
                    stake = INITIAL_BANKROLL * (edge_p1 / (p1_market_odds - 1)) * KELLY_FRACTION
                    total_staked += stake
                    bankroll += stake * (p1_market_odds - 1) if actual_winner == 1 else -stake

        elif edge_p2 > EDGE_THRESHOLD_MIN and edge_p2 < EDGE_THRESHOLD_MAX:
            if (1 - h2h_p1_win_rate) >= H2H_DISADVANTAGE_THRESHOLD:
                bet_count += 1
                if (p2_market_odds - 1) > 0:
                    stake = INITIAL_BANKROLL * (edge_p2 / (p2_market_odds - 1)) * KELLY_FRACTION
                    total_staked += stake
                    bankroll += stake * (p2_market_odds - 1) if actual_winner == 0 else -stake
        
        if bankroll <= 0: print("Bankroll busted!"); break

    print("\n--- Final Back-test Summary (v7.5 Intelligent Filters) ---")
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
    import traceback
    traceback.print_exc()
