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
ANALYSIS_LOG_FILE = "backtest_log_for_analysis_v7.3.csv"

INITIAL_BANKROLL = 1000
KELLY_FRACTION = 0.25
TRAIN_SPLIT_PERCENTAGE = 0.7
ROLLING_WINDOW = 10
SEQUENCE_LENGTH = 5

# --- v7.3 STRATEGIC FILTERS (Set for Wide Analysis) ---
EDGE_THRESHOLD_MIN = 0.02
EDGE_THRESHOLD_MAX = 0.99

# --- 2. Load Models and Prepare Data ---
try:
    print("--- Loading Data and Models for a New, Correct Analysis ---")
    df = pd.read_csv(FINAL_DATASET_FILE)
    df.dropna(subset=['P1_Win', 'Kickoff_P1_Odds', 'Kickoff_P2_Odds'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['Kickoff_P1_Odds'] = pd.to_numeric(df['Kickoff_P1_Odds'], errors='coerce')
    df['Kickoff_P2_Odds'] = pd.to_numeric(df['Kickoff_P2_Odds'], errors='coerce')
    df.dropna(subset=['Kickoff_P1_Odds', 'Kickoff_P2_Odds'], inplace=True)

    split_index = int(len(df) * TRAIN_SPLIT_PERCENTAGE)
    backtest_df = df.iloc[split_index:]
    print(f"Loaded {len(df)} total matches. Back-testing on {len(backtest_df)} matches.")

    gbm_model = joblib.load(GBM_MODEL_FILE)
    gbm_preprocessor = joblib.load(GBM_PREPROCESSOR_FILE)
    lstm_model = load_model(LSTM_MODEL_FILE)
    lstm_scaler = joblib.load(LSTM_SCALER_FILE)
    meta_model = joblib.load(META_MODEL_FILE)

    # --- 3. Main Back-testing and Logging Loop ---
    print(f"\n--- Generating a New, Accurate Log with Wide Filters ---")

    col_group1 = ['P1_Rolling_Win_Rate_L10', 'P1_Rolling_Pressure_Points_L10', 'P1_Rest_Days']
    col_group2 = ['P2_Rolling_Win_Rate_L10', 'P2_Rolling_Pressure_Points_L10', 'P2_Rest_Days']
    
    bankroll = INITIAL_BANKROLL
    bet_count = 0
    total_staked = 0
    bet_log = [] 

    for index, match in tqdm(backtest_df.iterrows(), total=backtest_df.shape[0]):
        history_df = df.iloc[:index]
        p1_id, p2_id = match['Player 1 ID'], match['Player 2 ID']
        p1_market_odds, p2_market_odds = match['Kickoff_P1_Odds'], match['Kickoff_P2_Odds']
        
        # (Feature calculation and prediction logic remains the same)
        p1_last_date = history_df[(history_df['Player 1 ID'] == p1_id) | (history_df['Player 2 ID'] == p1_id)]['Date'].max()
        p2_last_date = history_df[(history_df['Player 1 ID'] == p2_id) | (history_df['Player 2 ID'] == p2_id)]['Date'].max()
        p1_rest = (match['Date'] - p1_last_date).days if pd.notna(p1_last_date) else 30
        p2_rest = (match['Date'] - p2_last_date).days if pd.notna(p2_last_date) else 30
        rest_advantage = p1_rest - p2_rest

        h2h_df = history_df[((history_df['Player 1 ID'] == p1_id) & (history_df['Player 2 ID'] == p2_id)) | ((history_df['Player 1 ID'] == p2_id) & (history_df['Player 2 ID'] == p1_id))]
        p1_h2h_wins = len(h2h_df[((h2h_df['Player 1 ID'] == p1_id) & (h2h_df['P1_Win'] == 1)) | ((h2h_df['Player 2 ID'] == p1_id) & (h2h_df['P1_Win'] == 0))])
        h2h_p1_win_rate = p1_h2h_wins / len(h2h_df) if len(h2h_df) > 0 else 0.5
        
        p1_games = history_df[(history_df['Player 1 ID'] == p1_id) | (history_df['Player 2 ID'] == p1_id)].tail(ROLLING_WINDOW)
        p2_games = history_df[(history_df['Player 1 ID'] == p2_id) | (history_df['Player 2 ID'] == p2_id)].tail(ROLLING_WINDOW)
        
        p1_win_rate = p1_games.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p1_games.empty else 0.5
        p2_win_rate = p2_games.apply(lambda r: 1 if (r['Player 1 ID'] == p2_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p2_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p2_games.empty else 0.5
        win_rate_advantage = p1_win_rate - p2_win_rate
        
        p1_seq_df = history_df[(history_df['Player 1 ID'] == p1_id) | (history_df['Player 2 ID'] == p1_id)].tail(SEQUENCE_LENGTH)
        p2_seq_df = history_df[(history_df['Player 1 ID'] == p2_id) | (history_df['Player 2 ID'] == p2_id)].tail(SEQUENCE_LENGTH)
        if len(p1_seq_df) < SEQUENCE_LENGTH or len(p2_seq_df) < SEQUENCE_LENGTH: continue
        
        gbm_features = pd.DataFrame([{'Win_Rate_Advantage': win_rate_advantage, 'Pressure_Points_Advantage': 0, 'Rest_Advantage': rest_advantage, 'H2H_P1_Win_Rate': h2h_p1_win_rate, 'Player 1 ID': p1_id, 'Player 2 ID': p2_id}])
        X_gbm_processed = gbm_preprocessor.transform(gbm_features)
        gbm_pred = gbm_model.predict_proba(X_gbm_processed)[0, 1]

        p1_seq = [np.concatenate([r[col_group1].values, r[col_group2].values, [r['H2H_P1_Win_Rate']]]) for _, r in p1_seq_df.iterrows()]
        p2_seq = [np.concatenate([r[col_group2].values, r[col_group1].values, [1-r['H2H_P1_Win_Rate']]]) for _, r in p2_seq_df.iterrows()]
        
        X_p1, X_p2 = np.array([p1_seq]), np.array([p2_seq])
        nsamples, nsteps, nfeatures = X_p1.shape
        X_p1_scaled = lstm_scaler.transform(X_p1.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
        X_p2_scaled = lstm_scaler.transform(X_p2.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
        lstm_pred = lstm_model.predict([X_p1_scaled, X_p2_scaled], verbose=0)[0][0]
        
        X_meta = np.array([[gbm_pred, lstm_pred]])
        model_prob_p1 = meta_model.predict_proba(X_meta)[0, 1]
        
        p1_history_count = len(history_df[(history_df['Player 1 ID'] == p1_id) | (history_df['Player 2 ID'] == p1_id)])
        p2_history_count = len(history_df[(history_df['Player 1 ID'] == p2_id) | (history_df['Player 2 ID'] == p2_id)])
        
        edge_p1 = model_prob_p1 * p1_market_odds - 1
        edge_p2 = (1 - model_prob_p1) * p2_market_odds - 1
        actual_winner = match['P1_Win']
        
        bet_details = None

        if edge_p1 > EDGE_THRESHOLD_MIN and edge_p1 < EDGE_THRESHOLD_MAX:
            bet_count += 1
            if (p1_market_odds - 1) > 0:
                kelly_stake_fraction = edge_p1 / (p1_market_odds - 1)
                # --- FIX APPLIED HERE ---
                stake = INITIAL_BANKROLL * kelly_stake_fraction * KELLY_FRACTION
                total_staked += stake
                profit = stake * (p1_market_odds - 1) if actual_winner == 1 else -stake
                bankroll += profit
                bet_details = {'Bet_On_Player': match['Player 1'], 'Outcome': "Win" if actual_winner == 1 else "Loss", 'Profit': profit, 'Stake': stake, 'Model_Prob': model_prob_p1, 'Market_Odds': p1_market_odds, 'Edge': edge_p1}

        elif edge_p2 > EDGE_THRESHOLD_MIN and edge_p2 < EDGE_THRESHOLD_MAX:
            bet_count += 1
            if (p2_market_odds - 1) > 0:
                kelly_stake_fraction = edge_p2 / (p2_market_odds - 1)
                # --- FIX APPLIED HERE ---
                stake = INITIAL_BANKROLL * kelly_stake_fraction * KELLY_FRACTION
                total_staked += stake
                profit = stake * (p2_market_odds - 1) if actual_winner == 0 else -stake
                bankroll += profit
                bet_details = {'Bet_On_Player': match['Player 2'], 'Outcome': "Win" if actual_winner == 0 else "Loss", 'Profit': profit, 'Stake': stake, 'Model_Prob': (1 - model_prob_p1), 'Market_Odds': p2_market_odds, 'Edge': edge_p2}
        
        if bet_details:
            log_entry = {
                'Match_ID': match['Match ID'], 'Date': match['Date'].strftime('%Y-%m-%d'),
                'Player_1': match['Player 1'], 'Player_2': match['Player 2'],
                'P1_History': p1_history_count, 'P2_History': p2_history_count,
                'H2H_P1_Win_Rate': h2h_p1_win_rate, 'Win_Rate_Advantage': win_rate_advantage,
            }
            log_entry.update(bet_details)
            bet_log.append(log_entry)
        
        # Bankroll busting is not relevant for analysis staking, but kept for safety
        if bankroll <= 0:
            print("Bankroll busted!")
            break

    # --- 4. Final Results Summary & Save Log ---
    print("\n--- Final Back-test Summary (Wide Filters, Normalized Stake) ---")
    print(f"Initial Bankroll: ${INITIAL_BANKROLL:.2f}")
    print(f"Final Bankroll: ${bankroll:.2f}")
    final_profit = bankroll - INITIAL_BANKROLL
    roi = (final_profit / total_staked) * 100 if total_staked > 0 else 0
    print(f"Total Profit: ${final_profit:.2f}")
    print(f"Total Bets Placed: {bet_count}")
    print(f"Total Staked: ${total_staked:.2f}")
    print(f"Return on Investment (ROI): {roi:.2f}%")

    if bet_log:
        log_df = pd.DataFrame(bet_log)
        log_df.to_csv(ANALYSIS_LOG_FILE, index=False)
        print(f"\n✅ New, accurate analysis log saved to '{ANALYSIS_LOG_FILE}'")
        print(f"Total Bets Logged for Analysis: {len(log_df)}")
    else:
        print("No bets met the wide filter criteria.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

