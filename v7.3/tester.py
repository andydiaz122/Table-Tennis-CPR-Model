import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from tqdm import tqdm # TQDM is a great library for progress bars

# --- 1. Configuration ---
FINAL_DATASET_FILE = "final_dataset_v7.1.csv"
GBM_MODEL_FILE = "cpr_v7.1_gbm_specialist.joblib"
GBM_PREPROCESSOR_FILE = "gbm_preprocessor_v7.1.joblib"
LSTM_MODEL_FILE = "cpr_v7.1_lstm_specialist.h5"
LSTM_SCALER_FILE = "lstm_scaler_v7.1.joblib"
META_MODEL_FILE = "cpr_v7.1_meta_model.pkl"

INITIAL_BANKROLL = 1000
KELLY_FRACTION = 0.25
TRAIN_SPLIT_PERCENTAGE = 0.7
ROLLING_WINDOW = 10
SEQUENCE_LENGTH = 5

# --- v7.2 STRATEGIC FILTER PARAMETERS ---
EDGE_THRESHOLD_MIN = 0.10   # NEW: Raised from 0.03 to 0.10
EDGE_THRESHOLD_MAX = 0.99
ODDS_MAX_THRESHOLD = 3.0
H2H_ADVANTAGE_MAX_THRESHOLD = 0.667
WIN_RATE_ADVANTAGE_MAX_THRESHOLD = 0.233
# NEW: Player History Filter thresholds
HISTORY_MIN_THRESHOLD_1 = 10 # Bet if min history is LESS than this
HISTORY_MIN_THRESHOLD_2 = 25 # Bet if min history is GREATER than this...
HISTORY_MAX_THRESHOLD = 50   # ...and LESS than this

# --- 2. Load Models and Prepare Data ---
try:
    print("--- Loading Data and Models for Back-test ---")
    df = pd.read_csv(FINAL_DATASET_FILE)
    df.dropna(subset=['P1_Win', 'Kickoff_P1_Odds', 'Kickoff_P2_Odds'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['Kickoff_P1_Odds'] = pd.to_numeric(df['Kickoff_P1_Odds'], errors='coerce')
    df['Kickoff_P2_Odds'] = pd.to_numeric(df['Kickoff_P2_Odds'], errors='coerce')
    df.dropna(subset=['Kickoff_P1_Odds', 'Kickoff_P2_Odds'], inplace=True)

    split_index = int(len(df) * TRAIN_SPLIT_PERCENTAGE)
    train_df = df.iloc[:split_index]
    backtest_df = df.iloc[split_index:]

    print(f"Loaded {len(df)} total matches.")
    print(f"Training set size: {len(train_df)}")
    print(f"Back-test set size: {len(backtest_df)}")

    gbm_model = joblib.load(GBM_MODEL_FILE)
    gbm_preprocessor = joblib.load(GBM_PREPROCESSOR_FILE)
    lstm_model = load_model(LSTM_MODEL_FILE)
    lstm_scaler = joblib.load(LSTM_SCALER_FILE)
    meta_model = joblib.load(META_MODEL_FILE)
    print("All v7.2 models and filters loaded successfully.")

    # --- 3. Main Back-testing Loop ---
    print("\n--- Running Point-in-Time Back-test Simulation (v7.2 Filters) ---")

    col_group1 = ['P1_Rolling_Win_Rate_L10', 'P1_Rolling_Pressure_Points_L10', 'P1_Rest_Days']
    col_group2 = ['P2_Rolling_Win_Rate_L10', 'P2_Rolling_Pressure_Points_L10', 'P2_Rest_Days']
    
    bankroll = INITIAL_BANKROLL
    bet_count = 0
    total_staked = 0

    for index, match in tqdm(backtest_df.iterrows(), total=backtest_df.shape[0]):
        history_df = df.iloc[:index]
        p1_id, p2_id = match['Player 1 ID'], match['Player 2 ID']
        p1_market_odds, p2_market_odds = match['Kickoff_P1_Odds'], match['Kickoff_P2_Odds']
        
        # --- FEATURE RE-CALCULATION (Point-in-Time Correct) ---
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

        p1_pressure = p1_games.apply(lambda r: r['P1 Pressure Points'] if r['Player 1 ID'] == p1_id else r['P2 Pressure Points'], axis=1).mean() if not p1_games.empty else 0
        p2_pressure = p2_games.apply(lambda r: r['P1 Pressure Points'] if r['Player 1 ID'] == p2_id else r['P2 Pressure Points'], axis=1).mean() if not p2_games.empty else 0
        pressure_advantage = p1_pressure - p2_pressure
        
        # --- MODEL PREDICTION ---
        gbm_features = pd.DataFrame([{'Win_Rate_Advantage': win_rate_advantage, 'Pressure_Points_Advantage': pressure_advantage, 'Rest_Advantage': rest_advantage, 'H2H_P1_Win_Rate': h2h_p1_win_rate, 'Player 1 ID': p1_id, 'Player 2 ID': p2_id}])
        X_gbm_processed = gbm_preprocessor.transform(gbm_features)
        gbm_pred = gbm_model.predict_proba(X_gbm_processed)[0, 1]
        
        p1_seq_df = history_df[(history_df['Player 1 ID'] == p1_id) | (history_df['Player 2 ID'] == p1_id)].tail(SEQUENCE_LENGTH)
        p2_seq_df = history_df[(history_df['Player 1 ID'] == p2_id) | (history_df['Player 2 ID'] == p2_id)].tail(SEQUENCE_LENGTH)
        if len(p1_seq_df) < SEQUENCE_LENGTH or len(p2_seq_df) < SEQUENCE_LENGTH: continue

        p1_seq = [np.concatenate([r[col_group1].values, r[col_group2].values, [r['H2H_P1_Win_Rate']]]) for _, r in p1_seq_df.iterrows()]
        p2_seq = [np.concatenate([r[col_group2].values, r[col_group1].values, [1-r['H2H_P1_Win_Rate']]]) for _, r in p2_seq_df.iterrows()]
        
        X_p1, X_p2 = np.array([p1_seq]), np.array([p2_seq])
        nsamples, nsteps, nfeatures = X_p1.shape
        X_p1_scaled = lstm_scaler.transform(X_p1.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
        X_p2_scaled = lstm_scaler.transform(X_p2.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
        lstm_pred = lstm_model.predict([X_p1_scaled, X_p2_scaled], verbose=0)[0][0]

        X_meta = np.array([[gbm_pred, lstm_pred]])
        model_prob_p1 = meta_model.predict_proba(X_meta)[0, 1]
        
        # --- v7.2 BETTING LOGIC ---
        
        # NEW: Player History Filter Calculation
        p1_history_count = len(history_df[(history_df['Player 1 ID'] == p1_id) | (history_df['Player 2 ID'] == p1_id)])
        p2_history_count = len(history_df[(history_df['Player 1 ID'] == p2_id) | (history_df['Player 2 ID'] == p2_id)])
        min_history = min(p1_history_count, p2_history_count)
        
        passes_history_filter = (min_history < HISTORY_MIN_THRESHOLD_1) or \
                                (HISTORY_MIN_THRESHOLD_2 <= min_history <= HISTORY_MAX_THRESHOLD)

        edge_p1 = model_prob_p1 * p1_market_odds - 1
        edge_p2 = (1 - model_prob_p1) * p2_market_odds - 1

        bet_on_p1 = all([
            passes_history_filter, # The new history filter
            edge_p1 > EDGE_THRESHOLD_MIN, edge_p1 < EDGE_THRESHOLD_MAX,
            p1_market_odds < ODDS_MAX_THRESHOLD,
            h2h_p1_win_rate < H2H_ADVANTAGE_MAX_THRESHOLD,
            win_rate_advantage < WIN_RATE_ADVANTAGE_MAX_THRESHOLD
        ])

        bet_on_p2 = all([
            passes_history_filter, # The new history filter
            edge_p2 > EDGE_THRESHOLD_MIN, edge_p2 < EDGE_THRESHOLD_MAX,
            p2_market_odds < ODDS_MAX_THRESHOLD,
            (1 - h2h_p1_win_rate) < H2H_ADVANTAGE_MAX_THRESHOLD,
            (-win_rate_advantage) < WIN_RATE_ADVANTAGE_MAX_THRESHOLD
        ])

        stake, profit = 0, 0
        actual_winner = match['P1_Win']

        if bet_on_p1:
            bet_count += 1
            if (p1_market_odds - 1) <= 0: continue
            kelly_stake_fraction = edge_p1 / (p1_market_odds - 1)
            stake = bankroll * kelly_stake_fraction * KELLY_FRACTION
            total_staked += stake
            profit = stake * (p1_market_odds - 1) if actual_winner == 1 else -stake
        elif bet_on_p2:
            bet_count += 1
            if (p2_market_odds - 1) <= 0: continue
            kelly_stake_fraction = edge_p2 / (p2_market_odds - 1)
            stake = bankroll * kelly_stake_fraction * KELLY_FRACTION
            total_staked += stake
            profit = stake * (p2_market_odds - 1) if actual_winner == 0 else -stake

        bankroll += profit
        if bankroll <= 0:
            print("Bankroll busted!"); break

    # --- 4. Final Results ---
    print("\n--- Final Point-in-Time Back-test Complete (v7.2 Filters) ---")
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
    import traceback
    traceback.print_exc()

