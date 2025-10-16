import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 1. Configuration ---
FINAL_DATASET_FILE = "final_dataset_v7.4_no_duplicates.csv"
GBM_MODEL_FILE = "cpr_v7.4_gbm_specialist.joblib"
GBM_PREPROCESSOR_FILE = "gbm_preprocessor_v7.4.joblib"
LSTM_MODEL_FILE = "cpr_v7.1_lstm_specialist.h5"
LSTM_SCALER_FILE = "lstm_scaler_v7.1.joblib"
META_MODEL_FILE = "cpr_v7.4_meta_model.pkl"
ANALYSIS_LOG_FILE = "backtest_log_for_analysis_v7.4.csv"
EQUITY_CURVE_FILE = "equity_curve_v7.4.png"

INITIAL_BANKROLL = 1000
KELLY_FRACTION = 0.1
TRAIN_SPLIT_PERCENTAGE = 0.80
ROLLING_WINDOW = 10
SEQUENCE_LENGTH = 5

EDGE_THRESHOLD_MIN = 0.02
EDGE_THRESHOLD_MAX = 0.99
# --- FIX 1: Add a minimum odds denominator to prevent division by a small number ---
MIN_ODDS_DENOMINATOR = 0.10 # This means odds must be at least 1.10

# --- 2. Load Models and Prepare Data ---
try:
    print("--- Loading Data and Models for Symmetrical Analysis Back-test ---")
    df = pd.read_csv(FINAL_DATASET_FILE)
    df.dropna(subset=['P1_Win', 'Kickoff_P1_Odds', 'Kickoff_P2_Odds'], inplace=True)
#    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')

    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['Kickoff_P1_Odds'] = pd.to_numeric(df['Kickoff_P1_Odds'], errors='coerce')
    df['Kickoff_P2_Odds'] = pd.to_numeric(df['Kickoff_P2_Odds'], errors='coerce')
    df.dropna(subset=['Kickoff_P1_Odds', 'Kickoff_P2_Odds'], inplace=True)

    split_index = int(len(df) * TRAIN_SPLIT_PERCENTAGE)
    backtest_df = df.iloc[split_index:].copy() # Use .copy() to avoid SettingWithCopyWarning
    print(f"Loaded {len(df)} total matches. Back-testing on {len(backtest_df)} matches.")

    gbm_model = joblib.load(GBM_MODEL_FILE)
    gbm_preprocessor = joblib.load(GBM_PREPROCESSOR_FILE)
    lstm_model = load_model(LSTM_MODEL_FILE)
    lstm_scaler = joblib.load(LSTM_SCALER_FILE)
    meta_model = joblib.load(META_MODEL_FILE)

    # --- 3. Main Back-testing and Logging Loop ---
    print(f"\n--- Generating New Symmetrical Log with Wide Filters ---")

    col_group1 = ['P1_Rolling_Win_Rate_L10', 'P1_Rolling_Pressure_Points_L10', 'P1_Rest_Days']
    col_group2 = ['P2_Rolling_Win_Rate_L10', 'P2_Rolling_Pressure_Points_L10', 'P2_Rest_Days']
    
    bankroll = INITIAL_BANKROLL
    bet_count = 0
    total_staked = 0
    bet_log = [] 

    # --- FIX 2: Correctly log bankroll history after each bet ---
    bankroll_history_per_bet = [(backtest_df['Date'].iloc[0], INITIAL_BANKROLL)]

    for index, match in tqdm(backtest_df.iterrows(), total=backtest_df.shape[0]):
        history_df = df.iloc[:index]
        p1_id, p2_id = match['Player 1 ID'], match['Player 2 ID']
        p1_market_odds, p2_market_odds = match['Kickoff_P1_Odds'], match['Kickoff_P2_Odds']
        
        # (On-the-fly feature engineering code remains the same...)
        p1_last_date=history_df[(history_df['Player 1 ID']==p1_id)|(history_df['Player 2 ID']==p1_id)]['Date'].max()
        p2_last_date=history_df[(history_df['Player 1 ID']==p2_id)|(history_df['Player 2 ID']==p2_id)]['Date'].max()
        p1_rest=(match['Date']-p1_last_date).days if pd.notna(p1_last_date)else 30
        p2_rest=(match['Date']-p2_last_date).days if pd.notna(p2_last_date)else 30
        h2h_df=history_df[((history_df['Player 1 ID']==p1_id)&(history_df['Player 2 ID']==p2_id))|((history_df['Player 1 ID']==p2_id)&(history_df['Player 2 ID']==p1_id))]
        p1_h2h_wins=len(h2h_df[((h2h_df['Player 1 ID']==p1_id)&(h2h_df['P1_Win']==1))|((h2h_df['Player 2 ID']==p1_id)&(h2h_df['P1_Win']==0))])
        h2h_p1_win_rate=p1_h2h_wins/len(h2h_df) if len(h2h_df)>0 else 0.5
        p1_games=history_df[(history_df['Player 1 ID']==p1_id)|(history_df['Player 2 ID']==p1_id)].tail(ROLLING_WINDOW)
        p2_games=history_df[(history_df['Player 1 ID']==p2_id)|(history_df['Player 2 ID']==p2_id)].tail(ROLLING_WINDOW)
        p1_win_rate=p1_games.apply(lambda r:1 if(r['Player 1 ID']==p1_id and r['P1_Win']==1)or(r['Player 2 ID']==p1_id and r['P1_Win']==0)else 0,axis=1).mean()if not p1_games.empty else 0.5
        p2_win_rate=p2_games.apply(lambda r:1 if(r['Player 1 ID']==p2_id and r['P1_Win']==1)or(r['Player 2 ID']==p2_id and r['P1_Win']==0)else 0,axis=1).mean()if not p2_games.empty else 0.5
        win_rate_advantage=p1_win_rate - p2_win_rate
        p1_pressure_points = p1_games.apply(lambda r: r['P1 Pressure Points'] if r['Player 1 ID'] == p1_id else r['P2 Pressure Points'], axis=1).mean() if not p1_games.empty else 0.0
        p2_pressure_points = p2_games.apply(lambda r: r['P1 Pressure Points'] if r['Player 1 ID'] == p2_id else r['P2 Pressure Points'], axis=1).mean() if not p2_games.empty else 0.0
        pressure_points_advantage = p1_pressure_points - p2_pressure_points
        
        # (Prediction logic remains the same...)
        p1_seq_df = history_df[(history_df['Player 1 ID'] == p1_id) | (history_df['Player 2 ID'] == p1_id)].tail(SEQUENCE_LENGTH)
        p2_seq_df = history_df[(history_df['Player 1 ID'] == p2_id) | (history_df['Player 2 ID'] == p2_id)].tail(SEQUENCE_LENGTH)
        if len(p1_seq_df) < SEQUENCE_LENGTH or len(p2_seq_df) < SEQUENCE_LENGTH: continue
        p1_seq, p2_seq = [], []
        for _, row in p1_seq_df.iterrows(): p1_seq.append(np.concatenate([row[col_group1].values, row[col_group2].values, [row['H2H_P1_Win_Rate']]]) if row['Player 1 ID'] == p1_id else np.concatenate([row[col_group2].values, row[col_group1].values, [1 - row['H2H_P1_Win_Rate']]]))
        for _, row in p2_seq_df.iterrows(): p2_seq.append(np.concatenate([row[col_group1].values, row[col_group2].values, [row['H2H_P1_Win_Rate']]]) if row['Player 1 ID'] == p2_id else np.concatenate([row[col_group2].values, row[col_group1].values, [1 - row['H2H_P1_Win_Rate']]]))
        gbm_features = pd.DataFrame([{'Win_Rate_Advantage': win_rate_advantage, 'Pressure_Points_Advantage': pressure_points_advantage, 'Player 1 ID': p1_id, 'Player 2 ID': p2_id}])
        X_gbm_processed = gbm_preprocessor.transform(gbm_features)
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
        
        bet_details = None

        if edge_p1 > EDGE_THRESHOLD_MIN and edge_p1 < EDGE_THRESHOLD_MAX:
            if (p1_market_odds - 1) > MIN_ODDS_DENOMINATOR:
                bet_count += 1
                
                # --- ROBUST RISK CONTROL (3-STEP PROCESS) ---
                # 1. Calculate the raw Kelly-recommended fraction of bankroll
                kelly_fraction_rec = (edge_p1 / (p1_market_odds - 1)) * KELLY_FRACTION
                
                # 2. Cap this fraction at our hard limit (e.g., 5%)
                capped_fraction = min(kelly_fraction_rec, 0.05) # Using a safer 5% cap
                
                # 3. Calculate the final stake based on the capped, safe fraction
                stake = bankroll * capped_fraction
                
                total_staked += stake
                profit = stake * (p1_market_odds - 1) if actual_winner == 1 else -stake
                bankroll += profit
                bet_details = {'Bet_On_Player': match['Player 1'], 'Outcome': "Win" if actual_winner == 1 else "Loss", 'Profit': profit, 'Stake': stake, 'Model_Prob': model_prob_p1, 'Market_Odds': p1_market_odds, 'Edge': edge_p1}

        elif edge_p2 > EDGE_THRESHOLD_MIN and edge_p2 < EDGE_THRESHOLD_MAX:
            if (p2_market_odds - 1) > MIN_ODDS_DENOMINATOR:
                bet_count += 1

                # --- ROBUST RISK CONTROL (3-STEP PROCESS) ---
                # 1. Calculate the raw Kelly-recommended fraction of bankroll
                kelly_fraction_rec = (edge_p2 / (p2_market_odds - 1)) * KELLY_FRACTION
                
                # 2. Cap this fraction at our hard limit (e.g., 5%)
                capped_fraction = min(kelly_fraction_rec, 0.05) # Using a safer 5% cap
                
                # 3. Calculate the final stake based on the capped, safe fraction
                stake = bankroll * capped_fraction

                total_staked += stake
                profit = stake * (p2_market_odds - 1) if actual_winner == 0 else -stake
                bankroll += profit
                bet_details = {'Bet_On_Player': match['Player 2'], 'Outcome': "Win" if actual_winner == 0 else "Loss", 'Profit': profit, 'Stake': stake, 'Model_Prob': (1 - model_prob_p1), 'Market_Odds': p2_market_odds, 'Edge': edge_p2}

        # --- FIX 2 (CONTINUED): Log bankroll after every potential bet ---
        if bet_details:
            bankroll_history_per_bet.append((match['Date'], bankroll))
            
            p1_history_count = len(history_df[(history_df['Player 1 ID'] == p1_id) | (history_df['Player 2 ID'] == p1_id)])
            p2_history_count = len(history_df[(history_df['Player 1 ID'] == p2_id) | (history_df['Player 2 ID'] == p2_id)])
            log_entry = {'Match_ID': match['Match ID'],'Date': match['Date'].strftime('%Y-%m-%d'),'Player_1': match['Player 1'],'Player_2': match['Player 2'],'P1_History': p1_history_count,'P2_History': p2_history_count,'H2H_P1_Win_Rate': h2h_p1_win_rate,'Win_Rate_Advantage': win_rate_advantage}
            log_entry.update(bet_details)
            bet_log.append(log_entry)

    #    if bankroll <= 0:
    #        bankroll_history_per_bet.append((match['Date'], 0))
    #        print("Bankroll busted!")
    #        break

    # --- 4. Final Results Summary & Save Log ---
    print("\n--- Final Back-test Summary (Wide Filters, Symmetrical, Normalized Stake) ---")
    print(f"Initial Bankroll: ${INITIAL_BANKROLL:.2f}")
    print(f"Final Bankroll: ${bankroll:.2f}")
    final_profit = bankroll - INITIAL_BANKROLL
    roi = (final_profit / total_staked) * 100 if total_staked > 0 else 0
    print(f"Total Profit: ${final_profit:.2f}")
    print(f"Total Bets Placed: {bet_count}")
    print(f"Total Staked: ${total_staked:.2f}")
    print(f"Return on Investment (ROI): {roi:.2f}%")
    
    # --- Corrected Risk Analysis Calculation ---
    if len(bankroll_history_per_bet) > 1:
        history_df = pd.DataFrame(bankroll_history_per_bet, columns=['Date', 'Bankroll'])
        history_df['Date'] = pd.to_datetime(history_df['Date'])
        
        # Resample to get the last bankroll value for each day
        daily_history_df = history_df.set_index('Date').resample('D').last().ffill().reset_index()

        # 1. Maximum Drawdown Calculation
        daily_history_df['Peak'] = daily_history_df['Bankroll'].cummax()
        daily_history_df['Drawdown'] = (daily_history_df['Peak'] - daily_history_df['Bankroll']) / daily_history_df['Peak']
        max_drawdown = daily_history_df['Drawdown'].max()
        print(f"Maximum Drawdown: {max_drawdown:.2%}")

        # 2. Sharpe Ratio Calculation
        daily_history_df['Daily_Return'] = daily_history_df['Bankroll'].pct_change().fillna(0)
        if daily_history_df['Daily_Return'].std() > 0:
            sharpe_ratio = (daily_history_df['Daily_Return'].mean() / daily_history_df['Daily_Return'].std()) * np.sqrt(365) # Use 365 for daily data
            print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
        else:
            print("Sharpe Ratio: N/A (no volatility)")

        # 3. Equity Curve Plot (using the per-bet data for accuracy)
        plt.figure(figsize=(12, 6))
        plt.plot(history_df['Date'], history_df['Bankroll'])
        plt.title('Equity Curve (Bankroll Over Time)')
        plt.xlabel('Date')
        plt.ylabel('Bankroll ($)')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(EQUITY_CURVE_FILE)
        print(f"\n✅ Equity curve plot saved to '{EQUITY_CURVE_FILE}'")

    if bet_log:
        log_df = pd.DataFrame(bet_log)
        log_df.to_csv(ANALYSIS_LOG_FILE, index=False)
        print(f"\n✅ New, symmetrical analysis log saved to '{ANALYSIS_LOG_FILE}'")
        print(f"Total Bets Logged for Analysis: {len(log_df)}")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()