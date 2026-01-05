import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 1. Configuration ---
# FINAL_DATASET_FILE = "final_dataset_v7.4_no_duplicates.csv"
FINAL_DATASET_FILE = "final_dataset_v7.4.csv"
GBM_MODEL_FILE = "cpr_v7.4_gbm_specialist.joblib"
GBM_PREPROCESSOR_FILE = "gbm_preprocessor_v7.4.joblib"
# LSTM_MODEL_FILE = "cpr_v7.1_lstm_specialist.h5"
# LSTM_SCALER_FILE = "lstm_scaler_v7.1.joblib"
# META_MODEL_FILE = "cpr_v7.4_meta_model.pkl"
ANALYSIS_LOG_FILE = "backtest_log_for_analysis_v7.4.csv"
EQUITY_CURVE_FILE = "equity_curve_v7.4.png"
ROI_PLOT_FILE = "roi_per_bet_v7.4.png" # NEW: Filename for the ROI plot

INITIAL_BANKROLL = 1000
KELLY_FRACTION = 0.01
TRAIN_SPLIT_PERCENTAGE = 0.70
ROLLING_WINDOW = 20
SEQUENCE_LENGTH = 5

# ADDED: Warm-up period to ensure players have a minimum history before betting.
MIN_GAMES_THRESHOLD = 5

EDGE_THRESHOLD_MIN = 0.01
EDGE_THRESHOLD_MAX = 0.99
MIN_ODDS_DENOMINATOR = 0.10

# --- 2. Load Models and Prepare Data ---
try:
    print("--- Loading Data and Models for Symmetrical Analysis Back-test ---")
#    df = pd.read_csv(FINAL_DATASET_FILE)
    df = pd.read_csv(
        FINAL_DATASET_FILE,
        na_values=['-'], 
        keep_default_na=True, # Also respects empty strings/default missing values as NaN
        low_memory=False # Prevents Pandas from guessing types chunk-by-chunk
    )
    # --- DATA CLEANING AND PREPARATION ---
    df.drop_duplicates(subset=['Date', 'Player 1', 'Player 2'], keep='first', inplace=True)
    df.rename(columns={'P1 Pressure Points': 'P1_Pressure_Points', 'P2 Pressure Points': 'P2_Pressure_Points'}, inplace=True, errors='ignore')
    
    # MODIFIED: Added 'Line_Movement_P1' and corrected odds columns.
    required_cols = ['P1_Win', 'P1_Kickoff_Odds', 'P2_Kickoff_Odds', 'P1_Pressure_Points', 'P2_Pressure_Points', 'Line_Movement_P1']
    df.dropna(subset=required_cols, inplace=True)
    
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['P1_Kickoff_Odds'] = pd.to_numeric(df['P1_Kickoff_Odds'], errors='coerce')
    df['P2_Kickoff_Odds'] = pd.to_numeric(df['P2_Kickoff_Odds'], errors='coerce')
    df.dropna(subset=['P1_Kickoff_Odds', 'P2_Kickoff_Odds'], inplace=True)

    split_index = int(len(df) * TRAIN_SPLIT_PERCENTAGE)
    backtest_df = df.iloc[split_index:].copy()
    print(f"Loaded {len(df)} total matches. Back-testing on {len(backtest_df)} matches.")

    gbm_model = joblib.load(GBM_MODEL_FILE)
    gbm_preprocessor = joblib.load(GBM_PREPROCESSOR_FILE)
#    lstm_model = load_model(LSTM_MODEL_FILE)
#    lstm_scaler = joblib.load(LSTM_SCALER_FILE)
#    meta_model = joblib.load(META_MODEL_FILE)

    # --- 3. Main Back-testing and Logging Loop ---
    print(f"\n--- Generating New Symmetrical Log with Wide Filters ---")
    
    bankroll = INITIAL_BANKROLL
    bet_count = 0
    total_staked = 0
    bet_log = [] 

    bankroll_history_per_bet = [(backtest_df['Date'].iloc[0], INITIAL_BANKROLL)]

    daily_betting_bankroll = INITIAL_BANKROLL
    last_processed_date = None

    for index, match in tqdm(backtest_df.iterrows(), total=backtest_df.shape[0]):
        
        current_date = match['Date'].date()
        if last_processed_date != current_date:
            daily_betting_bankroll = bankroll
            last_processed_date = current_date

        history_df = df.iloc[:index]
        p1_id, p2_id = match['Player 1 ID'], match['Player 2 ID']
        p1_market_odds, p2_market_odds = match['P1_Kickoff_Odds'], match['P2_Kickoff_Odds']
        
        # --- On-the-fly Feature Engineering for GBM model ---
        p1_games_gbm = history_df[(history_df['Player 1 ID'] == p1_id) | (history_df['Player 2 ID'] == p1_id)].tail(ROLLING_WINDOW)
        p2_games_gbm = history_df[(history_df['Player 1 ID'] == p2_id) | (history_df['Player 2 ID'] == p2_id)].tail(ROLLING_WINDOW)
        
        # ADDED: Skip matches if players don't have enough historical data.
        if len(p1_games_gbm) < MIN_GAMES_THRESHOLD or len(p2_games_gbm) < MIN_GAMES_THRESHOLD:
            continue

        # MODIFIED: Get the pre-calculated line movement feature for Player 1.
        line_movement_p1 = match['Line_Movement_P1']

        p1_win_rate = p1_games_gbm.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p1_games_gbm.empty else 0.5
        p2_win_rate = p2_games_gbm.apply(lambda r: 1 if (r['Player 1 ID'] == p2_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p2_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p2_games_gbm.empty else 0.5
        win_rate_advantage = p1_win_rate - p2_win_rate
        
        p1_pressure_points = p1_games_gbm.apply(lambda r: r['P1_Pressure_Points'] if r['Player 1 ID'] == p1_id else r['P2_Pressure_Points'], axis=1).mean() if not p1_games_gbm.empty else 0.0
        p2_pressure_points = p2_games_gbm.apply(lambda r: r['P1_Pressure_Points'] if r['Player 1 ID'] == p2_id else r['P2_Pressure_Points'], axis=1).mean() if not p2_games_gbm.empty else 0.0
        pressure_points_advantage = p1_pressure_points - p2_pressure_points

        p1_set_comebacks = p1_games_gbm.apply(lambda r: r['P1 Set Comebacks'] if r['Player 1 ID'] == p1_id else r['P2 Set Comebacks'], axis=1).sum() if not p1_games_gbm.empty else 0.0
        p2_set_comebacks = p2_games_gbm.apply(lambda r: r['P1 Set Comebacks'] if r['Player 1 ID'] == p2_id else r['P2 Set Comebacks'], axis=1).sum() if not p2_games_gbm.empty else 0.0
        set_comebacks_advantage = p1_set_comebacks - p2_set_comebacks

        # --- Model Prediction ---
        gbm_features = pd.DataFrame([{'Line_Movement_P1': line_movement_p1, 'Win_Rate_Advantage': win_rate_advantage, 'Pressure_Points_Advantage': pressure_points_advantage, 'Set_Comebacks_Advantage': set_comebacks_advantage}])
        X_gbm_processed = gbm_preprocessor.transform(gbm_features)
        gbm_pred = gbm_model.predict_proba(X_gbm_processed)[0, 1]
        
        model_prob_p1 = gbm_pred 

        # --- Betting Logic ---
        edge_p1 = model_prob_p1 * p1_market_odds - 1
        edge_p2 = (1 - model_prob_p1) * p2_market_odds - 1
        actual_winner = match['P1_Win']
        bet_details = None
        if bankroll <= 0: continue

        if edge_p1 > EDGE_THRESHOLD_MIN and edge_p1 < EDGE_THRESHOLD_MAX:
            if (p1_market_odds - 1) > MIN_ODDS_DENOMINATOR:
                bet_count += 1
                kelly_fraction_rec = (edge_p1 / (p1_market_odds - 1)) * KELLY_FRACTION
                capped_fraction = min(kelly_fraction_rec, 0.05)
                if win_rate_advantage < 0: capped_fraction /= 4
                stake = daily_betting_bankroll * capped_fraction
                total_staked += stake
                profit = stake * (p1_market_odds - 1) if actual_winner == 1 else -stake
                bankroll += profit
                bet_details = {'Bet_On_Player': match['Player 1'], 'Outcome': "Win" if actual_winner == 1 else "Loss", 'Profit': profit, 'Stake': stake, 'Model_Prob': model_prob_p1, 'Market_Odds': p1_market_odds, 'Edge': edge_p1}
        elif edge_p2 > EDGE_THRESHOLD_MIN and edge_p2 < EDGE_THRESHOLD_MAX:
            if (p2_market_odds - 1) > MIN_ODDS_DENOMINATOR:
                bet_count += 1
                kelly_fraction_rec = (edge_p2 / (p2_market_odds - 1)) * KELLY_FRACTION
                capped_fraction = min(kelly_fraction_rec, 0.05)
                if win_rate_advantage > 0: capped_fraction /= 4
                stake = daily_betting_bankroll * capped_fraction
                total_staked += stake
                profit = stake * (p2_market_odds - 1) if actual_winner == 0 else -stake
                bankroll += profit
                bet_details = {'Bet_On_Player': match['Player 2'], 'Outcome': "Win" if actual_winner == 0 else "Loss", 'Profit': profit, 'Stake': stake, 'Model_Prob': (1 - model_prob_p1), 'Market_Odds': p2_market_odds, 'Edge': edge_p2}

        if bet_details:
            bankroll_history_per_bet.append((match['Date'], bankroll))
            h2h_df_log = history_df[((history_df['Player 1 ID']==p1_id)&(history_df['Player 2 ID']==p2_id))|((history_df['Player 1 ID']==p2_id)&(history_df['Player 2 ID']==p1_id))]
            p1_h2h_wins_log = len(h2h_df_log[((h2h_df_log['Player 1 ID']==p1_id)&(h2h_df_log['P1_Win']==1))|((h2h_df_log['Player 2 ID']==p1_id)&(h2h_df_log['P1_Win']==0))])
            h2h_p1_win_rate_log = p1_h2h_wins_log/len(h2h_df_log) if len(h2h_df_log)>0 else 0.5
            log_entry = {'Match_ID': match['Match_ID'],'Date': match['Date'].strftime('%Y-%m-%d'),'Player_1': match['Player 1'],'Player_2': match['Player 2'],'H2H_P1_Win_Rate': h2h_p1_win_rate_log,'Win_Rate_Advantage': win_rate_advantage, 'Line_Movement_P1': line_movement_p1}
            log_entry.update(bet_details)
            bet_log.append(log_entry)

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
    
    if len(bankroll_history_per_bet) > 1:
        history_df = pd.DataFrame(bankroll_history_per_bet, columns=['Date', 'Bankroll'])
        history_df['Date'] = pd.to_datetime(history_df['Date'])
        daily_history_df = history_df.set_index('Date').resample('D').last().ffill().reset_index()
        daily_history_df['Peak'] = daily_history_df['Bankroll'].cummax()
        daily_history_df['Drawdown'] = (daily_history_df['Peak'] - daily_history_df['Bankroll']) / daily_history_df['Peak']
        max_drawdown = daily_history_df['Drawdown'].max()
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        daily_history_df['Daily_Return'] = daily_history_df['Bankroll'].pct_change().fillna(0)
        if daily_history_df['Daily_Return'].std() > 0:
            sharpe_ratio = (daily_history_df['Daily_Return'].mean() / daily_history_df['Daily_Return'].std()) * np.sqrt(365)
            print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
        else:
            print("Sharpe Ratio: N/A (no volatility)")
        
        plt.figure(figsize=(12, 6))
        plt.plot(history_df.index, history_df['Bankroll'])
        plt.title('Equity Curve (Bankroll vs. Bet Count)')
        plt.xlabel('Bet Count')
        plt.ylabel('Bankroll ($)')
        plt.grid(True)
        plt.yscale('linear')
        plt.savefig(EQUITY_CURVE_FILE)
        print(f"\n✅ Equity curve plot saved to '{EQUITY_CURVE_FILE}'")

    if bet_log:
        log_df = pd.DataFrame(bet_log)
        log_df.to_csv(ANALYSIS_LOG_FILE, index=False)
        print(f"\n✅ New, symmetrical analysis log saved to '{ANALYSIS_LOG_FILE}'")
        print(f"Total Bets Logged for Analysis: {len(log_df)}")
        
        # NEW: Calculate and plot the ROI for each individual bet
        if not log_df.empty and 'Profit' in log_df.columns and 'Stake' in log_df.columns:
            log_df['Bet_ROI'] = (log_df['Profit'] / log_df['Stake']) * 100
            log_df['Color'] = log_df['Outcome'].apply(lambda x: 'g' if x == 'Win' else 'r')

            plt.figure(figsize=(12, 6))
            plt.scatter(log_df.index, log_df['Bet_ROI'], color=log_df['Color'], alpha=0.5, s=15)
            plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)
            plt.title('ROI per Individual Bet')
            plt.xlabel('Bet Number')
            plt.ylabel('Return on Investment (%)')
            plt.grid(True, linestyle=':', linewidth=0.5)
            plt.savefig(ROI_PLOT_FILE)
            print(f"✅ ROI per bet plot saved to '{ROI_PLOT_FILE}'")


except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()