import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys # --- NEW: For exiting gracefully
from feature_config import get_active_features, get_experiment_name, get_feature_count


# - Close set win rate calculation
def calculate_close_set_win_rate(player_id, rolling_games_df):
    """
    Calculates a player's win percentage in "close" sets (decided by 2 points).
    """
    if rolling_games_df.empty:
        return 0.5

    total_close_sets_played = 0
    total_close_sets_won = 0

    for _, game in rolling_games_df.iterrows():
        set_scores_str = game.get('Set Scores')
        if pd.isna(set_scores_str):
            continue

        for set_score in str(set_scores_str).split(','):
            try:
                p1_points, p2_points = map(int, set_score.split('-'))
                
                if abs(p1_points - p2_points) == 2:
                    total_close_sets_played += 1
                    
                    is_p1 = (game['Player 1 ID'] == player_id)
                    p1_won_set = p1_points > p2_points
                    
                    if (is_p1 and p1_won_set) or (not is_p1 and not p1_won_set):
                        total_close_sets_won += 1
            except (ValueError, IndexError):
                continue

    if total_close_sets_played == 0:
        return 0.5

    return total_close_sets_won / total_close_sets_played

# - Head-to-Head Dominance Score calculation
def calculate_h2h_dominance(p1_id, h2h_df, current_match_date, decay_factor):
    """
    Calculates a recency-weighted H2H dominance score based on point differentials.
    """
    if h2h_df.empty:
        return 0.0

    total_weighted_score = 0
    
    for _, game in h2h_df.iterrows():
        # Calculate recency weight
        days_ago = (current_match_date - game['Date']).days
        weight = decay_factor ** days_ago
        
        # Symmetrically calculate point differential from p1's perspective
        if game['Player 1 ID'] == p1_id:
            point_diff = game['P1 Total Points'] - game['P2 Total Points']
        else: # p1 was Player 2 in this historical match
            point_diff = game['P2 Total Points'] - game['P1 Total Points']
            
        total_weighted_score += (point_diff * weight)
        
    return total_weighted_score

def calculate_performance_slope(performance_history):
    """Calculates the slope of recent performance using linear regression."""
    if len(performance_history) < 2:
        return 0.0 # Cannot calculate a slope with fewer than 2 points

    y = np.array(performance_history)
    x = np.arange(len(y))
    
    # Use numpy's polyfit to find the slope of the best-fit line (degree 1)
    slope = np.polyfit(x, y, 1)[0]
    return slope

def calculate_pdr(player_id, rolling_games_df):
    """Calculates the Points Dominance Ratio for a single player."""
    if rolling_games_df.empty:
        return 0.5 # A neutral default value

    total_points_won = 0
    total_points_played = 0

    for _, game in rolling_games_df.iterrows():
        if 'P1 Total Points' not in game or 'P2 Total Points' not in game:
            continue # Skip if point data is missing
            
        if game['Player 1 ID'] == player_id:
            points_won = game['P1 Total Points']
            points_lost = game['P2 Total Points']
        else: # Player was P2
            points_won = game['P2 Total Points']
            points_lost = game['P1 Total Points']
        
        total_points_won += points_won
        total_points_played += (points_won + points_lost)
    
    if total_points_played == 0:
        return 0.5

    return total_points_won / total_points_played


# --- 1. Configuration ---
FINAL_DATASET_FILE = "final_dataset_v7.4_no_duplicates.csv"
GBM_MODEL_FILE = "cpr_v7.4_gbm_specialist.joblib"
GBM_PREPROCESSOR_FILE = "gbm_preprocessor_v7.4.joblib"
# LSTM_MODEL_FILE = "cpr_v7.1_lstm_specialist.h5"
# LSTM_SCALER_FILE = "lstm_scaler_v7.1.joblib"
# META_MODEL_FILE = "cpr_v7.4_meta_model.pkl"
ANALYSIS_LOG_FILE = "backtest_log_for_analysis_v7.4.csv"
EQUITY_CURVE_FILE = "equity_curve_v7.4.png"
ROI_PLOT_FILE = "roi_per_bet_v7.4.png" # NEW: Filename for the ROI plot

INITIAL_BANKROLL = 1000
KELLY_FRACTION = 0.035
TRAIN_SPLIT_PERCENTAGE = 0.70
ROLLING_WINDOW = 20
SHORT_ROLLING_WINDOW = 5
SLOPE_WINDOW = 10       # Used for PDR_Slope_Advantage
H2H_DECAY_FACTOR = 0.98  # --- NEW: Decay for H2H recency weighting ---

EDGE_THRESHOLD_MIN = .0001
EDGE_THRESHOLD_MAX = 0.99
MIN_ODDS_DENOMINATOR = 0.10

MIN_GAMES_THRESHOLD = 4 # A player must have at least 5 games in their history

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
#    df.rename(columns={'P1 Pressure Points': 'P1_Pressure_Points', 'P2 Pressure Points': 'P2_Pressure_Points'}, inplace=True, errors='ignore')

    # --- NEW: Add Elo_Advantage to the required columns check 
    required_cols = ['P1_Win', 'Kickoff_P1_Odds', 'Kickoff_P2_Odds', 'Win_Rate_L5_Advantage', 'Close_Set_Win_Rate_Advantage', 'PDR_Advantage', 'Daily_Fatigue_Advantage', 'PDR_Slope_Advantage', 'H2H_Dominance_Score', 'Time_Since_Last_Advantage', 'Matches_Last_24H_Advantage', 'Is_First_Match_Advantage']
    df.dropna(subset=required_cols, inplace=True)    

    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
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
#     lstm_model = load_model(LSTM_MODEL_FILE)
#     lstm_scaler = joblib.load(LSTM_SCALER_FILE)
#     meta_model = joblib.load(META_MODEL_FILE)

    # --- 3. Main Back-testing and Logging Loop ---
    print(f"\n--- Generating New Symmetrical Log with Wide Filters ---")
    
    bankroll = INITIAL_BANKROLL
    bet_count = 0
    total_staked = 0
    bet_log = [] 

    bankroll_history_per_bet = [(backtest_df['Date'].iloc[0], INITIAL_BANKROLL)]

    daily_betting_bankroll = INITIAL_BANKROLL
    last_processed_date = None

    player_pdr_history = {}     # Dictionary to track PDR history for the slope

    for index, match in tqdm(backtest_df.iterrows(), total=backtest_df.shape[0]):
        
        current_date = match['Date'].date()
        if last_processed_date != current_date:
            daily_betting_bankroll = bankroll
            last_processed_date = current_date

        history_df = df.iloc[:index]
        p1_id, p2_id = match['Player 1 ID'], match['Player 2 ID']
        p1_market_odds, p2_market_odds = match['Kickoff_P1_Odds'], match['Kickoff_P2_Odds']
        
        # --- On-the-fly Feature Engineering for GBM model ---
        # Get full history first, then the rolling window
        p1_games = history_df[(history_df['Player 1 ID'] == p1_id) | (history_df['Player 2 ID'] == p1_id)]
        p2_games = history_df[(history_df['Player 1 ID'] == p2_id) | (history_df['Player 2 ID'] == p2_id)]
        p1_games_gbm = history_df[(history_df['Player 1 ID'] == p1_id) | (history_df['Player 2 ID'] == p1_id)].tail(ROLLING_WINDOW)
        p2_games_gbm = history_df[(history_df['Player 1 ID'] == p2_id) | (history_df['Player 2 ID'] == p2_id)].tail(ROLLING_WINDOW)
        
        if len(p1_games_gbm) < MIN_GAMES_THRESHOLD or len(p2_games_gbm) < MIN_GAMES_THRESHOLD:
            continue

        # --- On-the-fly Fatigue and Recency Calculations --- #
        # 1. Time Since Last Match
        p1_last_game_date = p1_games['Date'].max()
        p1_time_since_last = (match['Date'] - p1_last_game_date).total_seconds() / 3600 if pd.notna(p1_last_game_date) else 72
        p2_last_game_date = p2_games['Date'].max()
        p2_time_since_last = (match['Date'] - p2_last_game_date).total_seconds() / 3600 if pd.notna(p2_last_game_date) else 72
        time_since_last_advantage = p1_time_since_last - p2_time_since_last

        # 2. Matches in Last 24 Hours
        p1_matches_last_24h = len(p1_games[p1_games['Date'] > (match['Date'] - pd.Timedelta(hours=24))])
        p2_matches_last_24h = len(p2_games[p2_games['Date'] > (match['Date'] - pd.Timedelta(hours=24))])
        matches_last_24h_advantage = p1_matches_last_24h - p2_matches_last_24h

        # 3. Is First Match of the Day
        today_history_df = history_df[history_df['Date'].dt.date == match['Date'].date()]
        p1_is_first_match = 1 if today_history_df[(today_history_df['Player 1 ID'] == p1_id) | (today_history_df['Player 2 ID'] == p1_id)].empty else 0
        p2_is_first_match = 1 if today_history_df[(today_history_df['Player 1 ID'] == p2_id) | (today_history_df['Player 2 ID'] == p2_id)].empty else 0
        is_first_match_advantage = p1_is_first_match - p2_is_first_match

        # --- On-the-fly "Hot Streak" Win Rate (L5) ---
        p1_rolling_games_short = p1_games.tail(SHORT_ROLLING_WINDOW)
        p1_win_rate_l5 = p1_rolling_games_short.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p1_rolling_games_short.empty else 0.5
        p2_rolling_games_short = p2_games.tail(SHORT_ROLLING_WINDOW)
        p2_win_rate_l5 = p2_rolling_games_short.apply(lambda r: 1 if (r['Player 1 ID'] == p2_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p2_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p2_rolling_games_short.empty else 0.5
        win_rate_l5_advantage = p1_win_rate_l5 - p2_win_rate_l5

        # --- On-the-fly PDR Slope Calculation ---    
        # Calculate current PDR for both players
        p1_pdr = calculate_pdr(p1_id, p1_games_gbm)
        p2_pdr = calculate_pdr(p2_id, p2_games_gbm)

        # Update and manage history for Player 1
        if p1_id not in player_pdr_history: player_pdr_history[p1_id] = []
        player_pdr_history[p1_id].append(p1_pdr)
        if len(player_pdr_history[p1_id]) > SLOPE_WINDOW:
            player_pdr_history[p1_id].pop(0)
        p1_pdr_slope = calculate_performance_slope(player_pdr_history[p1_id])

        # Update and manage history for Player 2
        if p2_id not in player_pdr_history: player_pdr_history[p2_id] = []
        player_pdr_history[p2_id].append(p2_pdr)
        if len(player_pdr_history[p2_id]) > SLOPE_WINDOW:
            player_pdr_history[p2_id].pop(0)
        p2_pdr_slope = calculate_performance_slope(player_pdr_history[p2_id])
        
        pdr_slope_advantage = p1_pdr_slope - p2_pdr_slope

        # NEW: Retrieve the pre-calculated Elo_Advantage from the current match row.
#        elo_advantage = match['Elo_Advantage']
        pdr_advantage = match['PDR_Advantage']
        daily_fatigue_advantage = match['Daily_Fatigue_Advantage']

        # 1. Calculate H2H on-the-fly using ONLY historical data to prevent leakage.
        h2h_df = history_df[((history_df['Player 1 ID'] == p1_id) & (history_df['Player 2 ID'] == p2_id)) | 
                              ((history_df['Player 1 ID'] == p2_id) & (history_df['Player 2 ID'] == p1_id))]
        
        if not h2h_df.empty:
            p1_h2h_wins = len(h2h_df[((h2h_df['Player 1 ID'] == p1_id) & (h2h_df['P1_Win'] == 1)) | 
                                     ((h2h_df['Player 2 ID'] == p1_id) & (h2h_df['P1_Win'] == 0))])
            h2h_p1_win_rate = p1_h2h_wins / len(h2h_df)
        else:
            h2h_p1_win_rate = 0.5 # Default value if they have no history

        # --- On-the-fly H2H Dominance Calculation ---
        h2h_dominance_score = calculate_h2h_dominance(p1_id, h2h_df, match['Date'], H2H_DECAY_FACTOR)

        p1_win_rate = p1_games_gbm.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p1_games_gbm.empty else 0.5
        p2_win_rate = p2_games_gbm.apply(lambda r: 1 if (r['Player 1 ID'] == p2_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p2_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p2_games_gbm.empty else 0.5
        win_rate_advantage = p1_win_rate - p2_win_rate
        
#        p1_pressure_points = p1_games_gbm.apply(lambda r: r['P1_Pressure_Points'] if r['Player 1 ID'] == p1_id else r['P2_Pressure_Points'], axis=1).mean() if not p1_games_gbm.empty else 0.0
#        p2_pressure_points = p2_games_gbm.apply(lambda r: r['P1_Pressure_Points'] if r['Player 1 ID'] == p2_id else r['P2_Pressure_Points'], axis=1).mean() if not p2_games_gbm.empty else 0.0
#        pressure_points_advantage = p1_pressure_points - p2_pressure_points

        # --- Replaced Pressure Points with Close Set Win Rate --- #
        p1_close_set_win_rate = calculate_close_set_win_rate(p1_id, p1_games_gbm)
        p2_close_set_win_rate = calculate_close_set_win_rate(p2_id, p2_games_gbm)
        close_set_win_rate_advantage = p1_close_set_win_rate - p2_close_set_win_rate

        p1_set_comebacks = p1_games_gbm.apply(lambda r: r['P1 Set Comebacks'] if r['Player 1 ID'] == p1_id else r['P2 Set Comebacks'], axis=1).sum() if not p1_games_gbm.empty else 0.0
        p2_set_comebacks = p2_games_gbm.apply(lambda r: r['P1 Set Comebacks'] if r['Player 1 ID'] == p2_id else r['P2 Set Comebacks'], axis=1).sum() if not p2_games_gbm.empty else 0.0
        set_comebacks_advantage = p1_set_comebacks - p2_set_comebacks

        # --- Model Prediction ---
        # Build full feature dictionary, then filter to active features from config
        all_features = {
            'Time_Since_Last_Advantage': time_since_last_advantage,
            'Matches_Last_24H_Advantage': matches_last_24h_advantage,
            'Is_First_Match_Advantage': is_first_match_advantage,
            'PDR_Slope_Advantage': pdr_slope_advantage,
            'H2H_P1_Win_Rate': h2h_p1_win_rate,
            'H2H_Dominance_Score': h2h_dominance_score,
            'Daily_Fatigue_Advantage': daily_fatigue_advantage,
            'PDR_Advantage': pdr_advantage,
            'Win_Rate_Advantage': win_rate_advantage,
            'Win_Rate_L5_Advantage': win_rate_l5_advantage,
            'Close_Set_Win_Rate_Advantage': close_set_win_rate_advantage,
            'Set_Comebacks_Advantage': set_comebacks_advantage
        }
        active_features = get_active_features()
        gbm_features = pd.DataFrame([{k: v for k, v in all_features.items() if k in active_features}])
        gbm_features = gbm_features[active_features]  # Ensure column order matches preprocessor

        X_gbm_processed = gbm_preprocessor.transform(gbm_features)
        gbm_pred = gbm_model.predict_proba(X_gbm_processed)[0, 1]
        
        # X_meta = np.array([[gbm_pred, lstm_pred]])
        # model_prob_p1 = meta_model.predict_proba(X_meta)[0, 1]
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
#                if win_rate_advantage < 0: capped_fraction /= 4
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
#                if win_rate_advantage > 0: capped_fraction /= 4
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
            log_entry = {'Match_ID': match['Match ID'],'Date': match['Date'].strftime('%Y-%m-%d'),'Player_1': match['Player 1'],'Player_2': match['Player 2'], 'Time_Since_Last_Advantage': time_since_last_advantage, 'Matches_Last_24H_Advantage': matches_last_24h_advantage, 'Is_First_Match_Advantage': is_first_match_advantage, 'H2H_P1_Win_Rate': h2h_p1_win_rate_log,'H2H_Dominance_Score': h2h_dominance_score, 'Win_Rate_Advantage': win_rate_advantage, 'Win_Rate_L5_Advantage': win_rate_l5_advantage,'PDR_Advantage': pdr_advantage, 'PDR_Slope_Advantage': pdr_slope_advantage, 'Daily_Fatigue_Advantage': daily_fatigue_advantage, 'Close_Set_Win_Rate_Advantage': close_set_win_rate_advantage}
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
        print(f"\n[OK] Equity curve plot saved to '{EQUITY_CURVE_FILE}'")

    if bet_log:
        log_df = pd.DataFrame(bet_log)
        log_df.to_csv(ANALYSIS_LOG_FILE, index=False)
        print(f"\n[OK] New, symmetrical analysis log saved to '{ANALYSIS_LOG_FILE}'")
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
            print(f"[OK] ROI per bet plot saved to '{ROI_PLOT_FILE}'")


except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()