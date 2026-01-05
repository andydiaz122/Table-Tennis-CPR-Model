import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import bisect

# ==============================================================================
# OPTIMIZED HELPER FUNCTIONS (Vectorized - No iterrows)
# ==============================================================================

def calculate_close_set_win_rate_vectorized(player_id, rolling_games_df):
    """Vectorized close set win rate calculation - eliminates iterrows loop."""
    if rolling_games_df.empty:
        return 0.5

    set_scores = rolling_games_df['Set Scores'].values
    p1_ids = rolling_games_df['Player 1 ID'].values

    total_close_sets_played = 0
    total_close_sets_won = 0

    for i in range(len(set_scores)):
        set_scores_str = set_scores[i]
        if pd.isna(set_scores_str):
            continue
        is_p1 = (p1_ids[i] == player_id)
        for set_score in str(set_scores_str).split(','):
            try:
                p1_points, p2_points = map(int, set_score.split('-'))
                if abs(p1_points - p2_points) == 2:
                    total_close_sets_played += 1
                    p1_won_set = p1_points > p2_points
                    if (is_p1 and p1_won_set) or (not is_p1 and not p1_won_set):
                        total_close_sets_won += 1
            except (ValueError, IndexError):
                continue

    if total_close_sets_played == 0:
        return 0.5
    return total_close_sets_won / total_close_sets_played


def calculate_h2h_dominance_vectorized(p1_id, h2h_indices, current_match_date, decay_factor,
                                        df_dates, df_p1_ids, df_p1_points, df_p2_points):
    """Vectorized H2H dominance using pre-indexed arrays."""
    if len(h2h_indices) == 0:
        return 0.0

    # Extract arrays for h2h matches only
    h2h_dates = df_dates[h2h_indices]
    h2h_p1_ids = df_p1_ids[h2h_indices]
    h2h_p1_pts = df_p1_points[h2h_indices]
    h2h_p2_pts = df_p2_points[h2h_indices]

    # Vectorized calculation
    days_ago = (current_match_date - h2h_dates).astype('timedelta64[D]').astype(float)
    weights = decay_factor ** days_ago

    # Point differential: positive if p1 advantage
    is_p1_player1 = (h2h_p1_ids == p1_id)
    point_diff = np.where(is_p1_player1, h2h_p1_pts - h2h_p2_pts, h2h_p2_pts - h2h_p1_pts)

    return np.sum(point_diff * weights)


def calculate_performance_slope(performance_history):
    """Unchanged - already efficient."""
    if len(performance_history) < 2:
        return 0.0
    y = np.array(performance_history)
    x = np.arange(len(y))
    return np.polyfit(x, y, 1)[0]


def calculate_pdr_vectorized(player_id, rolling_games_df):
    """Vectorized PDR calculation - eliminates iterrows loop."""
    if rolling_games_df.empty:
        return 0.5

    p1_ids = rolling_games_df['Player 1 ID'].values
    p1_points = rolling_games_df['P1 Total Points'].values
    p2_points = rolling_games_df['P2 Total Points'].values

    is_p1 = (p1_ids == player_id)
    points_won = np.where(is_p1, p1_points, p2_points)
    points_lost = np.where(is_p1, p2_points, p1_points)

    total_points_won = np.nansum(points_won)
    total_points_played = np.nansum(points_won) + np.nansum(points_lost)

    if total_points_played == 0:
        return 0.5
    return total_points_won / total_points_played


def calculate_win_rate_vectorized(player_id, games_df):
    """Vectorized win rate calculation - replaces apply(lambda, axis=1)."""
    if games_df.empty:
        return 0.5

    p1_ids = games_df['Player 1 ID'].values
    p1_wins = games_df['P1_Win'].values

    is_p1 = (p1_ids == player_id)
    wins = np.where(is_p1, p1_wins == 1, p1_wins == 0).astype(int)
    return wins.mean()


def calculate_set_comebacks_vectorized(player_id, games_df):
    """Vectorized set comebacks calculation - replaces apply(lambda, axis=1)."""
    if games_df.empty:
        return 0.0

    p1_ids = games_df['Player 1 ID'].values
    p1_comebacks = games_df['P1 Set Comebacks'].values
    p2_comebacks = games_df['P2 Set Comebacks'].values

    is_p1 = (p1_ids == player_id)
    comebacks = np.where(is_p1, p1_comebacks, p2_comebacks)
    return np.nansum(comebacks)

# --- 1. Configuration (Final Filtered) ---
FINAL_DATASET_FILE = "final_dataset_v7.4_no_duplicates.csv.gz"  # OPTIMIZATION: gzip
GBM_MODEL_FILE = "cpr_v7.4_gbm_specialist.joblib"
GBM_PREPROCESSOR_FILE = "gbm_preprocessor_v7.4.joblib"
ANALYSIS_LOG_FILE = "backtest_log_final_filtered.csv"
EQUITY_CURVE_FILE = "equity_curve_final_filtered.png"
ROI_PLOT_FILE = "roi_per_bet_final_filtered.png"

INITIAL_BANKROLL = 1000
KELLY_FRACTION = 0.02
TRAIN_SPLIT_PERCENTAGE = 0.70
ROLLING_WINDOW = 20
SHORT_ROLLING_WINDOW = 5
SLOPE_WINDOW = 10
H2H_DECAY_FACTOR = 0.98

EDGE_THRESHOLD_MIN = 0.0001
EDGE_THRESHOLD_MAX = 0.99
MIN_ODDS_DENOMINATOR = 0.10
MIN_GAMES_THRESHOLD = 4

# --- STRATEGIC FILTERS ---
ODDS_THRESHOLD_MAX = 3.0
FORM_L5_THRESHOLD = 0.1 # CORRECTION: Must be > 0.1, not != 0

# --- REGIME FILTER ---
CONFIDENCE_FILTER_EDGE_MIN = 0.50
CONFIDENCE_FILTER_ODDS_MAX = 2.0

# --- 2. Load Data & Models ---
try:
    print("--- Loading Data for Final Filtered Back-test ---")
    df = pd.read_csv(FINAL_DATASET_FILE, na_values=['-'], keep_default_na=True, low_memory=False)
    df.drop_duplicates(subset=['Date', 'Player 1', 'Player 2'], keep='first', inplace=True)
    base_required_cols = ['P1_Win', 'Kickoff_P1_Odds', 'Kickoff_P2_Odds', 'Set Scores', 'P1 Total Points', 'P2 Total Points', 'P1 Set Comebacks', 'P2 Set Comebacks', 'Daily_Fatigue_Advantage']
    df.dropna(subset=base_required_cols, inplace=True)
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

    # ===========================================================================
    # OPTIMIZATION: Pre-build indices for O(log N) lookups instead of O(N) filtering
    # ===========================================================================
    print("Building player and H2H indices for fast lookup...")

    # Pre-extract numpy arrays for vectorized operations (avoids DataFrame overhead)
    df_p1_ids = df['Player 1 ID'].values
    df_p2_ids = df['Player 2 ID'].values
    df_dates = df['Date'].values  # numpy datetime64
    df_dates_as_date = pd.to_datetime(df['Date']).dt.date.values
    df_p1_wins = df['P1_Win'].values
    df_p1_points = df['P1 Total Points'].values
    df_p2_points = df['P2 Total Points'].values
    df_p1_comebacks = df['P1 Set Comebacks'].values
    df_p2_comebacks = df['P2 Set Comebacks'].values
    df_set_scores = df['Set Scores'].values

    # Build player match index: player_id -> sorted list of match indices
    player_match_indices = defaultdict(list)
    for idx in range(len(df)):
        player_match_indices[df_p1_ids[idx]].append(idx)
        player_match_indices[df_p2_ids[idx]].append(idx)

    # Build H2H pair index: (min_id, max_id) -> sorted list of match indices
    h2h_pair_indices = defaultdict(list)
    for idx in range(len(df)):
        h2h_key = (min(df_p1_ids[idx], df_p2_ids[idx]), max(df_p1_ids[idx], df_p2_ids[idx]))
        h2h_pair_indices[h2h_key].append(idx)

    print(f"Indices built: {len(player_match_indices)} players, {len(h2h_pair_indices)} H2H pairs")

    # --- 3. Main Back-testing Loop ---
    print(f"\n--- Generating Log with Final Strategic Filters ---")

    bankroll, bet_count, total_staked = INITIAL_BANKROLL, 0, 0
    bet_log, bankroll_history_per_bet = [], [(backtest_df['Date'].iloc[0], INITIAL_BANKROLL)]
    daily_betting_bankroll, last_processed_date = INITIAL_BANKROLL, None
    player_pdr_history = {}

    # Get backtest indices for efficient iteration
    backtest_indices = backtest_df.index.tolist()

    for index in tqdm(backtest_indices, total=len(backtest_indices)):
        match = df.iloc[index]

        current_date = match['Date'].date()
        if last_processed_date != current_date:
            daily_betting_bankroll = bankroll
            last_processed_date = current_date

        p1_id, p2_id = match['Player 1 ID'], match['Player 2 ID']
        p1_market_odds, p2_market_odds = match['Kickoff_P1_Odds'], match['Kickoff_P2_Odds']

        # ===========================================================================
        # OPTIMIZATION: Use bisect for O(log N) index filtering instead of O(N)
        # ===========================================================================

        # Get player history indices using bisect (O(log N) instead of O(N) filter)
        p1_all_indices = player_match_indices[p1_id]
        p1_cutoff = bisect.bisect_left(p1_all_indices, index)
        p1_history_indices = p1_all_indices[:p1_cutoff]

        p2_all_indices = player_match_indices[p2_id]
        p2_cutoff = bisect.bisect_left(p2_all_indices, index)
        p2_history_indices = p2_all_indices[:p2_cutoff]

        # Get rolling window indices (last N matches)
        p1_gbm_indices = p1_history_indices[-ROLLING_WINDOW:] if len(p1_history_indices) >= ROLLING_WINDOW else p1_history_indices
        p2_gbm_indices = p2_history_indices[-ROLLING_WINDOW:] if len(p2_history_indices) >= ROLLING_WINDOW else p2_history_indices

        if len(p1_gbm_indices) < MIN_GAMES_THRESHOLD or len(p2_gbm_indices) < MIN_GAMES_THRESHOLD:
            continue

        # Create DataFrame slices only when needed (using .iloc with index list)
        p1_games = df.iloc[p1_history_indices]
        p2_games = df.iloc[p2_history_indices]
        p1_games_gbm = df.iloc[p1_gbm_indices]
        p2_games_gbm = df.iloc[p2_gbm_indices]

        # ===========================================================================
        # OPTIMIZATION: All feature calculations now vectorized (no iterrows/apply)
        # ===========================================================================

        # --- On-the-fly Fatigue and Recency Calculations --- #
        # 1. Time Since Last Match (using pre-extracted numpy arrays)
        if len(p1_history_indices) > 0:
            p1_last_game_date = df_dates[p1_history_indices[-1]]
            p1_time_since_last = (match['Date'] - pd.Timestamp(p1_last_game_date)).total_seconds() / 3600
        else:
            p1_time_since_last = 72

        if len(p2_history_indices) > 0:
            p2_last_game_date = df_dates[p2_history_indices[-1]]
            p2_time_since_last = (match['Date'] - pd.Timestamp(p2_last_game_date)).total_seconds() / 3600
        else:
            p2_time_since_last = 72
        time_since_last_advantage = p1_time_since_last - p2_time_since_last

        # 2. Matches in Last 24 Hours (vectorized using numpy arrays)
        cutoff_time = match['Date'] - pd.Timedelta(hours=24)
        p1_matches_last_24h = np.sum(df_dates[p1_history_indices] > np.datetime64(cutoff_time)) if len(p1_history_indices) > 0 else 0
        p2_matches_last_24h = np.sum(df_dates[p2_history_indices] > np.datetime64(cutoff_time)) if len(p2_history_indices) > 0 else 0
        matches_last_24h_advantage = p1_matches_last_24h - p2_matches_last_24h

        # 3. Is First Match of the Day (vectorized using pre-extracted date array)
        current_date_val = match['Date'].date()
        p1_today_matches = np.sum(df_dates_as_date[p1_history_indices] == current_date_val) if len(p1_history_indices) > 0 else 0
        p2_today_matches = np.sum(df_dates_as_date[p2_history_indices] == current_date_val) if len(p2_history_indices) > 0 else 0
        p1_is_first_match = 1 if p1_today_matches == 0 else 0
        p2_is_first_match = 1 if p2_today_matches == 0 else 0
        is_first_match_advantage = p1_is_first_match - p2_is_first_match

        # --- On-the-fly "Hot Streak" Win Rate (L5) - VECTORIZED ---
        p1_short_indices = p1_history_indices[-SHORT_ROLLING_WINDOW:] if len(p1_history_indices) >= SHORT_ROLLING_WINDOW else p1_history_indices
        p2_short_indices = p2_history_indices[-SHORT_ROLLING_WINDOW:] if len(p2_history_indices) >= SHORT_ROLLING_WINDOW else p2_history_indices

        if len(p1_short_indices) > 0:
            p1_short_p1ids = df_p1_ids[p1_short_indices]
            p1_short_wins = df_p1_wins[p1_short_indices]
            is_p1 = (p1_short_p1ids == p1_id)
            wins = np.where(is_p1, p1_short_wins == 1, p1_short_wins == 0).astype(int)
            p1_win_rate_l5 = wins.mean()
        else:
            p1_win_rate_l5 = 0.5

        if len(p2_short_indices) > 0:
            p2_short_p1ids = df_p1_ids[p2_short_indices]
            p2_short_wins = df_p1_wins[p2_short_indices]
            is_p1 = (p2_short_p1ids == p2_id)
            wins = np.where(is_p1, p2_short_wins == 1, p2_short_wins == 0).astype(int)
            p2_win_rate_l5 = wins.mean()
        else:
            p2_win_rate_l5 = 0.5
        win_rate_l5_advantage = p1_win_rate_l5 - p2_win_rate_l5

        # --- On-the-fly PDR Slope Calculation - VECTORIZED ---
        p1_pdr = calculate_pdr_vectorized(p1_id, p1_games_gbm)
        p2_pdr = calculate_pdr_vectorized(p2_id, p2_games_gbm)

        # Update and manage history for Player 1
        if p1_id not in player_pdr_history:
            player_pdr_history[p1_id] = []
        player_pdr_history[p1_id].append(p1_pdr)
        if len(player_pdr_history[p1_id]) > SLOPE_WINDOW:
            player_pdr_history[p1_id].pop(0)
        p1_pdr_slope = calculate_performance_slope(player_pdr_history[p1_id])

        # Update and manage history for Player 2
        if p2_id not in player_pdr_history:
            player_pdr_history[p2_id] = []
        player_pdr_history[p2_id].append(p2_pdr)
        if len(player_pdr_history[p2_id]) > SLOPE_WINDOW:
            player_pdr_history[p2_id].pop(0)
        p2_pdr_slope = calculate_performance_slope(player_pdr_history[p2_id])

        pdr_slope_advantage = p1_pdr_slope - p2_pdr_slope

        # Retrieve pre-calculated values from match row
        pdr_advantage = match['PDR_Advantage']
        daily_fatigue_advantage = match['Daily_Fatigue_Advantage']

        # ===========================================================================
        # OPTIMIZATION: H2H calculations using pre-built index (O(log N) vs O(N))
        # Calculate ONCE, reuse for both feature calculation and logging
        # ===========================================================================
        h2h_key = (min(p1_id, p2_id), max(p1_id, p2_id))
        h2h_all_indices = h2h_pair_indices[h2h_key]
        h2h_cutoff = bisect.bisect_left(h2h_all_indices, index)
        h2h_history_indices = h2h_all_indices[:h2h_cutoff]

        if len(h2h_history_indices) > 0:
            # Vectorized H2H win rate calculation
            h2h_p1ids = df_p1_ids[h2h_history_indices]
            h2h_wins = df_p1_wins[h2h_history_indices]
            is_p1_player1 = (h2h_p1ids == p1_id)
            p1_h2h_wins = np.sum(np.where(is_p1_player1, h2h_wins == 1, h2h_wins == 0))
            h2h_p1_win_rate = p1_h2h_wins / len(h2h_history_indices)
        else:
            h2h_p1_win_rate = 0.5  # Default value if no H2H history

        # --- On-the-fly H2H Dominance Calculation - VECTORIZED ---
        h2h_dominance_score = calculate_h2h_dominance_vectorized(
            p1_id, h2h_history_indices, match['Date'], H2H_DECAY_FACTOR,
            df_dates, df_p1_ids, df_p1_points, df_p2_points
        )

        # --- Win Rate (L20) - VECTORIZED ---
        p1_win_rate = calculate_win_rate_vectorized(p1_id, p1_games_gbm)
        p2_win_rate = calculate_win_rate_vectorized(p2_id, p2_games_gbm)
        win_rate_advantage = p1_win_rate - p2_win_rate

        # --- Close Set Win Rate - VECTORIZED ---
        p1_close_set_win_rate = calculate_close_set_win_rate_vectorized(p1_id, p1_games_gbm)
        p2_close_set_win_rate = calculate_close_set_win_rate_vectorized(p2_id, p2_games_gbm)
        close_set_win_rate_advantage = p1_close_set_win_rate - p2_close_set_win_rate

        # --- Set Comebacks - VECTORIZED ---
        p1_set_comebacks = calculate_set_comebacks_vectorized(p1_id, p1_games_gbm)
        p2_set_comebacks = calculate_set_comebacks_vectorized(p2_id, p2_games_gbm)
        set_comebacks_advantage = p1_set_comebacks - p2_set_comebacks

        # --- Model Prediction ---
        gbm_features = pd.DataFrame([{
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
        }])

        X_gbm_processed = gbm_preprocessor.transform(gbm_features)
        model_prob_p1 = gbm_model.predict_proba(X_gbm_processed)[0, 1]

        # --- Betting Logic with CORRECTED Strategic Filters ---
        p1_market_odds, p2_market_odds = match['Kickoff_P1_Odds'], match['Kickoff_P2_Odds']
        edge_p1 = model_prob_p1 * p1_market_odds - 1
        edge_p2 = (1 - model_prob_p1) * p2_market_odds - 1
        actual_winner = match['P1_Win']
        bet_details = None
        if bankroll <= 0: continue
        
        # Define filter conditions for Player 1
        p1_filters = {
            "Has_Edge": edge_p1 > EDGE_THRESHOLD_MIN and edge_p1 < EDGE_THRESHOLD_MAX,
            "Is_Betable": (p1_market_odds - 1) > MIN_ODDS_DENOMINATOR,
            "Prime_Directive": pdr_advantage > 0, # CORRECTION: Use on-the-fly variable
            "Risk_Off_Switch": p1_market_odds <= ODDS_THRESHOLD_MAX,
            "Clarity_Mandate": abs(win_rate_l5_advantage) > FORM_L5_THRESHOLD, # CORRECTION: Use correct logic
            # --- REGIME FILTER (Confidence Filter) ---
#            "Confidence_Filter": not (edge_p1 > CONFIDENCE_FILTER_EDGE_MIN and p1_market_odds < CONFIDENCE_FILTER_ODDS_MAX),
        }
        
        # Define filter conditions for Player 2
        p2_filters = {
            "Has_Edge": edge_p2 > EDGE_THRESHOLD_MIN and edge_p2 < EDGE_THRESHOLD_MAX,
            "Is_Betable": (p2_market_odds - 1) > MIN_ODDS_DENOMINATOR,
            "Prime_Directive": pdr_advantage < 0, # CORRECTION: Use on-the-fly variable
            "Risk_Off_Switch": p2_market_odds <= ODDS_THRESHOLD_MAX,
            "Clarity_Mandate": abs(win_rate_l5_advantage) > FORM_L5_THRESHOLD, # CORRECTION: Use correct logic
            # --- REGIME FILTER (Confidence Filter) ---
#            "Confidence_Filter": not (edge_p2 > CONFIDENCE_FILTER_EDGE_MIN and p2_market_odds < CONFIDENCE_FILTER_ODDS_MAX),
        }

        if all(p1_filters.values()):
            bet_count += 1
            kelly_fraction_rec = (edge_p1 / (p1_market_odds - 1)) * KELLY_FRACTION
            capped_fraction = min(kelly_fraction_rec, 0.05)
            if win_rate_advantage < 0: capped_fraction /= 4
            stake = daily_betting_bankroll * capped_fraction
            total_staked += stake
            profit = stake * (p1_market_odds - 1) if actual_winner == 1 else -stake
            bankroll += profit
            bet_details = {'Bet_On_Player': match['Player 1'], 'Outcome': "Win" if actual_winner == 1 else "Loss", 'Profit': profit, 'Stake': stake, 'Model_Prob': model_prob_p1, 'Market_Odds': p1_market_odds, 'Edge': edge_p1}
        elif all(p2_filters.values()):
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
            # OPTIMIZATION: Reuse already-calculated h2h_p1_win_rate (no duplicate calculation)
            log_entry = {
                'Match_ID': match['Match ID'],
                'Date': match['Date'].strftime('%Y-%m-%d'),
                'Player_1': match['Player 1'],
                'Player_2': match['Player 2'],
                'Time_Since_Last_Advantage': time_since_last_advantage,
                'Matches_Last_24H_Advantage': matches_last_24h_advantage,
                'Is_First_Match_Advantage': is_first_match_advantage,
                'H2H_P1_Win_Rate': h2h_p1_win_rate,  # Reuse calculated value
                'H2H_Dominance_Score': h2h_dominance_score,
                'Win_Rate_Advantage': win_rate_advantage,
                'Win_Rate_L5_Advantage': win_rate_l5_advantage,
                'PDR_Advantage': pdr_advantage,
                'PDR_Slope_Advantage': pdr_slope_advantage,
                'Daily_Fatigue_Advantage': daily_fatigue_advantage,
                'Close_Set_Win_Rate_Advantage': close_set_win_rate_advantage
            }
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
