import pandas as pd
from tqdm import tqdm
import numpy as np
import math


# - Close set win rate calculation (helper for pre-parsing)
def parse_set_scores_for_match(set_scores_str):
    """
    Pre-parse set scores string into structured format.
    Returns: (p1_close_won, p2_close_won, total_close) tuple
    """
    if pd.isna(set_scores_str):
        return (0, 0, 0)

    p1_close_won = 0
    p2_close_won = 0
    total_close = 0

    for set_score in str(set_scores_str).split(','):
        try:
            p1_points, p2_points = map(int, set_score.split('-'))
            # Check if the set was "close" (decided by 2 points)
            if abs(p1_points - p2_points) == 2:
                total_close += 1
                if p1_points > p2_points:
                    p1_close_won += 1
                else:
                    p2_close_won += 1
        except (ValueError, IndexError):
            continue

    return (p1_close_won, p2_close_won, total_close)


# OPTIMIZED: Vectorized close set win rate using pre-parsed data
def calculate_close_set_win_rate_optimized(player_id, rolling_indices, player_ids_p1, close_sets_p1_won, close_sets_p2_won, close_sets_total):
    """
    Calculates a player's win percentage in "close" sets using pre-parsed arrays.
    OPTIMIZED: No iterrows(), no string parsing in hot path.
    """
    if len(rolling_indices) == 0:
        return 0.5

    # Get pre-parsed close set data for these indices
    is_p1 = player_ids_p1[rolling_indices] == player_id
    p1_won = close_sets_p1_won[rolling_indices]
    p2_won = close_sets_p2_won[rolling_indices]
    total = close_sets_total[rolling_indices]

    # Player's close sets won = P1 wins where player is P1 + P2 wins where player is P2
    player_close_won = np.sum(np.where(is_p1, p1_won, p2_won))
    total_close_played = np.sum(total)

    if total_close_played == 0:
        return 0.5

    return player_close_won / total_close_played


# Legacy function kept for compatibility (not used in optimized path)
def calculate_close_set_win_rate(player_id, rolling_games_df):
    """
    Calculates a player's win percentage in "close" sets (decided by 2 points).
    NOTE: This is the legacy version with iterrows(). Use calculate_close_set_win_rate_optimized instead.
    """
    if rolling_games_df.empty:
        return 0.5

    total_close_sets_played = 0
    total_close_sets_won = 0

    for _, game in rolling_games_df.iterrows():
        set_scores_str = game.get('Set Scores')
        if pd.isna(set_scores_str):
            continue

        # Symmetrically parse the set scores
        for set_score in set_scores_str.split(','):
            try:
                p1_points, p2_points = map(int, set_score.split('-'))

                # Check if the set was "close"
                if abs(p1_points - p2_points) == 2:
                    total_close_sets_played += 1

                    # Check if our player won this close set
                    is_p1 = (game['Player 1 ID'] == player_id)
                    p1_won_set = p1_points > p2_points

                    if (is_p1 and p1_won_set) or (not is_p1 and not p1_won_set):
                        total_close_sets_won += 1
            except (ValueError, IndexError):
                continue # Skip malformed set scores

    if total_close_sets_played == 0:
        return 0.5 # Neutral default if no close sets were played

    return total_close_sets_won / total_close_sets_played

# - Head-to-Head Dominance Score calculation
# OPTIMIZED: Vectorized version - no iterrows()
def calculate_h2h_dominance(p1_id, h2h_df, current_date, decay_factor):
    """
    Calculates a recency-weighted H2H dominance score based on point differentials.
    OPTIMIZED: Uses vectorized numpy operations instead of iterrows().
    """
    if h2h_df.empty:
        return 0.0

    # Vectorized days_ago calculation
    days_ago = (current_date - h2h_df['Date']).dt.days.values
    weights = decay_factor ** days_ago

    # Vectorized point differential from p1's perspective
    p1_was_player1 = (h2h_df['Player 1 ID'].values == p1_id)
    p1_total = h2h_df['P1 Total Points'].values
    p2_total = h2h_df['P2 Total Points'].values

    # Where p1 was Player 1: P1 - P2, where p1 was Player 2: P2 - P1
    point_diff = np.where(p1_was_player1, p1_total - p2_total, p2_total - p1_total)

    return np.sum(point_diff * weights)

def calculate_performance_slope(performance_history):
    """Calculates the slope of recent performance using linear regression."""
    if len(performance_history) < 2:
        return 0.0 # Cannot calculate a slope with fewer than 2 points

    y = np.array(performance_history)
    x = np.arange(len(y))
    
    # Use numpy's polyfit to find the slope of the best-fit line (degree 1)
    slope = np.polyfit(x, y, 1)[0]
    return slope

# - Points Dominance Ratio calculation logic
# OPTIMIZED: Vectorized version - no iterrows()
def calculate_pdr(player_id, rolling_games_df):
    """
    Calculates the Points Dominance Ratio for a single player.
    OPTIMIZED: Uses vectorized numpy operations instead of iterrows().
    """
    if rolling_games_df.empty:
        return 0.5 # A neutral default value

    # Vectorized: determine where player was P1 vs P2
    player_was_p1 = (rolling_games_df['Player 1 ID'].values == player_id)
    p1_points = rolling_games_df['P1 Total Points'].values
    p2_points = rolling_games_df['P2 Total Points'].values

    # Points won by player (P1 points when player was P1, P2 points when player was P2)
    points_won = np.where(player_was_p1, p1_points, p2_points)
    points_lost = np.where(player_was_p1, p2_points, p1_points)

    total_points_won = np.sum(points_won)
    total_points_played = total_points_won + np.sum(points_lost)

    if total_points_played == 0:
        return 0.5

    return total_points_won / total_points_played

# - Elo calculation logic
def update_elo(p1_elo, p2_elo, p1_won, k_factor=32):
    expected_p1 = 1 / (1 + 10 ** ((p2_elo - p1_elo) / 400))
    expected_p2 = 1 / (1 + 10 ** ((p1_elo - p2_elo) / 400))
    score_p1 = 1 if p1_won else 0
    score_p2 = 0 if p1_won else 1
    new_p1_elo = p1_elo + k_factor * (score_p1 - expected_p1)
    new_p2_elo = p2_elo + k_factor * (score_p2 - expected_p2)
    return new_p1_elo, new_p2_elo


# --- 1. Configuration ---
RAW_STATS_FILE = "czech_liga_pro_advanced_stats_FIXED.csv"
OUTPUT_FILE = "final_engineered_features_v7.4.csv" # New, corrected output file
ROLLING_WINDOW = 20
SHORT_ROLLING_WINDOW = 5
SLOPE_WINDOW = 10       # - Number of recent matches to calculate the slope over
H2H_DECAY_FACTOR = 0.98    # - Decay for H2H recency weighting


# --- 2. Main Logic ---
try:
    print(f"--- Loading Raw Data from '{RAW_STATS_FILE}' ---")
    df = pd.read_csv(RAW_STATS_FILE)

    # --- Data Cleaning and Preparation ---
    df['Date'] = pd.to_datetime(df['Date'])
#    df.sort_values(by='Date', inplace=True)
    # Sort by date first, then by Match_ID to ensure chronological order for the same day
    df.sort_values(by=['Date', 'Match ID'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Convert IDs to integers for consistent matching
    df['Player 1 ID'] = df['Player 1 ID'].astype(int)
    df['Player 2 ID'] = df['Player 2 ID'].astype(int)
    
    # Create a robust P1_Win column
    def get_winner(score_str):
        try:
            cleaned_score = str(score_str).strip('="')
            p1_score, p2_score = map(int, cleaned_score.split('-'))
            return 1 if p1_score > p2_score else 0
        except (ValueError, IndexError):
            return np.nan
    df['P1_Win'] = df['Final Score'].apply(get_winner)
    df.dropna(subset=['P1_Win'], inplace=True)
    df['P1_Win'] = df['P1_Win'].astype(int)

    # OPTIMIZED: Pre-parse Set Scores once during load (not in hot path)
    print("--- Pre-parsing Set Scores for close set calculations ---")
    parsed_set_scores = df['Set Scores'].apply(parse_set_scores_for_match)
    close_sets_p1_won = np.array([x[0] for x in parsed_set_scores])
    close_sets_p2_won = np.array([x[1] for x in parsed_set_scores])
    close_sets_total = np.array([x[2] for x in parsed_set_scores])
    player_ids_p1 = df['Player 1 ID'].values  # Pre-extract for O(1) access

    # ... after df['P1_Win'] = df['P1_Win'].astype(int) ...

    ## NEW ## - Initialize Elo tracking
    print("--- Initializing Elo Rating System ---")
    elo_ratings = {}
    STARTING_ELO = 1500
    K_FACTOR = 32 # Common K-factor for Elo calculations

    # OPTIMIZED: Using deque(maxlen=N) instead of list.pop(0) - O(1) vs O(N)
    player_pdr_history = defaultdict(lambda: deque(maxlen=SLOPE_WINDOW))

    # =========================================================================
    # PHASE 3 OPTIMIZATION: Pre-compute rolling stats using vectorized operations
    # =========================================================================
    print("--- Pre-computing Rolling Statistics (vectorized) ---")
    from collections import defaultdict, deque
    import bisect

    # Step 1: Create long-form DataFrame with one row per (match, player) pair
    player_records = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        p1_id = row['Player 1 ID']
        p2_id = row['Player 2 ID']
        p1_won = row['P1_Win'] == 1

        # P1's perspective
        player_records.append({
            'match_idx': idx,
            'player_id': p1_id,
            'won': 1 if p1_won else 0,
            'points_won': row['P1 Total Points'],
            'points_lost': row['P2 Total Points'],
            'set_comebacks': row['P1 Set Comebacks'],
        })
        # P2's perspective
        player_records.append({
            'match_idx': idx,
            'player_id': p2_id,
            'won': 0 if p1_won else 1,
            'points_won': row['P2 Total Points'],
            'points_lost': row['P1 Total Points'],
            'set_comebacks': row['P2 Set Comebacks'],
        })

    player_df = pd.DataFrame(player_records)
    player_df = player_df.sort_values(['player_id', 'match_idx']).reset_index(drop=True)

    # Step 2: Calculate rolling stats with shift(1) to exclude current match (point-in-time)
    print("   Computing Win Rate L20...")
    player_df['win_rate_L20'] = player_df.groupby('player_id')['won'].transform(
        lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    ).fillna(0.5)

    print("   Computing Win Rate L5...")
    player_df['win_rate_L5'] = player_df.groupby('player_id')['won'].transform(
        lambda x: x.shift(1).rolling(SHORT_ROLLING_WINDOW, min_periods=1).mean()
    ).fillna(0.5)

    print("   Computing Set Comebacks L20...")
    player_df['set_comebacks_L20'] = player_df.groupby('player_id')['set_comebacks'].transform(
        lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).sum()
    ).fillna(0)

    print("   Computing PDR...")
    player_df['points_total'] = player_df['points_won'] + player_df['points_lost']
    player_df['rolling_points_won'] = player_df.groupby('player_id')['points_won'].transform(
        lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).sum()
    )
    player_df['rolling_points_total'] = player_df.groupby('player_id')['points_total'].transform(
        lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).sum()
    )
    player_df['pdr'] = (player_df['rolling_points_won'] / player_df['rolling_points_total']).fillna(0.5)

    # Step 3: Create fast lookup dictionary: (match_idx, player_id) -> stats
    print("   Building lookup tables...")
    player_stats_lookup = {}
    for _, row in player_df.iterrows():
        key = (row['match_idx'], row['player_id'])
        player_stats_lookup[key] = {
            'win_rate_L20': row['win_rate_L20'],
            'win_rate_L5': row['win_rate_L5'],
            'set_comebacks_L20': row['set_comebacks_L20'],
            'pdr': row['pdr'],
        }

    # Step 4: Pre-compute player match indices for remaining loop operations
    # player_match_indices[player_id] = list of match indices (chronological)
    player_match_indices = defaultdict(list)
    for idx in range(len(df)):
        row = df.iloc[idx]
        player_match_indices[row['Player 1 ID']].append(idx)
        player_match_indices[row['Player 2 ID']].append(idx)

    # h2h_pair_indices[(min_id, max_id)] = list of match indices
    h2h_pair_indices = defaultdict(list)
    for idx in range(len(df)):
        row = df.iloc[idx]
        h2h_key = (min(row['Player 1 ID'], row['Player 2 ID']), max(row['Player 1 ID'], row['Player 2 ID']))
        h2h_pair_indices[h2h_key].append(idx)

    # OPTIMIZED: Pre-compute date arrays for O(1) lookups instead of df.iloc[i]['Date']
    print("   Pre-computing date arrays for fast lookups...")
    df_dates = df['Date'].values  # numpy array for O(1) access
    df_dates_as_date = np.array([pd.Timestamp(d).date() for d in df_dates])  # date-only for today comparison
    df_p1_total_points = df['P1 Total Points'].values
    df_p2_total_points = df['P2 Total Points'].values

    print(f"   Pre-computed stats for {len(player_stats_lookup)} player-match pairs")
    print("--- Starting Feature Engineering Loop ---")
    engineered_rows = []

    # Iterate through each match to calculate point-in-time features symmetrically
    for index, match in tqdm(df.iterrows(), total=df.shape[0]):
        # OPTIMIZED: Removed history_df = df.iloc[:index] - using pre-computed indices

        p1_id = match['Player 1 ID']
        p2_id = match['Player 2 ID']
        current_date = match['Date'].date()

        # =====================================================================
        # PHASE 3: Use pre-computed rolling stats from lookup
        # =====================================================================
        p1_stats = player_stats_lookup.get((index, p1_id), {'win_rate_L20': 0.5, 'win_rate_L5': 0.5, 'set_comebacks_L20': 0, 'pdr': 0.5})
        p2_stats = player_stats_lookup.get((index, p2_id), {'win_rate_L20': 0.5, 'win_rate_L5': 0.5, 'set_comebacks_L20': 0, 'pdr': 0.5})

        p1_pdr = p1_stats['pdr']
        p2_pdr = p2_stats['pdr']
        pdr_advantage = p1_pdr - p2_pdr

        p1_win_rate = p1_stats['win_rate_L20']
        p2_win_rate = p2_stats['win_rate_L20']

        p1_win_rate_l5 = p1_stats['win_rate_L5']
        p2_win_rate_l5 = p2_stats['win_rate_L5']
        win_rate_advantage_l5 = p1_win_rate_l5 - p2_win_rate_l5

        p1_rolling_comebacks = p1_stats['set_comebacks_L20']
        p2_rolling_comebacks = p2_stats['set_comebacks_L20']

        # =====================================================================
        # PDR Slope - MUST stay in loop (STATEFUL calculation)
        # OPTIMIZED: deque(maxlen=N) auto-evicts oldest - no manual pop needed
        # =====================================================================
        player_pdr_history[p1_id].append(p1_pdr)
        p1_pdr_slope = calculate_performance_slope(list(player_pdr_history[p1_id]))

        player_pdr_history[p2_id].append(p2_pdr)
        p2_pdr_slope = calculate_performance_slope(list(player_pdr_history[p2_id]))
        pdr_slope_advantage = p1_pdr_slope - p2_pdr_slope

        # =====================================================================
        # Daily Fatigue & Time-based features - use pre-computed player indices
        # OPTIMIZED: Use bisect for O(log N) index filtering instead of O(N) list comprehension
        # =====================================================================
        p1_all_indices = player_match_indices[p1_id]
        p2_all_indices = player_match_indices[p2_id]

        # Binary search to find cutoff point - indices are sorted chronologically
        p1_cutoff = bisect.bisect_left(p1_all_indices, index)
        p2_cutoff = bisect.bisect_left(p2_all_indices, index)
        p1_history_indices = p1_all_indices[:p1_cutoff]
        p2_history_indices = p2_all_indices[:p2_cutoff]

        # OPTIMIZED: P1 today's games using pre-computed date arrays
        if p1_history_indices:
            p1_history_dates = df_dates_as_date[p1_history_indices]
            p1_today_mask = p1_history_dates == current_date
            if np.any(p1_today_mask):
                p1_today_indices = np.array(p1_history_indices)[p1_today_mask]
                p1_points_today = np.sum(df_p1_total_points[p1_today_indices] + df_p2_total_points[p1_today_indices])
                p1_is_first_match_of_day = 0
            else:
                p1_points_today = 0
                p1_is_first_match_of_day = 1
        else:
            p1_points_today = 0
            p1_is_first_match_of_day = 1

        # OPTIMIZED: P2 today's games using pre-computed date arrays
        if p2_history_indices:
            p2_history_dates = df_dates_as_date[p2_history_indices]
            p2_today_mask = p2_history_dates == current_date
            if np.any(p2_today_mask):
                p2_today_indices = np.array(p2_history_indices)[p2_today_mask]
                p2_points_today = np.sum(df_p1_total_points[p2_today_indices] + df_p2_total_points[p2_today_indices])
                p2_is_first_match_of_day = 0
            else:
                p2_points_today = 0
                p2_is_first_match_of_day = 1
        else:
            p2_points_today = 0
            p2_is_first_match_of_day = 1

        daily_fatigue_advantage = p1_points_today - p2_points_today

        # OPTIMIZED: Time since last match using pre-computed arrays
        if p1_history_indices:
            p1_last_game_date = pd.Timestamp(df_dates[p1_history_indices[-1]])
            p1_time_since_last_match_hours = (match['Date'] - p1_last_game_date).total_seconds() / 3600
        else:
            p1_time_since_last_match_hours = 72

        if p2_history_indices:
            p2_last_game_date = pd.Timestamp(df_dates[p2_history_indices[-1]])
            p2_time_since_last_match_hours = (match['Date'] - p2_last_game_date).total_seconds() / 3600
        else:
            p2_time_since_last_match_hours = 72

        time_since_last_advantage = p1_time_since_last_match_hours - p2_time_since_last_match_hours

        # OPTIMIZED: Matches in last 24 hours using numpy array comparisons
        cutoff_time = match['Date'] - pd.Timedelta(hours=24)
        cutoff_time_np = np.datetime64(cutoff_time)
        if p1_history_indices:
            p1_matches_last_24h = np.sum(df_dates[p1_history_indices] > cutoff_time_np)
        else:
            p1_matches_last_24h = 0
        if p2_history_indices:
            p2_matches_last_24h = np.sum(df_dates[p2_history_indices] > cutoff_time_np)
        else:
            p2_matches_last_24h = 0
        matches_last_24h_advantage = p1_matches_last_24h - p2_matches_last_24h

        is_first_match_advantage = p1_is_first_match_of_day - p2_is_first_match_of_day

        # =====================================================================
        # Close Set Win Rate - OPTIMIZED: uses pre-parsed set scores
        # =====================================================================
        p1_rolling_indices = p1_history_indices[-ROLLING_WINDOW:] if p1_history_indices else []
        p2_rolling_indices = p2_history_indices[-ROLLING_WINDOW:] if p2_history_indices else []

        # OPTIMIZED: Use pre-parsed close set data instead of parsing strings in loop
        p1_close_set_win_rate = calculate_close_set_win_rate_optimized(
            p1_id, p1_rolling_indices, player_ids_p1, close_sets_p1_won, close_sets_p2_won, close_sets_total
        )
        p2_close_set_win_rate = calculate_close_set_win_rate_optimized(
            p2_id, p2_rolling_indices, player_ids_p1, close_sets_p1_won, close_sets_p2_won, close_sets_total
        )
        close_set_win_rate_advantage = p1_close_set_win_rate - p2_close_set_win_rate

        # =====================================================================
        # H2H Calculation - use pre-computed H2H indices
        # OPTIMIZED: Use bisect for O(log N) instead of O(N) list comprehension
        # =====================================================================
        h2h_key = (min(p1_id, p2_id), max(p1_id, p2_id))
        h2h_all_indices = h2h_pair_indices[h2h_key]
        h2h_cutoff = bisect.bisect_left(h2h_all_indices, index)
        h2h_indices = h2h_all_indices[:h2h_cutoff]
        h2h_df = df.iloc[h2h_indices] if h2h_indices else df.iloc[[]]

        if len(h2h_df) > 0:
            # OPTIMIZED: Boolean masking instead of apply(axis=1) - vectorized operations
            p1_was_player1 = h2h_df['Player 1 ID'].values == p1_id
            p1_won_as_player1 = p1_was_player1 & (h2h_df['P1_Win'].values == 1)
            p1_won_as_player2 = (~p1_was_player1) & (h2h_df['P1_Win'].values == 0)
            p1_h2h_wins = np.sum(p1_won_as_player1 | p1_won_as_player2)
            h2h_p1_win_rate = p1_h2h_wins / len(h2h_df)
        else:
            h2h_p1_win_rate = 0.5

        # --- H2H Dominance Calculation ---
        h2h_dominance_score = calculate_h2h_dominance(p1_id, h2h_df, match['Date'], H2H_DECAY_FACTOR)
        
        # - Update Elo ratings based on the match outcome for the next iteration
        p1_won = match['P1_Win'] == 1
#        new_p1_elo, new_p2_elo = update_elo(p1_pre_match_elo, p2_pre_match_elo, p1_won, K_FACTOR)
#        elo_ratings[p1_id] = new_p1_elo
#        elo_ratings[p2_id] = new_p2_elo

        # Append the new, correct row
        new_row = match.to_dict()
        new_row.update({
            'PDR_Slope_Advantage': pdr_slope_advantage,
            'Daily_Fatigue_Advantage': daily_fatigue_advantage,
            'PDR_Advantage': pdr_advantage,
            'P1_Rolling_Win_Rate_L10': p1_win_rate,
#            'P1_Rolling_Pressure_Points_L10': p1_pressure_points,
            'P2_Rolling_Win_Rate_L10': p2_win_rate,
            'Win_Rate_L5_Advantage': win_rate_advantage_l5,
#            'P2_Rolling_Pressure_Points_L10': p2_pressure_points,
            'Close_Set_Win_Rate_Advantage': close_set_win_rate_advantage,
#            'P1_Rest_Days': p1_rest_days,
#            'P2_Rest_Days': p2_rest_days,
            'Time_Since_Last_Advantage': time_since_last_advantage, 
            'Matches_Last_24H_Advantage': matches_last_24h_advantage, 
            'Is_First_Match_Advantage': is_first_match_advantage, 
            'H2H_P1_Win_Rate': h2h_p1_win_rate,
            'H2H_Dominance_Score': h2h_dominance_score,
            'P1_Rolling_Set_Comebacks_L20': p1_rolling_comebacks, 
            'P2_Rolling_Set_Comebacks_L20': p2_rolling_comebacks  
        })
        engineered_rows.append(new_row)

    # --- 3. Save the Final Dataset ---
    final_df = pd.DataFrame(engineered_rows)
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Symmetrical feature engineering complete. Data saved to '{OUTPUT_FILE}'")

except FileNotFoundError:
    print(f"Error: The file '{RAW_STATS_FILE}' was not found. Please run the data collector first.")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()