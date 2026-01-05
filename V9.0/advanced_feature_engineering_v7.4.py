import pandas as pd
from tqdm import tqdm
import numpy as np
import math 


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
def calculate_h2h_dominance(p1_id, h2h_df, current_date, decay_factor):
    """
    Calculates a recency-weighted H2H dominance score based on point differentials.
    """
    if h2h_df.empty:
        return 0.0

    total_weighted_score = 0
    
    for _, game in h2h_df.iterrows():
        # Calculate recency weight
        days_ago = (current_date - game['Date']).days
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

# - Points Dominance Ratio calculation logic
def calculate_pdr(player_id, rolling_games_df):
    """Calculates the Points Dominance Ratio for a single player."""
    if rolling_games_df.empty:
        return 0.5 # A neutral default value

    total_points_won = 0
    total_points_played = 0

    # Symmetrically sum up points won and lost
    for _, game in rolling_games_df.iterrows():
        if game['Player 1 ID'] == player_id:
            points_won = game['P1 Total Points']
            points_lost = game['P2 Total Points'] # Opponent's points won are player's points lost
        else: # Player was P2
            points_won = game['P2 Total Points']
            points_lost = game['P1 Total Points']
        
        total_points_won += points_won
        total_points_played += (points_won + points_lost)
    
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

    # ... after df['P1_Win'] = df['P1_Win'].astype(int) ...

    ## NEW ## - Initialize Elo tracking
    print("--- Initializing Elo Rating System ---")
    elo_ratings = {}
    STARTING_ELO = 1500
    K_FACTOR = 32 # Common K-factor for Elo calculations

    player_pdr_history = {} ## NEW ## - To track recent PDRs for slope calculation

    # =========================================================================
    # PHASE 3 OPTIMIZATION: Pre-compute rolling stats using vectorized operations
    # =========================================================================
    print("--- Pre-computing Rolling Statistics (vectorized) ---")
    from collections import defaultdict

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
        # =====================================================================
        if p1_id not in player_pdr_history: player_pdr_history[p1_id] = []
        player_pdr_history[p1_id].append(p1_pdr)
        if len(player_pdr_history[p1_id]) > SLOPE_WINDOW:
            player_pdr_history[p1_id].pop(0)
        p1_pdr_slope = calculate_performance_slope(player_pdr_history[p1_id])

        if p2_id not in player_pdr_history: player_pdr_history[p2_id] = []
        player_pdr_history[p2_id].append(p2_pdr)
        if len(player_pdr_history[p2_id]) > SLOPE_WINDOW:
            player_pdr_history[p2_id].pop(0)
        p2_pdr_slope = calculate_performance_slope(player_pdr_history[p2_id])
        pdr_slope_advantage = p1_pdr_slope - p2_pdr_slope

        # =====================================================================
        # Daily Fatigue & Time-based features - use pre-computed player indices
        # =====================================================================
        p1_history_indices = [i for i in player_match_indices[p1_id] if i < index]
        p2_history_indices = [i for i in player_match_indices[p2_id] if i < index]

        # P1 today's games
        p1_today_indices = [i for i in p1_history_indices if df.iloc[i]['Date'].date() == current_date]
        if p1_today_indices:
            p1_games_today = df.iloc[p1_today_indices]
            p1_points_today = (p1_games_today['P1 Total Points'] + p1_games_today['P2 Total Points']).sum()
            p1_is_first_match_of_day = 0
        else:
            p1_points_today = 0
            p1_is_first_match_of_day = 1

        # P2 today's games
        p2_today_indices = [i for i in p2_history_indices if df.iloc[i]['Date'].date() == current_date]
        if p2_today_indices:
            p2_games_today = df.iloc[p2_today_indices]
            p2_points_today = (p2_games_today['P1 Total Points'] + p2_games_today['P2 Total Points']).sum()
            p2_is_first_match_of_day = 0
        else:
            p2_points_today = 0
            p2_is_first_match_of_day = 1

        daily_fatigue_advantage = p1_points_today - p2_points_today

        # Time since last match
        if p1_history_indices:
            p1_last_game_date = df.iloc[p1_history_indices[-1]]['Date']
            p1_time_since_last_match_hours = (match['Date'] - p1_last_game_date).total_seconds() / 3600
        else:
            p1_time_since_last_match_hours = 72

        if p2_history_indices:
            p2_last_game_date = df.iloc[p2_history_indices[-1]]['Date']
            p2_time_since_last_match_hours = (match['Date'] - p2_last_game_date).total_seconds() / 3600
        else:
            p2_time_since_last_match_hours = 72

        time_since_last_advantage = p1_time_since_last_match_hours - p2_time_since_last_match_hours

        # Matches in last 24 hours
        cutoff_time = match['Date'] - pd.Timedelta(hours=24)
        p1_matches_last_24h = sum(1 for i in p1_history_indices if df.iloc[i]['Date'] > cutoff_time)
        p2_matches_last_24h = sum(1 for i in p2_history_indices if df.iloc[i]['Date'] > cutoff_time)
        matches_last_24h_advantage = p1_matches_last_24h - p2_matches_last_24h

        is_first_match_advantage = p1_is_first_match_of_day - p2_is_first_match_of_day

        # =====================================================================
        # Close Set Win Rate - still needs rolling games (uses Set Scores string)
        # =====================================================================
        p1_rolling_indices = p1_history_indices[-ROLLING_WINDOW:] if p1_history_indices else []
        p2_rolling_indices = p2_history_indices[-ROLLING_WINDOW:] if p2_history_indices else []

        p1_rolling_games = df.iloc[p1_rolling_indices] if p1_rolling_indices else df.iloc[[]]
        p2_rolling_games = df.iloc[p2_rolling_indices] if p2_rolling_indices else df.iloc[[]]

        p1_close_set_win_rate = calculate_close_set_win_rate(p1_id, p1_rolling_games)
        p2_close_set_win_rate = calculate_close_set_win_rate(p2_id, p2_rolling_games)
        close_set_win_rate_advantage = p1_close_set_win_rate - p2_close_set_win_rate

        # =====================================================================
        # H2H Calculation - use pre-computed H2H indices
        # =====================================================================
        h2h_key = (min(p1_id, p2_id), max(p1_id, p2_id))
        h2h_indices = [i for i in h2h_pair_indices[h2h_key] if i < index]
        h2h_df = df.iloc[h2h_indices] if h2h_indices else df.iloc[[]]

        if len(h2h_df) > 0:
            p1_h2h_wins = h2h_df.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).sum()
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