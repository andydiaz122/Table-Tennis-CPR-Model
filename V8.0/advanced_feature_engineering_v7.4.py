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

    # --- OPTIMIZATION: Pre-build player index for O(1) lookups ---
    print("--- Building player match indices ---")
    player_match_indices = {}
    for idx in range(len(df)):
        row = df.iloc[idx]
        for pid in [row['Player 1 ID'], row['Player 2 ID']]:
            if pid not in player_match_indices:
                player_match_indices[pid] = []
            player_match_indices[pid].append(idx)

    # --- OPTIMIZATION: Pre-build H2H index for O(1) lookups ---
    print("--- Building H2H match indices ---")
    h2h_match_indices = {}
    for idx in range(len(df)):
        row = df.iloc[idx]
        key = frozenset([row['Player 1 ID'], row['Player 2 ID']])
        if key not in h2h_match_indices:
            h2h_match_indices[key] = []
        h2h_match_indices[key].append(idx)

    print("--- Starting Symmetrical Feature Engineering (this may take a few minutes) ---")
    engineered_rows = []

    # Iterate through each match to calculate point-in-time features symmetrically
    for index, match in tqdm(df.iterrows(), total=df.shape[0]):
        p1_id = match['Player 1 ID']
        p2_id = match['Player 2 ID']

        # --- OPTIMIZED: Use pre-built indices instead of filtering ---
        p1_indices = [i for i in player_match_indices.get(p1_id, []) if i < index]
        p2_indices = [i for i in player_match_indices.get(p2_id, []) if i < index]
        p1_games = df.iloc[p1_indices] if p1_indices else pd.DataFrame()
        p2_games = df.iloc[p2_indices] if p2_indices else pd.DataFrame()

        # - Daily Fatigue Calculation
        current_date = match['Date'].date()

        # Filter player games for matches played earlier today
        p1_games_today = p1_games[p1_games['Date'].dt.date == current_date] if not p1_games.empty else pd.DataFrame()
        p1_points_today = (p1_games_today['P1 Total Points'] + p1_games_today['P2 Total Points']).sum() if not p1_games_today.empty else 0

        # Check if it's the first match of the day
        p1_is_first_match_of_day = 1 if p1_games_today.empty else 0

        # Calculate P2's workload today
        p2_games_today = p2_games[p2_games['Date'].dt.date == current_date] if not p2_games.empty else pd.DataFrame()
        p2_points_today = (p2_games_today['P1 Total Points'] + p2_games_today['P2 Total Points']).sum() if not p2_games_today.empty else 0

        p2_is_first_match_of_day = 1 if p2_games_today.empty else 0

        daily_fatigue_advantage = p1_points_today - p2_points_today


        # --- Symmetrical Stat Calculation for Player 1 ---
        p1_rolling_games = p1_games.tail(ROLLING_WINDOW) if not p1_games.empty else pd.DataFrame()
        
        # --- OPTIMIZED: Vectorized PDR calculation ---
        if not p1_rolling_games.empty:
            p1_is_player1_pdr = p1_rolling_games['Player 1 ID'] == p1_id
            p1_points_won = np.where(p1_is_player1_pdr, p1_rolling_games['P1 Total Points'], p1_rolling_games['P2 Total Points']).sum()
            p1_points_played = p1_rolling_games['P1 Total Points'].sum() + p1_rolling_games['P2 Total Points'].sum()
            p1_pdr = p1_points_won / p1_points_played if p1_points_played > 0 else 0.5
        else:
            p1_pdr = 0.5

        # - Update PDR history and calculate slope
        if p1_id not in player_pdr_history: player_pdr_history[p1_id] = []
        player_pdr_history[p1_id].append(p1_pdr)
        if len(player_pdr_history[p1_id]) > SLOPE_WINDOW:
            player_pdr_history[p1_id].pop(0) # Keep the list at the desired size
        p1_pdr_slope = calculate_performance_slope(player_pdr_history[p1_id])

        # --- OPTIMIZED: Vectorized win rate calculation ---
        if not p1_rolling_games.empty:
            p1_wins = ((p1_rolling_games['Player 1 ID'] == p1_id) & (p1_rolling_games['P1_Win'] == 1)) | \
                      ((p1_rolling_games['Player 2 ID'] == p1_id) & (p1_rolling_games['P1_Win'] == 0))
            p1_win_rate = p1_wins.mean()
        else:
            p1_win_rate = 0.5
        # Calculate short-term "hot-streak" win rate
        p1_rolling_games_short = p1_games.tail(SHORT_ROLLING_WINDOW)
        if not p1_rolling_games_short.empty:
            p1_wins_l5 = ((p1_rolling_games_short['Player 1 ID'] == p1_id) & (p1_rolling_games_short['P1_Win'] == 1)) | \
                         ((p1_rolling_games_short['Player 2 ID'] == p1_id) & (p1_rolling_games_short['P1_Win'] == 0))
            p1_win_rate_l5 = p1_wins_l5.mean()
        else:
            p1_win_rate_l5 = 0.5
        
        # --- OPTIMIZED: Vectorized pressure points and comebacks ---
        p1_pressure_points = 0.0
        p1_rolling_comebacks = 0
        if not p1_rolling_games.empty:
            p1_is_player1 = p1_rolling_games['Player 1 ID'] == p1_id
            p1_pressure_points = np.where(p1_is_player1, p1_rolling_games['P1 Pressure Points'], p1_rolling_games['P2 Pressure Points']).mean()
            p1_rolling_comebacks = np.where(p1_is_player1, p1_rolling_games['P1 Set Comebacks'], p1_rolling_games['P2 Set Comebacks']).sum()
        p1_close_set_win_rate = calculate_close_set_win_rate(p1_id, p1_rolling_games)

        # --- Calculate rest in hours, not days ---
        if not p1_games.empty:
            p1_last_game_date = p1_games['Date'].max()
            p1_time_since_last_match_hours = (match['Date'] - p1_last_game_date).total_seconds() / 3600
        else:
            p1_time_since_last_match_hours = 72 # Default to 3 days for new players

        # --- Calculate matches in the last 24 hours ---
        p1_matches_last_24h = len(p1_games[p1_games['Date'] > (match['Date'] - pd.Timedelta(hours=24))]) if not p1_games.empty else 0

        # --- Symmetrical Stat Calculation for Player 2 ---
        # (p2_games already computed at top of loop using pre-built indices)
        p2_rolling_games = p2_games.tail(ROLLING_WINDOW) if not p2_games.empty else pd.DataFrame()

        # --- OPTIMIZED: Vectorized PDR calculation ---
        if not p2_rolling_games.empty:
            p2_is_player1_pdr = p2_rolling_games['Player 1 ID'] == p2_id
            p2_points_won = np.where(p2_is_player1_pdr, p2_rolling_games['P1 Total Points'], p2_rolling_games['P2 Total Points']).sum()
            p2_points_played = p2_rolling_games['P1 Total Points'].sum() + p2_rolling_games['P2 Total Points'].sum()
            p2_pdr = p2_points_won / p2_points_played if p2_points_played > 0 else 0.5
        else:
            p2_pdr = 0.5
        pdr_advantage = p1_pdr - p2_pdr

        # - Update PDR history and calculate slope
        if p2_id not in player_pdr_history: player_pdr_history[p2_id] = []
        player_pdr_history[p2_id].append(p2_pdr)
        if len(player_pdr_history[p2_id]) > SLOPE_WINDOW:
            player_pdr_history[p2_id].pop(0)
        p2_pdr_slope = calculate_performance_slope(player_pdr_history[p2_id])
        pdr_slope_advantage = p1_pdr_slope - p2_pdr_slope 

        # --- OPTIMIZED: Vectorized win rate calculation ---
        if not p2_rolling_games.empty:
            p2_wins = ((p2_rolling_games['Player 1 ID'] == p2_id) & (p2_rolling_games['P1_Win'] == 1)) | \
                      ((p2_rolling_games['Player 2 ID'] == p2_id) & (p2_rolling_games['P1_Win'] == 0))
            p2_win_rate = p2_wins.mean()
        else:
            p2_win_rate = 0.5
        # Calculate short-term "hot-streak" win rate
        p2_rolling_games_short = p2_games.tail(SHORT_ROLLING_WINDOW)
        if not p2_rolling_games_short.empty:
            p2_wins_l5 = ((p2_rolling_games_short['Player 1 ID'] == p2_id) & (p2_rolling_games_short['P1_Win'] == 1)) | \
                         ((p2_rolling_games_short['Player 2 ID'] == p2_id) & (p2_rolling_games_short['P1_Win'] == 0))
            p2_win_rate_l5 = p2_wins_l5.mean()
        else:
            p2_win_rate_l5 = 0.5
        # After all individual player calculations
        win_rate_advantage_l5 = p1_win_rate_l5 - p2_win_rate_l5

        # --- OPTIMIZED: Vectorized pressure points and comebacks ---
        p2_pressure_points = 0.0
        p2_rolling_comebacks = 0
        if not p2_rolling_games.empty:
            p2_is_player1 = p2_rolling_games['Player 1 ID'] == p2_id
            p2_pressure_points = np.where(p2_is_player1, p2_rolling_games['P1 Pressure Points'], p2_rolling_games['P2 Pressure Points']).mean()
            p2_rolling_comebacks = np.where(p2_is_player1, p2_rolling_games['P1 Set Comebacks'], p2_rolling_games['P2 Set Comebacks']).sum()
        p2_close_set_win_rate = calculate_close_set_win_rate(p2_id, p2_rolling_games)
        close_set_win_rate_advantage = p1_close_set_win_rate - p2_close_set_win_rate

        # Calculate rest in hours, not days
        if not p2_games.empty:
            p2_last_game_date = p2_games['Date'].max()
            p2_time_since_last_match_hours = (match['Date'] - p2_last_game_date).total_seconds() / 3600
        else:
            p2_time_since_last_match_hours = 72 # Default to 3 days for new players

        # --- Calculate matches in the last 24 hours ---
        p2_matches_last_24h = len(p2_games[p2_games['Date'] > (match['Date'] - pd.Timedelta(hours=24))]) if not p2_games.empty else 0

        time_since_last_advantage = p1_time_since_last_match_hours - p2_time_since_last_match_hours
        matches_last_24h_advantage = p1_matches_last_24h - p2_matches_last_24h
        is_first_match_advantage = p1_is_first_match_of_day - p2_is_first_match_of_day

        # --- OPTIMIZED: H2H Calculation using pre-built index ---
        h2h_key = frozenset([p1_id, p2_id])
        h2h_indices = [i for i in h2h_match_indices.get(h2h_key, []) if i < index]
        h2h_df = df.iloc[h2h_indices] if h2h_indices else pd.DataFrame()
        # --- OPTIMIZED: Vectorized H2H win rate calculation ---
        if not h2h_df.empty:
            h2h_p1_wins = ((h2h_df['Player 1 ID'] == p1_id) & (h2h_df['P1_Win'] == 1)) | \
                          ((h2h_df['Player 2 ID'] == p1_id) & (h2h_df['P1_Win'] == 0))
            h2h_p1_win_rate = h2h_p1_wins.mean()
        else:
            h2h_p1_win_rate = 0.5

        # --- H2H Dominance Calculation --- ## MODIFIED ##
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