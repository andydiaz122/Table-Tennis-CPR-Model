import pandas as pd
from tqdm import tqdm
import numpy as np
import math


# - Robust score parsing utility
def parse_set_scores(set_scores_str):
    """
    Parses set scores string into list of (p1_score, p2_score) tuples.
    Handles both ", " and "," delimiters.

    Args:
        set_scores_str: e.g., "2-11, 11-3, 7-11" OR "2-11,11-3,7-11"

    Returns:
        [(2, 11), (11, 3), (7, 11)]
    """
    if pd.isna(set_scores_str) or not set_scores_str:
        return []

    try:
        # Normalize delimiters: replace ", " with ","
        normalized = str(set_scores_str).replace(", ", ",")
        sets = []
        for set_score in normalized.split(","):
            set_score = set_score.strip()
            if "-" in set_score:
                p1, p2 = map(int, set_score.split("-"))
                sets.append((p1, p2))
        return sets
    except (ValueError, AttributeError):
        return []


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


# - Clutch Factor calculation (deuce sets only: 10-10 or beyond)
def calculate_clutch_factor(player_id, rolling_games_df):
    """
    Calculates a player's win percentage in DEUCE sets (10-10 or beyond).
    A deuce set is when both players reached 10+ points and winner won by exactly 2.

    Returns: (clutch_factor, deuce_sets_count)
    """
    if rolling_games_df.empty:
        return 0.5, 0

    total_deuce_sets = 0
    deuce_sets_won = 0

    for _, game in rolling_games_df.iterrows():
        set_scores = parse_set_scores(game.get('Set Scores'))

        for p1_points, p2_points in set_scores:
            min_score = min(p1_points, p2_points)
            max_score = max(p1_points, p2_points)

            # Deuce set: both reached 10+, winner won by exactly 2
            if min_score >= 10 and max_score == min_score + 2:
                total_deuce_sets += 1

                is_p1 = (game['Player 1 ID'] == player_id)
                p1_won_set = p1_points > p2_points

                if (is_p1 and p1_won_set) or (not is_p1 and not p1_won_set):
                    deuce_sets_won += 1

    if total_deuce_sets == 0:
        return 0.5, 0

    return deuce_sets_won / total_deuce_sets, total_deuce_sets


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


# - Pythagorean Expectation calculation
def calculate_pythagorean_win_rate(player_id, rolling_games_df, gamma=9.5):
    """
    Calculates Pythagorean expected win rate based on points scored/allowed.
    Formula: PW^gamma / (PW^gamma + PA^gamma)
    """
    if rolling_games_df.empty:
        return 0.5

    total_points_won = 0
    total_points_allowed = 0

    for _, game in rolling_games_df.iterrows():
        if game['Player 1 ID'] == player_id:
            total_points_won += game['P1 Total Points']
            total_points_allowed += game['P2 Total Points']
        else:
            total_points_won += game['P2 Total Points']
            total_points_allowed += game['P1 Total Points']

    if total_points_won + total_points_allowed == 0:
        return 0.5

    pw_gamma = total_points_won ** gamma
    pa_gamma = total_points_allowed ** gamma

    return pw_gamma / (pw_gamma + pa_gamma)


# - Exponential fatigue calculation
# NOTE: Since data only has date (not timestamps), we use match index position
# to estimate fatigue. Assume ~30 min average gap between matches on same day.
ESTIMATED_MATCH_GAP_MINUTES = 30

def calculate_exponential_fatigue(player_id, current_match_index, player_games_today, decay_lambda=0.02):
    """
    Calculates exponential fatigue factor based on same-day matches.
    Uses match index position to estimate time gaps since no timestamps available.
    More recent matches contribute more to fatigue.

    Args:
        player_id: The player to calculate fatigue for
        current_match_index: The index of the current match in the dataset
        player_games_today: DataFrame of this player's matches earlier today
        decay_lambda: Decay rate per minute (default 0.02 = ~35 min half-life)

    Returns:
        float: Fatigue factor (weighted sum of points played, with decay)
    """
    if player_games_today.empty:
        return 0.0

    fatigue = 0.0
    num_earlier_matches = len(player_games_today)

    # Process matches from oldest to newest (by their position in the day)
    for i, (idx, game) in enumerate(player_games_today.iterrows()):
        # Estimate minutes elapsed based on match position
        # Most recent match = position 0, next most recent = position 1, etc.
        matches_ago = num_earlier_matches - i  # 1 for most recent, 2 for 2nd most recent, etc.
        estimated_minutes_elapsed = matches_ago * ESTIMATED_MATCH_GAP_MINUTES

        # Total points played in that match
        total_points = game['P1 Total Points'] + game['P2 Total Points']

        # Exponential decay: recent matches contribute more
        fatigue += total_points * math.exp(-decay_lambda * estimated_minutes_elapsed)

    return fatigue


# - Elo calculation logic
def update_elo(p1_elo, p2_elo, p1_won, k_factor=32):
    expected_p1 = 1 / (1 + 10 ** ((p2_elo - p1_elo) / 400))
    expected_p2 = 1 / (1 + 10 ** ((p1_elo - p2_elo) / 400))
    score_p1 = 1 if p1_won else 0
    score_p2 = 0 if p1_won else 1
    new_p1_elo = p1_elo + k_factor * (score_p1 - expected_p1)
    new_p2_elo = p2_elo + k_factor * (score_p2 - expected_p2)
    return new_p1_elo, new_p2_elo


# - Glicko-2 rating system
def glicko2_update(player_mu, player_phi, player_sigma,
                   opponent_mu, opponent_phi, outcome):
    """
    Updates Glicko-2 ratings after a single match.

    Args:
        player_mu: Player's current rating (Glicko-2 scale, 0 = 1500 Elo)
        player_phi: Player's current rating deviation (uncertainty)
        player_sigma: Player's current volatility
        opponent_mu: Opponent's rating
        opponent_phi: Opponent's rating deviation
        outcome: 1 for win, 0 for loss

    Returns: (new_mu, new_phi, new_sigma)
    """
    # Step 1: Compute g(phi) and E (expected score)
    g = 1 / math.sqrt(1 + 3 * opponent_phi**2 / (math.pi**2))
    E = 1 / (1 + math.exp(-g * (player_mu - opponent_mu)))

    # Step 2: Compute variance v
    v = 1 / (g**2 * E * (1 - E))

    # Step 3: Compute delta (performance vs expectation)
    delta = v * g * (outcome - E)

    # Step 4: Simplified volatility update (keep stable for speed)
    new_sigma = player_sigma

    # Step 5: Update RD (phi) - incorporates uncertainty decay
    phi_star = math.sqrt(player_phi**2 + new_sigma**2)
    new_phi = 1 / math.sqrt(1/phi_star**2 + 1/v)

    # Step 6: Update rating (mu)
    new_mu = player_mu + new_phi**2 * g * (outcome - E)

    return new_mu, new_phi, new_sigma


# --- 1. Configuration ---
RAW_STATS_FILE = "../../czech_liga_pro_advanced_stats_FIXED.csv"
OUTPUT_FILE = "final_engineered_features_v7.4.csv" # New, corrected output file
ROLLING_WINDOW = 20
SHORT_ROLLING_WINDOW = 5
SLOPE_WINDOW = 10       # - Number of recent matches to calculate the slope over
H2H_DECAY_FACTOR = 0.98    # - Decay for H2H recency weighting

# Glicko-2 Constants
GLICKO_INITIAL_MU = 0.0       # Starting rating (0 = 1500 Elo equivalent)
GLICKO_INITIAL_PHI = 2.0148   # Initial rating deviation (~350 Elo points uncertainty)
GLICKO_INITIAL_SIGMA = 0.06   # Initial volatility

# Pythagorean Expectation
PYTHAGOREAN_GAMMA = 9.5  # Optimized exponent for table tennis

# Fatigue Decay
FATIGUE_DECAY_LAMBDA = 0.02  # Per minute, ~35 min half-life


# --- 2. Main Logic ---
try:
    print(f"--- Loading Raw Data from '{RAW_STATS_FILE}' ---")
    df = pd.read_csv(RAW_STATS_FILE)

    # --- Data Cleaning and Preparation ---
    df['Date'] = pd.to_datetime(df['Date'])
    # CRITICAL FIX: Create proper DateTime column for sorting
    # String sorting of Time fails: '8:30:00' > '18:30:00' in ASCII!
    # This caused matches at 8-9am to see future matches in their history
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'])
    df.sort_values(by='DateTime', inplace=True)
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

    # CRITICAL FIX: Reset index after dropna to maintain sequential indices
    # Without this, df.iloc[:index] in the main loop includes future matches!
    # (rows dropped creates gaps in index → iloc uses position, not label)
    df.reset_index(drop=True, inplace=True)

    # ... after df['P1_Win'] = df['P1_Win'].astype(int) ...
    
    ## NEW ## - Initialize Elo tracking
    print("--- Initializing Elo Rating System ---")
    elo_ratings = {}
    STARTING_ELO = 1500
    K_FACTOR = 32 # Common K-factor for Elo calculations

    ## NEW ## - Initialize Glicko-2 tracking
    print("--- Initializing Glicko-2 Rating System ---")
    glicko_ratings = {}  # {player_id: (mu, phi, sigma)}

    player_pdr_history = {} ## NEW ## - To track recent PDRs for slope calculation

    print("--- Starting Symmetrical Feature Engineering (this may take a few minutes) ---")
    engineered_rows = []

    # Iterate through each match to calculate point-in-time features symmetrically
    for index, match in tqdm(df.iterrows(), total=df.shape[0]):
        history_df = df.iloc[:index]

        p1_id = match['Player 1 ID']
        p2_id = match['Player 2 ID']

        # - Get pre-match Elo ratings and calculate the advantage feature
        p1_pre_match_elo = elo_ratings.get(p1_id, STARTING_ELO)
        p2_pre_match_elo = elo_ratings.get(p2_id, STARTING_ELO)
        elo_advantage = p1_pre_match_elo - p2_pre_match_elo

        # - Get pre-match Glicko-2 ratings
        p1_glicko = glicko_ratings.get(p1_id, (GLICKO_INITIAL_MU, GLICKO_INITIAL_PHI, GLICKO_INITIAL_SIGMA))
        p2_glicko = glicko_ratings.get(p2_id, (GLICKO_INITIAL_MU, GLICKO_INITIAL_PHI, GLICKO_INITIAL_SIGMA))
        p1_mu, p1_phi, p1_sigma = p1_glicko
        p2_mu, p2_phi, p2_sigma = p2_glicko

        # - Daily Fatigue Calculation
        current_date = match['Date'].date()
        
        # Filter history for matches played earlier today
        today_history_df = history_df[history_df['Date'].dt.date == current_date]
        
        # Calculate P1's workload today
        p1_games_today = today_history_df[(today_history_df['Player 1 ID'] == p1_id) | (today_history_df['Player 2 ID'] == p1_id)]
        p1_points_today = (p1_games_today['P1 Total Points'] + p1_games_today['P2 Total Points']).sum()

        # Check if it's the first match of the day
        p1_is_first_match_of_day = 1 if p1_games_today.empty else 0

        # Calculate P2's workload today
        p2_games_today = today_history_df[(today_history_df['Player 1 ID'] == p2_id) | (today_history_df['Player 2 ID'] == p2_id)]
        p2_points_today = (p2_games_today['P1 Total Points'] + p2_games_today['P2 Total Points']).sum()
        
        p2_is_first_match_of_day = 1 if p2_games_today.empty else 0

        # Linear fatigue (kept for reference but will be removed from output)
        daily_fatigue_advantage = p1_points_today - p2_points_today

        # Exponential fatigue calculation (replaces linear)
        # Uses player's earlier matches today, not just total points
        p1_fatigue_factor = calculate_exponential_fatigue(p1_id, index, p1_games_today, FATIGUE_DECAY_LAMBDA)
        p2_fatigue_factor = calculate_exponential_fatigue(p2_id, index, p2_games_today, FATIGUE_DECAY_LAMBDA)
        fatigue_factor_diff = p1_fatigue_factor - p2_fatigue_factor

        # --- Symmetrical Stat Calculation for Player 1 ---
        p1_games = history_df[(history_df['Player 1 ID'] == p1_id) | (history_df['Player 2 ID'] == p1_id)]
        p1_rolling_games = p1_games.tail(ROLLING_WINDOW)
        
        p1_pdr = calculate_pdr(p1_id, p1_rolling_games)

        # - Update PDR history and calculate slope
        if p1_id not in player_pdr_history: player_pdr_history[p1_id] = []
        player_pdr_history[p1_id].append(p1_pdr)
        if len(player_pdr_history[p1_id]) > SLOPE_WINDOW:
            player_pdr_history[p1_id].pop(0) # Keep the list at the desired size
        p1_pdr_slope = calculate_performance_slope(player_pdr_history[p1_id])
        p1_pdr_variance = np.var(player_pdr_history[p1_id]) if len(player_pdr_history[p1_id]) > 1 else 0.0

        # CORRECTED: Changed from .sum() / len() to .mean() to perfectly match the backtest script logic.
        p1_win_rate = p1_rolling_games.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or \
                                             (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).mean() \
                                             if not p1_rolling_games.empty else 0.5
        # Calculate short-term "hot-streak" win rate
        p1_rolling_games_short = p1_games.tail(SHORT_ROLLING_WINDOW)
        p1_win_rate_l5 = p1_rolling_games_short.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or \
                                                      (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).mean() \
                                                      if not p1_rolling_games_short.empty else 0.5
        
        p1_pressure_points = 0.0
        if not p1_rolling_games.empty:
            p1_pressure_points = p1_rolling_games.apply(lambda r: r['P1 Pressure Points'] if r['Player 1 ID'] == p1_id else r['P2 Pressure Points'], axis=1).mean()
        # - Calculate rolling sum of set comebacks for Player 1
        p1_rolling_comebacks = p1_rolling_games.apply(lambda r: r['P1 Set Comebacks'] if r['Player 1 ID'] == p1_id else r['P2 Set Comebacks'], axis=1).sum() \
                                                if not p1_rolling_games.empty else 0
        p1_close_set_win_rate = calculate_close_set_win_rate(p1_id, p1_rolling_games)
        p1_clutch_factor, p1_deuce_sets = calculate_clutch_factor(p1_id, p1_rolling_games)
        p1_pythagorean = calculate_pythagorean_win_rate(p1_id, p1_rolling_games, PYTHAGOREAN_GAMMA)
        p1_pythagorean_delta = p1_win_rate - p1_pythagorean  # Luck factor

        p1_last_game_date = p1_games['Date'].max()
#        p1_rest_days = (match['Date'] - p1_last_game_date).days if pd.notna(p1_last_game_date) else 30
        # --- Calculate rest in hours, not days ---
        if pd.notna(p1_last_game_date):
            p1_time_since_last_match_hours = (match['Date'] - p1_last_game_date).total_seconds() / 3600
        else:
            p1_time_since_last_match_hours = 72 # Default to 3 days for new players

        # --- Calculate matches in the last 24 hours ---
        p1_matches_last_24h = len(p1_games[p1_games['Date'] > (match['Date'] - pd.Timedelta(hours=24))])

        # --- Symmetrical Stat Calculation for Player 2 ---
        p2_games = history_df[(history_df['Player 1 ID'] == p2_id) | (history_df['Player 2 ID'] == p2_id)]
        p2_rolling_games = p2_games.tail(ROLLING_WINDOW)

        p2_pdr = calculate_pdr(p2_id, p2_rolling_games)
        pdr_advantage = p1_pdr - p2_pdr

        # - Update PDR history and calculate slope
        if p2_id not in player_pdr_history: player_pdr_history[p2_id] = []
        player_pdr_history[p2_id].append(p2_pdr)
        if len(player_pdr_history[p2_id]) > SLOPE_WINDOW:
            player_pdr_history[p2_id].pop(0)
        p2_pdr_slope = calculate_performance_slope(player_pdr_history[p2_id])
        p2_pdr_variance = np.var(player_pdr_history[p2_id]) if len(player_pdr_history[p2_id]) > 1 else 0.0
        pdr_slope_advantage = p1_pdr_slope - p2_pdr_slope
        pdr_variance_diff = p1_pdr_variance - p2_pdr_variance

        # CORRECTED: Changed from .sum() / len() to .mean() to perfectly match the backtest script logic.
        p2_win_rate = p2_rolling_games.apply(lambda r: 1 if (r['Player 1 ID'] == p2_id and r['P1_Win'] == 1) or \
                                             (r['Player 2 ID'] == p2_id and r['P1_Win'] == 0) else 0, axis=1).mean() \
                                             if not p2_rolling_games.empty else 0.5
        # Calculate short-term "hot-streak" win rate
        p2_rolling_games_short = p2_games.tail(SHORT_ROLLING_WINDOW)
        p2_win_rate_l5 = p2_rolling_games_short.apply(lambda r: 1 if (r['Player 1 ID'] == p2_id and r['P1_Win'] == 1) or \
                                                      (r['Player 2 ID'] == p2_id and r['P1_Win'] == 0) else 0, axis=1).mean() \
                                                      if not p2_rolling_games_short.empty else 0.5
        # After all individual player calculations
        win_rate_advantage_l5 = p1_win_rate_l5 - p2_win_rate_l5

        p2_pressure_points = 0.0
        if not p2_rolling_games.empty:
            p2_pressure_points = p2_rolling_games.apply(lambda r: r['P1 Pressure Points'] if r['Player 1 ID'] == p2_id else r['P2 Pressure Points'], axis=1).mean()
        # - Calculate rolling sum of set comebacks for Player 2
        p2_rolling_comebacks = p2_rolling_games.apply(lambda r: r['P1 Set Comebacks'] if r['Player 1 ID'] == p2_id else r['P2 Set Comebacks'], axis=1).sum() \
                                                if not p2_rolling_games.empty else 0
        p2_close_set_win_rate = calculate_close_set_win_rate(p2_id, p2_rolling_games)
        close_set_win_rate_advantage = p1_close_set_win_rate - p2_close_set_win_rate
        p2_clutch_factor, p2_deuce_sets = calculate_clutch_factor(p2_id, p2_rolling_games)
        clutch_factor_diff = p1_clutch_factor - p2_clutch_factor
        p2_pythagorean = calculate_pythagorean_win_rate(p2_id, p2_rolling_games, PYTHAGOREAN_GAMMA)
        p2_pythagorean_delta = p2_win_rate - p2_pythagorean  # Luck factor
        pythagorean_delta_diff = p1_pythagorean_delta - p2_pythagorean_delta

        p2_last_game_date = p2_games['Date'].max()
#        p2_rest_days = (match['Date'] - p2_last_game_date).days if pd.notna(p2_last_game_date) else 30
        # Calculate rest in hours, not days
        if pd.notna(p2_last_game_date):
            p2_time_since_last_match_hours = (match['Date'] - p2_last_game_date).total_seconds() / 3600
        else:
            p2_time_since_last_match_hours = 72 # Default to 3 days for new players

        # --- Calculate matches in the last 24 hours ---
        p2_matches_last_24h = len(p2_games[p2_games['Date'] > (match['Date'] - pd.Timedelta(hours=24))])
        
        time_since_last_advantage = p1_time_since_last_match_hours - p2_time_since_last_match_hours
        matches_last_24h_advantage = p1_matches_last_24h - p2_matches_last_24h
        is_first_match_advantage = p1_is_first_match_of_day - p2_is_first_match_of_day

        # --- H2H Calculation ---
        h2h_df = history_df[((history_df['Player 1 ID'] == p1_id) & (history_df['Player 2 ID'] == p2_id)) | ((history_df['Player 1 ID'] == p2_id) & (history_df['Player 2 ID'] == p1_id))]
        p1_h2h_wins = h2h_df.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).sum()
        h2h_p1_win_rate = p1_h2h_wins / len(h2h_df) if len(h2h_df) > 0 else 0.5
        h2h_matches_count = len(h2h_df)

        # --- H2H Dominance Calculation --- ## MODIFIED ##
        h2h_dominance_score = calculate_h2h_dominance(p1_id, h2h_df, match['Date'], H2H_DECAY_FACTOR)
        
        # - Update Elo ratings based on the match outcome for the next iteration
        p1_won = match['P1_Win'] == 1
        new_p1_elo, new_p2_elo = update_elo(p1_pre_match_elo, p2_pre_match_elo, p1_won, K_FACTOR)
        elo_ratings[p1_id] = new_p1_elo
        elo_ratings[p2_id] = new_p2_elo

        # - Update Glicko-2 ratings based on the match outcome
        new_p1_mu, new_p1_phi, new_p1_sigma = glicko2_update(
            p1_mu, p1_phi, p1_sigma, p2_mu, p2_phi, 1 if p1_won else 0
        )
        new_p2_mu, new_p2_phi, new_p2_sigma = glicko2_update(
            p2_mu, p2_phi, p2_sigma, p1_mu, p1_phi, 0 if p1_won else 1
        )
        glicko_ratings[p1_id] = (new_p1_mu, new_p1_phi, new_p1_sigma)
        glicko_ratings[p2_id] = (new_p2_mu, new_p2_phi, new_p2_sigma)

        # Append the new, correct row
        new_row = match.to_dict()
        new_row.update({
            # Elo features
            'P1_Elo': p1_pre_match_elo,
            'P2_Elo': p2_pre_match_elo,
            'Elo_Diff': elo_advantage,
            'PDR_Slope_Advantage': pdr_slope_advantage,
            'PDR_Advantage': pdr_advantage,
            'P1_Rolling_Win_Rate_L10': p1_win_rate,
            'P2_Rolling_Win_Rate_L10': p2_win_rate,
            'Win_Rate_L5_Advantage': win_rate_advantage_l5,
            'Close_Set_Win_Rate_Advantage': close_set_win_rate_advantage,
            'Time_Since_Last_Advantage': time_since_last_advantage,
            'Matches_Last_24H_Advantage': matches_last_24h_advantage,
            'Is_First_Match_Advantage': is_first_match_advantage,
            'H2H_P1_Win_Rate': h2h_p1_win_rate,
            'H2H_Dominance_Score': h2h_dominance_score,
            'H2H_Matches': h2h_matches_count,
            'P1_Rolling_Set_Comebacks_L20': p1_rolling_comebacks,
            'P2_Rolling_Set_Comebacks_L20': p2_rolling_comebacks,
            # Glicko-2 features
            'P1_Glicko_Mu': p1_mu,
            'P2_Glicko_Mu': p2_mu,
            'Glicko_Mu_Diff': p1_mu - p2_mu,
            'P1_Glicko_Phi': p1_phi,
            'P2_Glicko_Phi': p2_phi,
            'Glicko_Phi_Sum': p1_phi + p2_phi,
            'P1_Glicko_Sigma': p1_sigma,
            'P2_Glicko_Sigma': p2_sigma,
            # Clutch Factor features (deuce sets: 10-10+)
            'P1_Clutch_Factor_L20': p1_clutch_factor,
            'P2_Clutch_Factor_L20': p2_clutch_factor,
            'Clutch_Factor_Diff': clutch_factor_diff,
            'P1_Deuce_Sets_L20': p1_deuce_sets,
            'P2_Deuce_Sets_L20': p2_deuce_sets,
            # Pythagorean Expectation features
            'P1_Pythagorean_Win_Rate_L20': p1_pythagorean,
            'P2_Pythagorean_Win_Rate_L20': p2_pythagorean,
            'P1_Pythagorean_Delta_L20': p1_pythagorean_delta,
            'P2_Pythagorean_Delta_L20': p2_pythagorean_delta,
            'Pythagorean_Delta_Diff': pythagorean_delta_diff,
            # Exponential Fatigue features (replaces Daily_Fatigue_Advantage)
            'P1_Fatigue_Factor': p1_fatigue_factor,
            'P2_Fatigue_Factor': p2_fatigue_factor,
            'Fatigue_Factor_Diff': fatigue_factor_diff,
            # PDR Variance features
            'P1_PDR_Variance_L20': p1_pdr_variance,
            'P2_PDR_Variance_L20': p2_pdr_variance,
            'PDR_Variance_Diff': pdr_variance_diff,
        })
        engineered_rows.append(new_row)

    # --- 3. Save the Final Dataset ---
    final_df = pd.DataFrame(engineered_rows)
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[SUCCESS] Symmetrical feature engineering complete. Data saved to '{OUTPUT_FILE}'")

    # --- 4. Generate Feature Statistics ---
    print("\n--- Generating Feature Statistics ---")
    feature_cols = [
        # Elo features
        'P1_Elo', 'P2_Elo', 'Elo_Diff',
        # Glicko-2 features
        'P1_Glicko_Mu', 'P2_Glicko_Mu', 'Glicko_Mu_Diff',
        'P1_Glicko_Phi', 'P2_Glicko_Phi', 'Glicko_Phi_Sum',
        'P1_Glicko_Sigma', 'P2_Glicko_Sigma',
        # Clutch Factor features
        'P1_Clutch_Factor_L20', 'P2_Clutch_Factor_L20', 'Clutch_Factor_Diff',
        'P1_Deuce_Sets_L20', 'P2_Deuce_Sets_L20',
        # Pythagorean features
        'P1_Pythagorean_Win_Rate_L20', 'P2_Pythagorean_Win_Rate_L20',
        'P1_Pythagorean_Delta_L20', 'P2_Pythagorean_Delta_L20', 'Pythagorean_Delta_Diff',
        # Fatigue features
        'P1_Fatigue_Factor', 'P2_Fatigue_Factor', 'Fatigue_Factor_Diff',
        # PDR Variance features
        'P1_PDR_Variance_L20', 'P2_PDR_Variance_L20', 'PDR_Variance_Diff',
        # H2H features
        'H2H_Matches'
    ]
    # Filter to only existing columns
    existing_feature_cols = [col for col in feature_cols if col in final_df.columns]
    stats_df = final_df[existing_feature_cols].describe().T
    stats_df['non_null'] = final_df[existing_feature_cols].count()
    stats_df['null_count'] = final_df[existing_feature_cols].isnull().sum()
    stats_df.to_csv('feature_statistics_v7.4.csv')
    print(f"Feature statistics saved to 'feature_statistics_v7.4.csv'")
    print(f"Total new features added: {len(existing_feature_cols)}")

except FileNotFoundError:
    print(f"Error: The file '{RAW_STATS_FILE}' was not found. Please run the data collector first.")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()