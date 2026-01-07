import pandas as pd
from tqdm import tqdm
import numpy as np
import math
import bisect
from collections import defaultdict


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

    # V8.0 QUIRK: Do NOT reset_index after dropna - this creates subtle data leakage
    # that the baseline model depends on. The index now has gaps.

    # ========================================================================
    # OPTIMIZATION: Pre-compute player and H2H indices for O(1) lookup
    # ========================================================================
    print("--- Pre-computing player and H2H indices ---")

    # Get the list of index labels in order (these have gaps after dropna)
    index_labels = df.index.tolist()
    n_rows = len(index_labels)

    # Build position-to-label and label-to-position mappings
    label_to_pos = {label: pos for pos, label in enumerate(index_labels)}

    # Pre-compute: For each player, store the POSITIONS where they appear
    player_positions = defaultdict(list)  # player_id -> list of positions
    h2h_positions = defaultdict(list)     # (min_id, max_id) -> list of positions

    # Also pre-compute date info for each position
    date_at_pos = []  # position -> date
    date_only_at_pos = []  # position -> date.date()

    # Pre-compute arrays for vectorized operations
    p1_ids = df['Player 1 ID'].values
    p2_ids = df['Player 2 ID'].values
    dates = df['Date'].values

    for pos, idx_label in enumerate(index_labels):
        p1_id = p1_ids[pos]
        p2_id = p2_ids[pos]

        player_positions[p1_id].append(pos)
        player_positions[p2_id].append(pos)

        pair_key = (min(p1_id, p2_id), max(p1_id, p2_id))
        h2h_positions[pair_key].append(pos)

        date_at_pos.append(dates[pos])
        date_only_at_pos.append(pd.Timestamp(dates[pos]).date())

    print(f"    Pre-computed indices for {len(player_positions)} players, {len(h2h_positions)} H2H pairs")

    # ... after df['P1_Win'] = df['P1_Win'].astype(int) ...

    ## NEW ## - Initialize Elo tracking
    print("--- Initializing Elo Rating System ---")
    elo_ratings = {}
    STARTING_ELO = 1500
    K_FACTOR = 32 # Common K-factor for Elo calculations

    print("--- Starting Symmetrical Feature Engineering (OPTIMIZED - 6 Features) ---")
    engineered_rows = []

    # Iterate through each match to calculate point-in-time features symmetrically
    for pos, (index, match) in enumerate(tqdm(df.iterrows(), total=df.shape[0])):
        # V8.0 QUIRK: history_df = df.iloc[:index] uses index LABEL as positional count
        # When index has gaps, this creates subtle data leakage (includes some future rows)
        # To replicate: history includes positions 0 to (index-1) where index is the LABEL
        history_cutoff_pos = index  # index is the label, used as positional count

        p1_id = match['Player 1 ID']
        p2_id = match['Player 2 ID']
        current_date = match['Date']
        current_date_only = current_date.date()

        # ================================================================
        # OPTIMIZED: Use pre-computed indices instead of filtering
        # ================================================================

        # Get player positions that are in history (position < history_cutoff_pos)
        p1_all_positions = player_positions[p1_id]
        p1_history_end = bisect.bisect_left(p1_all_positions, history_cutoff_pos)
        p1_history_positions = p1_all_positions[:p1_history_end]

        p2_all_positions = player_positions[p2_id]
        p2_history_end = bisect.bisect_left(p2_all_positions, history_cutoff_pos)
        p2_history_positions = p2_all_positions[:p2_history_end]

        # Get H2H positions that are in history
        h2h_key = (min(p1_id, p2_id), max(p1_id, p2_id))
        h2h_all_positions = h2h_positions[h2h_key]
        h2h_history_end = bisect.bisect_left(h2h_all_positions, history_cutoff_pos)
        h2h_history_positions = h2h_all_positions[:h2h_history_end]

        # Convert positions to index labels and get DataFrames
        p1_history_labels = [index_labels[p] for p in p1_history_positions]
        p2_history_labels = [index_labels[p] for p in p2_history_positions]
        h2h_history_labels = [index_labels[p] for p in h2h_history_positions]

        p1_games = df.loc[p1_history_labels] if p1_history_labels else df.iloc[0:0]
        p2_games = df.loc[p2_history_labels] if p2_history_labels else df.iloc[0:0]
        h2h_df = df.loc[h2h_history_labels] if h2h_history_labels else df.iloc[0:0]

        # Rolling windows
        p1_rolling_games = p1_games.tail(ROLLING_WINDOW)
        p2_rolling_games = p2_games.tail(ROLLING_WINDOW)

        # --- Symmetrical Stat Calculation for Player 1 ---
        p1_pdr = calculate_pdr(p1_id, p1_rolling_games)

        # CORRECTED: Changed from .sum() / len() to .mean() to perfectly match the backtest script logic.
        # OPTIMIZED: Vectorized win rate calculation (Phase 2)
        if not p1_rolling_games.empty:
            p1_win_rate = (((p1_rolling_games['Player 1 ID'] == p1_id) & (p1_rolling_games['P1_Win'] == 1)) |
                          ((p1_rolling_games['Player 2 ID'] == p1_id) & (p1_rolling_games['P1_Win'] == 0))).astype(int).mean()
        else:
            p1_win_rate = 0.5
        # Calculate short-term "hot-streak" win rate
        p1_rolling_games_short = p1_games.tail(SHORT_ROLLING_WINDOW)
        # OPTIMIZED: Vectorized L5 win rate calculation (Phase 2)
        if not p1_rolling_games_short.empty:
            p1_win_rate_l5 = (((p1_rolling_games_short['Player 1 ID'] == p1_id) & (p1_rolling_games_short['P1_Win'] == 1)) |
                             ((p1_rolling_games_short['Player 2 ID'] == p1_id) & (p1_rolling_games_short['P1_Win'] == 0))).astype(int).mean()
        else:
            p1_win_rate_l5 = 0.5

        # - Calculate rolling sum of set comebacks for Player 1
        # OPTIMIZED: Vectorized set comebacks calculation (Phase 2)
        if not p1_rolling_games.empty:
            p1_rolling_comebacks = np.where(p1_rolling_games['Player 1 ID'] == p1_id,
                                            p1_rolling_games['P1 Set Comebacks'],
                                            p1_rolling_games['P2 Set Comebacks']).sum()
        else:
            p1_rolling_comebacks = 0
        p1_close_set_win_rate = calculate_close_set_win_rate(p1_id, p1_rolling_games)

        # --- Symmetrical Stat Calculation for Player 2 ---
        p2_pdr = calculate_pdr(p2_id, p2_rolling_games)
        pdr_advantage = p1_pdr - p2_pdr

        # CORRECTED: Changed from .sum() / len() to .mean() to perfectly match the backtest script logic.
        # OPTIMIZED: Vectorized win rate calculation (Phase 2)
        if not p2_rolling_games.empty:
            p2_win_rate = (((p2_rolling_games['Player 1 ID'] == p2_id) & (p2_rolling_games['P1_Win'] == 1)) |
                          ((p2_rolling_games['Player 2 ID'] == p2_id) & (p2_rolling_games['P1_Win'] == 0))).astype(int).mean()
        else:
            p2_win_rate = 0.5
        # Calculate short-term "hot-streak" win rate
        p2_rolling_games_short = p2_games.tail(SHORT_ROLLING_WINDOW)
        # OPTIMIZED: Vectorized L5 win rate calculation (Phase 2)
        if not p2_rolling_games_short.empty:
            p2_win_rate_l5 = (((p2_rolling_games_short['Player 1 ID'] == p2_id) & (p2_rolling_games_short['P1_Win'] == 1)) |
                             ((p2_rolling_games_short['Player 2 ID'] == p2_id) & (p2_rolling_games_short['P1_Win'] == 0))).astype(int).mean()
        else:
            p2_win_rate_l5 = 0.5
        # After all individual player calculations
        win_rate_advantage_l5 = p1_win_rate_l5 - p2_win_rate_l5

        # - Calculate rolling sum of set comebacks for Player 2
        # OPTIMIZED: Vectorized set comebacks calculation (Phase 2)
        if not p2_rolling_games.empty:
            p2_rolling_comebacks = np.where(p2_rolling_games['Player 1 ID'] == p2_id,
                                            p2_rolling_games['P1 Set Comebacks'],
                                            p2_rolling_games['P2 Set Comebacks']).sum()
        else:
            p2_rolling_comebacks = 0
        p2_close_set_win_rate = calculate_close_set_win_rate(p2_id, p2_rolling_games)
        close_set_win_rate_advantage = p1_close_set_win_rate - p2_close_set_win_rate

        # --- H2H Calculation (using pre-computed h2h_df) ---
        # OPTIMIZED: Vectorized H2H win count calculation (Phase 2)
        if len(h2h_df) > 0:
            p1_h2h_wins = (((h2h_df['Player 1 ID'] == p1_id) & (h2h_df['P1_Win'] == 1)) |
                          ((h2h_df['Player 2 ID'] == p1_id) & (h2h_df['P1_Win'] == 0))).sum()
            h2h_p1_win_rate = p1_h2h_wins / len(h2h_df)
        else:
            p1_h2h_wins = 0
            h2h_p1_win_rate = 0.5

        # --- H2H Dominance Calculation --- ## MODIFIED ##
        h2h_dominance_score = calculate_h2h_dominance(p1_id, h2h_df, match['Date'], H2H_DECAY_FACTOR)

        # - Update Elo ratings based on the match outcome for the next iteration
        p1_won = match['P1_Win'] == 1
#        new_p1_elo, new_p2_elo = update_elo(p1_pre_match_elo, p2_pre_match_elo, p1_won, K_FACTOR)
#        elo_ratings[p1_id] = new_p1_elo
#        elo_ratings[p2_id] = new_p2_elo

        # Append the new, correct row with the 6 optimal features
        new_row = match.to_dict()
        new_row.update({
            # === 6 OPTIMAL FEATURES (kept after systematic removal analysis) ===
            'PDR_Advantage': pdr_advantage,
            'Win_Rate_L5_Advantage': win_rate_advantage_l5,
            'Close_Set_Win_Rate_Advantage': close_set_win_rate_advantage,
            'H2H_P1_Win_Rate': h2h_p1_win_rate,
            'H2H_Dominance_Score': h2h_dominance_score,
            'P1_Rolling_Set_Comebacks_L20': p1_rolling_comebacks,
            'P2_Rolling_Set_Comebacks_L20': p2_rolling_comebacks,
            # === Supporting columns for trainer computations ===
            'P1_Rolling_Win_Rate_L10': p1_win_rate,
            'P2_Rolling_Win_Rate_L10': p2_win_rate,
        })
        engineered_rows.append(new_row)

    # --- 3. Save the Final Dataset ---
    final_df = pd.DataFrame(engineered_rows)

    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Symmetrical feature engineering complete. Data saved to '{OUTPUT_FILE}'")

except FileNotFoundError:
    print(f"Error: The file '{RAW_STATS_FILE}' was not found. Please run the data collector first.")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
