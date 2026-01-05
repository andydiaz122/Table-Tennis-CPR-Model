import pandas as pd
from tqdm import tqdm
import numpy as np

# - General function to get P1's vig-free probability
def get_vig_free_prob(p1_odds, p2_odds):
    if pd.isna(p1_odds) or pd.isna(p2_odds) or p1_odds <= 1 or p2_odds <= 1:
        return np.nan
    raw_p1_prob = 1 / p1_odds
    raw_p2_prob = 1 / p2_odds
    market_overround = raw_p1_prob + raw_p2_prob
    if market_overround == 0:
        return np.nan
    return raw_p1_prob / market_overround

# --- 1. Configuration ---
# RAW_STATS_FILE = "czech_liga_pro_advanced_stats_FIXED.csv"
RAW_STATS_FILE = "czech_liga_pro_advanced_stats_2024.csv"
RAW_ODDS_FILE = "historical_odds_v7.0.csv"
# OUTPUT_FILE = "final_engineered_features_v7.4.csv" # New, corrected output file
OUTPUT_FILE = "final_dataset_v7.4.csv"
ROLLING_WINDOW = 20



# --- 2. Main Logic ---
try:
    print(f"--- Loading Raw Data from '{RAW_STATS_FILE}' ---")
    df = pd.read_csv(RAW_STATS_FILE)

# --- Load and Prepare Opening and Kickoff Odds Data ---
    print(f"--- Loading and Preparing Data from '{RAW_ODDS_FILE}' ---")
    odds_df = pd.read_csv(RAW_ODDS_FILE)
    odds_df = odds_df[odds_df['Market_Name'] == 'Match Winner'].copy()
    odds_df.sort_values(by=['Match_ID', 'Odds_Timestamp'], inplace=True)

    opening_odds_df = odds_df.groupby('Match_ID').nth(0).reset_index()
    opening_odds_df = opening_odds_df[['Match_ID', 'P1_Odds', 'P2_Odds']]
    opening_odds_df.rename(columns={'P1_Odds': 'P1_Opening_Odds', 'P2_Odds': 'P2_Opening_Odds'}, inplace=True)
    
    kickoff_odds_df = odds_df.groupby('Match_ID').nth(1).reset_index()
    kickoff_odds_df = kickoff_odds_df[['Match_ID', 'P1_Odds', 'P2_Odds']]
    kickoff_odds_df.rename(columns={'P1_Odds': 'P1_Kickoff_Odds', 'P2_Odds': 'P2_Kickoff_Odds'}, inplace=True)

    # --- Merge Both Sets of Odds with Main Stats Data ---
    df.rename(columns={'Match ID': 'Match_ID'}, inplace=True)
    df['Match_ID'] = df['Match_ID'].astype(int)
    
    df = pd.merge(df, opening_odds_df, on='Match_ID', how='inner')
    df = pd.merge(df, kickoff_odds_df, on='Match_ID', how='inner')
    print(f"--- Data merged. Total matches with stats and both odds: {len(df)} ---")

    # --- Data Cleaning and Preparation ---
    # Convert all odds columns to numeric, coercing errors
    for col in ['P1_Opening_Odds', 'P2_Opening_Odds', 'P1_Kickoff_Odds', 'P2_Kickoff_Odds']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['P1_Opening_Odds', 'P2_Opening_Odds', 'P1_Kickoff_Odds', 'P2_Kickoff_Odds'], inplace=True) # Drop rows where odds are invalid

    df['Date'] = pd.to_datetime(df['Date'])
#    df.sort_values(by='Date', inplace=True)
    # Sort by date first, then by Match_ID to ensure chronological order for the same day
    df.sort_values(by=['Date', 'Match_ID'], inplace=True)
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

    print("--- Starting Symmetrical Feature Engineering (this may take a few minutes) ---")
    
    engineered_rows = []

    # Iterate through each match to calculate point-in-time features symmetrically
    for index, match in tqdm(df.iterrows(), total=df.shape[0]):
        history_df = df.iloc[:index]

        p1_id = match['Player 1 ID']
        p2_id = match['Player 2 ID']

        # --- Market Odds Calculation ---
        # - Calculate line movement
        opening_prob_p1 = get_vig_free_prob(match['P1_Opening_Odds'], match['P2_Opening_Odds'])
        kickoff_prob_p1 = get_vig_free_prob(match['P1_Kickoff_Odds'], match['P2_Kickoff_Odds'])
        if pd.isna(opening_prob_p1) or pd.isna(kickoff_prob_p1):
            line_movement_p1 = 0.0
        else:
            line_movement_p1 = kickoff_prob_p1 - opening_prob_p1
            
        
        # --- Symmetrical Stat Calculation for Player 1 ---
        p1_games = history_df[(history_df['Player 1 ID'] == p1_id) | (history_df['Player 2 ID'] == p1_id)]
        p1_rolling_games = p1_games.tail(ROLLING_WINDOW)
        
        # CORRECTED: Changed from .sum() / len() to .mean() to perfectly match the backtest script logic.
        p1_win_rate = p1_rolling_games.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or \
                                             (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).mean() \
                                             if not p1_rolling_games.empty else 0.5
        p1_pressure_points = 0.0
        if not p1_rolling_games.empty:
            p1_pressure_points = p1_rolling_games.apply(lambda r: r['P1 Pressure Points'] if r['Player 1 ID'] == p1_id else r['P2 Pressure Points'], axis=1).mean()
        
        # - Calculate rolling sum of set comebacks for Player 1
        p1_rolling_comebacks = p1_rolling_games.apply(lambda r: r['P1 Set Comebacks'] if r['Player 1 ID'] == p1_id else r['P2 Set Comebacks'], axis=1).sum() \
                                                if not p1_rolling_games.empty else 0

        p1_last_game_date = p1_games['Date'].max()
        p1_rest_days = (match['Date'] - p1_last_game_date).days if pd.notna(p1_last_game_date) else 30

        # --- Symmetrical Stat Calculation for Player 2 ---
        p2_games = history_df[(history_df['Player 1 ID'] == p2_id) | (history_df['Player 2 ID'] == p2_id)]
        p2_rolling_games = p2_games.tail(ROLLING_WINDOW)

        # CORRECTED: Changed from .sum() / len() to .mean() to perfectly match the backtest script logic.
        p2_win_rate = p2_rolling_games.apply(lambda r: 1 if (r['Player 1 ID'] == p2_id and r['P1_Win'] == 1) or \
                                             (r['Player 2 ID'] == p2_id and r['P1_Win'] == 0) else 0, axis=1).mean() \
                                             if not p2_rolling_games.empty else 0.5 
        p2_pressure_points = 0.0
        if not p2_rolling_games.empty:
            p2_pressure_points = p2_rolling_games.apply(lambda r: r['P1 Pressure Points'] if r['Player 1 ID'] == p2_id else r['P2 Pressure Points'], axis=1).mean()

        # - Calculate rolling sum of set comebacks for Player 2
        p2_rolling_comebacks = p2_rolling_games.apply(lambda r: r['P1 Set Comebacks'] if r['Player 1 ID'] == p2_id else r['P2 Set Comebacks'], axis=1).sum() \
                                                if not p2_rolling_games.empty else 0

        p2_last_game_date = p2_games['Date'].max()
        p2_rest_days = (match['Date'] - p2_last_game_date).days if pd.notna(p2_last_game_date) else 30
        
        # --- H2H Calculation ---
        h2h_df = history_df[((history_df['Player 1 ID'] == p1_id) & (history_df['Player 2 ID'] == p2_id)) | ((history_df['Player 1 ID'] == p2_id) & (history_df['Player 2 ID'] == p1_id))]
        p1_h2h_wins = h2h_df.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).sum()
        h2h_p1_win_rate = p1_h2h_wins / len(h2h_df) if len(h2h_df) > 0 else 0.5
        
        # Append the new, correct row
        new_row = match.to_dict()
        new_row.update({
            'Line_Movement_P1': line_movement_p1,
            'P1_Rolling_Win_Rate_L10': p1_win_rate,
            'P1_Rolling_Pressure_Points_L10': p1_pressure_points,
            'P2_Rolling_Win_Rate_L10': p2_win_rate,
            'P2_Rolling_Pressure_Points_L10': p2_pressure_points,
            'P1_Rest_Days': p1_rest_days,
            'P2_Rest_Days': p2_rest_days,
            'H2H_P1_Win_Rate': h2h_p1_win_rate,
            'P1_Rolling_Set_Comebacks_L20': p1_rolling_comebacks, ## NEW ##
            'P2_Rolling_Set_Comebacks_L20': p2_rolling_comebacks  ## NEW ##
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