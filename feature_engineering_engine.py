import pandas as pd
import numpy as np

print("Starting Feature Engineering for CPR Model v6.1...")

# --- Configuration ---
RAW_DATA_FILE = 'czech_liga_pro_advanced_stats_FIXED.csv'
OUTPUT_FILE = 'full_feature_database_v6.1.csv'
ROLLING_WINDOW = 10 # For TW-BPR and other rolling stats

# --- Load Data ---
try:
    df = pd.read_csv(RAW_DATA_FILE)
    print(f"Successfully loaded {len(df)} matches from {RAW_DATA_FILE}")
except FileNotFoundError:
    print(f"FATAL ERROR: The data file '{RAW_DATA_FILE}' was not found.")
    exit()

# --- Data Cleaning and Preparation ---
df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.sort_values(by='Date', inplace=True)
df.dropna(subset=['Final Score', 'Set Scores'], inplace=True)

# Extract set wins for each player
def parse_scores(row):
    try:
        p1_sets = int(row['Final Score'].split('-')[0].replace('"', '').replace('=', ''))
        p2_sets = int(row['Final Score'].split('-')[1].replace('"', '').replace('=', ''))
        return p1_sets, p2_sets
    except:
        return np.nan, np.nan

df['P1_Sets'], df['P2_Sets'] = zip(*df.apply(parse_scores, axis=1))
df.dropna(subset=['P1_Sets', 'P2_Sets'], inplace=True)

# Create a long-format DataFrame (one row per player per match)
p1_df = df[['Date', 'Player 1', 'Player 2', 'P1_Sets', 'P2_Sets', 'Set Scores']].rename(columns={'Player 1': 'Player', 'Player 2': 'Opponent', 'P1_Sets': 'Sets_Won', 'P2_Sets': 'Sets_Lost'})
p1_df['Win'] = (p1_df['Sets_Won'] > p1_df['Sets_Lost']).astype(int)

p2_df = df[['Date', 'Player 2', 'Player 1', 'P2_Sets', 'P1_Sets', 'Set Scores']].rename(columns={'Player 2': 'Player', 'Player 1': 'Opponent', 'P2_Sets': 'Sets_Won', 'P1_Sets': 'Sets_Lost'})
p2_df['Win'] = (p2_df['Sets_Won'] > p2_df['Sets_Lost']).astype(int)

player_match_data = pd.concat([p1_df, p2_df]).sort_values(by='Date').reset_index(drop=True)

# --- Feature Engineering Functions ---

def calculate_time_weighted_bpr(series):
    # Exponentially Weighted Moving Average for Win Percentage
    ewma_win_pct = series['Win'].ewm(span=ROLLING_WINDOW, adjust=False).mean().iloc[-1]
    
    # Calculate Set Win % for the rolling window
    window_sets_won = series['Sets_Won'].sum()
    window_sets_lost = series['Sets_Lost'].sum()
    total_sets = window_sets_won + window_sets_lost
    set_win_pct = window_sets_won / total_sets if total_sets > 0 else 0
    
    # Final TW-BPR Formula
    return (ewma_win_pct * 0.4) + (set_win_pct * 0.6) * 100

def parse_set_scores_narrative(set_scores_str, player_is_p1):
    try:
        sets = set_scores_str.replace('"', '').split(', ')
        lost_first_set = False
        was_up_2_0 = False
        
        p1_set_wins = 0
        p2_set_wins = 0

        # Check first set
        s1_p1, s1_p2 = map(int, sets[0].split('-'))
        if (player_is_p1 and s1_p1 < s1_p2) or (not player_is_p1 and s1_p1 > s1_p2):
            lost_first_set = True

        # Check 2-0 lead
        for i, s in enumerate(sets):
            if i < 2: # Check after 2 sets
                p1, p2 = map(int, s.split('-'))
                if p1 > p2: p1_set_wins += 1
                else: p2_set_wins += 1
        
        if (player_is_p1 and p1_set_wins == 2) or (not player_is_p1 and p2_set_wins == 2):
            was_up_2_0 = True
            
        return lost_first_set, was_up_2_0
    except:
        return None, None

        
# --- Main Processing Loop ---
all_players = pd.concat([df['Player 1'], df['Player 2']]).unique()
player_features_list = []

for player_name in all_players:
    player_df = player_match_data[player_match_data['Player'] == player_name]
    if len(player_df) < ROLLING_WINDOW:
        continue

    # 1. Calculate the EWMA Win Percentage directly on the 'Win' series.
    # This is more efficient and less prone to errors.
    ewma_win_pct = player_df['Win'].ewm(span=ROLLING_WINDOW, adjust=False).mean().iloc[-1]

    # 2. Calculate the rolling Set Win Percentage using rolling sums.
    rolling_sets_won = player_df['Sets_Won'].rolling(window=ROLLING_WINDOW).sum()
    rolling_sets_lost = player_df['Sets_Lost'].rolling(window=ROLLING_WINDOW).sum()
    rolling_total_sets = rolling_sets_won + rolling_sets_lost
    rolling_set_win_pct = rolling_sets_won / rolling_total_sets
    set_win_pct = rolling_set_win_pct.iloc[-1]
    
    # 3. Combine them using your formula
    final_tw_bpr = (ewma_win_pct * 0.4) + (set_win_pct * 0.6) * 100

    # ... (rest of the code for other features)

    # Calculate narrative features (Comeback and Finisher scores)
    comeback_opportunities = 0
    comeback_wins = 0
    finisher_opportunities = 0
    finisher_wins = 0

    for idx, row in player_df.iterrows():
        is_p1 = (df.loc[df['Date'] == row['Date'], 'Player 1'] == player_name).any()
        lost_first, up_2_0 = parse_set_scores_narrative(row['Set Scores'], is_p1)
        
        if lost_first is not None and lost_first:
            comeback_opportunities += 1
            if row['Win'] == 1:
                comeback_wins += 1
        
        if up_2_0 is not None and up_2_0:
            finisher_opportunities += 1
            if row['Win'] == 1:
                finisher_wins += 1

    comeback_score = (comeback_wins / comeback_opportunities) * 100 if comeback_opportunities > 5 else 50.0
    finisher_score = (finisher_wins / finisher_opportunities) * 100 if finisher_opportunities > 5 else 90.0

    player_features_list.append({
        'Player': player_name,
        'TW_BPR': final_tw_bpr,
        'Comeback_Score': comeback_score,
        'Finisher_Score': finisher_score,
        # Placeholder for other features calculated similarly (SDR, PPI, etc.)
        # In a full implementation, these would be calculated with rolling windows too.
        'SDR': np.random.uniform(2.0, 6.0),
        'PPI': np.random.uniform(-5.0, 5.0),
        'URS': np.random.uniform(20.0, 80.0),
        'SoS': np.random.uniform(50.0, 75.0),
        'VCI': np.random.uniform(2.0, 5.0),
        'Fatigue': 1 # Default fatigue for profile
    })

# --- Save to CSV ---
feature_df = pd.DataFrame(player_features_list)
feature_df.to_csv(OUTPUT_FILE, index=False)

print(f"Feature engineering complete. Saved {len(feature_df)} player profiles to {OUTPUT_FILE}")