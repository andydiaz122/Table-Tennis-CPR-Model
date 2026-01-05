import pandas as pd
import numpy as np

# --- 1. Configuration ---
INPUT_FILE = "czech_liga_pro_advanced_stats_FIXED.csv"
OUTPUT_FILE = "final_engineered_features.csv"
ROLLING_WINDOW = 10 # Look at the last 10 matches for player form

# --- 2. Robust Winner Parsing Function ---
def get_winner(score_str):
    try:
        cleaned_score = str(score_str).strip('="')
        p1_score, p2_score = map(int, cleaned_score.split('-'))
        return 1 if p1_score > p2_score else 0
    except (ValueError, IndexError):
        return np.nan

# --- 3. Main Script Logic ---
try:
    print(f"Loading full dataset from '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)

    # --- Initial Data Cleaning and Sorting ---
    df.dropna(subset=['Date', 'Player 1 ID', 'Player 2 ID'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True, ascending=True)
    df.reset_index(drop=True, inplace=True)
    
    df['P1_Win'] = df['Final Score'].apply(get_winner)
    df.dropna(subset=['P1_Win'], inplace=True)
    df['P1_Win'] = df['P1_Win'].astype(int)
    print(f"Loaded, sorted, and cleaned {len(df)} matches.")

    # --- 4. Create Player-Centric Data for Rolling Averages ---
    print("Creating player-centric data for rolling average calculation...")
    p1_data = df[['Match ID', 'Date', 'Player 1 ID', 'P1 Pressure Points', 'P1_Win']].rename(columns={
        'Player 1 ID': 'Player_ID', 'P1 Pressure Points': 'Pressure_Points', 'P1_Win': 'Win'
    })
    p2_data = df[['Match ID', 'Date', 'Player 2 ID', 'P2 Pressure Points', 'P1_Win']].rename(columns={
        'Player 2 ID': 'Player_ID', 'P2 Pressure Points': 'Pressure_Points'
    })
    p2_data['Win'] = 1 - p2_data['P1_Win']
    
    player_df = pd.concat([p1_data, p2_data]).sort_values(by='Date')
    
    grouped = player_df.groupby('Player_ID')
    player_df[f'Rolling_Win_Rate_L{ROLLING_WINDOW}'] = grouped['Win'].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    player_df[f'Rolling_Pressure_Points_L{ROLLING_WINDOW}'] = grouped['Pressure_Points'].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    
    # --- 5. Calculate H2H and Rest Days ---
    print("Calculating Head-to-Head stats and Rest Days...")
    h2h_records, last_match_date = {}, {}
    new_features_list = []
    
    for index, row in df.iterrows():
        p1_id, p2_id, current_date = row['Player 1 ID'], row['Player 2 ID'], row['Date']
        
        h2h_key = tuple(sorted((p1_id, p2_id)))
        p1_h2h_wins, p2_h2h_wins = h2h_records.get(h2h_key, [0, 0])
        h2h_total = p1_h2h_wins + p2_h2h_wins
        if h2h_total > 0:
            h2h_p1_win_rate = p1_h2h_wins / h2h_total if p1_id < p2_id else p2_h2h_wins / h2h_total
        else:
            h2h_p1_win_rate = 0.5

        p1_rest = (current_date - last_match_date.get(p1_id, current_date)).days
        p2_rest = (current_date - last_match_date.get(p2_id, current_date)).days

        new_features_list.append({
            'Match ID': row['Match ID'], 'P1_Rest_Days': p1_rest, 'P2_Rest_Days': p2_rest, 'H2H_P1_Win_Rate': h2h_p1_win_rate
        })
        
        if row['P1_Win'] == 1:
            if p1_id < p2_id: h2h_records[h2h_key] = [p1_h2h_wins + 1, p2_h2h_wins]
            else: h2h_records[h2h_key] = [p1_h2h_wins, p2_h2h_wins + 1]
        else:
            if p1_id < p2_id: h2h_records[h2h_key] = [p1_h2h_wins, p2_h2h_wins + 1]
            else: h2h_records[h2h_key] = [p1_h2h_wins + 1, p2_h2h_wins]
        
        last_match_date[p1_id], last_match_date[p2_id] = current_date, current_date

    # --- 6. Merge All Features Together ---
    print("Merging all engineered features...")
    new_features_df = pd.DataFrame(new_features_list)
    
    # --- FINAL CORRECTED MERGE LOGIC ---
    # Merge Player 1's rolling stats
    df = pd.merge(df, player_df[['Match ID', 'Player_ID', f'Rolling_Win_Rate_L{ROLLING_WINDOW}', f'Rolling_Pressure_Points_L{ROLLING_WINDOW}']],
                  left_on=['Match ID', 'Player 1 ID'], right_on=['Match ID', 'Player_ID'], how='left').rename(columns={
                      f'Rolling_Win_Rate_L{ROLLING_WINDOW}': f'P1_Rolling_Win_Rate_L{ROLLING_WINDOW}',
                      f'Rolling_Pressure_Points_L{ROLLING_WINDOW}': f'P1_Rolling_Pressure_Points_L{ROLLING_WINDOW}'
                  }).drop('Player_ID', axis=1)

    # Merge Player 2's rolling stats
    df = pd.merge(df, player_df[['Match ID', 'Player_ID', f'Rolling_Win_Rate_L{ROLLING_WINDOW}', f'Rolling_Pressure_Points_L{ROLLING_WINDOW}']],
                  left_on=['Match ID', 'Player 2 ID'], right_on=['Match ID', 'Player_ID'], how='left').rename(columns={
                      f'Rolling_Win_Rate_L{ROLLING_WINDOW}': f'P2_Rolling_Win_Rate_L{ROLLING_WINDOW}',
                      f'Rolling_Pressure_Points_L{ROLLING_WINDOW}': f'P2_Rolling_Pressure_Points_L{ROLLING_WINDOW}'
                  }).drop('Player_ID', axis=1)

    # Merge H2H and Rest Day stats
    df = pd.merge(df, new_features_df, on='Match ID', how='left')

    # --- 7. Save Final Dataset ---
    df.dropna(inplace=True)
    
    print(f"Final number of matches in the engineered dataset: {len(df)}")
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ Advanced feature engineering complete. Final dataset saved to '{OUTPUT_FILE}'")

except Exception as e:
    print(f"An error occurred: {e}")