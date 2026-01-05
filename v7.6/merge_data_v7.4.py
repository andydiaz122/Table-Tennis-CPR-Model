import pandas as pd
import os

# --- 1. Configuration ---
FEATURES_FILE = "final_engineered_features_v7.4.csv"
ODDS_FILE = "historical_odds_v7.0.csv"
OUTPUT_FILE = "final_dataset_v7.4.csv"

# --- 2. Main Script Logic ---
try:
    # Load both datasets
    print(f"Loading features from '{FEATURES_FILE}'...")
    df_features = pd.read_csv(FEATURES_FILE)
    print(f"Loaded {len(df_features)} matches with engineered features.")

    print(f"Loading odds from '{ODDS_FILE}'...")
    df_odds = pd.read_csv(ODDS_FILE)
    print(f"Loaded {len(df_odds)} historical odds records.")

    # --- THIS IS THE FIX ---
    # Standardize the 'Match ID' column name in both dataframes to prevent key errors.
    if 'Match_ID' in df_odds.columns:
        df_odds.rename(columns={'Match_ID': 'Match ID'}, inplace=True)
    
    # Create the timestamp for merging
    df_features['Match_Start_Time'] = pd.to_datetime(df_features['Date'] + ' ' + df_features['Time'])

    # --- 3. Isolate Kickoff Odds ---
    print("\nIsolating pre-match 'Match Winner' odds...")
    
    df_winner_odds = df_odds[df_odds['Market_Name'] == 'Match Winner'].copy()
    df_winner_odds['Odds_Timestamp'] = pd.to_datetime(df_winner_odds['Odds_Timestamp'])
    
    # Now the merge key 'Match ID' is consistent across both dataframes
    merged_odds = pd.merge(df_winner_odds, df_features[['Match ID', 'Match_Start_Time']], on='Match ID', how='left')
    
    merged_odds.dropna(subset=['Match_Start_Time'], inplace=True)
    prematch_odds = merged_odds[merged_odds['Odds_Timestamp'] <= merged_odds['Match_Start_Time']]
    
    # Find the index of the latest prematch odd for each match
    kickoff_indices = prematch_odds.groupby('Match ID')['Odds_Timestamp'].idxmax()
    kickoff_odds = prematch_odds.loc[kickoff_indices]
    
    # Select and rename the final columns we need
    kickoff_odds = kickoff_odds[['Match ID', 'P1_Odds', 'P2_Odds']].rename(columns={
        'P1_Odds': 'Kickoff_P1_Odds',
        'P2_Odds': 'Kickoff_P2_Odds'
    })
    print(f"Successfully isolated kickoff odds for {len(kickoff_odds)} matches.")

    # --- 4. Merge Kickoff Odds with Features ---
    print(f"\nMerging kickoff odds with the main feature dataset...")
    # This merge will now work correctly
    df_final = pd.merge(df_features, kickoff_odds, on='Match ID', how='left')
    
    df_final.drop(columns=['Match_Start_Time'], inplace=True)
    
    print(f"Final merged dataset contains {len(df_final)} matches.")

    # --- 5. Save the Final Dataset ---
    # Check if the output file already exists to decide whether to write headers
    file_exists = os.path.exists(OUTPUT_FILE)

    print(f"\nAppending data to '{OUTPUT_FILE}'...")

    df_final.to_csv(
        OUTPUT_FILE,
        mode='a',          # 'a' stands for append
        index=False,
        header=not file_exists # Only write header if the file does NOT exist
    )

    print(f"\n✅ Data successfully appended to the final dataset!")
    print(f" -> {OUTPUT_FILE}")

except FileNotFoundError as e:
    print("Error: An input file was not found. Please ensure both CSV files are in the same folder.")
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")    