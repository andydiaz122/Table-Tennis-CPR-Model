# FILE: 2_prepare_test_matches.py (Final Diagnostic Version)
import pandas as pd

# --- Configuration ---
START_DATE = '2025-09-17'
END_DATE = '2025-09-18'
FULL_DATASET_FILE = "final_dataset_v7.4.csv"

# --- Helper function ---
def decimal_to_moneyline(decimal_odds):
    if pd.isna(decimal_odds) or decimal_odds <= 1:
        return 'None'
    if decimal_odds >= 2.0:
        return f"{int((decimal_odds - 1) * 100)}"
    else:
        return f"{int(-100 / (decimal_odds - 1))}"

# --- Main Logic ---
try:
    print(f"--- Preparing Matches for Paper-Trading from {START_DATE} to {END_DATE} ---")
    df = pd.read_csv(FULL_DATASET_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Isolate the matches for the paper-trading period first
    test_matches_df = df[(df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)].copy()

    # --- FINAL DIAGNOSTIC: Data Quality Report ---
    print("\n--- 🕵️ Data Quality Report on Kickoff Odds ---")
    p1_odds_raw = test_matches_df['Kickoff_P1_Odds']
    p2_odds_raw = test_matches_df['Kickoff_P2_Odds']
    
    # We will test the conversion directly without assuming the cause.
    p1_odds_converted = pd.to_numeric(p1_odds_raw, errors='coerce')
    p2_odds_converted = pd.to_numeric(p2_odds_raw, errors='coerce')

    total_odds_pairs = len(test_matches_df)
    invalid_p1_count = p1_odds_converted.isna().sum()
    invalid_p2_count = p2_odds_converted.isna().sum()

    print(f"Found {total_odds_pairs} matches in the date range.")
    print(f"Player 1 Odds: {total_odds_pairs - invalid_p1_count} valid, {invalid_p1_count} invalid (could not be converted to a number).")
    print(f"Player 2 Odds: {total_odds_pairs - invalid_p2_count} valid, {invalid_p2_count} invalid (could not be converted to a number).")
    print("---------------------------------------------")
    
    # Overwrite the original columns with the cleaned versions for the rest of the script
    test_matches_df['Kickoff_P1_Odds'] = p1_odds_converted
    test_matches_df['Kickoff_P2_Odds'] = p2_odds_converted

    print("\n✅ Copy and paste the matches below into your 'upcoming_matches' list:\n")
    
    for date, matches_on_day in test_matches_df.groupby(test_matches_df['Date'].dt.date):
        print(f"\n# --- Matches for {date} ---")
        for _, row in matches_on_day.iterrows():
            # FIXED a typo here: 'Kickoff_P1_ Odds' -> 'Kickoff_P1_Odds'
            p1_ml = decimal_to_moneyline(row['Kickoff_P1_Odds'])
            p2_ml = decimal_to_moneyline(row['Kickoff_P2_Odds'])
            
            p1_odds_str = f"{row['Kickoff_P1_Odds']:.2f}" if pd.notna(row['Kickoff_P1_Odds']) else "N/A"
            p2_odds_str = f"{row['Kickoff_P2_Odds']:.2f}" if pd.notna(row['Kickoff_P2_Odds']) else "N/A"
            winner = f"Winner: {'P1' if row['P1_Win']==1 else 'P2'}"

            print("{")
            print(f"    'Match ID': {row['Match ID']},")
            print(f"    'Player 1 ID': {row['Player 1 ID']}, 'Player 1 Name': '{row['Player 1']}',")
            print(f"    'Player 2 ID': {row['Player 2 ID']}, 'Player 2 Name': '{row['Player 2']}',")
            print(f"    'P1_ML': {p1_ml}, # Actual Decimal: {p1_odds_str}, {winner}")
            print(f"    'P2_ML': {p2_ml}  # Actual Decimal: {p2_odds_str}")
            print("},")

except FileNotFoundError:
    print(f"Error: The file '{FULL_DATASET_FILE}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")