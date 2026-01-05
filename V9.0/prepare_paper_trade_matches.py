import pandas as pd
import numpy as np

# --- 1. Configuration ---
HISTORICAL_DATA_FILE = "final_dataset_v7.1.csv"
START_DATE = '2025-09-18'
END_DATE = '2025-09-22'

# --- 2. Helper function to convert decimal to moneyline ---
def decimal_to_moneyline(decimal_odds):
    """Converts decimal odds to American Moneyline odds."""
    if pd.isna(decimal_odds) or decimal_odds <= 1:
        return 'None'
    
    if decimal_odds >= 2.0:
        return f"{int((decimal_odds - 1) * 100)}"
    else:
        return f"{int(-100 / (decimal_odds - 1))}"

# --- 3. Main Script Logic ---
try:
    print(f"--- Preparing Matches for Paper-Trading from {START_DATE} to {END_DATE} ---")
    
    # Load the complete, up-to-date dataset
    df = pd.read_csv(HISTORICAL_DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])

    # --- FIX: Convert odds columns to numeric, coercing errors (like '-') to NaN ---
    df['Kickoff_P1_Odds'] = pd.to_numeric(df['Kickoff_P1_Odds'], errors='coerce')
    df['Kickoff_P2_Odds'] = pd.to_numeric(df['Kickoff_P2_Odds'], errors='coerce')

    # Isolate the matches for the paper-trading period
    paper_trade_matches = df[(df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)].copy()

    print("\nCopy and paste the relevant day's matches into your predictor's 'upcoming_matches' list:\n")
    
    # Group by date to make it easy to copy day by day
    for date, matches_on_day in paper_trade_matches.groupby(df['Date'].dt.date):
        print(f"\n# --- Matches for {date} ---")
        for _, row in matches_on_day.iterrows():
            p1_ml = decimal_to_moneyline(row['Kickoff_P1_Odds'])
            p2_ml = decimal_to_moneyline(row['Kickoff_P2_Odds'])
            
            # --- FIX: Check for NaN before formatting to prevent errors ---
            p1_odds_str = f"{row['Kickoff_P1_Odds']:.2f}" if pd.notna(row['Kickoff_P1_Odds']) else "N/A"
            p2_odds_str = f"{row['Kickoff_P2_Odds']:.2f}" if pd.notna(row['Kickoff_P2_Odds']) else "N/A"

            print("{")
            print(f"    'Player 1 ID': {row['Player 1 ID']}, 'Player 1 Name': '{row['Player 1']}',")
            print(f"    'Player 2 ID': {row['Player 2 ID']}, 'Player 2 Name': '{row['Player 2']}',")
            print(f"    'P1_ML': {p1_ml}, # Actual Decimal: {p1_odds_str}, Winner: {'P1' if row['P1_Win']==1 else 'P2'}")
            print(f"    'P2_ML': {p2_ml}  # Actual Decimal: {p2_odds_str}")
            print("},")

except FileNotFoundError:
    print(f"Error: The file '{HISTORICAL_DATA_FILE}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
