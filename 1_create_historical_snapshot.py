# FILE: 1_create_historical_snapshot.py
import pandas as pd

# --- Configuration ---
# Set this to the FIRST day of your paper-trading period.
# The script will create a history file containing all data BEFORE this date.
CUTOFF_DATE = '2025-09-17'

# --- File Paths ---
FULL_DATASET_FILE = "final_dataset_v7.4.csv" # Using your latest dataset
SNAPSHOT_OUTPUT_FILE = "paper_trade_history.csv"

# --- Main Logic ---
try:
    print(f"--- Creating Historical Snapshot before {CUTOFF_DATE} ---")
    df = pd.read_csv(FULL_DATASET_FILE)
    df['Date'] = pd.to_datetime(df['Date'])

    # The crucial step: Filter out all data on or after the cutoff date
    historical_df = df[df['Date'] < CUTOFF_DATE].copy()

    historical_df.to_csv(SNAPSHOT_OUTPUT_FILE, index=False)
    print(f"✅ Successfully created '{SNAPSHOT_OUTPUT_FILE}' with {len(historical_df)} historical records.")
    print("Your predictor will now use this safe, point-in-time data.")

except FileNotFoundError:
    print(f"Error: The file '{FULL_DATASET_FILE}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")