import pandas as pd

# --- Configuration ---
FINAL_DATASET_FILE = "final_dataset_v7.4_no_duplicates.csv"
TRAIN_SPLIT_PERCENTAGE = 0.70
SEQUENCE_LENGTH = 5
GAP_END_DATE = '2025-08-03'

# --- Load and Split Data ---
df = pd.read_csv(FINAL_DATASET_FILE)
df['Date'] = pd.to_datetime(df['Date'], format='mixed')
df.sort_values(by='Date', inplace=True)
df.reset_index(drop=True, inplace=True)

split_index = int(len(df) * TRAIN_SPLIT_PERCENTAGE)
backtest_df = df.iloc[split_index:].copy()

# Filter for the "gap period"
gap_df = backtest_df[backtest_df['Date'] < GAP_END_DATE]

print(f"Analyzing {len(gap_df)} matches in the gap period...\n")

# --- Run Diagnostic ---
for index, match in gap_df.iterrows():
    history_df = df.iloc[:index]
    p1_id, p2_id = match['Player 1 ID'], match['Player 2 ID']
    
    p1_hist_count = len(history_df[(history_df['Player 1 ID'] == p1_id) | (history_df['Player 2 ID'] == p1_id)])
    p2_hist_count = len(history_df[(history_df['Player 1 ID'] == p2_id) | (history_df['Player 2 ID'] == p2_id)])
    
    if p1_hist_count < SEQUENCE_LENGTH or p2_hist_count < SEQUENCE_LENGTH:
        print(f"Date: {match['Date'].date()} - Match: {match['Player 1']} vs. {match['Player 2']}")
        print(f"  -> SKIPPED: P1 History={p1_hist_count}, P2 History={p2_hist_count} (Required: {SEQUENCE_LENGTH})")