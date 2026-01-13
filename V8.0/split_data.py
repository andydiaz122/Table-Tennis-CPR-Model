import pandas as pd

# --- 1. Configuration ---
INPUT_FILE = "final_dataset_v7.4_no_duplicates.csv"
TRAINING_FILE = "training_dataset.csv"
TESTING_FILE = "testing_dataset.csv"
# We will use a 75/25 split for training and testing; .999 when running system live (no testing data needed)
TRAIN_SPLIT_PERCENTAGE = 0.70

# --- EMBARGO CONFIGURATION (Phase 1.1.2) ---
EMBARGO_HOURS = 24  # Minimum gap between train and test sets


def create_embargo_split(df, train_pct=0.70, embargo_hours=24):
    """
    Creates train/test split with temporal embargo buffer.

    This prevents information leakage by ensuring a minimum time gap
    between the last training sample and first test sample.

    Args:
        df: DataFrame sorted by Date
        train_pct: Percentage of data for training (0.0 to 1.0)
        embargo_hours: Minimum hours between train end and test start

    Returns:
        train_df, test_df, embargo_stats dict
    """
    df = df.sort_values('Date').reset_index(drop=True)

    # Initial split point
    initial_split_idx = int(len(df) * train_pct)

    # Get the train end timestamp
    train_end_date = df.iloc[initial_split_idx - 1]['Date']
    embargo_cutoff = train_end_date + pd.Timedelta(hours=embargo_hours)

    # Find first test sample AFTER embargo
    test_candidates = df[df['Date'] > embargo_cutoff]

    if len(test_candidates) == 0:
        raise ValueError(f"No test samples remain after {embargo_hours}h embargo. Reduce embargo or train percentage.")

    test_start_idx = test_candidates.index[0]

    # Create final splits
    train_df = df.iloc[:initial_split_idx]
    test_df = df.iloc[test_start_idx:]

    # Calculate embargo stats
    matches_in_embargo = test_start_idx - initial_split_idx
    actual_gap_hours = (test_df.iloc[0]['Date'] - train_df.iloc[-1]['Date']).total_seconds() / 3600

    # Check for overlapping players in boundary windows
    last_train_hour = train_df[train_df['Date'] >= train_end_date - pd.Timedelta(hours=1)]
    first_test_hour = test_df[test_df['Date'] <= test_df.iloc[0]['Date'] + pd.Timedelta(hours=1)]

    train_players = set(last_train_hour['Player 1 ID'].tolist() + last_train_hour['Player 2 ID'].tolist())
    test_players = set(first_test_hour['Player 1 ID'].tolist() + first_test_hour['Player 2 ID'].tolist())
    overlapping_players = train_players & test_players

    embargo_stats = {
        'initial_split_idx': initial_split_idx,
        'final_test_start_idx': test_start_idx,
        'matches_in_embargo_zone': matches_in_embargo,
        'train_end_date': str(train_end_date),
        'test_start_date': str(test_df.iloc[0]['Date']),
        'actual_gap_hours': actual_gap_hours,
        'embargo_met': actual_gap_hours >= embargo_hours,
        'overlapping_players_count': len(overlapping_players),
        'overlapping_player_ids': list(overlapping_players)[:10]  # First 10 for logging
    }

    return train_df, test_df, embargo_stats


# --- 2. Main Script Logic ---
try:
    # Load the entire dataset from your CSV file.
    print(f"Loading data from '{INPUT_FILE}'...")
    df = pd.read_csv(
        INPUT_FILE,
        na_values=['-'],
        keep_default_na=True,
        low_memory=False
    )
    print(f"Successfully loaded {len(df)} total matches.")

    # Convert the 'Date' column to a proper datetime format to ensure correct sorting.
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort the entire dataset by date, from oldest to newest.
    df = df.sort_values(by='Date', ascending=True)

    # --- Use embargo-aware split ---
    print(f"\n--- Applying {EMBARGO_HOURS}-hour Embargo Split ---")
    training_df, testing_df, embargo_stats = create_embargo_split(
        df,
        train_pct=TRAIN_SPLIT_PERCENTAGE,
        embargo_hours=EMBARGO_HOURS
    )

    print(f"\nEmbargo Statistics:")
    print(f"  Matches skipped in embargo zone: {embargo_stats['matches_in_embargo_zone']}")
    print(f"  Train end date: {embargo_stats['train_end_date']}")
    print(f"  Test start date: {embargo_stats['test_start_date']}")
    print(f"  Actual gap: {embargo_stats['actual_gap_hours']:.1f} hours")
    print(f"  Embargo requirement met: {'✓' if embargo_stats['embargo_met'] else '✗'}")
    print(f"  Overlapping players at boundary: {embargo_stats['overlapping_players_count']}")

    print(f"\nFinal Split:")
    print(f"  Training set size: {len(training_df)} matches")
    print(f"  Testing set size: {len(testing_df)} matches")

    # Save the two new datasets to separate CSV files.
    training_df.to_csv(TRAINING_FILE, index=False)
    testing_df.to_csv(TESTING_FILE, index=False)

    print(f"\n✅ Successfully created training and testing files:")
    print(f" -> {TRAINING_FILE}")
    print(f" -> {TESTING_FILE}")

except FileNotFoundError:
    print(f"Error: The input file '{INPUT_FILE}' was not found. Please make sure it's in the same folder as the script.")
except Exception as e:
    print(f"An error occurred: {e}")