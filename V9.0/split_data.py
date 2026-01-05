import pandas as pd

# --- 1. Configuration ---
INPUT_FILE = "final_dataset_v7.4_no_duplicates.csv.gz"  # OPTIMIZATION: uses gzip compression
TRAINING_FILE = "training_dataset.csv"
TESTING_FILE = "testing_dataset.csv"
# We will use a 75/25 split for training and testing; .999 when running system live (no testing data needed)
TRAIN_SPLIT_PERCENTAGE = 0.70

# --- 2. Main Script Logic ---
try:
    # Load the entire dataset from your CSV file.
    print(f"Loading data from '{INPUT_FILE}'...")
#    df = pd.read_csv(INPUT_FILE)
#    df = pd.read_csv(INPUT_FILE, parse_dates=['Date'], date_format='mixed')
#    df = pd.read_csv(INPUT_FILE, parse_dates=['Date'], date_format='mixed', low_memory=False)
    # Define the data types for the problematic columns
    column_types = {
        'Kickoff_P1_Odds': str,
        'Kickoff_P2_Odds': str
    }
    df = pd.read_csv(
        INPUT_FILE,
        na_values=['-'], 
        keep_default_na=True, # Also respects empty strings/default missing values as NaN
        low_memory=False # Prevents Pandas from guessing types chunk-by-chunk
    )
    print(f"Successfully loaded {len(df)} total matches.")

    # Convert the 'Date' column to a proper datetime format to ensure correct sorting.
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort the entire dataset by date, from oldest to newest.
    # This is crucial to ensure our test set contains the most recent matches.
    df = df.sort_values(by='Date', ascending=True)

    # Calculate the split point.
    split_index = int(len(df) * TRAIN_SPLIT_PERCENTAGE)

    # Split the data into training and testing sets.
    training_df = df.iloc[:split_index]
    testing_df = df.iloc[split_index:]

    print(f"\nSplitting data...")
    print(f"Training set size: {len(training_df)} matches")
    print(f"Testing set size: {len(testing_df)} matches")

    # Save the two new datasets to separate CSV files.
    training_df.to_csv(TRAINING_FILE, index=False)
    testing_df.to_csv(TESTING_FILE, index=False)
    
    print(f"\n[OK] Successfully created training and testing files:")
    print(f" -> {TRAINING_FILE}")
    print(f" -> {TESTING_FILE}")

except FileNotFoundError:
    print(f"Error: The input file '{INPUT_FILE}' was not found. Please make sure it's in the same folder as the script.")
except Exception as e:
    print(f"An error occurred: {e}")