import pandas as pd

# Load the dataset
file_path = 'final_dataset_v7.4.csv'
# df = pd.read_csv(file_path)

try:
    # Attempt to read the file using the Python engine and skip bad lines.
    # This combination of parameters handles the "Expected X fields, saw Y" error.
    df = pd.read_csv(
        file_path, 
        sep=',', 
        engine='python', 
        on_bad_lines='skip', 
        encoding='utf-8',
        na_values=['-'], 
        keep_default_na=True, # Also respects empty strings/default missing values as NaN
    )
    
    # Print the shape before removing duplicates
    print(f"Dataset shape before removing duplicates (after skipping errors): {df.shape}")

    # Remove duplicates based on all columns and keep the first occurrence
    df.drop_duplicates(subset='Match ID', keep='last', inplace=True)

    # Print the shape after removing duplicates
    print(f"Dataset shape after removing duplicates: {df.shape}")

    # --- FIXED: Only drop rows missing CRITICAL columns for betting ---
    # Previously: df.dropna(inplace=True) dropped ALL rows with ANY NaN
    # This was too aggressive - dropped rows missing Match Format, Set Scores, etc.
    # which are not needed for prediction. Only drop rows missing odds or outcome.
    critical_cols = ['Kickoff_P1_Odds', 'Kickoff_P2_Odds', 'P1_Win']
    before_dropna = len(df)
    df.dropna(subset=critical_cols, inplace=True)
    after_dropna = len(df)
    print(f"Dropped {before_dropna - after_dropna} rows missing critical columns (odds/outcome)")

    # Save the cleaned data to a new CSV file
    df.to_csv('final_dataset_v7.4_no_duplicates.csv', index=False)

    print("Duplicates removed and data saved to 'final_dataset_v7.4_no_duplicates.csv'")

except Exception as e:
    print(f"An unexpected error occurred during processing: {e}")