import pandas as pd

# Load the dataset
# OPTIMIZATION: Read from gzip-compressed file (output of merge_data_v7.4.py)
file_path = 'final_dataset_v7.4.csv.gz'
output_path = 'final_dataset_v7.4_no_duplicates.csv.gz'

try:
    # Attempt to read the file using the Python engine and skip bad lines.
    # This combination of parameters handles the "Expected X fields, saw Y" error.
    # OPTIMIZATION: pandas auto-detects .gz compression, no explicit param needed
    df = pd.read_csv(
        file_path,
        sep=',',
        engine='python',
        on_bad_lines='skip',
        encoding='utf-8',
        na_values=['-'],
        keep_default_na=True,
    )

    # Print the shape before removing duplicates
    print(f"Dataset shape before removing duplicates (after skipping errors): {df.shape}")

    # OPTIMIZATION: Chain operations instead of inplace=True for better performance
    df = df.drop_duplicates(subset='Match ID', keep='last')

    # Print the shape after removing duplicates
    print(f"Dataset shape after removing duplicates: {df.shape}")

    # OPTIMIZATION: Chain operations instead of inplace=True
    df = df.dropna()

    # Save the cleaned data to a new CSV file
    # OPTIMIZATION: Output gzip-compressed for ~70% smaller file
    df.to_csv(output_path, index=False, compression='gzip')

    print(f"Duplicates removed and data saved to '{output_path}'")

except Exception as e:
    print(f"An unexpected error occurred during processing: {e}")