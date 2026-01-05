import pandas as pd

# Load the dataset with error handling for misaligned lines
file_path = 'final_dataset_v7.4.csv'

try:
    # Attempt to read the file using the Python engine and skip bad lines.
    # This combination of parameters handles the "Expected X fields, saw Y" error.
    df = pd.read_csv(
        file_path, 
        sep=',', 
        engine='python', 
        on_bad_lines='skip', 
        encoding='utf-8'
    )
    
    # Print the shape before removing duplicates
    print(f"Dataset shape before removing duplicates (after skipping errors): {df.shape}")

    # Remove duplicates based on all columns and keep the first occurrence
    df.drop_duplicates(subset='Match ID', keep='last', inplace=True)

    # Print the shape after removing duplicates
    print(f"Dataset shape after removing duplicates: {df.shape}")

    # Save the cleaned data to a new CSV file
    df.to_csv('final_dataset_v7.4_no_duplicates.csv', index=False)

    print("Duplicates removed and data saved to 'final_dataset_v7.4_no_duplicates.csv'")

except Exception as e:
    print(f"An unexpected error occurred during processing: {e}")
