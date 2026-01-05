import pandas as pd

# --- 1. Configuration ---
# Make sure this filename matches your data file exactly.
DATA_FILE = "backtest_data_with_odds.csv"

# --- 2. Main Script Logic ---
try:
    # Load the backtest data from your CSV file.
    df = pd.read_csv(DATA_FILE)

    print("--- Statistical Sanity Check ---")
    # Get descriptive statistics for odds and probability columns.
    stats_check = df[['P1_Win_Prob', 'P1_Simulated_Odds', 'P2_Simulated_Odds']].describe()
    print(stats_check)
    
    print("\n--- Predictive Power Check ---")
    # Determine the actual winner (1 if Player 1 won, 0 if Player 2 won).
    # This handles potential missing scores and ensures the column is a string.
    df.dropna(subset=['Final Score'], inplace=True)
    df['Final Score'] = df['Final Score'].astype(str)
    df['Actual_Winner'] = df['Final Score'].apply(lambda x: 1 if int(x.strip('="').split('-')[0]) > int(x.strip('="').split('-')[1]) else 0)

    # Determine the model's predicted winner (1 if P1 was the favorite, 0 if P2 was the favorite).
    df['Predicted_Winner'] = df['P1_Win_Prob'].apply(lambda x: 1 if x > 0.5 else 0)

    # Calculate the accuracy of the model's predictions on this test set.
    correct_predictions = (df['Actual_Winner'] == df['Predicted_Winner']).sum()
    total_predictions = len(df)
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

    print(f"Total Matches in Test Set: {total_predictions}")
    print(f"Correct Predictions (Favorite Won): {correct_predictions}")
    print(f"Model Accuracy on Test Set: {accuracy:.2f}%")
    print("---------------------------------")


except FileNotFoundError:
    print(f"Error: The input file '{DATA_FILE}' was not found. Please make sure it's in the same folder as the script.")
except Exception as e:
    print(f"An error occurred: {e}")