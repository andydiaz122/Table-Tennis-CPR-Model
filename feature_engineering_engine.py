import pandas as pd

# --- 1. Configuration ---
TRAINING_INPUT = "training_dataset.csv"
TESTING_INPUT = "testing_dataset.csv"

TRAINING_OUTPUT = "training_features.csv"
TESTING_OUTPUT = "testing_features.csv"

# --- 2. Feature Engineering Function ---
def create_features(df):
    """
    Takes a dataframe of match data and engineers new features for modeling.
    """
    print("Engineering new features...")
    
    # Create "delta" features that show the advantage of Player 1 over Player 2
    df['Total_Points_Advantage'] = df['P1 Total Points'] - df['P2 Total Points']
    df['Pressure_Points_Advantage'] = df['P1 Pressure Points'] - df['P2 Pressure Points']
    df['Comeback_Advantage'] = df['P1 Set Comebacks'] - df['P2 Set Comebacks']
    
    # Define the target variable (the actual outcome)
    df.dropna(subset=['Final Score'], inplace=True)
    df['Final Score'] = df['Final Score'].astype(str)
    df['P1_Win'] = df['Final Score'].apply(lambda x: 1 if int(x.strip('="').split('-')[0]) > int(x.strip('="').split('-')[1]) else 0)

    # Select the final columns needed for the models
    # We keep the IDs for the LSTM and the engineered features for the GBM
    final_columns = [
        'Match ID', 'Date',
        'Player 1 ID', 'Player 2 ID',
        'Total_Points_Advantage',
        'Pressure_Points_Advantage',
        'Comeback_Advantage',
        'P1_Win' # The target
    ]
    
    # Ensure all required columns exist
    for col in final_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataframe.")
            
    return df[final_columns]

# --- 3. Main Script Logic ---
try:
    # Process the Training Data
    print(f"--- Processing Training Data ---")
    df_train = pd.read_csv(TRAINING_INPUT)
    features_train = create_features(df_train)
    features_train.to_csv(TRAINING_OUTPUT, index=False)
    print(f"Successfully created '{TRAINING_OUTPUT}' with {len(features_train)} matches.")

    # Process the Testing Data
    print(f"\n--- Processing Testing Data ---")
    df_test = pd.read_csv(TESTING_INPUT)
    features_test = create_features(df_test)
    features_test.to_csv(TESTING_OUTPUT, index=False)
    print(f"Successfully created '{TESTING_OUTPUT}' with {len(features_test)} matches.")

    print("\n✅ Feature engineering complete.")

except FileNotFoundError as e:
    print(f"Error: An input file was not found. Please ensure '{TRAINING_INPUT}' and '{TESTING_INPUT}' exist.")
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")