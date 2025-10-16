import pandas as pd
import joblib

# --- 1. Configuration ---
TESTING_FILE = "testing_dataset.csv"
MODEL_FILE = "odds_model_v6.4.joblib"
SCALER_FILE = "scaler.joblib"
# This is the final output file you will use for your back-test.
OUTPUT_FILE = "backtest_data_with_odds.csv"

# --- 2. Main Script Logic ---
try:
    # Load the testing dataset, the trained model, and the scaler.
    print(f"Loading data from '{TESTING_FILE}'...")
    df_test = pd.read_csv(TESTING_FILE)
    print(f"Loading model from '{MODEL_FILE}' and scaler from '{SCALER_FILE}'...")
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print(f"Successfully loaded {len(df_test)} matches for testing.")

    # --- 3. Data Cleaning (same as in the training script) ---
    # It's important to apply the exact same cleaning steps to the test data.
    df_test.dropna(subset=['Final Score'], inplace=True)
    df_test['Final Score'] = df_test['Final Score'].astype(str)

    # --- 4. Feature Preparation ---
    # Define the same features the model was trained on.
    features = [
        'P1 Total Points', 'P2 Total Points',
        'P1 Pressure Points', 'P2 Pressure Points',
        'P1 Set Comebacks', 'P2 Set Comebacks'
    ]
    
    # Separate the features (X) from the test data.
    X_test = df_test[features]

    # Apply the same scaling that was used on the training data.
    # IMPORTANT: Use .transform() here, not .fit_transform().
    X_test_scaled = scaler.transform(X_test)

    # --- 5. Generate Probabilities ---
    print("\nGenerating win probabilities for the test set...")
    # Use model.predict_proba() to get the probabilities for each outcome.
    # It returns probabilities for [class 0 (P2 Win), class 1 (P1 Win)].
    win_probabilities = model.predict_proba(X_test_scaled)

    # Extract the probability of Player 1 winning (the second column).
    df_test['P1_Win_Prob'] = win_probabilities[:, 1]
    df_test['P2_Win_Prob'] = 1 - df_test['P1_Win_Prob']
    print("Probabilities generated.")

    # --- 6. Calculate Simulated Odds ---
    print("Calculating simulated odds...")
    # Convert probabilities to decimal odds. Add a small epsilon to avoid division by zero.
    epsilon = 1e-9 
    df_test['P1_Simulated_Odds'] = 1 / (df_test['P1_Win_Prob'] + epsilon)
    df_test['P2_Simulated_Odds'] = 1 / (df_test['P2_Win_Prob'] + epsilon)
    print("Odds calculated.")

    # --- 7. Save the Final Dataset ---
    df_test.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ Successfully created the final back-testing file:")
    print(f" -> {OUTPUT_FILE}")
    print("\nThis file now contains the actual match outcomes and your model's simulated odds.")

except FileNotFoundError as e:
    print(f"Error: A required file was not found. Please make sure '{TESTING_FILE}', '{MODEL_FILE}', and '{SCALER_FILE}' are in the same folder.")
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")