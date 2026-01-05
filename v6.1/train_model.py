import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib # For saving the trained model

# --- 1. Configuration ---
TRAINING_FILE = "training_dataset.csv"
MODEL_FILE = "odds_model.joblib" # The output file for our trained model

# --- 2. Main Script Logic ---
try:
    # Load the training dataset
    print(f"Loading data from '{TRAINING_FILE}'...")
    df = pd.read_csv(TRAINING_FILE)
    print(f"Successfully loaded {len(df)} matches for training.")

    # --- NEW: Data Cleaning & Preparation ---
    print("Cleaning data...")
    # Drop any rows where the 'Final Score' is missing.
    df.dropna(subset=['Final Score'], inplace=True)
    # Ensure the 'Final Score' column is treated as a string to prevent errors.
    df['Final Score'] = df['Final Score'].astype(str)
    print(f"Removed rows with missing scores. {len(df)} matches remaining.")

    # --- 3. Feature Engineering ---
    # Select the features (columns) you want the model to learn from.
    features = [
        'P1 Total Points', 'P2 Total Points',
        'P1 Pressure Points', 'P2 Pressure Points',
        'P1 Set Comebacks', 'P2 Set Comebacks'
    ]
    
    # Define the target variable (what we want to predict).
    # We'll create a new column 'P1_Win'. It will be 1 if Player 1 won, and 0 if Player 2 won.
    df['P1_Win'] = df['Final Score'].apply(lambda x: 1 if int(x.strip('="').split('-')[0]) > int(x.strip('="').split('-')[1]) else 0)

    # Separate our data into features (X) and the target (y)
    X = df[features]
    y = df['P1_Win']
    
    # It's good practice to scale the features. This helps the model perform better.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 4. Model Training ---
    print("\nTraining the Logistic Regression model...")
    # We create an instance of the model.
    model = LogisticRegression()
    # We 'fit' the model to our data, which is the training process.
    model.fit(X_scaled, y)
    print("Model training complete.")

    # --- 5. Save the Model and Scaler ---
    # We save the trained model and the scaler to files so we can use them in the next step.
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, 'scaler.joblib') # Also save the scaler
    
    print(f"\n✅ Successfully trained and saved the model to '{MODEL_FILE}'")

except FileNotFoundError:
    print(f"Error: The input file '{TRAINING_FILE}' was not found. Please run the data splitting script first.")
except Exception as e:
    print(f"An error occurred: {e}")