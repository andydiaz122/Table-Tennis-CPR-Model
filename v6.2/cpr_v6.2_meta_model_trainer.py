import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.linear_model import LogisticRegression
import joblib

# --- 1. Configuration ---
FEATURES_FILE = "training_features.csv"
# Input Model Files
GBM_MODEL_FILE = "cpr_v6.2_gbm_specialist.joblib"
GBM_PREPROCESSOR_FILE = "gbm_preprocessor.joblib"
LSTM_MODEL_FILE = "cpr_v6.2_lstm_specialist.h5"
LSTM_SCALER_FILE = "lstm_scaler.joblib"
# Output Model File
META_MODEL_OUTPUT_FILE = "cpr_v6.2_meta_model.pkl"

SEQUENCE_LENGTH = 5 # Must be the same as in the LSTM trainer

# --- 2. Main Script Logic ---
try:
    # --- Load Data and Trained Models ---
    print("Loading data and all specialist models...")
    df = pd.read_csv(FEATURES_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    df.dropna(inplace=True)

    gbm_model = joblib.load(GBM_MODEL_FILE)
    gbm_preprocessor = joblib.load(GBM_PREPROCESSOR_FILE)
    lstm_model = load_model(LSTM_MODEL_FILE)
    lstm_scaler = joblib.load(LSTM_SCALER_FILE)
    print("All models and data loaded successfully.")

    # --- 3. Generate Predictions from Specialists ---
    # We need to create the exact same data structures used for training each model.
    
    # --- A: Prepare data for LSTM and get its predictions ---
    print("Preparing sequences for LSTM...")
    player_history = {}
    
    # These lists will hold the data that has enough history for the LSTM
    aligned_indices, X_p1_sequences, X_p2_sequences = [], [], []
    features_to_sequence = ['Total_Points_Advantage', 'Pressure_Points_Advantage', 'Comeback_Advantage']

    for index, row in df.iterrows():
        p1_id, p2_id = row['Player 1 ID'], row['Player 2 ID']
        p1_hist = player_history.get(p1_id, [])
        p2_hist = player_history.get(p2_id, [])
        
        if len(p1_hist) >= SEQUENCE_LENGTH and len(p2_hist) >= SEQUENCE_LENGTH:
            aligned_indices.append(index)
            X_p1_sequences.append(p1_hist[-SEQUENCE_LENGTH:])
            X_p2_sequences.append(p2_hist[-SEQUENCE_LENGTH:])
            
        current_p1_stats = row[features_to_sequence].values
        if p1_id not in player_history: player_history[p1_id] = []
        player_history[p1_id].append(current_p1_stats)
        
        current_p2_stats = -row[features_to_sequence].values
        if p2_id not in player_history: player_history[p2_id] = []
        player_history[p2_id].append(current_p2_stats)

    X_p1 = np.array(X_p1_sequences)
    X_p2 = np.array(X_p2_sequences)
    
    nsamples, nsteps, nfeatures = X_p1.shape
    X_p1_scaled = lstm_scaler.transform(X_p1.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
    X_p2_scaled = lstm_scaler.transform(X_p2.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
    
    print("Generating predictions from LSTM...")
    lstm_predictions = lstm_model.predict([X_p1_scaled, X_p2_scaled]).flatten()

    # --- B: Get GBM predictions ONLY for the aligned data ---
    # It's crucial we use the same matches that the LSTM could predict on.
    aligned_df = df.loc[aligned_indices]
    
    X_gbm = aligned_df[['Total_Points_Advantage', 'Pressure_Points_Advantage', 'Comeback_Advantage', 'Player 1 ID', 'Player 2 ID']]
    X_gbm_processed = gbm_preprocessor.transform(X_gbm)
    
    print("Generating predictions from GBM...")
    gbm_predictions = gbm_model.predict_proba(X_gbm_processed)[:, 1]

    # --- C: Get the actual outcomes (the target) ---
    y_meta = aligned_df['P1_Win'].values

    # --- 4. Train the Meta-Model ---
    # The features for our meta-model are the predictions from the specialists.
    X_meta = np.vstack([gbm_predictions, lstm_predictions]).T

    print(f"\nTraining the meta-model on {len(X_meta)} samples...")
    # A simple Logistic Regression model is perfect for a manager model.
    meta_model = LogisticRegression()
    meta_model.fit(X_meta, y_meta)
    print("Meta-model training complete.")
    
    # --- 5. Save the Final Meta-Model ---
    joblib.dump(meta_model, META_MODEL_OUTPUT_FILE)
    print(f"\n✅ Successfully trained and saved the Meta-Model to '{META_MODEL_OUTPUT_FILE}'")

except FileNotFoundError as e:
    print(f"Error: A required model or data file was not found. Please ensure all trainers have been run.")
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")