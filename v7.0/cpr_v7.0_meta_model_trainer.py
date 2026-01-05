import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.linear_model import LogisticRegression
import joblib

# --- 1. Configuration ---
FEATURES_FILE = "final_engineered_features.csv"
# --- NEW: Updated Input Model Files ---
GBM_MODEL_FILE = "cpr_v7.0_gbm_specialist.joblib"
GBM_PREPROCESSOR_FILE = "gbm_preprocessor_v7.0.joblib"
LSTM_MODEL_FILE = "cpr_v7.0_lstm_specialist.h5"
LSTM_SCALER_FILE = "lstm_scaler_v7.0.joblib"
# --- NEW: Updated Output Model File ---
META_MODEL_OUTPUT_FILE = "cpr_v7.0_meta_model.pkl"

SEQUENCE_LENGTH = 5 # Must be the same as in the LSTM trainer

# --- 2. Main Script Logic ---
try:
    # --- Load Data and Trained Models ---
    print("Loading data and all re-trained specialist models...")
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
    # --- A: Prepare data for LSTM and get its predictions ---
    print("Preparing sequences for LSTM...")
    player_history = {}
    aligned_indices, X_p1_sequences, X_p2_sequences = [], [], []
    
    features_to_sequence = [
        'P1_Rolling_Win_Rate_L10', 'P1_Rolling_Pressure_Points_L10', 'P1_Rest_Days',
        'P2_Rolling_Win_Rate_L10', 'P2_Rolling_Pressure_Points_L10', 'P2_Rest_Days',
        'H2H_P1_Win_Rate'
    ]
    df_sequence_features = df[['Player 1 ID', 'Player 2 ID', 'P1_Win'] + features_to_sequence]

    for index, row in df_sequence_features.iterrows():
        p1_id, p2_id = row['Player 1 ID'], row['Player 2 ID']
        p1_hist, p2_hist = player_history.get(p1_id, []), player_history.get(p2_id, [])
        
        if len(p1_hist) >= SEQUENCE_LENGTH and len(p2_hist) >= SEQUENCE_LENGTH:
            aligned_indices.append(index)
            X_p1_sequences.append(p1_hist[-SEQUENCE_LENGTH:])
            X_p2_sequences.append(p2_hist[-SEQUENCE_LENGTH:])
            
        p1_perspective_vector = row[features_to_sequence[:3]].values
        p2_opponent_features = row[features_to_sequence[3:6]].values
        p1_full_vector = np.concatenate([p1_perspective_vector, p2_opponent_features, [row['H2H_P1_Win_Rate']]])
        if p1_id not in player_history: player_history[p1_id] = []
        player_history[p1_id].append(p1_full_vector)

        p2_full_vector = np.concatenate([p2_opponent_features, p1_perspective_vector, [1 - row['H2H_P1_Win_Rate']]])
        if p2_id not in player_history: player_history[p2_id] = []
        player_history[p2_id].append(p2_full_vector)

    X_p1, X_p2 = np.array(X_p1_sequences), np.array(X_p2_sequences)
    
    nsamples, nsteps, nfeatures = X_p1.shape
    X_p1_scaled = lstm_scaler.transform(X_p1.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
    X_p2_scaled = lstm_scaler.transform(X_p2.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
    
    print("Generating predictions from LSTM...")
    lstm_predictions = lstm_model.predict([X_p1_scaled, X_p2_scaled]).flatten()

    # --- B: Prepare data for GBM and get its predictions ---
    aligned_df = df.loc[aligned_indices].copy()
    aligned_df['Win_Rate_Advantage'] = aligned_df['P1_Rolling_Win_Rate_L10'] - aligned_df['P2_Rolling_Win_Rate_L10']
    aligned_df['Pressure_Points_Advantage'] = aligned_df['P1_Rolling_Pressure_Points_L10'] - aligned_df['P2_Rolling_Pressure_Points_L10']
    aligned_df['Rest_Advantage'] = aligned_df['P1_Rest_Days'] - aligned_df['P2_Rest_Days']
    
    X_gbm = aligned_df[['Win_Rate_Advantage', 'Pressure_Points_Advantage', 'Rest_Advantage', 'H2H_P1_Win_Rate', 'Player 1 ID', 'Player 2 ID']]
    X_gbm_processed = gbm_preprocessor.transform(X_gbm)
    
    print("Generating predictions from GBM...")
    gbm_predictions = gbm_model.predict_proba(X_gbm_processed)[:, 1]

    # --- C: Get the actual outcomes (the target) ---
    y_meta = aligned_df['P1_Win'].values

    # --- 4. Train the Meta-Model ---
    X_meta = np.vstack([gbm_predictions, lstm_predictions]).T

    print(f"\nTraining the meta-model on {len(X_meta)} samples...")
    meta_model = LogisticRegression()
    meta_model.fit(X_meta, y_meta)
    print("Meta-model training complete.")
    
    # --- 5. Save the Final Meta-Model ---
    joblib.dump(meta_model, META_MODEL_OUTPUT_FILE)
    print(f"\n✅ Successfully re-trained and saved the Meta-Model to '{META_MODEL_OUTPUT_FILE}'")

except FileNotFoundError as e:
    print(f"Error: A required model or data file was not found. Please ensure all trainers have been run.")
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")