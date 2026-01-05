import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

# --- 1. Configuration ---
# ---!!!--- Manually input the new matches you want to predict ---!!!
upcoming_matches = [
{'Player 1 ID': 373963, 'Player 1 Name': 'Tomas Janata', 'Player 2 ID': 800199, 'Player 2 Name': 'Tomas Dousa'},
{'Player 1 ID': 391465, 'Player 1 Name': 'Vaclav Dolezal', 'Player 2 ID': 1082004, 'Player 2 Name': 'Petr Bruzek'},
{'Player 1 ID': 1158350, 'Player 1 Name': 'Josef Pantak', 'Player 2 ID': 689526, 'Player 2 Name': 'Jaroslav Strnad 1961'},
{'Player 1 ID': 548705, 'Player 1 Name': 'Jiri Plachy', 'Player 2 ID': 917524, 'Player 2 Name': 'Jakub Tazler'},
]
# ---!!!--- ---!!!--- ---!!!--- ---!!!--- ---!!!--- ---!!!---

HISTORICAL_DATA_FILE = "final_engineered_features.csv" 
GBM_MODEL_FILE = "cpr_v6.4_gbm_specialist.joblib"
GBM_PREPROCESSOR_FILE = "gbm_preprocessor_v6.4.joblib"
LSTM_MODEL_FILE = "cpr_v6.4_lstm_specialist.h5"
LSTM_SCALER_FILE = "lstm_scaler_v6.4.joblib"
META_MODEL_FILE = "cpr_v6.4_meta_model.pkl"

SEQUENCE_LENGTH = 5
ROLLING_WINDOW = 10

# --- 2. Load All Models and Historical Data ---
try:
    print("--- Loading All Models and Historical Data for Prediction ---")
    df_history = pd.read_csv(HISTORICAL_DATA_FILE)
    df_history['Date'] = pd.to_datetime(df_history['Date'])

    gbm_model = joblib.load(GBM_MODEL_FILE)
    gbm_preprocessor = joblib.load(GBM_PREPROCESSOR_FILE)
    lstm_model = load_model(LSTM_MODEL_FILE)
    lstm_scaler = joblib.load(LSTM_SCALER_FILE)
    meta_model = joblib.load(META_MODEL_FILE)
    print("All v6.2 models and data loaded successfully.")

    # --- 3. Feature Engineering and Sequence Building for New Matches ---
    print("\n--- Engineering Features & Building Sequences for New Matches ---")
    
    new_match_features = []
    X_p1_sequences, X_p2_sequences = [], []

    features_for_lstm_sequence = [
        'P1_Rolling_Win_Rate_L10', 'P1_Rolling_Pressure_Points_L10', 'P1_Rest_Days',
        'P2_Rolling_Win_Rate_L10', 'P2_Rolling_Pressure_Points_L10', 'P2_Rest_Days',
        'H2H_P1_Win_Rate'
    ]
    
    for match in upcoming_matches:
        p1_id = match['Player 1 ID']
        p2_id = match['Player 2 ID']
        
        # --- A. Engineer Features for the GBM ---
        p1_last_match_date = df_history[((df_history['Player 1 ID'] == p1_id) | (df_history['Player 2 ID'] == p1_id))]['Date'].max()
        p2_last_match_date = df_history[((df_history['Player 1 ID'] == p2_id) | (df_history['Player 2 ID'] == p2_id))]['Date'].max()
        p1_rest_days = (datetime.now() - p1_last_match_date).days if pd.notna(p1_last_match_date) else 30
        
        # --- CORRECTED LINE ---
        p2_rest_days = (datetime.now() - p2_last_match_date).days if pd.notna(p2_last_match_date) else 30

        h2h_df = df_history[((df_history['Player 1 ID'] == p1_id) & (df_history['Player 2 ID'] == p2_id)) | 
                            ((df_history['Player 1 ID'] == p2_id) & (df_history['Player 2 ID'] == p1_id))]
        p1_h2h_wins = len(h2h_df[((h2h_df['Player 1 ID'] == p1_id) & (h2h_df['P1_Win'] == 1)) | 
                                 ((h2h_df['Player 2 ID'] == p1_id) & (h2h_df['P1_Win'] == 0))])
        h2h_total = len(h2h_df)
        h2h_p1_win_rate = p1_h2h_wins / h2h_total if h2h_total > 0 else 0.5

        p1_recent_games = df_history[((df_history['Player 1 ID'] == p1_id) | (df_history['Player 2 ID'] == p1_id))].tail(ROLLING_WINDOW)
        p2_recent_games = df_history[((df_history['Player 1 ID'] == p2_id) | (df_history['Player 2 ID'] == p2_id))].tail(ROLLING_WINDOW)
        
        p1_rolling_win_rate = p1_recent_games.apply(lambda row: 1 if (row['Player 1 ID'] == p1_id and row['P1_Win'] == 1) or (row['Player 2 ID'] == p1_id and row['P1_Win'] == 0) else 0, axis=1).mean()
        p2_rolling_win_rate = p2_recent_games.apply(lambda row: 1 if (row['Player 1 ID'] == p2_id and row['P1_Win'] == 1) or (row['Player 2 ID'] == p2_id and row['P1_Win'] == 0) else 0, axis=1).mean()
        p1_pressure_points = p1_recent_games.apply(lambda row: row['P1 Pressure Points'] if row['Player 1 ID'] == p1_id else row['P2 Pressure Points'], axis=1).mean()
        p2_pressure_points = p2_recent_games.apply(lambda row: row['P1 Pressure Points'] if row['Player 1 ID'] == p2_id else row['P2 Pressure Points'], axis=1).mean()

        match_data = match.copy()
        match_data.update({
            'Win_Rate_Advantage': p1_rolling_win_rate - p2_rolling_win_rate,
            'Pressure_Points_Advantage': p1_pressure_points - p2_pressure_points,
            'Rest_Advantage': p1_rest_days - p2_rest_days,
            'H2H_P1_Win_Rate': h2h_p1_win_rate
        })
        new_match_features.append(match_data)
        
        # --- B. Build the Sequences for the LSTM ---
        p1_sequence_df = df_history[((df_history['Player 1 ID'] == p1_id) | (df_history['Player 2 ID'] == p1_id))].tail(SEQUENCE_LENGTH)
        p2_sequence_df = df_history[((df_history['Player 1 ID'] == p2_id) | (df_history['Player 2 ID'] == p2_id))].tail(SEQUENCE_LENGTH)

        p1_sequence, p2_sequence = [], []
        for idx, row in p1_sequence_df.iterrows():
            p1_perspective = np.concatenate([row[features_for_lstm_sequence[:3]].values, row[features_for_lstm_sequence[3:6]].values, [row['H2H_P1_Win_Rate']]])
            p1_sequence.append(p1_perspective)
        
        for idx, row in p2_sequence_df.iterrows():
            p2_perspective = np.concatenate([row[features_for_lstm_sequence[3:6]].values, row[features_for_lstm_sequence[:3]].values, [1 - row['H2H_P1_Win_Rate']]])
            p2_sequence.append(p2_perspective)

        X_p1_sequences.append(p1_sequence)
        X_p2_sequences.append(p2_sequence)

    df_predict = pd.DataFrame(new_match_features)

    # --- 4. Generate Predictions ---
    print("\n--- Generating Final Predictions ---")
    # A. GBM Predictions
    X_gbm = df_predict[['Win_Rate_Advantage', 'Pressure_Points_Advantage', 'Rest_Advantage', 'H2H_P1_Win_Rate', 'Player 1 ID', 'Player 2 ID']]
    X_gbm_processed = gbm_preprocessor.transform(X_gbm)
    gbm_preds = gbm_model.predict_proba(X_gbm_processed)[:, 1]

    # B. LSTM Predictions with real sequences
    X_p1 = np.array(X_p1_sequences)
    X_p2 = np.array(X_p2_sequences)
    nsamples, nsteps, nfeatures = X_p1.shape
    X_p1_scaled = lstm_scaler.transform(X_p1.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
    X_p2_scaled = lstm_scaler.transform(X_p2.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
    lstm_preds = lstm_model.predict([X_p1_scaled, X_p2_scaled]).flatten()

    # C. Meta-Model Final Prediction
    X_meta = np.vstack([gbm_preds, lstm_preds]).T
    final_probs = meta_model.predict_proba(X_meta)[:, 1]
    
    df_predict['P1_Win_Probability'] = final_probs

    # --- 5. Display Results ---
    for index, row in df_predict.iterrows():
        p1_name = row['Player 1 Name']
        p2_name = row['Player 2 Name']
        p1_prob = row['P1_Win_Probability']
        
        print("\n---------------------------------")
        print(f"Matchup: {p1_name} vs. {p2_name}")
        print(f"Predicted Win Probabilities:")
        print(f"  - {p1_name}: {p1_prob:.2%}")
        print(f"  - {p2_name}: {1-p1_prob:.2%}")
        print("---------------------------------")

except Exception as e:
    print(f"An error occurred: {e}")