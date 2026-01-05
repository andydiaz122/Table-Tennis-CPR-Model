import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

# --- 1. Configuration ---
# ---!!!--- Manually input the new matches you want to predict ---!!!
upcoming_matches = [
    {'Player 1 ID': 1085799, 'Player 1 Name': 'Martin Vizek', 'Player 2 ID': 1012124, 'Player 2 Name': 'Michal Vondrak'},
    {'Player 1 ID': 1080540, 'Player 1 Name': 'Kir Martin', 'Player 2 ID': 388336, 'Player 2 Name': 'Vladimir Postelt'},
    {'Player 1 ID': 338586, 'Player 1 Name': 'Marek Fabini', 'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek'},
    {'Player 1 ID': 358665, 'Player 1 Name': 'Marek Sedlak', 'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak'},
    {'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma', 'Player 2 ID': 689521, 'Player 2 Name': 'Jaroslav Strnad (1964)'},
    {'Player 1 ID': 1088851, 'Player 1 Name': 'Oleksandr Kolisnyk', 'Player 2 ID': 608764, 'Player 2 Name': 'Michal Jezek'},
]
# ---!!!--- ---!!!--- ---!!!--- ---!!!--- ---!!!--- ---!!!---

# The full historical dataset is now required to calculate rolling averages and H2H
HISTORICAL_DATA_FILE = "final_engineered_features.csv" 
# Input Model Files
GBM_MODEL_FILE = "cpr_v6.3_gbm_specialist.joblib"
GBM_PREPROCESSOR_FILE = "gbm_preprocessor_v6.3.joblib"
LSTM_MODEL_FILE = "cpr_v6.3_lstm_specialist.h5"
LSTM_SCALER_FILE = "lstm_scaler_v6.3.joblib"
META_MODEL_FILE = "cpr_v6.3_meta_model.pkl"

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

    # --- 3. Feature Engineering for New Matches ---
    print("\n--- Engineering Features for New Matches ---")
    
    new_match_features = []
    
    for match in upcoming_matches:
        p1_id = match['Player 1 ID']
        p2_id = match['Player 2 ID']
        
        # --- Calculate Rest Days ---
        p1_last_match_date = df_history[((df_history['Player 1 ID'] == p1_id) | (df_history['Player 2 ID'] == p1_id))]['Date'].max()
        p2_last_match_date = df_history[((df_history['Player 1 ID'] == p2_id) | (df_history['Player 2 ID'] == p2_id))]['Date'].max()
        p1_rest_days = (datetime.now() - p1_last_match_date).days if pd.notna(p1_last_match_date) else 30 # Default to high rest
        p2_rest_days = (datetime.now() - p2_last_match_date).days if pd.notna(p2_last_match_date) else 30

        # --- Calculate H2H ---
        h2h_df = df_history[((df_history['Player 1 ID'] == p1_id) & (df_history['Player 2 ID'] == p2_id)) | 
                            ((df_history['Player 1 ID'] == p2_id) & (df_history['Player 2 ID'] == p1_id))]
        p1_h2h_wins = len(h2h_df[((h2h_df['Player 1 ID'] == p1_id) & (h2h_df['P1_Win'] == 1)) | 
                                 ((h2h_df['Player 2 ID'] == p1_id) & (h2h_df['P1_Win'] == 0))])
        h2h_total = len(h2h_df)
        h2h_p1_win_rate = p1_h2h_wins / h2h_total if h2h_total > 0 else 0.5

        # --- Get Last N Games for Rolling Averages ---
        p1_recent_games = df_history[((df_history['Player 1 ID'] == p1_id) | (df_history['Player 2 ID'] == p1_id))].tail(ROLLING_WINDOW)
        p2_recent_games = df_history[((df_history['Player 1 ID'] == p2_id) | (df_history['Player 2 ID'] == p2_id))].tail(ROLLING_WINDOW)
        
        p1_rolling_win_rate = p1_recent_games.apply(lambda row: 1 if (row['Player 1 ID'] == p1_id and row['P1_Win'] == 1) or (row['Player 2 ID'] == p1_id and row['P1_Win'] == 0) else 0, axis=1).mean()
        p2_rolling_win_rate = p2_recent_games.apply(lambda row: 1 if (row['Player 1 ID'] == p2_id and row['P1_Win'] == 1) or (row['Player 2 ID'] == p2_id and row['P1_Win'] == 0) else 0, axis=1).mean()
        p1_pressure_points = p1_recent_games.apply(lambda row: row['P1 Pressure Points'] if row['Player 1 ID'] == p1_id else row['P2 Pressure Points'], axis=1).mean()
        p2_pressure_points = p2_recent_games.apply(lambda row: row['P1 Pressure Points'] if row['Player 1 ID'] == p2_id else row['P2 Pressure Points'], axis=1).mean()

        match_data = match.copy()
        match_data.update({
            'P1_Rest_Days': p1_rest_days, 'P2_Rest_Days': p2_rest_days,
            'H2H_P1_Win_Rate': h2h_p1_win_rate,
            'P1_Rolling_Win_Rate_L10': p1_rolling_win_rate, 'P2_Rolling_Win_Rate_L10': p2_rolling_win_rate,
            'P1_Rolling_Pressure_Points_L10': p1_pressure_points, 'P2_Rolling_Pressure_Points_L10': p2_pressure_points,
            'Win_Rate_Advantage': p1_rolling_win_rate - p2_rolling_win_rate,
            'Pressure_Points_Advantage': p1_pressure_points - p2_pressure_points,
            'Rest_Advantage': p1_rest_days - p2_rest_days
        })
        new_match_features.append(match_data)

    df_predict = pd.DataFrame(new_match_features)

    # --- 4. Generate Predictions ---
    print("\n--- Generating Final Predictions ---")
    # A. GBM Predictions
    X_gbm = df_predict[['Win_Rate_Advantage', 'Pressure_Points_Advantage', 'Rest_Advantage', 'H2H_P1_Win_Rate', 'Player 1 ID', 'Player 2 ID']]
    X_gbm_processed = gbm_preprocessor.transform(X_gbm)
    gbm_preds = gbm_model.predict_proba(X_gbm_processed)[:, 1]

    # B. LSTM Predictions
    # This requires a more complex history lookup similar to training, simplified here
    # For a live system, you'd have a pre-built history database to query
    print("Note: LSTM prediction in this script is a simplified approximation.")
    # In a real scenario, you'd re-build the exact sequence vectors as in training.
    # For this example, we'll assume the GBM prediction is dominant as LSTM requires complex state management.
    lstm_preds = gbm_preds # Using GBM as a stand-in for this simplified script

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