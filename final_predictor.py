import pandas as pd
import numpy as np
import xgboost as xgb
from tensorflow.keras.models import load_model
import joblib

print("--- CPR Model v6.1 Daily Predictor ---")

# --- Configuration ---
FEATURES_FILE = 'full_feature_database_v6.1.csv'
GBM_MODEL_FILE = 'cpr_v6.1_gbm_specialist.json'
LSTM_MODEL_FILE = 'cpr_v6.1_lstm_specialist.h5'
META_MODEL_FILE = 'cpr_v6.1_meta_model.pkl'

# --- MANUALLY EDIT THIS SECTION FOR DAILY MATCHES ---
upcoming_matches = [
    #{'PlayerA': 'Kolenic Tibor', 'PlayerB': 'Briza Frantisek'},
    #{'PlayerA': 'Mares Erik', 'PlayerB': 'Macela Petr'},
    # Add other matches here
    {'PlayerA': 'David Heczko', 'PlayerB': 'Martin Stefek'},
    {'PlayerA': 'Petr Bradach', 'PlayerB': 'Vratislav Petracek'},
    {'PlayerA': 'Lukas Tonar', 'PlayerB': 'Marek Chlebecek'},
    {'PlayerA': 'Michal Vedmoch', 'PlayerB': 'Pavel Vondra'},
    {'PlayerA': 'Milan Fisera', 'PlayerB': 'Michal Vedmoch'},
    {'PlayerA': 'Simon Kadavy', 'PlayerB': 'Lukas Tonar'},
    {'PlayerA': 'Lubor Sulava', 'PlayerB': 'Tomas Regner'},
    {'PlayerA': 'Michal Vavrecka', 'PlayerB': 'Petr Bradach'},
    {'PlayerA': 'Vitezslav Bosak', 'PlayerB': 'David Heczko'},
    {'PlayerA': 'Martin Stefek', 'PlayerB': 'Radomir Vidlicka'},
    {'PlayerA': 'Pavel Vondra', 'PlayerB': 'Jiri Dedek'},
    {'PlayerA': 'Marek Chlebecek', 'PlayerB': 'Karel Kapras'},
]
# ---------------------------------------------------


# --- Load Models and Data ---
try:
    gbm = xgb.XGBClassifier()
    gbm.load_model(GBM_MODEL_FILE)
    lstm = load_model(LSTM_MODEL_FILE)
    meta_model = joblib.load(META_MODEL_FILE)
    features_db = pd.read_csv(FEATURES_FILE).set_index('Player')
    print("All models and data loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load model files. Make sure all trainers have been run. Error: {e}")
    exit()

# --- Prediction Loop ---
print("\n--- Match Predictions ---")
for match in upcoming_matches:
    p1 = match['PlayerA']
    p2 = match['PlayerB']
    
    if p1 not in features_db.index or p2 not in features_db.index:
        print(f"Skipping {p1} vs {p2}: One or both players not in feature database.")
        continue
        
    # 1. Get Specialist Predictions
    # GBM Prediction
    feature_diff = features_db.loc[p1] - features_db.loc[p2]
    gbm_win_prob = gbm.predict_proba(pd.DataFrame([feature_diff]))[0][0]

    # LSTM Prediction (Simplified placeholder)
    # A real implementation would fetch the last 15 performance scores for each player
    lstm_momentum_p1 = np.random.uniform(0.4, 0.6) # Placeholder
    
    # 2. Get Meta-Model Final Prediction
    specialist_predictions = np.array([[gbm_win_prob, lstm_momentum_p1]])
    final_win_prob = meta_model.predict_proba(specialist_predictions)[0][0]
    
    print(f"\nMatch: {p1} vs. {p2}")
    print(f"  -> GBM Specialist (Situational): {p1} Win Prob = {gbm_win_prob:.2%}")
    print(f"  -> LSTM Specialist (Momentum): {p1} Momentum Score = {lstm_momentum_p1:.2f}")
    print(f"  --> FINAL ENSEMBLE PREDICTION: {p1} Win Prob = {final_win_prob:.2%}")
