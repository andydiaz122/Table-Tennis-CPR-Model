import pandas as pd
import numpy as np
import xgboost as xgb
from tensorflow.keras.models import load_model
from sklearn.linear_model import LogisticRegression
import joblib

print("Training CPR v6.1 Meta-Model 'Manager'...")

# --- Configuration ---
FEATURES_FILE = 'full_feature_database_v6.1.csv'
RAW_DATA_FILE = 'czech_liga_pro_advanced_stats_FIXED.csv'
GBM_MODEL_FILE = 'cpr_v6.1_gbm_specialist.json'
LSTM_MODEL_FILE = 'cpr_v6.1_lstm_specialist.h5'
META_MODEL_OUTPUT_FILE = 'cpr_v6.1_meta_model.pkl'

# --- Load Models and Data ---
try:
    gbm = xgb.XGBClassifier()
    gbm.load_model(GBM_MODEL_FILE)
    lstm = load_model(LSTM_MODEL_FILE)
    features_db = pd.read_csv(FEATURES_FILE).set_index('Player')
    matches_df = pd.read_csv(RAW_DATA_FILE)
    print("Successfully loaded specialist models and data.")
except Exception as e:
    print(f"FATAL ERROR: Could not load required files. Make sure specialists are trained. Error: {e}")
    exit()

# --- Generate Specialist Predictions for Meta-Training ---
meta_X = []
meta_y = []

# This is a simplified process. A real one would use cross-validation.
for _, row in matches_df.head(5000).iterrows(): # Use a subset for speed
    p1 = row['Player 1']
    p2 = row['Player 2']
    
    if p1 in features_db.index and p2 in features_db.index:
        # GBM Prediction
        feature_diff = features_db.loc[p1] - features_db.loc[p2]
        gbm_pred_proba = gbm.predict_proba(pd.DataFrame([feature_diff]))[0][0] # P(P1 Win)

        # LSTM Prediction (Simplified - real one needs recent sequences)
        lstm_pred = np.random.rand() # Placeholder for actual LSTM prediction logic
        
        # Actual Result
        try:
            p1_sets, p2_sets = map(int, row['Final Score'].replace('"', '').replace('=', '').split('-'))
            winner = 0 if p1_sets > p2_sets else 1
            
            meta_X.append([gbm_pred_proba, lstm_pred])
            meta_y.append(winner)
        except:
            continue

# --- Train Meta-Model ---
meta_model = LogisticRegression()
meta_model.fit(meta_X, meta_y)

# --- Save Meta-Model ---
joblib.dump(meta_model, META_MODEL_OUTPUT_FILE)
print(f"Meta-Model 'Manager' trained and saved to '{META_MODEL_OUTPUT_FILE}'")
