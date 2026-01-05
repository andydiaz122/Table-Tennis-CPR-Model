import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Training CPR v6.1 'Situational Analyst' (GBM)...")

# --- Configuration ---
FEATURES_FILE = 'full_feature_database_v6.1.csv'
RAW_DATA_FILE = 'czech_liga_pro_advanced_stats_FIXED.csv'
MODEL_OUTPUT_FILE = 'cpr_v6.1_gbm_specialist.json'

# --- Load Data ---
try:
    features_db = pd.read_csv(FEATURES_FILE).set_index('Player')
    matches_df = pd.read_csv(RAW_DATA_FILE)
    print("Successfully loaded feature database and raw match data.")
except FileNotFoundError:
    print(f"FATAL ERROR: Make sure '{FEATURES_FILE}' and '{RAW_DATA_FILE}' exist first.")
    exit()

# --- Prepare Training Data ---
matches_df.dropna(subset=['Final Score'], inplace=True)
training_data = []

for _, row in matches_df.iterrows():
    p1 = row['Player 1']
    p2 = row['Player 2']
    
    if p1 in features_db.index and p2 in features_db.index:
        p1_features = features_db.loc[p1]
        p2_features = features_db.loc[p2]
        
        # Create feature vector (difference between player features)
        feature_diff = p1_features - p2_features
        
        p1_sets, p2_sets = map(int, row['Final Score'].replace('"', '').replace('=', '').split('-'))
        winner = 0 if p1_sets > p2_sets else 1 # 0 for P1 win, 1 for P2 win
        
        training_data.append({**feature_diff, 'winner': winner})

train_df = pd.DataFrame(training_data)
X = train_df.drop('winner', axis=1)
y = train_df['winner']

# Using a simple train/test split for this script's validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train XGBoost Model ---
gbm = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    use_label_encoder=False
)

gbm.fit(X_train, y_train)

# --- Validate and Save ---
preds = gbm.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"GBM Specialist validation accuracy: {accuracy * 100:.2f}%")

gbm.save_model(MODEL_OUTPUT_FILE)
print(f"GBM Specialist model trained and saved to '{MODEL_OUTPUT_FILE}'")
