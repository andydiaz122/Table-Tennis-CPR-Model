import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib
from feature_config import get_active_features, get_experiment_name, get_feature_count

# --- 1. Configuration ---
FEATURES_FILE = "training_dataset.csv"
MODEL_OUTPUT_FILE = "cpr_v7.4_gbm_specialist.joblib"
PREPROCESSOR_FILE = "gbm_preprocessor_v7.4.joblib"

# --- 2. Main Script Logic ---
try:
    print(f"Loading feature data from '{FEATURES_FILE}'...")
    df = pd.read_csv(FEATURES_FILE)
    df.dropna(inplace=True)
    print(f"Loaded {len(df)} matches for GBM re-training.")

    # --- 3. Feature Engineering ---
    # Create Set_Comebacks_Advantage from rolling stats
    # (Win_Rate_Advantage removed - L5 hot streak captures signal better than L20 form)
    df['Set_Comebacks_Advantage'] = df['P1_Rolling_Set_Comebacks_L20'] - df['P2_Rolling_Set_Comebacks_L20']

    # Define feature types - loaded from feature_config.py for systematic testing
    # CORRECTED: Removed categorical_features list entirely.
    numerical_features = get_active_features()

    print(f"=" * 60)
    print(f"EXPERIMENT: {get_experiment_name()}")
    print(f"Training with {get_feature_count()} features:")
    for f in numerical_features:
        print(f"  - {f}")
    print(f"=" * 60)

    # CORRECTED: The preprocessor now ONLY handles numerical features.
    # The ('cat', OneHotEncoder...) transformer has been removed.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features)
        ])

    # CORRECTED: The feature matrix X now ONLY contains the symmetrical numerical features.
    X = df[numerical_features]
    y = df['P1_Win']

    print("Preprocessing data with symmetrical features...")
    X_processed = preprocessor.fit_transform(X)

    # --- 4. Hyperparameter Tuning ---
    print("\nStarting GridSearchCV to find the best model parameters...")
    # NOTE: Using your updated hyperparameter grid
    param_grid = {
        'n_estimators': [100],
        'learning_rate': [0.01, 0.05],
        'max_depth': [2],
        'min_samples_leaf': [40, 50],
        'subsample': [0.7, 0.8]
    }

    gbm = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_processed, y)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    best_gbm_model = grid_search.best_estimator_

    # --- 5. Save the Final Model ---
    joblib.dump(best_gbm_model, MODEL_OUTPUT_FILE)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    
    print(f"\n[OK] Successfully re-trained and saved the GBM model to '{MODEL_OUTPUT_FILE}'")

except FileNotFoundError:
    print(f"Error: The input file '{FEATURES_FILE}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")