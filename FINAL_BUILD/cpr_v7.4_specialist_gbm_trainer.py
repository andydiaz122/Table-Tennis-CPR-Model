import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib

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
    # Create new 'advantage' features from the rolling stats
    df['Win_Rate_Advantage'] = df['P1_Rolling_Win_Rate_L10'] - df['P2_Rolling_Win_Rate_L10']
#    df['Pressure_Points_Advantage'] = df['P1_Rolling_Pressure_Points_L10'] - df['P2_Rolling_Pressure_Points_L10']
#    df['Rest_Advantage'] = df['P1_Rest_Days'] - df['P2_Rest_Days']
    df['Set_Comebacks_Advantage'] = df['P1_Rolling_Set_Comebacks_L20'] - df['P2_Rolling_Set_Comebacks_L20']

    # Define feature types
    # CORRECTED: Removed categorical_features list entirely.
#    numerical_features = ['Win_Rate_Advantage', 'Pressure_Points_Advantage', 'Rest_Advantage']
    # Updated feature list with new v7.4 features
    numerical_features = [
        # Original features
        'Time_Since_Last_Advantage', 'Matches_Last_24H_Advantage', 'Is_First_Match_Advantage',
        'PDR_Slope_Advantage', 'H2H_P1_Win_Rate', 'H2H_Dominance_Score', 'PDR_Advantage',
        'Win_Rate_Advantage', 'Win_Rate_L5_Advantage', 'Close_Set_Win_Rate_Advantage', 'Set_Comebacks_Advantage',
        # New v7.4 features
        'Elo_Diff',                    # Elo rating difference
        'Glicko_Mu_Diff',              # Glicko-2 rating difference
        'Glicko_Phi_Sum',              # Combined uncertainty
        'Clutch_Factor_Diff',          # Deuce set win rate difference
        'Pythagorean_Delta_Diff',      # Luck factor difference
        'Fatigue_Factor_Diff',         # Exponential fatigue (replaces Daily_Fatigue_Advantage)
        'PDR_Variance_Diff',           # PDR consistency difference
        'H2H_Matches',                 # H2H sample size
    ]

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
    
    print(f"\n[SUCCESS] Successfully re-trained and saved the GBM model to '{MODEL_OUTPUT_FILE}'")

except FileNotFoundError:
    print(f"Error: The input file '{FEATURES_FILE}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")