import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib

# --- 1. Configuration ---
FEATURES_FILE = "training_features.csv"
MODEL_OUTPUT_FILE = "cpr_v6.2_gbm_specialist.joblib" # Using joblib for consistency
PREPROCESSOR_FILE = "gbm_preprocessor.joblib"

# --- 2. Main Script Logic ---
try:
    print(f"Loading feature data from '{FEATURES_FILE}'...")
    df = pd.read_csv(FEATURES_FILE)
    df.dropna(inplace=True) # Drop rows with any missing data
    print(f"Loaded {len(df)} matches for GBM training.")

    # --- 3. Feature Preparation ---
    # Define which columns are categorical (Player IDs) and numerical (Advantage stats)
    categorical_features = ['Player 1 ID', 'Player 2 ID']
    numerical_features = ['Total_Points_Advantage', 'Pressure_Points_Advantage', 'Comeback_Advantage']
    
    # Create a preprocessor to handle both feature types
    # OneHotEncoder handles non-numerical IDs, StandardScaler standardizes numerical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Separate features (X) from the target (y)
    X = df[numerical_features + categorical_features]
    y = df['P1_Win']

    # Apply the preprocessing steps to the data
    print("Preprocessing data...")
    X_processed = preprocessor.fit_transform(X)

    # --- 4. Hyperparameter Tuning with GridSearchCV ---
    print("\nStarting GridSearchCV to find the best model parameters...")
    # Define a grid of parameters to test
    param_grid = {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4]
    }

    # Initialize the GBM model
    gbm = GradientBoostingClassifier(random_state=42)
    # Initialize GridSearchCV to test all parameter combinations
    grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    
    # Run the search
    grid_search.fit(X_processed, y)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    best_gbm_model = grid_search.best_estimator_

    # --- 5. Save the Final Model ---
    joblib.dump(best_gbm_model, MODEL_OUTPUT_FILE)
    joblib.dump(preprocessor, PREPROCESSOR_FILE) # Also save the preprocessor
    
    print(f"\n✅ Successfully trained and saved the GBM model to '{MODEL_OUTPUT_FILE}'")
    print(f"✅ Saved the data preprocessor to '{PREPROCESSOR_FILE}'")

except FileNotFoundError:
    print(f"Error: The input file '{FEATURES_FILE}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")