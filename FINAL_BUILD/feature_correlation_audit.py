"""
Feature Correlation Audit for CPR Model v9.1

Identifies highly correlated features (|r| > 0.7) that may be candidates for removal.
This helps reduce filter bloat and potential multicollinearity issues.
"""
import pandas as pd
import numpy as np

# --- Configuration ---
TRAINING_FILE = "training_dataset.csv"
CORRELATION_THRESHOLD = 0.7

# 19 GBM features from cpr_v7.4_specialist_gbm_trainer.py
GBM_FEATURES = [
    # Original features
    'Time_Since_Last_Advantage', 'Matches_Last_24H_Advantage', 'Is_First_Match_Advantage',
    'PDR_Slope_Advantage', 'H2H_P1_Win_Rate', 'H2H_Dominance_Score', 'PDR_Advantage',
    'Win_Rate_Advantage', 'Win_Rate_L5_Advantage', 'Close_Set_Win_Rate_Advantage', 'Set_Comebacks_Advantage',
    # New v7.4 features
    'Elo_Diff',
    'Glicko_Mu_Diff',
    'Glicko_Phi_Sum',
    'Clutch_Factor_Diff',
    'Pythagorean_Delta_Diff',
    'Fatigue_Factor_Diff',
    'PDR_Variance_Diff',
    'H2H_Matches',
]

def main():
    print("--- Feature Correlation Audit v9.1 ---\n")

    try:
        df = pd.read_csv(TRAINING_FILE)
        print(f"Loaded {len(df)} rows from {TRAINING_FILE}")
    except FileNotFoundError:
        print(f"Error: {TRAINING_FILE} not found. Run the pipeline first.")
        return

    # Filter to only existing GBM features
    existing_features = [f for f in GBM_FEATURES if f in df.columns]
    missing_features = [f for f in GBM_FEATURES if f not in df.columns]

    if missing_features:
        print(f"\nWarning: Missing features: {missing_features}")

    print(f"\nAnalyzing {len(existing_features)} features...\n")

    # Compute correlation matrix
    corr_matrix = df[existing_features].corr()

    # Identify highly correlated pairs
    high_corr_pairs = []
    for i, col1 in enumerate(existing_features):
        for j, col2 in enumerate(existing_features):
            if i < j:  # Only check upper triangle
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > CORRELATION_THRESHOLD:
                    high_corr_pairs.append((col1, col2, corr))

    # Sort by absolute correlation
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # Print results
    print(f"=== Highly Correlated Feature Pairs (|r| > {CORRELATION_THRESHOLD}) ===\n")

    if not high_corr_pairs:
        print("No highly correlated feature pairs found.")
    else:
        print(f"{'Feature 1':<35} {'Feature 2':<35} {'Correlation':>12}")
        print("-" * 85)
        for col1, col2, corr in high_corr_pairs:
            print(f"{col1:<35} {col2:<35} {corr:>12.4f}")

    print(f"\n=== Correlation Summary Statistics ===\n")

    # Get all pairwise correlations (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    all_corrs = corr_matrix.where(mask).stack()

    print(f"Mean absolute correlation: {all_corrs.abs().mean():.4f}")
    print(f"Max absolute correlation:  {all_corrs.abs().max():.4f}")
    print(f"Pairs with |r| > 0.5:      {(all_corrs.abs() > 0.5).sum()}")
    print(f"Pairs with |r| > 0.7:      {(all_corrs.abs() > 0.7).sum()}")
    print(f"Pairs with |r| > 0.9:      {(all_corrs.abs() > 0.9).sum()}")

    # Feature variance check (near-constant features)
    print(f"\n=== Feature Variance Check (potential near-constant features) ===\n")
    variances = df[existing_features].var()
    low_var_features = variances[variances < 0.001].sort_values()

    if len(low_var_features) > 0:
        print(f"{'Feature':<40} {'Variance':>12}")
        print("-" * 55)
        for feat, var in low_var_features.items():
            print(f"{feat:<40} {var:>12.6f}")
    else:
        print("No low-variance features found.")

    # Recommendations
    print(f"\n=== Recommendations ===\n")

    candidates_for_removal = set()

    # From correlation pairs, recommend removing one from each pair
    for col1, col2, corr in high_corr_pairs:
        # Prefer keeping the simpler/more interpretable feature
        if 'Elo' in col1 and 'Glicko' in col2:
            candidates_for_removal.add(col1)  # Keep Glicko over Elo
            print(f"- Consider removing {col1} (r={corr:.4f} with {col2})")
        elif 'Elo' in col2 and 'Glicko' in col1:
            candidates_for_removal.add(col2)
            print(f"- Consider removing {col2} (r={corr:.4f} with {col1})")
        elif 'Variance' in col1:
            candidates_for_removal.add(col1)
            print(f"- Consider removing {col1} (r={corr:.4f} with {col2})")
        elif 'Variance' in col2:
            candidates_for_removal.add(col2)
            print(f"- Consider removing {col2} (r={corr:.4f} with {col1})")
        else:
            print(f"- High correlation: {col1} <-> {col2} (r={corr:.4f})")

    # Add low-variance features
    for feat in low_var_features.index:
        if feat not in candidates_for_removal:
            candidates_for_removal.add(feat)
            print(f"- Consider removing {feat} (low variance: {low_var_features[feat]:.6f})")

    if candidates_for_removal:
        print(f"\nTotal features recommended for removal: {len(candidates_for_removal)}")
        print(f"Candidates: {sorted(candidates_for_removal)}")
    else:
        print("\nNo features recommended for removal.")

    # Save correlation matrix to CSV
    corr_matrix.to_csv('feature_correlation_matrix.csv')
    print(f"\nCorrelation matrix saved to 'feature_correlation_matrix.csv'")

if __name__ == "__main__":
    main()
