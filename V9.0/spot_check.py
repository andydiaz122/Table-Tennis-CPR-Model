"""
Spot-check validation: Compare specific rows from optimized output vs baseline.
This runs the optimized script and compares rows 25000 and 35000.
"""
import pandas as pd
import subprocess
import time
import os

# Configuration
BASELINE_PATH = r"C:\Users\PC\Documents\Coding\Predictive Modeling\CPR_Model_Local_Build\V8.0\final_engineered_features_v7.4.csv"
OPTIMIZED_PATH = r"C:\Users\PC\Documents\Coding\Predictive Modeling\CPR_Model_Local_Build\V9.0-speed-bottleneck\V9.0\final_engineered_features_v7.4.csv"
FEATURE_SCRIPT = r"C:\Users\PC\Documents\Coding\Predictive Modeling\CPR_Model_Local_Build\V9.0-speed-bottleneck\V9.0\advanced_feature_engineering_v7.4.py"

# Rows to check (middle of dataset)
CHECK_ROWS = [25000, 35000]

# Key feature columns to compare
FEATURE_COLS = [
    'PDR_Advantage',
    'P1_Rolling_Win_Rate_L10',
    'P2_Rolling_Win_Rate_L10',
    'Win_Rate_L5_Advantage',
    'Close_Set_Win_Rate_Advantage',
    'H2H_P1_Win_Rate',
    'H2H_Dominance_Score',
    'Daily_Fatigue_Advantage',
    'Time_Since_Last_Advantage',
    'PDR_Slope_Advantage',
]

TOLERANCE = 1e-10

def main():
    print("="*70)
    print("SPOT-CHECK VALIDATION")
    print(f"Checking rows: {CHECK_ROWS}")
    print("="*70)

    # Step 1: Run optimized script
    print("\n[1] Running optimized feature engineering script...")
    print("    (This may take 30-40 minutes for full dataset)")

    start_time = time.time()
    result = subprocess.run(
        ['python', FEATURE_SCRIPT],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(FEATURE_SCRIPT)
    )
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n[FAIL] Script failed after {elapsed:.1f}s")
        print("STDERR:", result.stderr[-1000:])
        return 1

    print(f"    Completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

    # Step 2: Load both outputs
    print("\n[2] Loading outputs for comparison...")
    baseline_df = pd.read_csv(BASELINE_PATH)
    optimized_df = pd.read_csv(OPTIMIZED_PATH)

    print(f"    Baseline shape: {baseline_df.shape}")
    print(f"    Optimized shape: {optimized_df.shape}")

    if len(optimized_df) != len(baseline_df):
        print(f"\n[FAIL] Row count mismatch: {len(optimized_df)} vs {len(baseline_df)}")
        return 1

    # Step 3: Compare specific rows
    print("\n[3] Comparing specified rows...")
    all_match = True

    for row_idx in CHECK_ROWS:
        print(f"\n--- Row {row_idx} ---")
        baseline_row = baseline_df.iloc[row_idx]
        optimized_row = optimized_df.iloc[row_idx]

        row_match = True
        for col in FEATURE_COLS:
            base_val = baseline_row[col]
            opt_val = optimized_row[col]
            diff = abs(base_val - opt_val)

            if diff > TOLERANCE:
                print(f"  [FAIL] {col}")
                print(f"         Baseline:  {base_val}")
                print(f"         Optimized: {opt_val}")
                print(f"         Diff:      {diff}")
                row_match = False
                all_match = False
            else:
                print(f"  [OK] {col}: {opt_val:.10f}")

        if row_match:
            print(f"  >>> Row {row_idx}: ALL FEATURES MATCH")
        else:
            print(f"  >>> Row {row_idx}: MISMATCH DETECTED")

    # Final verdict
    print("\n" + "="*70)
    if all_match:
        print("[SUCCESS] SPOT-CHECK PASSED - All checked rows match baseline exactly!")
        print("          Safe to proceed with full pipeline validation.")
        return 0
    else:
        print("[FAIL] SPOT-CHECK FAILED - Optimizations changed calculation results!")
        print("       DO NOT proceed. Debug needed.")
        return 1

if __name__ == "__main__":
    exit(main())
