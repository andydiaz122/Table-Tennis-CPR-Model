"""
Validation script for 2000-row sub-sample testing.
Compares optimized output against baseline with numerical tolerance.

Success Criteria:
- 100% logic parity on 2,000 rows
- Max absolute diff < 1e-8
- Execution time (2k rows) < 10 seconds

Focus Columns:
- H2H_Dominance_Score
- Close_Set_Win_Rate_Advantage
"""

import pandas as pd
import numpy as np
import subprocess
import time
import sys

# Paths
BASELINE_PATH = r"C:\Users\PC\Documents\Coding\Predictive Modeling\CPR_Model_Local_Build\V8.0\final_engineered_features_v7.4.csv"
OPTIMIZED_PATH = r"C:\Users\PC\Documents\Coding\Predictive Modeling\CPR_Model_Local_Build\V9.0-speed-bottleneck\V9.0\final_engineered_features_v7.4.csv"
FEATURE_SCRIPT = r"C:\Users\PC\Documents\Coding\Predictive Modeling\CPR_Model_Local_Build\V9.0-speed-bottleneck\V9.0\advanced_feature_engineering_v7.4.py"

# Focus columns for detailed analysis
FOCUS_COLUMNS = [
    'H2H_Dominance_Score',
    'Close_Set_Win_Rate_Advantage',
    'H2H_P1_Win_Rate',
    'PDR_Advantage',
    'P1_Rolling_Win_Rate_L10',
    'P2_Rolling_Win_Rate_L10',
    'Win_Rate_L5_Advantage'
]

TOLERANCE = 1e-8
TEST_ROWS = 2000


def run_optimized_script():
    """Run the optimized feature engineering script and time it."""
    print(f"\n{'='*60}")
    print("STEP 1: Running optimized feature engineering script...")
    print(f"{'='*60}")

    start_time = time.time()
    result = subprocess.run(
        ['python', FEATURE_SCRIPT],
        capture_output=True,
        text=True,
        cwd=r"C:\Users\PC\Documents\Coding\Predictive Modeling\CPR_Model_Local_Build\V9.0-speed-bottleneck\V9.0"
    )
    elapsed = time.time() - start_time

    print(f"Execution time: {elapsed:.2f} seconds")

    if result.returncode != 0:
        print("ERROR: Script failed!")
        print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        return None

    print("Script completed successfully!")
    return elapsed


def load_and_compare():
    """Load baseline and optimized outputs, compare with tolerance."""
    print(f"\n{'='*60}")
    print("STEP 2: Loading data for comparison...")
    print(f"{'='*60}")

    # Load baseline (first 2000 rows)
    print(f"Loading baseline from: {BASELINE_PATH}")
    baseline_df = pd.read_csv(BASELINE_PATH, nrows=TEST_ROWS)
    print(f"  Baseline shape: {baseline_df.shape}")

    # Load optimized output
    print(f"Loading optimized from: {OPTIMIZED_PATH}")
    optimized_df = pd.read_csv(OPTIMIZED_PATH)
    print(f"  Optimized shape: {optimized_df.shape}")

    # Verify row count
    if len(optimized_df) != TEST_ROWS:
        print(f"WARNING: Expected {TEST_ROWS} rows, got {len(optimized_df)}")

    return baseline_df, optimized_df


def compare_columns(baseline_df, optimized_df):
    """Compare columns with detailed analysis for focus columns."""
    print(f"\n{'='*60}")
    print("STEP 3: Column-by-column comparison...")
    print(f"{'='*60}")

    # Get numeric columns for comparison
    numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns.tolist()

    all_passed = True
    results = []

    for col in numeric_cols:
        if col not in optimized_df.columns:
            print(f"  MISSING: {col} not in optimized output")
            all_passed = False
            continue

        baseline_vals = baseline_df[col].values
        optimized_vals = optimized_df[col].values

        # Handle NaN values
        mask = ~(np.isnan(baseline_vals) | np.isnan(optimized_vals))
        if mask.sum() == 0:
            results.append((col, 0.0, "OK (all NaN)"))
            continue

        max_diff = np.max(np.abs(baseline_vals[mask] - optimized_vals[mask]))
        passed = max_diff <= TOLERANCE

        status = "OK" if passed else "FAIL"
        results.append((col, max_diff, status))

        if not passed:
            all_passed = False

    # Print results
    print(f"\n{'Column':<40} {'Max Diff':<20} {'Status':<10}")
    print("-" * 70)

    # Print focus columns first
    print("\nFOCUS COLUMNS:")
    for col, max_diff, status in results:
        if col in FOCUS_COLUMNS:
            emoji = "[OK]" if status == "OK" else "[FAIL]"
            print(f"  {col:<38} {max_diff:<20.2e} {emoji}")

    print("\nOTHER NUMERIC COLUMNS:")
    for col, max_diff, status in results:
        if col not in FOCUS_COLUMNS:
            emoji = "[OK]" if status == "OK" else "[FAIL]"
            print(f"  {col:<38} {max_diff:<20.2e} {emoji}")

    return all_passed


def run_assert_frame_equal(baseline_df, optimized_df):
    """Run pd.testing.assert_frame_equal for comprehensive comparison."""
    print(f"\n{'='*60}")
    print("STEP 4: Running assert_frame_equal (atol={TOLERANCE})...")
    print(f"{'='*60}")

    # Select only numeric columns for comparison
    numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns.tolist()
    common_cols = [c for c in numeric_cols if c in optimized_df.columns]

    baseline_numeric = baseline_df[common_cols].astype(float)
    optimized_numeric = optimized_df[common_cols].astype(float)

    try:
        pd.testing.assert_frame_equal(
            baseline_numeric,
            optimized_numeric,
            atol=TOLERANCE,
            rtol=0,
            check_exact=False,
            check_dtype=False
        )
        print("[OK] assert_frame_equal PASSED!")
        return True
    except AssertionError as e:
        print("[FAIL] assert_frame_equal FAILED!")
        print(f"Error: {str(e)[:500]}...")
        return False


def main():
    print("\n" + "="*70)
    print("   2000-ROW VALIDATION TEST")
    print("   Comparing optimized output against baseline")
    print("="*70)

    # Step 1: Run optimized script
    elapsed = run_optimized_script()
    if elapsed is None:
        print("\n[FAIL] VALIDATION FAILED - Script error")
        return 1

    # Step 2: Load data
    baseline_df, optimized_df = load_and_compare()

    # Step 3: Column comparison
    columns_passed = compare_columns(baseline_df, optimized_df)

    # Step 4: Assert frame equal
    assert_passed = run_assert_frame_equal(baseline_df, optimized_df)

    # Final summary
    print(f"\n{'='*70}")
    print("   VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Execution time:    {elapsed:.2f}s {'[OK]' if elapsed < 10 else '[SLOW]'} (target: <10s)")
    print(f"  Column comparison: {'[OK]' if columns_passed else '[FAIL]'}")
    print(f"  Frame equality:    {'[OK]' if assert_passed else '[FAIL]'}")
    print(f"  Row count:         {len(optimized_df)} / {TEST_ROWS}")

    overall_passed = columns_passed and assert_passed and elapsed < 10

    print(f"\n  OVERALL: {'[OK] PASSED - Ready for full run' if overall_passed else '[FAIL] FAILED - Debug needed'}")
    print("="*70)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
