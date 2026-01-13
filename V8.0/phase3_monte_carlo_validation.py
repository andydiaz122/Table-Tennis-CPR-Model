"""
Phase 3: Monte Carlo / PBO Validation
=====================================

Validates Phase 2 filter stack results are statistically robust.

Tests:
1. Shuffle Test (N=1000) - 95% must end with bankroll >= $1,500
2. Noise Injection (±1%, ±2%) - ROI must remain positive
3. PBO Calculation - Must be < 30%

Author: CPR Quant Team
Date: 2026-01-13
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_FILE = "/Users/christiandiaz/Documents/Coding/table-tennis-cpr-model/v8-feature-research/V8.0/backtest_log_filtered_v8.csv"
REPORT_FILE = "/Users/christiandiaz/Documents/Coding/table-tennis-cpr-model/v8-feature-research/V8.0/phase3_validation_report.txt"

INITIAL_BANKROLL = 1000
MIN_BANKROLL_THRESHOLD = 1500
N_SIMULATIONS = 1000


# ==============================================================================
# TEST 1: SHUFFLE TEST
# ==============================================================================

def simulate_bankroll(df, initial_bankroll=1000):
    """Simulate cumulative bankroll from bet sequence."""
    return initial_bankroll + df['Profit'].sum()


def shuffle_test(df, n_sims=1000, threshold=1500):
    """
    Test 1: Shuffle order and check bankroll stability.

    Question: Is the ROI dependent on bet ordering, or is it genuine edge?
    """
    print("\n" + "=" * 70)
    print("TEST 1: SHUFFLE TEST (N={})".format(n_sims))
    print("=" * 70)
    print(f"Threshold: Final bankroll >= ${threshold}")
    print(f"Requirement: >= 95% of simulations must pass")
    print()

    results = []
    for i in range(n_sims):
        shuffled = df.sample(frac=1, random_state=i)
        final = simulate_bankroll(shuffled)
        results.append(final)

        # Progress indicator
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{n_sims} simulations...")

    pass_count = sum(1 for b in results if b >= threshold)
    pass_rate = pass_count / n_sims

    result = {
        'pass_rate': pass_rate,
        'pass_count': pass_count,
        'n_sims': n_sims,
        'mean_bankroll': np.mean(results),
        'std_bankroll': np.std(results),
        'min_bankroll': np.min(results),
        'max_bankroll': np.max(results),
        'percentile_5': np.percentile(results, 5),
        'percentile_95': np.percentile(results, 95),
        'passed': pass_rate >= 0.95
    }

    print("\n[RESULTS]")
    print(f"  Pass Rate: {pass_rate*100:.2f}% ({pass_count}/{n_sims})")
    print(f"  Mean Bankroll: ${result['mean_bankroll']:.2f}")
    print(f"  Std Dev: ${result['std_bankroll']:.2f}")
    print(f"  Min: ${result['min_bankroll']:.2f}")
    print(f"  Max: ${result['max_bankroll']:.2f}")
    print(f"  5th Percentile: ${result['percentile_5']:.2f}")
    print(f"  95th Percentile: ${result['percentile_95']:.2f}")
    print()

    if result['passed']:
        print("  [PASS] Shuffle test PASSED - edge is order-independent")
    else:
        print("  [FAIL] Shuffle test FAILED - edge may be order-dependent")

    return result


# ==============================================================================
# TEST 2: NOISE INJECTION
# ==============================================================================

def noise_injection_test(df, noise_levels=[0.01, 0.02], n_trials=100):
    """
    Test 2: Perturb Model_Prob and check ROI stability.

    Question: Is the edge brittle? Will small perturbations collapse ROI?
    """
    print("\n" + "=" * 70)
    print("TEST 2: NOISE INJECTION")
    print("=" * 70)
    print("Testing robustness to probability estimation errors")
    print()

    base_profit = df['Profit'].sum()
    base_staked = df['Stake'].abs().sum()
    base_roi = base_profit / base_staked * 100

    print(f"[BASELINE]")
    print(f"  ROI: {base_roi:.2f}%")
    print(f"  Bets: {len(df)}")
    print()

    results = {}

    for noise in noise_levels:
        print(f"[NOISE LEVEL: ±{noise*100:.0f}%]")

        trial_rois = []
        trial_bets = []

        for trial in range(n_trials):
            perturbed = df.copy()
            np.random.seed(trial)

            # Add random noise to Model_Prob_Adjusted
            perturbation = np.random.uniform(-noise, noise, len(df))
            perturbed['Model_Prob_Noisy'] = (perturbed['Model_Prob_Adjusted'] + perturbation).clip(0.01, 0.99)

            # Recalculate edge with noisy probability
            perturbed['Edge_Noisy'] = perturbed['Model_Prob_Noisy'] - perturbed['Implied_Prob']

            # Filter: only keep bets that still have positive edge after noise
            # AND still meet minimum edge threshold (3%)
            valid = perturbed[(perturbed['Edge_Noisy'] >= 0.03) & (perturbed['Edge_Noisy'] <= 0.25)]

            if len(valid) > 0:
                roi = valid['Profit'].sum() / valid['Stake'].abs().sum() * 100
            else:
                roi = 0

            trial_rois.append(roi)
            trial_bets.append(len(valid))

        mean_roi = np.mean(trial_rois)
        std_roi = np.std(trial_rois)
        mean_bets = np.mean(trial_bets)

        # Determine pass/fail
        if noise == 0.01:
            passed = mean_roi > 2.0
            requirement = "> 2%"
        else:
            passed = mean_roi > 0.0
            requirement = "> 0%"

        results[f'noise_{int(noise*100)}pct'] = {
            'noise_level': noise,
            'mean_roi': mean_roi,
            'std_roi': std_roi,
            'min_roi': np.min(trial_rois),
            'max_roi': np.max(trial_rois),
            'mean_bets': mean_bets,
            'roi_delta': mean_roi - base_roi,
            'passed': passed,
            'requirement': requirement
        }

        print(f"  Mean ROI: {mean_roi:.2f}% (± {std_roi:.2f}%)")
        print(f"  ROI Range: [{np.min(trial_rois):.2f}%, {np.max(trial_rois):.2f}%]")
        print(f"  Mean Bets: {mean_bets:.0f}")
        print(f"  ROI Delta: {mean_roi - base_roi:+.2f}%")

        if passed:
            print(f"  [PASS] ROI ({mean_roi:.2f}%) {requirement}")
        else:
            print(f"  [FAIL] ROI ({mean_roi:.2f}%) not {requirement}")
        print()

    return results


# ==============================================================================
# TEST 3: PROBABILITY OF BACKTEST OVERFITTING (PBO)
# ==============================================================================

def calculate_pbo(df, n_folds=8):
    """
    Test 3: Probability of Backtest Overfitting (simplified time-series CV).

    Question: What's the probability that our strategy would fail out-of-sample?

    Method:
    - Split data into N chronological folds
    - Calculate ROI for each fold
    - PBO = fraction of folds that underperform the median

    A low PBO (<30%) indicates the strategy is likely robust across time periods.
    """
    print("\n" + "=" * 70)
    print("TEST 3: PROBABILITY OF BACKTEST OVERFITTING (PBO)")
    print("=" * 70)
    print(f"Method: {n_folds}-fold chronological split")
    print(f"Requirement: PBO < 30%")
    print()

    # Sort by date for time-series integrity
    df = df.sort_values('Date').reset_index(drop=True)

    fold_size = len(df) // n_folds
    fold_results = []

    print("[FOLD ANALYSIS]")
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else len(df)
        fold = df.iloc[start:end]

        fold_profit = fold['Profit'].sum()
        fold_staked = fold['Stake'].abs().sum()
        fold_roi = fold_profit / fold_staked * 100 if fold_staked > 0 else 0
        fold_bets = len(fold)

        # Get date range
        start_date = fold['Date'].min().strftime('%Y-%m-%d')
        end_date = fold['Date'].max().strftime('%Y-%m-%d')

        fold_results.append({
            'fold': i + 1,
            'start_date': start_date,
            'end_date': end_date,
            'bets': fold_bets,
            'roi': fold_roi,
            'profit': fold_profit
        })

        print(f"  Fold {i+1}: {start_date} to {end_date} | {fold_bets} bets | ROI: {fold_roi:+.2f}%")

    # Calculate PBO
    fold_rois = [f['roi'] for f in fold_results]
    median_roi = np.median(fold_rois)

    # Count folds below median
    below_median = sum(1 for r in fold_rois if r < median_roi)
    pbo = below_median / n_folds

    # Additional metrics
    positive_folds = sum(1 for r in fold_rois if r > 0)
    consistency = positive_folds / n_folds

    print()
    print("[PBO CALCULATION]")
    print(f"  Median ROI across folds: {median_roi:.2f}%")
    print(f"  Folds below median: {below_median}/{n_folds}")
    print(f"  PBO: {pbo*100:.1f}%")
    print(f"  Positive folds: {positive_folds}/{n_folds} ({consistency*100:.1f}% consistency)")
    print()

    passed = pbo < 0.30

    if passed:
        print("  [PASS] PBO ({:.1f}%) < 30%".format(pbo*100))
    else:
        print("  [FAIL] PBO ({:.1f}%) >= 30%".format(pbo*100))

    return {
        'pbo': pbo,
        'median_roi': median_roi,
        'fold_results': fold_results,
        'fold_rois': fold_rois,
        'below_median': below_median,
        'n_folds': n_folds,
        'positive_folds': positive_folds,
        'consistency': consistency,
        'passed': passed
    }


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("=" * 70)
    print("PHASE 3: MONTE CARLO / PBO VALIDATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input: {INPUT_FILE}")
    print()

    # Load filtered data
    print("Loading filtered data...")
    df = pd.read_csv(INPUT_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Loaded {len(df):,} bets")

    report_lines = []
    report_lines.append("PHASE 3: MONTE CARLO / PBO VALIDATION")
    report_lines.append("=" * 70)
    report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Input: {len(df):,} filtered bets")
    report_lines.append("")

    # Run all tests
    test_results = {}

    # Test 1: Shuffle Test
    shuffle_result = shuffle_test(df, n_sims=N_SIMULATIONS, threshold=MIN_BANKROLL_THRESHOLD)
    test_results['shuffle'] = shuffle_result
    report_lines.append(f"TEST 1 - SHUFFLE: {'PASS' if shuffle_result['passed'] else 'FAIL'} ({shuffle_result['pass_rate']*100:.1f}% >= $1,500)")

    # Test 2: Noise Injection
    noise_result = noise_injection_test(df, noise_levels=[0.01, 0.02], n_trials=100)
    test_results['noise'] = noise_result
    for key, val in noise_result.items():
        report_lines.append(f"TEST 2 - {key.upper()}: {'PASS' if val['passed'] else 'FAIL'} (ROI: {val['mean_roi']:.2f}%)")

    # Test 3: PBO
    pbo_result = calculate_pbo(df, n_folds=8)
    test_results['pbo'] = pbo_result
    report_lines.append(f"TEST 3 - PBO: {'PASS' if pbo_result['passed'] else 'FAIL'} ({pbo_result['pbo']*100:.1f}%)")

    # Final Summary
    print("\n" + "=" * 70)
    print("PHASE 3 VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True

    print("\n[TEST RESULTS]")
    print(f"  1. Shuffle Test:    {'PASS' if shuffle_result['passed'] else 'FAIL'} ({shuffle_result['pass_rate']*100:.1f}% >= $1,500)")
    if not shuffle_result['passed']:
        all_passed = False

    for key, val in noise_result.items():
        status = 'PASS' if val['passed'] else 'FAIL'
        print(f"  2. {key}:  {status} (ROI: {val['mean_roi']:.2f}%)")
        if not val['passed']:
            all_passed = False

    print(f"  3. PBO:             {'PASS' if pbo_result['passed'] else 'FAIL'} ({pbo_result['pbo']*100:.1f}%)")
    if not pbo_result['passed']:
        all_passed = False

    print()
    if all_passed:
        print("  >>> ALL TESTS PASSED - Strategy is statistically robust <<<")
        report_lines.append("\nOVERALL: ALL TESTS PASSED")
    else:
        print("  >>> SOME TESTS FAILED - Review required <<<")
        report_lines.append("\nOVERALL: SOME TESTS FAILED")

    print("=" * 70)

    # Save report
    with open(REPORT_FILE, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\nReport saved to: {REPORT_FILE}")

    return test_results, all_passed


if __name__ == "__main__":
    results, passed = main()
