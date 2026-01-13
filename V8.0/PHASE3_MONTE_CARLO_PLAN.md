# Phase 3: Monte Carlo / PBO Validation

## Current State
- **Commit:** `4241a28` (Phase 2 complete)
- **Branch:** `feature-research-github-scrape`
- **Input:** `backtest_log_filtered_v8.csv` (2,056 bets)
- **Phase 2 Results:** ROI 4.33%, Sharpe 3.63, Max DD 25.26%, Final Bankroll $1,865

---

## Objective

Validate that Phase 2 filter stack results are **statistically robust** and not overfitted. Three tests required:

### Test 1: Shuffle Test (N=1000)
**Question:** Is the 4.33% ROI dependent on bet ordering, or is it genuine edge?

**Method:**
```python
for i in range(1000):
    shuffled_df = df.sample(frac=1, random_state=i)
    final_bankroll = simulate_bankroll(shuffled_df, initial=1000)
    results.append(final_bankroll)

pass_rate = sum(1 for b in results if b >= 1500) / 1000
# REQUIREMENT: pass_rate >= 0.95 (95% of simulations > $1,500)
```

**Success Criteria:** ≥95% of simulations end with bankroll ≥ $1,500

---

### Test 2: Noise Injection
**Question:** Is the edge brittle? Will small perturbations collapse ROI?

**Method:**
```python
for noise_level in [0.01, 0.02]:  # ±1%, ±2%
    perturbed_df = df.copy()
    perturbed_df['Model_Prob_Noisy'] = df['Model_Prob_Adjusted'] + np.random.uniform(-noise_level, noise_level, len(df))
    perturbed_df['Edge_Noisy'] = perturbed_df['Model_Prob_Noisy'] - perturbed_df['Implied_Prob']

    # Recalculate which bets still pass filters
    # Simulate and calculate ROI
```

**Success Criteria:**
- ±1% noise: ROI should remain > 2%
- ±2% noise: ROI should remain > 0% (still profitable)

---

### Test 3: Probability of Backtest Overfitting (PBO)
**Question:** What's the probability that our in-sample optimization would fail out-of-sample?

**Method (Combinatorially Symmetric Cross-Validation):**
```python
# Split 2,056 bets into S subsets (e.g., S=8 for 257 bets each)
# For each combination of S/2 subsets as "in-sample":
#   1. Optimize on in-sample
#   2. Test on out-of-sample
#   3. Calculate rank of IS-optimal strategy in OOS
# PBO = fraction of times IS-optimal performs below median OOS
```

**Simplified Approach (Time-Based Split):**
```python
# Split by time into N folds (chronological)
# For each fold as holdout:
#   - Train/validate on remaining folds
#   - Test on holdout
# PBO = fraction of holdouts where strategy underperforms random
```

**Success Criteria:** PBO < 30%

---

## Implementation

### File: `V8.0/phase3_monte_carlo_validation.py`

```python
"""
Phase 3: Monte Carlo / PBO Validation
=====================================

Tests:
1. Shuffle Test (N=1000) - 95% must end with bankroll >= $1,500
2. Noise Injection (±1%, ±2%) - ROI must remain positive
3. PBO Calculation - Must be < 30%
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

INPUT_FILE = "backtest_log_filtered_v8.csv"
INITIAL_BANKROLL = 1000
MIN_BANKROLL_THRESHOLD = 1500
N_SIMULATIONS = 1000

def simulate_bankroll(df, initial_bankroll=1000):
    """Simulate cumulative bankroll from bet sequence."""
    return initial_bankroll + df['Profit'].sum()

def shuffle_test(df, n_sims=1000, threshold=1500):
    """Test 1: Shuffle order and check bankroll stability."""
    results = []
    for i in tqdm(range(n_sims), desc="Shuffle Test"):
        shuffled = df.sample(frac=1, random_state=i)
        final = simulate_bankroll(shuffled)
        results.append(final)

    pass_count = sum(1 for b in results if b >= threshold)
    pass_rate = pass_count / n_sims

    return {
        'pass_rate': pass_rate,
        'mean_bankroll': np.mean(results),
        'std_bankroll': np.std(results),
        'min_bankroll': np.min(results),
        'max_bankroll': np.max(results),
        'percentile_5': np.percentile(results, 5),
        'passed': pass_rate >= 0.95
    }

def noise_injection_test(df, noise_levels=[0.01, 0.02]):
    """Test 2: Perturb Model_Prob and check ROI stability."""
    results = {}
    base_roi = df['Profit'].sum() / df['Stake'].sum() * 100

    for noise in noise_levels:
        perturbed = df.copy()
        np.random.seed(42)
        perturbation = np.random.uniform(-noise, noise, len(df))
        perturbed['Model_Prob_Noisy'] = (perturbed['Model_Prob_Adjusted'] + perturbation).clip(0, 1)

        # Recalculate edge with noise
        perturbed['Edge_Noisy'] = perturbed['Model_Prob_Noisy'] - perturbed['Implied_Prob']

        # Only keep bets that still have positive edge after noise
        valid = perturbed[perturbed['Edge_Noisy'] > 0]

        if len(valid) > 0:
            roi = valid['Profit'].sum() / valid['Stake'].sum() * 100
        else:
            roi = 0

        results[f'noise_{int(noise*100)}pct'] = {
            'roi': roi,
            'bets_remaining': len(valid),
            'roi_delta': roi - base_roi
        }

    return results

def calculate_pbo(df, n_folds=8):
    """Test 3: Probability of Backtest Overfitting (simplified)."""
    # Sort by date for time-series integrity
    df = df.sort_values('Date').reset_index(drop=True)

    fold_size = len(df) // n_folds
    fold_returns = []

    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else len(df)
        fold = df.iloc[start:end]
        fold_roi = fold['Profit'].sum() / fold['Stake'].sum() * 100 if fold['Stake'].sum() > 0 else 0
        fold_returns.append(fold_roi)

    # PBO = fraction of folds below median
    median_return = np.median(fold_returns)
    below_median = sum(1 for r in fold_returns if r < median_return)
    pbo = below_median / n_folds

    return {
        'pbo': pbo,
        'fold_returns': fold_returns,
        'median_return': median_return,
        'passed': pbo < 0.30
    }
```

---

## Success Criteria Summary

| Test | Requirement | Action if Fail |
|------|-------------|----------------|
| Shuffle Test | ≥95% above $1,500 | Investigate bet clustering |
| Noise ±1% | ROI > 2% | Edge too brittle - loosen filters |
| Noise ±2% | ROI > 0% | Edge too brittle - reduce filter count |
| PBO | < 30% | Overfitting - simplify strategy |

---

## Deliverables

- [ ] `phase3_monte_carlo_validation.py` script
- [ ] Shuffle test results (pass rate, distribution)
- [ ] Noise injection results (ROI at ±1%, ±2%)
- [ ] PBO calculation with fold breakdown
- [ ] Summary report with PASS/FAIL for each test
- [ ] Git commit and push
- [ ] Phase 4 (Shadow Mode) handoff if all tests pass

---

## Failure Recovery

| Failure | Root Cause | Recovery |
|---------|------------|----------|
| Shuffle < 95% | Bet order dependency | Check for time-series leakage |
| Noise collapses ROI | Brittle edge | Widen filter bands |
| PBO > 30% | Overfitting | Remove most restrictive filter |
