# Feature Research Recovery Log

**Date:** 2026-01-13
**Branch Analyzed:** feature-research-github-scrape

---

## Golden Baseline (Commit 2e7b695)

| Metric | Value |
|--------|-------|
| Total Bets | 4,509 |
| ROI | 1.50% |
| Sharpe Ratio | 2.02 |
| Max Drawdown | 34.12% |
| Final Bankroll | $1,600.48 |

---

## After Filter Stack (filter_stack_v8.py)

| Metric | Value | Change |
|--------|-------|--------|
| Total Bets | 2,056 | -54.4% |
| ROI | 4.33% | +189% |
| Sharpe Ratio | 3.63 | +80% |
| Max Drawdown | 25.26% | -26% |

---

## Safe Scripts (Don't Change Baseline)

These scripts work ON TOP of the golden baseline:

| Script | Location | Purpose |
|--------|----------|---------|
| `filter_stack_v8.py` | V8.0/ | 3-tier filter stack (4.33% ROI) |
| `phase3_monte_carlo_validation.py` | V8.0/ | Statistical robustness testing |
| `temporal_leakage_audit.py` | audit_reports/ | Validate no data leakage |
| `loss_pattern_audit.py` | audit_reports/ | Identify losing patterns |
| `calibration_audit.py` | audit_reports/ | Probability calibration analysis |
| `overconfidence_forensic.py` | audit_reports/ | Model overconfidence root cause |

---

## Experimental Scripts (Change Baseline)

These scripts produce different baseline metrics and require testing before integration:

| Script | Location | Impact |
|--------|----------|--------|
| `backtest_final_v7.4_calibrated.py` | experimental/ | MaxDD 34% -> 9.5%, different ROI |
| `advanced_feature_engineering_dynamic_elo.py` | experimental/ | +9.2% ROI, changes baseline to 4,563 bets |
| `backtest_columns_patch.diff` | experimental/ | Adds Hour, P1/P2_Games_Count columns |

---

## Dynamic K-Factor Elo Implementation

**Commits:** e1b8ae1 -> 4b5a073

Replaces static K=32 with experience-based Dynamic K-Factor:

```python
if player_matches < 10:
    K = 70   # Placement phase (high volatility)
elif player_matches < 30:
    K = 35   # Development phase
else:
    K = 20   # Established phase (stable)
```

**Results vs Static K=32:**
- ROI: 1.73% -> 1.89% (+9.2%)
- Sharpe: 2.52 -> 2.80 (+11%)
- Max Drawdown: 37.96% -> 37.82% (-0.14%)
- Total Bets: 4,537 -> 4,523

---

## 24-Hour Embargo Split

**Commit:** d474e96

Added to `split_data.py` with `EMBARGO_HOURS = 0` (disabled by default).

When enabled (`EMBARGO_HOURS = 24`):
- Creates minimum 24-hour gap between train and test sets
- Logs matches skipped in embargo zone
- Reports overlapping players at boundary

---

## Key Insight: Order of Operations

The 4.33% ROI breakthrough is the **order of operations**:

1. **TRANSFORM FIRST** - Cap probabilities:
   - `H2H_PERFECT_CAP = 0.70` (cap H2H=100% records)
   - `DEAD_ZONE_CAP = 0.75` (max probability cap)

2. **RECALCULATE EDGE** - After caps applied

3. **THEN FILTER** - Apply boolean masks:
   - Edge: 3% - 25%
   - Dominance: >= 4
   - Odds: 1.10 - 4.00
   - Model Prob: 0.35 - 0.75
   - Divergence: 3% - 20%

---

## Dangling Commits Reviewed

| Commit | Description | Value |
|--------|-------------|-------|
| `03121c5` | WIP stash from Dynamic K-Factor work | LOW |
| `b09ad89` | Adds Hour, P1/P2_Games_Count columns | MEDIUM (extracted) |
| `5cb72c8` | 2.44% ROI investigation conclusion | LOW (docs only) |

---

## Phase Summary

| Phase | Commit | Finding |
|-------|--------|---------|
| 1.1 | 95edb38 | No look-ahead bias detected |
| 1.1.2 | d474e96 | 24-hour embargo implemented |
| 1.2 | 681ad84 | H2H 95.6% of model importance, 25pp miscalibration |
| 1.3 | 51f47ee | Late night profitable, circuit breakers hurt |
| 2 | 4241a28 | 3-tier filter stack: 4.33% ROI |
| 3 | e574f1b | Monte Carlo validation: 100% shuffle pass |

---

## Reproduction Commands

```bash
# Step 1: Run golden pipeline (baseline)
cd /Users/christiandiaz/Documents/Coding/table-tennis-cpr-model/V8.0
python advanced_feature_engineering_v7.4.py && \
python merge_data_v7.4.py && \
python remove_duplicates_from_final_dataset.py && \
python split_data.py && \
python cpr_v7.4_specialist_gbm_trainer.py && \
python backtest_final_v7.4.py

# Step 2: Apply filter stack (4.33% ROI)
python filter_stack_v8.py

# Step 3: Validate with Monte Carlo
python phase3_monte_carlo_validation.py
```

---

## Future Experiments

1. **Test Dynamic K-Factor Elo:**
   ```bash
   cp experimental/advanced_feature_engineering_dynamic_elo.py advanced_feature_engineering_v7.4.py
   # Run full pipeline, compare results
   ```

2. **Test 24-hour Embargo:**
   ```bash
   # Edit split_data.py: EMBARGO_HOURS = 24
   # Run full pipeline, compare results
   ```

3. **Test Calibration-Weighted Kelly:**
   ```bash
   cp experimental/backtest_final_v7.4_calibrated.py backtest_final_v7.4.py
   # Run pipeline, compare MaxDD reduction
   ```
