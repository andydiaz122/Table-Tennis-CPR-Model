# V9.0 Feature Testing Branch

> **This is a standalone experimental branch** - it does not share history with the main branch and is maintained separately for feature testing and experimentation.

## Branch Purpose

This branch contains V9.0 feature testing experiments for the Czech Liga Pro table tennis betting model. It serves as a testbed for new inter-match feature engineering approaches before potential integration into the main codebase.

## Key Result

| Metric | Baseline | After Features | Improvement |
|--------|----------|----------------|-------------|
| **ROI** | 2.44% | **3.67%** | +1.23% |
| Sharpe Ratio | - | 4.46 | - |
| Max Drawdown | - | 23.51% | - |
| Total Bets | - | 1,023 | - |

## Commit History

### `6c4f127` - Add 28 new inter-match features, improve ROI from 2.44% to 3.67%

Implemented 28 new inter-match features in `advanced_feature_engineering_v7.4.py`:

| Category | Features | Count |
|----------|----------|-------|
| **Elo Rating** | P1_Elo, P2_Elo, Elo_Diff | 3 |
| **Glicko-2** | P1/P2_Glicko_Mu, P1/P2_Glicko_Phi, P1/P2_Glicko_Sigma, Glicko_Mu_Diff, Glicko_Phi_Sum | 8 |
| **Clutch Factor** | P1/P2_Clutch_Factor_L20, Clutch_Factor_Diff, P1/P2_Deuce_Sets_L20 | 5 |
| **Pythagorean** | P1/P2_Pythagorean_Win_Rate_L20, P1/P2_Pythagorean_Delta_L20, Pythagorean_Delta_Diff | 5 |
| **Fatigue** | P1/P2_Fatigue_Factor, Fatigue_Factor_Diff (replaces Daily_Fatigue_Advantage) | 3 |
| **PDR Variance** | P1/P2_PDR_Variance_L20, PDR_Variance_Diff | 3 |
| **H2H Matches** | H2H_Matches (sample size) | 1 |

### `e4d54b7` - Initial commit: V9.0 baseline with 2.44% ROI

Baseline V9.0 codebase before feature enhancements.

## Tags

- **`v9.0-roi-3.67`** - Milestone tag marking the 3.67% ROI achievement with 28 new features

## Key Files

- `FINAL_BUILD/advanced_feature_engineering_v7.4.py` - Feature engineering with all 28 new features
- `FINAL_BUILD/cpr_v7.4_specialist_gbm_trainer.py` - GBM model trainer (19 features)
- `FINAL_BUILD/backtest_with_compounding_logic_v7.6.py` - Backtest with compounding
- `FINAL_BUILD/backtest_final_v7.4.py` - Final filtered backtest
- `FINAL_BUILD/feature_statistics_v7.4.csv` - Feature distribution statistics

## Test Pipeline

The full test pipeline (in order):
1. `advanced_feature_engineering_v7.4.py` - Engineer features
2. `merge_data_v7.4.py` - Merge with odds
3. `remove_duplicates_from_final_dataset.py` - Clean dataset
4. `split_data.py` - Train/test split (70/30)
5. `cpr_v7.4_specialist_gbm_trainer.py` - Train GBM model
6. `backtest_with_compounding_logic_v7.6.py` - Run backtest
7. `analyze_performance.py` - Analyze results
8. `backtest_final_v7.4.py` - Final validation

## Note

This branch is intentionally kept separate from the main development line to preserve the experimental state and allow for independent iteration on feature engineering approaches.
