# CPR Czech Liga Pro Table Tennis Model (V8.0+)

## Project Overview
A machine learning pipeline for predicting Czech Liga Pro Table Tennis match outcomes.
**Goal**: Maximize trade performance (ROI, Sharpe Ratio, Drawdown) while avoiding overfitting.

---
## Project Structure
CPR_Model_Local_Build/
├── V8.0/                    ← Main working directory (run all scripts from here)
│   ├── *.py                 ← Pipeline scripts
│   ├── *.joblib             ← Trained models
│   ├── 2024_season_data/    ← Season data
│   └── back_test_forensic_analysis/
├── FINAL_BUILD/             ← Production deployment
├── tests/                   ← Unit and performance tests
├── checkpoints/             ← Training checkpoints
└── logs/                    ← Performance logs

**Git Workflow**: Development uses feature branches (not version folders). Releases are tagged (e.g., `v8.0.0`). Run `git tag -l` to see all releases.
---
## Global Configuration

**Model**: All agents use `claude-opus-4-5-20250114` (Opus 4.5)
**Reason**: Maximum reasoning capability for quantitative analysis

---

## Core Principles & Constraints

- **MINIMIZE FILTERS**: Avoid adding unnecessary filters in backtesting to prevent overfitting
- **TEST MODE ONLY**: Unless specified, do not run `run_pipeline.bat` or extract data from BetsAPI
- **OBJECTIVE TARGETS**:
  - ROI > 2%
  - Sharpe Ratio > 2.5
  - Maximum Drawdown < 35%

---

## Pipeline Execution Order (Test Mode Workflow)

**EXACT sequence when running full system analysis or training:**

```bash
python advanced_feature_engineering_v7.4.py && \
python merge_data_v7.4.py && \
python remove_duplicates_from_final_dataset.py && \
python split_data.py && \
python cpr_v7.4_specialist_gbm_trainer.py && \
python backtest_with_compounding_logic_v7.6.py && \
python analyze_performance.py && \
python backtest_final_v7.4.py
```

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | advanced_feature_engineering_v7.4.py | czech_liga_pro_advanced_stats_FIXED.csv | final_engineered_features_v7.4.csv |
| 2 | merge_data_v7.4.py | features + historical_odds_v7.0.csv | final_dataset_v7.4.csv |
| 3 | remove_duplicates_from_final_dataset.py | final_dataset_v7.4.csv | final_dataset_v7.4_no_duplicates.csv |
| 4 | split_data.py | cleaned dataset | training_dataset.csv + testing_dataset.csv |
| 5 | cpr_v7.4_specialist_gbm_trainer.py | training_dataset.csv | cpr_v7.4_gbm_specialist.joblib |
| 6 | backtest_with_compounding_logic_v7.6.py | dataset + models | backtest_log_for_analysis_v7.4.csv |
| 7 | analyze_performance.py | backtest log | Console report (ROI by category) |
| 8 | backtest_final_v7.4.py | dataset + models | backtest_log_final_filtered.csv |

---

## The CPR Quantitative Alpha Team (10 Subagents)

All agents use **Opus 4.5** for maximum reasoning capability.

### Agent Roster

| # | Agent | Role | Primary Files |
|---|-------|------|---------------|
| 1 | **Chief Strategist** | Orchestrator, task delegation | All files |
| 2 | **Statistical Arbitrageur** | P-values, significance testing | analyze_performance.py, backtests |
| 3 | **Feature Architect** | Feature engineering | advanced_feature_engineering_v7.4.py |
| 4 | **GBM Specialist** | Model training, hyperparameter tuning | cpr_v7.4_specialist_gbm_trainer.py |
| 5 | **Low-Latency Engineer** | Performance optimization | merge_data, remove_duplicates |
| 6 | **Forensic Debugger** | Root cause analysis | All files (when errors occur) |
| 7 | **Backtest Integrity Officer** | Bias/overfitting detection | backtest_final_v7.4.py |
| 8 | **Risk & Portfolio Manager** | Kelly Criterion, drawdown | backtest_with_compounding_logic_v7.6.py |
| 9 | **Data Integrity Guard** | Data quality, ETL | merge_data, remove_duplicates |
| 10 | **Documentation Librarian** | Docs, logging | CLAUDE.md, analyze_performance.py |

### Core Team Skills

The team collectively possesses expertise in:

- Advanced Stochastic Calculus (Brownian motion, Ito calculus)
- Low-Latency C++ Development (Template metaprogramming, lock-free structures)
- Python for Data Science (Pandas, NumPy, SciPy mastery)
- Market Microstructure Knowledge (Order book dynamics, exchange protocols)
- Statistical Arbitrage & Mean Reversion
- Machine Learning & Deep Learning (TensorFlow/PyTorch)
- High-Performance Distributed Computing
- Time-Series Analysis (ARIMA/GARCH, cointegration)
- Risk Management & Factor Modeling (VaR, Barra models)
- Bayesian Inference & Probability Theory (MCMC methods)
- Natural Language Processing (Sentiment analysis, LLM integration)
- Rigorous Backtesting Methodologies (Look-ahead bias prevention)
- Linux/Unix Kernel Optimization
- Portfolio Construction & Optimization (Kelly Criterion)
- Large-Scale Data Engineering (ETL pipelines)
- Algorithm Design & Data Structures
- Signal Processing (Fourier transforms, wavelets)
- Execution Algorithms (VWAP/TWAP, market impact)
- Econometrics (Causal inference, hypothesis testing)
- Database Architecture (KDB+/q for time-series)
- Game Theory & Strategic Behavior
- Machine Learning: Avoiding Overfitting

### Invocation Examples

```
"Use the Feature Architect to design a new momentum feature"
"Have the Backtest Integrity Officer audit the latest backtest"
"Ask the Statistical Arbitrageur to validate the ROI significance"
"Invoke the GBM Specialist to tune learning rate"
"Get the Risk & Portfolio Manager to optimize Kelly fraction"
```

---

## Data Architecture

| File | Size | Description |
|------|------|-------------|
| czech_liga_pro_advanced_stats_FIXED.csv | 12 MB | Raw match data (~50K matches) |
| historical_odds_v7.0.csv | 980 MB | Historical betting odds |
| final_engineered_features_v7.4.csv | 24 MB | Engineered features |
| final_dataset_v7.4_no_duplicates.csv | 13 MB | Clean merged dataset |
| training_dataset.csv | ~9 MB | 70% oldest (for training) |
| testing_dataset.csv | ~4 MB | 30% newest (for testing) |

---

## 6 GBM Features (Optimized)

After systematic feature removal analysis, we retained only the 6 features with significant predictive value.
All features are calculated as "advantages" (Player 1 - Player 2):

1. `H2H_P1_Win_Rate` - Head-to-head win rate (**CRITICAL** - 0.56% ROI impact)
2. `H2H_Dominance_Score` - Decay-weighted H2H point differential
3. `PDR_Advantage` - Points Dominance Ratio
4. `Win_Rate_L5_Advantage` - L5 "hot streak" win rate
5. `Close_Set_Win_Rate_Advantage` - Clutch performance in tight sets
6. `Set_Comebacks_Advantage` - Comeback ability

### Removed Features (Zero Predictive Value)
- `Time_Since_Last_Advantage`, `Matches_Last_24H_Advantage`, `Is_First_Match_Advantage`
- `PDR_Slope_Advantage`, `Daily_Fatigue_Advantage`, `Win_Rate_Advantage` (L20)

---

## Two-Stage Backtesting

### Stage 1: Exploration (backtest_with_compounding_logic_v7.6.py)
- Wide filters, KELLY_FRACTION = 0.035
- Output: `backtest_log_for_analysis_v7.4.csv`

### Stage 2: Analysis (analyze_performance.py)
- Identify profitable categories
- Output: Console report with ROI by category

### Stage 3: Validation (backtest_final_v7.4.py)
- Strategic filters applied, KELLY_FRACTION = 0.02
- Output: `backtest_log_final_filtered.csv`

---

## Development Guidelines

1. **Do NOT add filters without explicit user approval**
2. Focus on model/feature improvements over filter tuning
3. Any changes must be validated against objectives (ROI >2%, Sharpe >2.5, DD <35%)
4. All code changes require backtest validation before commit
5. Document all significant changes in code comments

---

## CRITICAL: Development Rules (Lessons Learned - Jan 2026)

These rules were established after a failed optimization attempt where baseline metrics could not be reproduced.

### Rule 1: BASELINE FIRST
**Before ANY code changes, run the full pipeline and record exact baseline metrics:**
```bash
# Run full pipeline BEFORE changes
python advanced_feature_engineering_v7.4.py && \
python merge_data_v7.4.py && \
python remove_duplicates_from_final_dataset.py && \
python split_data.py && \
python cpr_v7.4_specialist_gbm_trainer.py && \
python backtest_with_compounding_logic_v7.6.py && \
python analyze_performance.py && \
python backtest_final_v7.4.py

# Record baseline with commit hash
echo "Baseline at commit $(git rev-parse --short HEAD):"
echo "ROI: X.XX%, Sharpe: X.XX, MaxDD: X.XX%, Bets: XXXX"
```

### Rule 2: INCREMENTAL CHANGES
- Make ONE small change at a time
- Test IMMEDIATELY after each change
- If metrics shift unexpectedly, STOP and investigate
- Never batch multiple changes before testing

### Rule 3: END-TO-END VERIFICATION
- Matching intermediate outputs (CSVs) is NOT sufficient
- MUST verify final backtest ROI/Sharpe/MaxDD after EVERY change
- If numbers don't match baseline, DO NOT proceed

### Rule 4: DOCUMENT BASELINES
When establishing a baseline, document:
- Exact git commit hash
- ROI, Sharpe, MaxDD, Total Bets
- Date verified
- Any configuration (split ratio, Kelly fraction, etc.)

### Rule 5: ABORT ON DISCREPANCY
If expected baseline cannot be reproduced:
1. STOP all work immediately
2. Do not commit changes
3. Investigate the discrepancy first
4. Reset to known-good state if necessary

---

## Verified Baseline (Jan 2026)

**Commit:** `4de4728` (master state after V9.0 cleanup)
**Date Verified:** 2026-01-10
**Configuration:** 70/30 train/test split, KELLY_FRACTION = 0.02

| Metric | Value |
|--------|-------|
| ROI | **1.50%** |
| Sharpe Ratio | **2.02** |
| Max Drawdown | **34.12%** |
| Total Bets | **4,509** |
| Final Bankroll | $1,600.48 (from $1,000) |

**Note:** This is the TRUE baseline from master. Previous references to "2.44% ROI" were from a different commit/configuration that could not be reproduced. All future optimization work should target improvement over these verified numbers.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v7.4 | 2025 | Current stable version with GBM specialist |
| v8.0.0 | Jan 2025 | Git tag `v8.0.0` - Consolidated as main working version. 10-agent team structure, Opus 4.5 integration. Legacy version folders (v6.x-v7.x, V9.0) removed; using git branches for development going forward. |
