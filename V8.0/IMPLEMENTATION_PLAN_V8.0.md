# CPR Model V8.0 Implementation Plan
## Backtest Integrity & Filter Optimization

**Version:** 8.0
**Date:** January 2026
**Status:** Ready for Execution

---

## Executive Summary

This plan implements a rigorous, multi-phase validation and filtering system for the CPR table tennis prediction model. The goal is to ensure backtest integrity for LIVE trading deployment while avoiding overfitting.

**Critical Success Metrics:**
- Zero temporal leakage confirmed
- Calibration drift < 5% across all probability bins
- Minimum 1,000 bets post-filtering for statistical significance
- Probability of Backtest Overfitting (PBO) < 30%
- Strategy survives Monte Carlo shuffle test (p < 0.05)

---

## Sub-Agent Registry

| Agent ID | Specialty | Phase |
|----------|-----------|-------|
| `temporal-leakage-forensic-specialist` | Data leakage detection, embargo validation | 1.1 |
| `probability-calibration-auditor` | Brier score analysis, calibration curves | 1.2 |
| `residual-loss-pattern-detective` | Loss clustering, regime detection | 1.3 |
| `hierarchical-risk-filter-architect` | Filter stack design, sample preservation | 2 |
| `monte-carlo-stress-test-engineer` | PBO calculation, sensitivity analysis | 3 |
| `chief-validator` | Independent oversight, final sign-off | All |

---

## PHASE 1: HIGH-FIDELITY FORENSIC ANALYSIS & LEAKAGE AUDIT

### 1.1 Temporal Leakage & Embargo Validation

**Assigned Agent:** `temporal-leakage-forensic-specialist`

**Mission:** Prove the model is NOT cheating by detecting any future data contamination.

#### Task 1.1.1: Rolling Window Integrity Audit

**Objective:** Verify L5/L20 rolling windows contain ONLY past data relative to prediction time.

**Input Files:**
- `FINAL_BUILD/advanced_feature_engineering_v7.4.py`
- `FINAL_BUILD/final_dataset_v7.4_no_duplicates.csv`
- `FINAL_BUILD/backtest_with_compounding_logic_v7.6.py`

**Procedure:**
```python
# LOOK-AHEAD TEST ALGORITHM
for each match M at timestamp T:
    rolling_window = get_L20_window(player_id, T)
    for game G in rolling_window:
        assert G.timestamp < T, f"LEAKAGE: Game {G.id} at {G.timestamp} > Match time {T}"

    # Check same-day ordering
    same_day_games = [g for g in rolling_window if g.date == M.date]
    for game G in same_day_games:
        assert G.timestamp < T, f"SAME-DAY LEAKAGE: {G.id} played after prediction"
```

**Deliverable:** `V8.0/audit_reports/temporal_leakage_report.csv`
- Columns: `match_id`, `prediction_time`, `feature_name`, `leakage_detected`, `offending_game_id`, `time_delta`

**Acceptance Criteria:**
- [ ] Zero rows with `leakage_detected = True`
- [ ] All rolling windows respect chronological order
- [ ] Same-day matches properly ordered by time (not just date)

---

#### Task 1.1.2: Implement 24-Hour Embargo Buffer

**Objective:** Modify train/test split to enforce temporal gap.

**Current State:** Simple 70/30 percentage split
**Target State:** 70/30 split WITH 24-hour buffer zone

**Implementation:**
```python
# IN: backtest_with_compounding_logic_v7.6.py (or new split logic)

EMBARGO_HOURS = 24

def create_embargo_split(df, train_pct=0.70):
    """
    Creates train/test split with temporal embargo.
    """
    df = df.sort_values('Date').reset_index(drop=True)
    split_idx = int(len(df) * train_pct)

    train_end_date = df.iloc[split_idx - 1]['Date']
    embargo_cutoff = train_end_date + pd.Timedelta(hours=EMBARGO_HOURS)

    # Find first test sample AFTER embargo
    test_start_idx = df[df['Date'] > embargo_cutoff].index[0]

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[test_start_idx:]

    # Log embargo stats
    embargo_matches_skipped = test_start_idx - split_idx
    print(f"Embargo: Skipped {embargo_matches_skipped} matches in buffer zone")

    return train_df, test_df
```

**Deliverable:**
- Modified split function in `V8.0/backtest_v8.0_embargo.py`
- `V8.0/audit_reports/embargo_validation.json`

**Acceptance Criteria:**
- [ ] Minimum 24-hour gap between last training sample and first test sample
- [ ] Embargo stats logged for transparency
- [ ] No player appears in both final training hour and first test hour

---

#### Task 1.1.3: Player-Level Embargo Validation

**Objective:** Ensure no player's recent match leaks into their next prediction.

**Procedure:**
```python
# PLAYER EMBARGO TEST
MIN_PLAYER_EMBARGO_HOURS = 2  # At minimum, 2 hours between games

for player_id in unique_players:
    player_matches = df[df['Player 1 ID'] == player_id | df['Player 2 ID'] == player_id]
    player_matches = player_matches.sort_values('Date')

    for i in range(1, len(player_matches)):
        prev_match = player_matches.iloc[i-1]
        curr_match = player_matches.iloc[i]

        time_gap = (curr_match['Date'] - prev_match['Date']).total_seconds() / 3600

        if time_gap < MIN_PLAYER_EMBARGO_HOURS:
            # Check if prev_match is in curr_match's rolling window
            flag_potential_leakage(prev_match, curr_match, time_gap)
```

**Deliverable:** `V8.0/audit_reports/player_embargo_violations.csv`

**Acceptance Criteria:**
- [ ] All rapid-fire matches (< 2 hours apart) flagged for review
- [ ] Rolling windows exclude matches within embargo period

---

### CHECKPOINT 1A: Chief Validator Review

**Agent:** `chief-validator`

**Review Items:**
1. Verify `temporal_leakage_report.csv` shows zero leakage
2. Confirm embargo implementation matches specification
3. Validate player-level embargo logic
4. Sign-off required before proceeding to Phase 1.2

**Sign-off Document:** `V8.0/validation/checkpoint_1a_signoff.md`

---

### 1.2 Calibration Mapping (Brier Score Deep Dive)

**Assigned Agent:** `probability-calibration-auditor`

**Mission:** Map model probability outputs to empirical win rates and identify "Dead Zones."

#### Task 1.2.1: Probability Bin Analysis

**Objective:** Calculate empirical win rate for each probability bin.

**Procedure:**
```python
# CALIBRATION ANALYSIS
PROB_BINS = [(0.30, 0.35), (0.35, 0.40), (0.40, 0.45), (0.45, 0.50),
             (0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
             (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90)]

calibration_results = []

for bin_low, bin_high in PROB_BINS:
    bin_bets = df[(df['Model_Prob'] >= bin_low) & (df['Model_Prob'] < bin_high)]

    if len(bin_bets) < 30:  # Insufficient sample
        continue

    expected_prob = bin_bets['Model_Prob'].mean()
    empirical_win_rate = bin_bets['Win'].mean()
    calibration_error = empirical_win_rate - expected_prob

    brier_score = ((bin_bets['Model_Prob'] - bin_bets['Win']) ** 2).mean()

    calibration_results.append({
        'bin': f"{bin_low:.2f}-{bin_high:.2f}",
        'n_bets': len(bin_bets),
        'expected_prob': expected_prob,
        'empirical_win_rate': empirical_win_rate,
        'calibration_error': calibration_error,
        'brier_score': brier_score,
        'is_dead_zone': abs(calibration_error) > 0.10  # >10% miscalibration
    })
```

**Deliverable:**
- `V8.0/audit_reports/calibration_by_bin.csv`
- `V8.0/audit_reports/calibration_curve.png` (visual)

**Acceptance Criteria:**
- [ ] All bins with n >= 30 analyzed
- [ ] Dead zones (>10% calibration error) identified
- [ ] Overall Brier score < 0.25 (baseline for useful model)

---

#### Task 1.2.2: Dead Zone Identification & Mapping

**Objective:** Identify probability ranges where model systematically over/under-estimates.

**Dead Zone Definition:**
- **Overconfident Zone:** Model predicts 70%+, empirical < 60%
- **Underconfident Zone:** Model predicts 55%, empirical > 65%
- **Noise Zone:** Model predicts 48-52%, essentially coin-flip

**Deliverable:** `V8.0/audit_reports/dead_zones.json`
```json
{
  "dead_zones": [
    {
      "range": "0.75-0.80",
      "type": "overconfident",
      "expected": 0.775,
      "empirical": 0.62,
      "recommendation": "EXCLUDE or apply calibration correction"
    }
  ],
  "safe_zones": [
    {
      "range": "0.55-0.65",
      "calibration_error": 0.02,
      "recommendation": "PRIMARY BETTING ZONE"
    }
  ]
}
```

**Acceptance Criteria:**
- [ ] All dead zones documented with corrective recommendations
- [ ] Safe zones identified for primary betting focus
- [ ] Calibration correction factors calculated if needed

---

#### Task 1.2.3: Temporal Calibration Drift Analysis

**Objective:** Check if calibration degrades over time (regime drift).

**Procedure:**
```python
# ROLLING CALIBRATION CHECK
ROLLING_WINDOW_BETS = 200

for i in range(ROLLING_WINDOW_BETS, len(df)):
    window = df.iloc[i-ROLLING_WINDOW_BETS:i]

    brier_score = ((window['Model_Prob'] - window['Win']) ** 2).mean()
    calibration_error = window['Win'].mean() - window['Model_Prob'].mean()

    rolling_calibration.append({
        'end_date': window.iloc[-1]['Date'],
        'brier_score': brier_score,
        'calibration_error': calibration_error
    })

# FLAG: If Brier score increases >50% from baseline, model is degrading
```

**Deliverable:**
- `V8.0/audit_reports/calibration_drift.csv`
- `V8.0/audit_reports/calibration_drift.png` (time series plot)

**Acceptance Criteria:**
- [ ] No period with Brier score > 1.5x baseline
- [ ] Calibration error remains within ±5% over time
- [ ] Any drift periods flagged for investigation

---

### 1.3 Residual Analysis (Loss Pattern Detection)

**Assigned Agent:** `residual-loss-pattern-detective`

**Mission:** Forensically diagnose WHY specific bets failed and identify toxic patterns.

#### Task 1.3.1: Loss Clustering by Player Experience

**Objective:** Determine if losses concentrate on "new players" (<15 matches).

**Procedure:**
```python
# NEW PLAYER ANALYSIS
NEW_PLAYER_THRESHOLD = 15

losses = df[df['Outcome'] == 'Loss']

for _, loss in losses.iterrows():
    p1_history = len(history_df[history_df involves player 1])
    p2_history = len(history_df[history_df involves player 2])

    loss['bet_on_new_player'] = (
        (loss['Bet_On'] == 'P1' and p1_history < NEW_PLAYER_THRESHOLD) or
        (loss['Bet_On'] == 'P2' and p2_history < NEW_PLAYER_THRESHOLD)
    )

new_player_loss_rate = losses['bet_on_new_player'].mean()
```

**Deliverable:** `V8.0/audit_reports/loss_by_player_experience.csv`

**Key Metrics:**
- Loss rate when betting on player with < 15 games
- Loss rate when betting on player with >= 15 games
- Chi-square test for significance

**Acceptance Criteria:**
- [ ] If new player loss rate > baseline + 10%, implement MIN_GAMES filter
- [ ] Optimal MIN_GAMES threshold identified via ROC analysis

---

#### Task 1.3.2: Loss Clustering by Time-of-Day / League

**Objective:** Identify if losses concentrate in specific time windows or sub-leagues.

**Procedure:**
```python
# TEMPORAL REGIME ANALYSIS
losses['hour'] = losses['Date'].dt.hour
losses['day_of_week'] = losses['Date'].dt.dayofweek

# Group by hour
hourly_loss_rate = df.groupby(df['Date'].dt.hour).apply(
    lambda x: (x['Outcome'] == 'Loss').mean()
)

# Group by league/tournament if available
if 'League' in df.columns:
    league_loss_rate = df.groupby('League').apply(
        lambda x: (x['Outcome'] == 'Loss').mean()
    )
```

**Deliverable:**
- `V8.0/audit_reports/loss_by_hour.csv`
- `V8.0/audit_reports/loss_by_league.csv` (if applicable)
- `V8.0/audit_reports/loss_heatmap.png`

**Acceptance Criteria:**
- [ ] Toxic time windows identified (loss rate > baseline + 15%)
- [ ] Toxic leagues/tournaments flagged
- [ ] Recommendations for time-based filters if warranted

---

#### Task 1.3.3: Loss Clustering by Odds Movement / Liquidity

**Objective:** Determine if losses correlate with odds volatility or thin markets.

**Procedure:**
```python
# ODDS STABILITY ANALYSIS (if opening/closing odds available)
if 'Opening_Odds' in df.columns and 'Closing_Odds' in df.columns:
    df['odds_movement'] = df['Closing_Odds'] - df['Opening_Odds']
    df['odds_moved_against'] = (
        (df['Bet_On'] == 'P1' and df['odds_movement'] > 0.05) or
        (df['Bet_On'] == 'P2' and df['odds_movement'] < -0.05)
    )

    # Loss rate when odds moved against us
    adverse_movement_loss_rate = df[df['odds_moved_against']]['Outcome'].apply(
        lambda x: 1 if x == 'Loss' else 0
    ).mean()
```

**Deliverable:** `V8.0/audit_reports/loss_by_odds_movement.csv`

**Acceptance Criteria:**
- [ ] If adverse movement loss rate > 60%, implement odds stability filter
- [ ] Quantify "phantom edge" from odds that moved before execution

---

#### Task 1.3.4: Consecutive Loss Streak Analysis

**Objective:** Identify if loss streaks are random or indicative of regime breakdown.

**Procedure:**
```python
# STREAK ANALYSIS
def find_loss_streaks(outcomes):
    streaks = []
    current_streak = 0
    for outcome in outcomes:
        if outcome == 'Loss':
            current_streak += 1
        else:
            if current_streak >= 5:
                streaks.append(current_streak)
            current_streak = 0
    return streaks

loss_streaks = find_loss_streaks(df['Outcome'].tolist())

# Compare to random baseline (binomial)
expected_max_streak = calculate_expected_max_streak(n_bets, loss_rate)
```

**Deliverable:** `V8.0/audit_reports/loss_streak_analysis.csv`

**Acceptance Criteria:**
- [ ] Observed max streak vs expected max streak compared
- [ ] If observed >> expected, investigate regime breakdown
- [ ] Consecutive loss circuit breaker threshold determined

---

### CHECKPOINT 1B: Chief Validator Review

**Agent:** `chief-validator`

**Review Items:**
1. Verify calibration analysis methodology is sound
2. Confirm dead zones are correctly identified
3. Validate loss clustering analysis is not cherry-picking
4. Ensure residual patterns are statistically significant (not noise)
5. Sign-off required before proceeding to Phase 2

**Sign-off Document:** `V8.0/validation/checkpoint_1b_signoff.md`

---

## PHASE 2: SYSTEMATIC FILTER SYNTHESIS & OPTIMIZATION

**Assigned Agent:** `hierarchical-risk-filter-architect`

**Mission:** Construct the "Alpha-Preserving Filter Stack" without p-hacking.

### 2.1 Tier 1: Institutional Constraints (Non-Negotiable)

**Objective:** Implement base-level filters that are industry-standard.

#### Task 2.1.1: Edge Floor & Ceiling Implementation

**Specification:**
```python
# TIER 1 FILTERS - NON-NEGOTIABLE
EDGE_FLOOR = 0.03      # 3% minimum edge
EDGE_CEILING = 0.25    # 25% maximum edge (flags data errors)

def tier1_edge_filter(edge):
    """
    Rejects bets with insufficient or suspicious edge.
    """
    if edge < EDGE_FLOOR:
        return False, "REJECT: Edge below floor (noise)"
    if edge > EDGE_CEILING:
        return False, "REJECT: Edge above ceiling (suspicious)"
    return True, "PASS"
```

**Rationale:**
- 3% floor: Compensates for execution slippage (~1-2%) and model uncertainty
- 25% ceiling: Edges >25% typically indicate data errors or extreme outliers

---

#### Task 2.1.2: Sample Size Floor Implementation

**Specification:**
```python
# MINIMUM GAMES THRESHOLD
MIN_GAMES_TOTAL = 12       # Total historical games
MIN_GAMES_RECENT = 5       # Games in last 30 days

def tier1_sample_filter(player_id, history_df, current_date):
    """
    Rejects bets on players with insufficient history.
    """
    player_games = history_df[involves(player_id)]

    total_games = len(player_games)
    if total_games < MIN_GAMES_TOTAL:
        return False, f"REJECT: Only {total_games} total games"

    recent_games = player_games[
        player_games['Date'] > current_date - pd.Timedelta(days=30)
    ]
    if len(recent_games) < MIN_GAMES_RECENT:
        return False, f"REJECT: Only {len(recent_games)} recent games"

    return True, "PASS"
```

**Rationale:**
- 12 total games: Table tennis is high-variance; <12 games is insufficient for pattern detection
- 5 recent games: Ensures player is active and form is assessable

---

#### Task 2.1.3: Odds Sanity Filter

**Specification:**
```python
# ODDS BOUNDS
MIN_ODDS = 1.10   # Minimum 10% return potential
MAX_ODDS = 4.00   # Maximum 300% implied underdog

def tier1_odds_filter(odds):
    """
    Rejects bets with extreme odds.
    """
    if odds < MIN_ODDS:
        return False, "REJECT: Odds too low (heavy favorite, no value)"
    if odds > MAX_ODDS:
        return False, "REJECT: Odds too high (extreme underdog, high variance)"
    return True, "PASS"
```

**Rationale:**
- Odds < 1.10: Even with edge, profit potential is minimal
- Odds > 4.00: High variance destroys Kelly-based bankroll management

---

### 2.2 Tier 2: Calibration & Confidence Bands

**Objective:** Restrict betting to calibrated probability zones.

#### Task 2.2.1: Probability Band Filter

**Specification:**
```python
# CALIBRATED PROBABILITY BANDS
# (Informed by Task 1.2.2 Dead Zone Analysis)
PROB_FLOOR = 0.35    # Below this: underdog too uncertain
PROB_CEILING = 0.80  # Above this: model overconfident

def tier2_probability_filter(model_prob):
    """
    Rejects bets outside calibrated confidence zones.
    """
    if model_prob < PROB_FLOOR:
        return False, "REJECT: Probability below floor (high uncertainty)"
    if model_prob > PROB_CEILING:
        return False, "REJECT: Probability above ceiling (overconfidence zone)"
    return True, "PASS"
```

**Dynamic Adjustment:**
- These bounds should be informed by Phase 1.2 calibration analysis
- If dead zones identified, adjust accordingly

---

#### Task 2.2.2: Model-Market Divergence Filter

**Specification:**
```python
# MAXIMUM DISAGREEMENT WITH MARKET
MAX_DIVERGENCE = 0.20  # 20% maximum disagreement

def tier2_divergence_filter(model_prob, market_implied_prob):
    """
    Rejects bets where model disagrees too strongly with market.
    Large disagreements may indicate model error or hidden information.
    """
    divergence = abs(model_prob - market_implied_prob)

    if divergence > MAX_DIVERGENCE:
        return False, f"REJECT: {divergence:.1%} divergence exceeds {MAX_DIVERGENCE:.1%} limit"

    # Also require MINIMUM divergence (otherwise no edge)
    MIN_DIVERGENCE = 0.03
    if divergence < MIN_DIVERGENCE:
        return False, f"REJECT: {divergence:.1%} divergence below minimum edge"

    return True, "PASS"
```

**Rationale:**
- Market incorporates information model may not have (injuries, motivation)
- >20% disagreement: Model is likely wrong, not the market
- <3% disagreement: Insufficient edge after slippage

---

### 2.3 Tier 3: Dynamic Risk Management (Circuit Breakers)

**Objective:** Implement adaptive risk controls.

#### Task 2.3.1: Volatility-Adjusted Kelly

**Specification:**
```python
# DYNAMIC KELLY FRACTION
BASE_KELLY = 0.035

def tier3_dynamic_kelly(edge, odds, recent_roi_variance):
    """
    Adjusts Kelly fraction based on recent performance volatility.
    """
    # Standard Kelly
    kelly_raw = edge / (odds - 1)

    # Volatility adjustment: Higher variance = lower fraction
    volatility_multiplier = 1 / (1 + recent_roi_variance)

    # Apply base Kelly fraction
    kelly_adjusted = kelly_raw * BASE_KELLY * volatility_multiplier

    # Hard cap at 5%
    return min(kelly_adjusted, 0.05)
```

**Rationale:**
- During high-variance periods, reduce exposure
- Prevents aggressive betting during regime uncertainty

---

#### Task 2.3.2: Drawdown Circuit Breaker

**Specification:**
```python
# DRAWDOWN LIMITS
DRAWDOWN_WARNING = 0.10    # 10% - reduce stake by 50%
DRAWDOWN_PAUSE = 0.15      # 15% - pause betting for 24 hours
DRAWDOWN_STOP = 0.25       # 25% - full stop, manual review required

def tier3_drawdown_breaker(current_bankroll, peak_bankroll):
    """
    Implements circuit breakers based on drawdown.
    """
    drawdown = (peak_bankroll - current_bankroll) / peak_bankroll

    if drawdown >= DRAWDOWN_STOP:
        return "STOP", "CRITICAL: 25% drawdown - full stop"
    elif drawdown >= DRAWDOWN_PAUSE:
        return "PAUSE", "WARNING: 15% drawdown - pause 24 hours"
    elif drawdown >= DRAWDOWN_WARNING:
        return "REDUCE", "CAUTION: 10% drawdown - reduce stake 50%"
    else:
        return "NORMAL", "OK"
```

---

#### Task 2.3.3: Consecutive Loss Breaker

**Specification:**
```python
# CONSECUTIVE LOSS LIMITS
CONSECUTIVE_LOSS_WARNING = 5   # Reduce stake
CONSECUTIVE_LOSS_PAUSE = 8     # Pause betting

def tier3_streak_breaker(consecutive_losses):
    """
    Implements breakers based on loss streaks.
    """
    if consecutive_losses >= CONSECUTIVE_LOSS_PAUSE:
        return "PAUSE", f"WARNING: {consecutive_losses} consecutive losses"
    elif consecutive_losses >= CONSECUTIVE_LOSS_WARNING:
        return "REDUCE", f"CAUTION: {consecutive_losses} consecutive losses"
    else:
        return "NORMAL", "OK"
```

---

### 2.4 Vanishing Sample Analysis

**CRITICAL TASK:** Ensure filters don't eliminate statistical significance.

#### Task 2.4.1: Filter Impact Assessment

**Procedure:**
```python
# FILTER FUNNEL ANALYSIS
filter_funnel = []

# Start with raw backtest
raw_bets = len(df)
filter_funnel.append({'stage': 'Raw', 'bets': raw_bets, 'pct': 100})

# Apply Tier 1
tier1_passed = df[apply_tier1_filters(df)]
filter_funnel.append({
    'stage': 'Tier 1 (Edge, Sample, Odds)',
    'bets': len(tier1_passed),
    'pct': len(tier1_passed) / raw_bets * 100
})

# Apply Tier 2
tier2_passed = tier1_passed[apply_tier2_filters(tier1_passed)]
filter_funnel.append({
    'stage': 'Tier 2 (Probability, Divergence)',
    'bets': len(tier2_passed),
    'pct': len(tier2_passed) / raw_bets * 100
})

# CRITICAL CHECK
MIN_SAMPLE_SIZE = 500  # Absolute minimum for any statistical inference
IDEAL_SAMPLE_SIZE = 1000  # Required for p < 0.05 on ROI

if len(tier2_passed) < MIN_SAMPLE_SIZE:
    raise ValueError(f"VANISHING SAMPLE: Only {len(tier2_passed)} bets remain. Loosen Tier 2 filters.")
```

**Deliverable:** `V8.0/audit_reports/filter_funnel.csv`

**Acceptance Criteria:**
- [ ] Final sample size >= 500 (minimum)
- [ ] Ideally >= 1,000 for statistical significance
- [ ] If below threshold, identify most restrictive filter and loosen

---

### CHECKPOINT 2: Chief Validator Review

**Agent:** `chief-validator`

**Review Items:**
1. Verify filter hierarchy is logically sound (Tier 1 before Tier 2)
2. Confirm no p-hacking in filter threshold selection
3. Validate vanishing sample analysis
4. Ensure filter thresholds use round numbers (not overfit)
5. Compare filtered vs unfiltered ROI (should improve, not just reduce variance)
6. Sign-off required before proceeding to Phase 3

**Sign-off Document:** `V8.0/validation/checkpoint_2_signoff.md`

**Critical Questions:**
- Are the filter thresholds based on prior research, or tuned to this dataset?
- Would a ±20% change in any threshold collapse the strategy?
- Is the final sample size sufficient for statistical inference?

---

## PHASE 3: STRESS TEST BACKTEST (VALIDATION)

**Assigned Agent:** `monte-carlo-stress-test-engineer`

**Mission:** Differentiate between skill and luck using rigorous statistical tests.

### 3.1 Monte Carlo Shuffle Test

**Objective:** Verify ROI isn't dependent on bet order (path dependency).

#### Task 3.1.1: Bet Order Randomization

**Procedure:**
```python
# MONTE CARLO SHUFFLE TEST
N_SIMULATIONS = 1000

def shuffle_test(filtered_bets, bankroll_initial=1000):
    """
    Randomizes bet order and simulates outcomes.
    """
    simulation_results = []

    for sim in range(N_SIMULATIONS):
        shuffled = filtered_bets.sample(frac=1, random_state=sim)

        bankroll = bankroll_initial
        for _, bet in shuffled.iterrows():
            stake = calculate_stake(bankroll, bet)
            profit = calculate_profit(bet, stake)
            bankroll += profit

            if bankroll <= 0:
                break  # Bankruptcy

        final_roi = (bankroll - bankroll_initial) / total_staked
        simulation_results.append({
            'sim_id': sim,
            'final_bankroll': bankroll,
            'roi': final_roi,
            'bankruptcy': bankroll <= 0
        })

    return pd.DataFrame(simulation_results)

results = shuffle_test(filtered_bets)

# STATISTICAL ANALYSIS
mean_roi = results['roi'].mean()
std_roi = results['roi'].std()
pct_profitable = (results['roi'] > 0).mean()
pct_bankruptcy = results['bankruptcy'].mean()
```

**Deliverable:**
- `V8.0/audit_reports/monte_carlo_results.csv`
- `V8.0/audit_reports/roi_distribution.png`

**Acceptance Criteria:**
- [ ] Mean ROI > 0 across simulations
- [ ] >= 95% of simulations profitable (p < 0.05)
- [ ] Bankruptcy rate < 5%
- [ ] ROI distribution is roughly normal (not skewed by outliers)

---

### 3.2 Sensitivity Analysis

**Objective:** Verify strategy is robust to parameter perturbations.

#### Task 3.2.1: Edge Threshold Sensitivity

**Procedure:**
```python
# EDGE SENSITIVITY ANALYSIS
EDGE_THRESHOLDS = [0.02, 0.025, 0.03, 0.035, 0.04, 0.05]

sensitivity_results = []

for threshold in EDGE_THRESHOLDS:
    filtered = df[df['Edge'] >= threshold]

    # Run backtest
    roi = run_backtest(filtered)
    n_bets = len(filtered)

    sensitivity_results.append({
        'edge_threshold': threshold,
        'n_bets': n_bets,
        'roi': roi,
        'sharpe': calculate_sharpe(filtered)
    })

# CHECK: Does ROI collapse with small parameter changes?
baseline_roi = sensitivity_results[2]['roi']  # 0.03 threshold
for result in sensitivity_results:
    if result['edge_threshold'] != 0.03:
        roi_change = abs(result['roi'] - baseline_roi) / baseline_roi
        if roi_change > 0.50:  # >50% ROI change from 0.5% threshold change
            flag_overfit(result['edge_threshold'])
```

**Deliverable:** `V8.0/audit_reports/sensitivity_edge.csv`

**Acceptance Criteria:**
- [ ] ROI changes < 30% for ±0.5% edge threshold changes
- [ ] No single threshold is dramatically better (overfit signal)

---

#### Task 3.2.2: Sample Size Threshold Sensitivity

**Procedure:**
```python
# MIN_GAMES SENSITIVITY
MIN_GAMES_THRESHOLDS = [5, 8, 10, 12, 15, 20]

for threshold in MIN_GAMES_THRESHOLDS:
    # Run backtest with this threshold
    ...
```

**Deliverable:** `V8.0/audit_reports/sensitivity_min_games.csv`

---

#### Task 3.2.3: Probability Band Sensitivity

**Procedure:**
```python
# PROBABILITY BAND SENSITIVITY
PROB_FLOORS = [0.30, 0.35, 0.40]
PROB_CEILINGS = [0.75, 0.80, 0.85]

for floor in PROB_FLOORS:
    for ceiling in PROB_CEILINGS:
        # Run backtest with these bounds
        ...
```

**Deliverable:** `V8.0/audit_reports/sensitivity_prob_bands.csv`

---

### 3.3 Probability of Backtest Overfitting (PBO)

**Objective:** Calculate the probability that the strategy is overfit.

#### Task 3.3.1: PBO Calculation (Lopez de Prado Method)

**Procedure:**
```python
# PBO CALCULATION (Simplified)
# Full implementation requires Combinatorial Symmetric Cross-Validation

def calculate_pbo(df, n_splits=10):
    """
    Estimates Probability of Backtest Overfitting using
    out-of-sample vs in-sample performance comparison.
    """
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)

    is_sharpe_ratios = []
    oos_sharpe_ratios = []

    for train_idx, test_idx in tscv.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        is_sharpe = calculate_sharpe(train)
        oos_sharpe = calculate_sharpe(test)

        is_sharpe_ratios.append(is_sharpe)
        oos_sharpe_ratios.append(oos_sharpe)

    # PBO = proportion of folds where IS rank != OOS rank
    # Simplified: correlation between IS and OOS performance
    correlation = np.corrcoef(is_sharpe_ratios, oos_sharpe_ratios)[0, 1]

    # Low correlation = high PBO (in-sample performance doesn't predict out-of-sample)
    pbo_estimate = 1 - max(0, correlation)

    return pbo_estimate

pbo = calculate_pbo(filtered_bets)
```

**Deliverable:** `V8.0/audit_reports/pbo_analysis.json`

**Acceptance Criteria:**
- [ ] PBO < 30% (strong evidence of genuine edge)
- [ ] PBO < 50% (acceptable, proceed with caution)
- [ ] PBO >= 50%: FAIL - strategy likely overfit

---

### 3.4 Temporal Stability Test

**Objective:** Verify strategy works across different time periods.

#### Task 3.4.1: Rolling Window Performance

**Procedure:**
```python
# TEMPORAL STABILITY
WINDOW_SIZE = 200  # bets
STEP_SIZE = 50

rolling_performance = []

for start in range(0, len(filtered_bets) - WINDOW_SIZE, STEP_SIZE):
    window = filtered_bets.iloc[start:start + WINDOW_SIZE]

    roi = calculate_roi(window)
    win_rate = calculate_win_rate(window)
    sharpe = calculate_sharpe(window)

    rolling_performance.append({
        'start_idx': start,
        'end_date': window.iloc[-1]['Date'],
        'roi': roi,
        'win_rate': win_rate,
        'sharpe': sharpe
    })

# CHECK: Is performance stable or does it swing wildly?
roi_volatility = np.std([r['roi'] for r in rolling_performance])
```

**Deliverable:**
- `V8.0/audit_reports/temporal_stability.csv`
- `V8.0/audit_reports/rolling_roi.png`

**Acceptance Criteria:**
- [ ] No 200-bet window with ROI < -20%
- [ ] At least 80% of windows profitable
- [ ] No obvious regime breaks (sudden performance collapse)

---

### CHECKPOINT 3: Chief Validator Final Review

**Agent:** `chief-validator`

**Final Review Items:**
1. Monte Carlo: >= 95% simulations profitable
2. Sensitivity: Strategy survives ±20% parameter changes
3. PBO: < 30% probability of overfitting
4. Temporal: Stable across time periods

**Final Sign-off Document:** `V8.0/validation/final_signoff.md`

**GO / NO-GO Decision:**
- All checkpoints passed: **GO** for shadow mode testing
- Any checkpoint failed: **NO-GO** - return to relevant phase

---

## PHASE 4: SHADOW MODE VALIDATION (Post-Implementation)

**Objective:** Compare backtest predictions to real-world outcomes without risking capital.

### 4.1 Live Prediction Logging

**Duration:** Minimum 3 days, ideally 7 days

**Procedure:**
1. Run model on live matches
2. Log predictions WITHOUT placing bets
3. Compare predicted edge vs actual outcome
4. Track odds at prediction time vs closing odds (CLV validation)

**Deliverable:** `V8.0/shadow_mode/predictions_log.csv`

### 4.2 Backtest vs Reality Comparison

**Metrics to Compare:**
- Predicted win rate vs actual win rate
- Predicted ROI vs actual ROI (if bets were placed)
- Edge at prediction time vs edge at market close

**Acceptance Criteria:**
- [ ] Actual win rate within 5% of predicted
- [ ] No systematic CLV degradation (odds moving against us)
- [ ] Performance matches backtest expectations

---

## EXECUTION TIMELINE

| Phase | Tasks | Agent | Estimated Duration |
|-------|-------|-------|-------------------|
| 1.1 | Temporal Leakage Audit | temporal-leakage-forensic-specialist | Day 1 |
| 1.2 | Calibration Mapping | probability-calibration-auditor | Day 1-2 |
| 1.3 | Residual Analysis | residual-loss-pattern-detective | Day 2 |
| **CP1** | Checkpoint 1 Review | chief-validator | Day 2 |
| 2.1-2.4 | Filter Implementation | hierarchical-risk-filter-architect | Day 3 |
| **CP2** | Checkpoint 2 Review | chief-validator | Day 3 |
| 3.1-3.4 | Stress Testing | monte-carlo-stress-test-engineer | Day 4 |
| **CP3** | Final Review | chief-validator | Day 4 |
| 4.1-4.2 | Shadow Mode | All agents | Day 5-11 |

---

## DELIVERABLES SUMMARY

### Phase 1 Deliverables
- `V8.0/audit_reports/temporal_leakage_report.csv`
- `V8.0/audit_reports/embargo_validation.json`
- `V8.0/audit_reports/player_embargo_violations.csv`
- `V8.0/audit_reports/calibration_by_bin.csv`
- `V8.0/audit_reports/calibration_curve.png`
- `V8.0/audit_reports/dead_zones.json`
- `V8.0/audit_reports/calibration_drift.csv`
- `V8.0/audit_reports/loss_by_player_experience.csv`
- `V8.0/audit_reports/loss_by_hour.csv`
- `V8.0/audit_reports/loss_streak_analysis.csv`

### Phase 2 Deliverables
- `V8.0/backtest_v8.0_filtered.py`
- `V8.0/audit_reports/filter_funnel.csv`
- `V8.0/validation/checkpoint_2_signoff.md`

### Phase 3 Deliverables
- `V8.0/audit_reports/monte_carlo_results.csv`
- `V8.0/audit_reports/roi_distribution.png`
- `V8.0/audit_reports/sensitivity_edge.csv`
- `V8.0/audit_reports/sensitivity_min_games.csv`
- `V8.0/audit_reports/pbo_analysis.json`
- `V8.0/audit_reports/temporal_stability.csv`
- `V8.0/validation/final_signoff.md`

### Phase 4 Deliverables
- `V8.0/shadow_mode/predictions_log.csv`
- `V8.0/shadow_mode/backtest_vs_reality.csv`

---

## RISK REGISTER

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Vanishing Sample | High | Critical | Loosen Tier 2 filters if <500 bets |
| Calibration Dead Zones | Medium | High | Implement probability band filters |
| Temporal Leakage | Low | Critical | Rigorous Phase 1.1 audit |
| Regime Drift | Medium | Medium | Temporal stability monitoring |
| Over-filtering | Medium | High | PBO and sensitivity analysis |
| Execution Slippage | High | Medium | 3% edge floor accounts for this |

---

## APPENDIX A: Sub-Agent Detailed Specifications

### A.1 temporal-leakage-forensic-specialist

**Tools Required:**
- pandas, numpy for data manipulation
- Custom temporal validation functions

**Input Files:**
- `FINAL_BUILD/advanced_feature_engineering_v7.4.py`
- `FINAL_BUILD/final_dataset_v7.4_no_duplicates.csv`

**Output Files:**
- `V8.0/audit_reports/temporal_leakage_report.csv`
- `V8.0/audit_reports/embargo_validation.json`
- `V8.0/backtest_v8.0_embargo.py` (modified split logic)

**Success Criteria:**
- Zero temporal leakage detected
- Embargo implementation validated

---

### A.2 probability-calibration-auditor

**Tools Required:**
- pandas, numpy, matplotlib
- Brier score calculation
- Calibration curve plotting

**Input Files:**
- Backtest results with model probabilities and outcomes

**Output Files:**
- `V8.0/audit_reports/calibration_by_bin.csv`
- `V8.0/audit_reports/dead_zones.json`
- `V8.0/audit_reports/calibration_curve.png`

**Success Criteria:**
- All probability bins analyzed
- Dead zones identified with recommendations

---

### A.3 residual-loss-pattern-detective

**Tools Required:**
- pandas, numpy, scipy
- Statistical clustering
- Chi-square tests

**Input Files:**
- Backtest results with all features

**Output Files:**
- `V8.0/audit_reports/loss_by_player_experience.csv`
- `V8.0/audit_reports/loss_by_hour.csv`
- `V8.0/audit_reports/loss_streak_analysis.csv`

**Success Criteria:**
- Loss patterns identified with statistical significance
- Toxic regimes flagged

---

### A.4 hierarchical-risk-filter-architect

**Tools Required:**
- pandas, numpy
- Filter logic implementation

**Input Files:**
- Raw backtest results
- Phase 1 audit reports (to inform filter thresholds)

**Output Files:**
- `V8.0/backtest_v8.0_filtered.py`
- `V8.0/audit_reports/filter_funnel.csv`

**Success Criteria:**
- Filter hierarchy implemented
- Sample size preserved (>= 500 bets)

---

### A.5 monte-carlo-stress-test-engineer

**Tools Required:**
- pandas, numpy, scipy
- Monte Carlo simulation
- Statistical testing

**Input Files:**
- Filtered backtest results

**Output Files:**
- `V8.0/audit_reports/monte_carlo_results.csv`
- `V8.0/audit_reports/sensitivity_*.csv`
- `V8.0/audit_reports/pbo_analysis.json`

**Success Criteria:**
- PBO < 30%
- Strategy survives shuffle test
- Robust to parameter perturbations

---

### A.6 chief-validator

**Role:** Independent oversight and final sign-off

**Responsibilities:**
1. Review all sub-agent deliverables
2. Verify methodological soundness
3. Challenge assumptions
4. Identify blind spots
5. Provide GO/NO-GO decision

**Checkpoints:**
- Checkpoint 1A: After Phase 1.1
- Checkpoint 1B: After Phase 1.3
- Checkpoint 2: After Phase 2
- Checkpoint 3: After Phase 3 (Final)

**Output Files:**
- `V8.0/validation/checkpoint_1a_signoff.md`
- `V8.0/validation/checkpoint_1b_signoff.md`
- `V8.0/validation/checkpoint_2_signoff.md`
- `V8.0/validation/final_signoff.md`

---

*Document Version: 1.0*
*Last Updated: January 2026*
*Status: Ready for Execution*
