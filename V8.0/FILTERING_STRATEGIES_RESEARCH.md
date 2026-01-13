# Filtering Strategies Research for Table Tennis CPR Model Backtest
## Academic Literature & GitHub Repository Analysis

**Research Date:** January 2026
**Purpose:** Identify filtering strategies to prevent overfitting and maintain integrity for LIVE trading

---

## Executive Summary

This research compiles filtering strategies from academic papers and production-grade GitHub sports prediction models that are **NOT currently implemented** in `backtest_final_v7.4.py`. The goal is to determine optimal filtering levels that balance:
- **Signal preservation** (not filtering out profitable bets)
- **Overfitting prevention** (not curve-fitting to historical patterns)
- **Live trading robustness** (strategies that survive real market conditions)

### Current Filters in backtest_final_v7.4.py
| Filter | Threshold | Purpose |
|--------|-----------|---------|
| Edge Bounds | 0.0001 < edge < 0.99 | Basic positive EV requirement |
| Odds Denominator | (odds - 1) > 0.10 | Avoid ultra-low margin bets |
| Prime Directive | PDR_Advantage alignment | Bet with point dominance |
| Risk-Off Switch | Odds <= 3.0 | Avoid high-volatility longshots |
| Clarity Mandate | abs(Win_Rate_L5) > 0.1 | Require form clarity |
| Min Games | >= 4 games history | Player must have history |
| Stake Reduction | /4 when against form | Reduce risk on contrarian bets |

---

## SECTION 1: ACADEMIC RESEARCH FINDINGS

### 1.1 Model Calibration > Accuracy (Critical Finding)

**Source:** [Machine Learning for Sports Betting - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S266682702400015X)

> "Using calibration, rather than accuracy, as the basis for model selection leads to greater returns, on average (return on investment of +34.69% versus -35.17%)."

**Filter Recommendation:**
```python
# CALIBRATION CONFIDENCE FILTER
# Only bet when model probability is in calibrated range
CALIBRATION_RANGES = {
    'high_confidence': (0.65, 0.85),  # Most reliable predictions
    'medium_confidence': (0.55, 0.65),  # Acceptable predictions
    # Avoid: (0.85, 1.0) - overconfident, often miscalibrated
    # Avoid: (0.50, 0.55) - near coin-flip, high noise
}
```

**Implementation Gap:** Your model outputs probability but doesn't validate calibration reliability zones.

---

### 1.2 Purged Cross-Validation & Embargo (Critical for Time-Series)

**Source:** [Marcos Lopez de Prado - Advances in Financial Machine Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3104847)

> "Traditional time-series backtesting is sample inefficient, and leakage occurs. Purging and embargoing are necessary to avoid extreme leakage of information."

**Key Concept:** Combinatorial Purged Cross-Validation (CPCV) shows "marked superiority in mitigating overfitting risks" with "lower Probability of Backtest Overfitting (PBO)."

**Filter Recommendation:**
```python
# TEMPORAL EMBARGO FILTER
# Skip bets on matches occurring within X hours of training data cutoff
EMBARGO_HOURS = 24  # Minimum gap between train end and test start

# PLAYER RECENCY EMBARGO
# Don't bet on player if their most recent training data is too recent
PLAYER_DATA_EMBARGO_MATCHES = 3  # Wait 3 matches after last training sample
```

**Implementation Gap:** Your backtest uses a simple 70/30 split without temporal embargo.

---

### 1.3 Sample Size Requirements for Statistical Significance

**Source:** [Sports Investing Statistical Significance - Sports Insights](https://www.sportsinsights.com/sports-investing-statistical-significance/)

> "If a system is producing a better than 57% winning percentage, the cutoff is around 2,000 games to prove statistical significance."

**Source:** [Sample Size Betting Analysis - Punter2Pro](https://punter2pro.com/sample-size-betting-results-analysis/)

> "If you after 1100 bets still have 5% yield – you're most likely on to something. The risk of these results being pure chance is now just 5% (p < 0.05)."

**Filter Recommendation:**
```python
# PLAYER SAMPLE SIZE FILTER
MIN_HISTORICAL_MATCHES_TOTAL = 20  # Overall history for reliable patterns
MIN_RECENT_FORM_MATCHES = 5       # Recent matches for form assessment
MIN_H2H_ENCOUNTERS = 3            # Minimum H2H for H2H-weighted decisions

# BACKTEST VALIDATION THRESHOLD
MIN_BETS_FOR_SIGNIFICANCE = 1000  # Minimum bets before trusting ROI
```

**Implementation Gap:** Your MIN_GAMES_THRESHOLD = 4 may be too low for reliable predictions.

---

### 1.4 Kelly Criterion Under Uncertainty

**Source:** [Wharton - Betting Kelly Study](https://wsb.wharton.upenn.edu/wp-content/uploads/2023/05/Beggy_2023__Betting_Kelly.pdf)

> "Full Kelly simply does not work in a realistic betting environment. It led to bankruptcy in 100% of the scenarios."

> "Partial Kelly with coefficient 0.50 and a conservative 10% threshold as the most profitable strategy."

**Filter Recommendation:**
```python
# KELLY FRACTION UNCERTAINTY ADJUSTMENT
BASE_KELLY = 0.035  # Current setting
UNCERTAINTY_DISCOUNT = 0.50  # Apply half-Kelly for edge uncertainty

# EDGE CONFIDENCE TIERS
EDGE_TIERS = {
    'high': {'min_edge': 0.10, 'kelly_mult': 1.0},    # Full (adjusted) Kelly
    'medium': {'min_edge': 0.05, 'kelly_mult': 0.75}, # 75% Kelly
    'low': {'min_edge': 0.02, 'kelly_mult': 0.50},    # Half Kelly
    # Below 2% edge: No bet (noise zone)
}
```

**Implementation Gap:** Your 0.0001 edge threshold is too low - likely noise.

---

### 1.5 Maximum Drawdown Circuit Breakers

**Source:** [Drawdown Management - QuantifiedStrategies](https://www.quantifiedstrategies.com/drawdown/)

> "Maximum Drawdown (MDD) highlights the largest loss from peak to trough. Lower drawdowns (<15%) indicate better capital preservation."

> "A lot of traders believe that anything over a 20% MDD becomes devastating to a portfolio."

**Filter Recommendation:**
```python
# DRAWDOWN CIRCUIT BREAKERS
MAX_DRAWDOWN_PAUSE = 0.15      # Pause betting at 15% drawdown
MAX_DRAWDOWN_STOP = 0.25       # Full stop at 25% drawdown
RECOVERY_THRESHOLD = 0.10     # Resume at 10% drawdown

# CONSECUTIVE LOSS FILTER
MAX_CONSECUTIVE_LOSSES = 8    # Trigger review after 8 straight losses
STAKE_REDUCTION_AFTER_LOSSES = 0.50  # Reduce stake by 50% during streak
```

**Implementation Gap:** No drawdown-based position sizing or circuit breakers.

---

## SECTION 2: GITHUB REPOSITORY FILTERING STRATEGIES

### 2.1 Closing Line Value (CLV) Validation

**Source:** [VSiN - Closing Line Value](https://vsin.com/how-to-bet/the-importance-of-closing-line-value/)

> "Sample size is everything when it comes to gauging your true success and edge, and it's a lot bigger than most realize. Results only start to show more signal than noise north of 2,000 to 3,000 wagers."

> "Consistently beating the closing line is a mark of a sharp bettor. It validates their skill."

**Filter Recommendation:**
```python
# CLV-BASED BET FILTERING (Requires closing line data)
def clv_filter(opening_odds, closing_odds, bet_side):
    """
    Only take bets where you're getting better value than closing line.
    """
    if bet_side == 'P1':
        clv = (1/opening_odds) - (1/closing_odds)
    return clv > 0.02  # Require 2%+ CLV

# If you don't have closing lines, use market consensus
def consensus_filter(your_odds, market_avg_odds):
    """
    Ensure your odds beat market average by significant margin.
    """
    edge_vs_market = (1/your_odds) - (1/market_avg_odds)
    return edge_vs_market > 0.03  # 3% better than market
```

**Implementation Gap:** No CLV or market consensus validation.

---

### 2.2 Value Betting Expected Value Thresholds

**Source:** [georgedouzas/sports-betting](https://github.com/georgedouzas/sports-betting)

> "The bettor should aim to systematically estimate the value bets, backtest their performance, and not create arbitrarily accurate predictive models."

**Source:** [clemsage/SportsBet](https://github.com/clemsage/SportsBet)

> "If the do_value_betting argument is enabled, the system only places bets where predicted probability × bookmaker's odds > 1."

**Filter Recommendation:**
```python
# MINIMUM EDGE THRESHOLDS (Academic-backed)
MIN_EDGE_THRESHOLD = 0.03     # 3% minimum edge (not 0.0001)
MAX_EDGE_THRESHOLD = 0.25     # 25% max (higher = suspicious)

# VALUE RATIO FILTER
def value_ratio_filter(model_prob, market_odds):
    """
    Require meaningful value ratio, not just positive edge.
    """
    implied_prob = 1 / market_odds
    value_ratio = model_prob / implied_prob
    return 1.05 <= value_ratio <= 1.30  # 5-30% better than market
```

**Implementation Gap:** Current 0.0001 threshold allows near-zero edge bets (noise).

---

### 2.3 Model Confidence Banding

**Source:** [sports-ai.dev - AI Model Calibration](https://www.sports-ai.dev/blog/ai-model-calibration-brier-score)

> "Thresholds (e.g., min edge 3%) should be recomputed using calibrated probabilities only to avoid inflated position sizes."

**Source:** [kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting)

> "Outputs expected value for teams money lines to provide better insight. The fraction of your bankroll to bet based on the Kelly Criterion is also outputted."

**Filter Recommendation:**
```python
# MODEL CONFIDENCE BANDS
def confidence_band_filter(model_prob):
    """
    Only bet in calibrated confidence zones.
    Extreme probabilities (near 0 or 1) are often miscalibrated.
    """
    RELIABLE_RANGE_LOW = 0.35   # Below this: underdog too uncertain
    RELIABLE_RANGE_HIGH = 0.80  # Above this: favorite overconfident

    return RELIABLE_RANGE_LOW <= model_prob <= RELIABLE_RANGE_HIGH

# DISAGREEMENT FILTER (Model vs Market)
def model_market_disagreement_filter(model_prob, implied_prob):
    """
    Large disagreements may indicate model error, not edge.
    """
    disagreement = abs(model_prob - implied_prob)
    MAX_DISAGREEMENT = 0.20  # Don't trust >20% disagreement
    MIN_DISAGREEMENT = 0.03  # Require >3% disagreement for edge

    return MIN_DISAGREEMENT <= disagreement <= MAX_DISAGREEMENT
```

**Implementation Gap:** No confidence banding or model-market disagreement limits.

---

### 2.4 Form Regime Detection

**Source:** [Tennis-Prediction (Matyyas)](https://github.com/Matyyas/Tennis-Prediction)

> "The final purpose is to build a betting strategy with positive ROI, and they explored 2 betting strategies and found a lucrative one."

**Filter Recommendation:**
```python
# FORM REGIME FILTER
def form_regime_filter(p1_win_rate_l5, p2_win_rate_l5, p1_win_rate_l20, p2_win_rate_l20):
    """
    Detect when recent form diverges significantly from baseline.
    Extreme divergence may be noise or unsustainable.
    """
    p1_form_divergence = p1_win_rate_l5 - p1_win_rate_l20
    p2_form_divergence = p2_win_rate_l5 - p2_win_rate_l20

    MAX_FORM_DIVERGENCE = 0.30  # Don't trust >30% form swings

    return (abs(p1_form_divergence) < MAX_FORM_DIVERGENCE and
            abs(p2_form_divergence) < MAX_FORM_DIVERGENCE)

# FORM CONSISTENCY FILTER
def form_consistency_filter(win_rate_l5, win_rate_l10, win_rate_l20):
    """
    Require monotonic or stable form trajectory, not erratic.
    """
    trajectory = [win_rate_l5, win_rate_l10, win_rate_l20]
    is_monotonic_up = all(trajectory[i] <= trajectory[i+1] for i in range(len(trajectory)-1))
    is_monotonic_down = all(trajectory[i] >= trajectory[i+1] for i in range(len(trajectory)-1))
    is_stable = all(abs(trajectory[i] - trajectory[i+1]) < 0.10 for i in range(len(trajectory)-1))

    return is_monotonic_up or is_monotonic_down or is_stable
```

**Implementation Gap:** Your Clarity Mandate only requires non-zero form difference.

---

### 2.5 Reverse Line Movement / Steam Detection

**Source:** [Sports Insights - Steam Moves](https://www.sportsinsights.com/betting-systems/steam-moves/)

> "Steam Move [is] sudden, drastic and uniform line movement across the entire sports betting marketplace... the result of betting groups, betting syndicates."

**Filter Recommendation:**
```python
# ODDS MOVEMENT FILTER (Requires odds history)
def odds_movement_filter(opening_odds, current_odds, bet_side):
    """
    Detect if odds have moved against you (sharp money disagreement).
    """
    if bet_side == 'P1':
        odds_drift = opening_odds - current_odds
        # Negative drift means odds shortened (more confident market)
        # If we're betting P1 and odds lengthened, sharps disagree
        return odds_drift > -0.10  # Allow max 10% odds lengthening

# ODDS STABILITY FILTER
def odds_stability_filter(odds_history):
    """
    Avoid bets where odds are highly volatile (uncertain market).
    """
    if len(odds_history) < 3:
        return True

    volatility = np.std(odds_history)
    return volatility < 0.15  # Max 15% odds volatility
```

**Implementation Gap:** No odds movement or market sentiment validation (though may require additional data).

---

## SECTION 3: RECOMMENDED NEW FILTERS FOR IMPLEMENTATION

### Priority 1: Essential (High Impact, Low Complexity)

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| **Minimum Edge** | edge > 0.03 (3%) | Current 0.0001 is noise; literature recommends 3%+ |
| **Maximum Edge** | edge < 0.25 (25%) | Very high edges often indicate model error |
| **Model Probability Bounds** | 0.35 < prob < 0.80 | Extreme probabilities miscalibrated |
| **Sample Size Increase** | MIN_GAMES >= 10 | Current 4 is insufficient for pattern reliability |
| **Recent Form Requirement** | >= 3 matches in last 30 days | Avoid stale player data |

### Priority 2: Important (Medium Impact, Medium Complexity)

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| **Model-Market Disagreement Cap** | < 20% disagreement | Large disagreements may be model errors |
| **Drawdown Circuit Breaker** | Pause at 15% DD | Protect bankroll during losing streaks |
| **Consecutive Loss Limit** | Reduce stake after 6 losses | Adapt to potential regime change |
| **Form Divergence Cap** | L5 vs L20 < 30% | Extreme form swings are noise |
| **H2H Minimum** | >= 2 prior meetings | H2H features unreliable without history |

### Priority 3: Advanced (High Impact, High Complexity)

| Filter | Requirement | Rationale |
|--------|-------------|-----------|
| **CLV Validation** | Requires closing odds data | Best validation of true edge |
| **Calibration Monitoring** | Track Brier Score rolling | Detect when model decalibrates |
| **Embargo Period** | 24h gap between train/test | Prevent temporal leakage |
| **Regime Detection** | Monitor win rate across bins | Detect market condition changes |

---

## SECTION 4: OVERFITTING WARNING SIGNS

### Signs Your Backtest Is Overfit:

1. **Too Many Filters** - Each filter reduces sample size; >5-6 active filters likely overfit
2. **Perfect Parameter Values** - Round numbers (0.1, 0.5) more robust than precise (0.0734)
3. **High ROI on Low Sample** - 40% ROI on 200 bets = likely luck; 8% ROI on 2000 bets = more reliable
4. **Asymmetric Performance** - Works great on favorites, fails on underdogs (or vice versa)
5. **Time Period Sensitivity** - Dramatically different results in different time periods
6. **Sharpe Ratio > 2.0** - Very high Sharpe in backtests often doesn't replicate

### Lopez de Prado's "Probability of Backtest Overfitting" (PBO)

> "A strategy with PBO > 50% is more likely overfit than not. Target PBO < 30%."

---

## SECTION 5: RECOMMENDED FILTER IMPLEMENTATION ORDER

### Phase 1: Tighten Core Filters (Immediate)
```python
EDGE_THRESHOLD_MIN = 0.03  # Was 0.0001 - INCREASE 300x
EDGE_THRESHOLD_MAX = 0.25  # Was 0.99 - DECREASE to catch errors
MIN_GAMES_THRESHOLD = 10   # Was 4 - INCREASE for reliability
```

### Phase 2: Add Confidence Bounds (Week 1)
```python
MODEL_PROB_MIN = 0.35      # NEW - avoid extreme underdog bets
MODEL_PROB_MAX = 0.80      # NEW - avoid overconfident favorite bets
MAX_MODEL_MARKET_DISAGREEMENT = 0.20  # NEW - cap disagreement
```

### Phase 3: Add Risk Management (Week 2)
```python
MAX_DRAWDOWN_PAUSE = 0.15  # NEW - circuit breaker
MAX_CONSECUTIVE_LOSSES = 6 # NEW - streak detection
DAILY_BET_LIMIT = 20       # NEW - prevent over-betting
```

### Phase 4: Add Temporal Filters (Week 3)
```python
MIN_DAYS_SINCE_LAST_MATCH = 0.5  # NEW - avoid players on break
MAX_DAYS_SINCE_LAST_MATCH = 14   # NEW - avoid inactive players
MIN_RECENT_MATCHES_30_DAYS = 3   # NEW - require active players
```

---

## SECTION 6: SAMPLE SIZE GUIDANCE FOR FILTER VALIDATION

| Metric | Minimum Sample | Confidence Level |
|--------|---------------|------------------|
| Win Rate Significance | 500 bets | 90% |
| ROI Significance | 1,000 bets | 95% |
| Filter Effectiveness | 200 bets per filter state | Compare on/off |
| Sharpe Ratio Reliability | 250+ daily observations | Annualized estimate |
| Drawdown Statistics | Full equity curve | Requires complete history |

**Rule of Thumb:** If changing a filter threshold by ±20% dramatically changes results, it's likely overfit to that parameter.

---

## REFERENCES

### Academic Papers
- [A Systematic Review of Machine Learning in Sports Betting (2024)](https://arxiv.org/abs/2410.21484)
- [Machine Learning for Sports Betting: Calibration vs Accuracy](https://www.sciencedirect.com/science/article/pii/S266682702400015X)
- [Advances in Financial Machine Learning - Lopez de Prado](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3104847)
- [A Statistical Theory of Optimal Decision-Making in Sports Betting](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0287601)
- [Optimal Betting Under Parameter Uncertainty](https://www.academia.edu/20341527/Optimal_Betting_Under_Parameter_Uncertainty_Improving_the_Kelly_Criterion)

### GitHub Repositories
- [georgedouzas/sports-betting](https://github.com/georgedouzas/sports-betting) - Value betting framework
- [kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting) - Kelly Criterion integration
- [Matyyas/Tennis-Prediction](https://github.com/Matyyas/Tennis-Prediction) - Tennis-specific strategies
- [BrandoPolistirolo/Tennis-Betting-ML](https://github.com/BrandoPolistirolo/Tennis-Betting-ML) - Logistic regression for tennis
- [clemsage/SportsBet](https://github.com/clemsage/SportsBet) - Value betting implementation

### Industry Resources
- [Closing Line Value - VSiN](https://vsin.com/how-to-bet/the-importance-of-closing-line-value/)
- [Statistical Significance - Sports Insights](https://www.sportsinsights.com/sports-investing-statistical-significance/)
- [Model Calibration - Neptune.ai](https://neptune.ai/blog/brier-score-and-model-calibration)
- [Drawdown Management - QuantifiedStrategies](https://www.quantifiedstrategies.com/drawdown/)
- [Purged Cross-Validation - Wikipedia](https://en.wikipedia.org/wiki/Purged_cross-validation)

---

## CONCLUSION

The research reveals that your current `backtest_final_v7.4.py` has reasonable structural filters but several critical gaps:

1. **Edge threshold too low** (0.0001 vs recommended 3%)
2. **No model probability bounds** (extreme predictions unreliable)
3. **No drawdown circuit breakers** (risk management)
4. **Sample size requirements too low** (4 vs recommended 10+)
5. **No form regime validation** (erratic form unreliable)

**Key Insight:** Academic literature strongly suggests that **calibration is more important than accuracy** for sports betting profitability. The model probability output should be trusted within specific bands, not across the full 0-1 range.

**Recommended Approach:** Start with Phase 1 filters (tighten edge, increase sample size) and validate performance before adding complexity. Each additional filter should be justified by improved Sharpe ratio or reduced max drawdown, not just higher ROI.

---

*Document generated for CPR Model v8.0 optimization*
