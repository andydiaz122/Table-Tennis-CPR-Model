# Momentum Features Research Report
## New Features for Table Tennis CPR Model

**Research Date:** 2026-01-12
**Sources:** 25+ GitHub repositories and academic papers on sports prediction models

---

## Executive Summary

This research analyzed highly-rated sports prediction models on GitHub and academic research to identify momentum features **NOT currently implemented** in our table tennis model. The current model uses 6 momentum-related features. This report documents **35+ new potential features** categorized by type and implementation complexity.

---

## Current Model Baseline

### Active Features (6)
| Feature | Type | Window |
|---------|------|--------|
| Win_Rate_L5_Advantage | Hot Streak | 5 matches |
| H2H_P1_Win_Rate | H2H Record | All-time |
| H2H_Dominance_Score | H2H Momentum | All-time (decayed) |
| PDR_Advantage | Rolling Performance | 20 matches |
| Close_Set_Win_Rate_Advantage | Clutch | 20 matches |
| Set_Comebacks_Advantage | Resilience | 20 matches |

### Previously Tested & Removed (Zero Value)
- Win_Rate_Advantage (L20), PDR_Slope_Advantage, Time_Since_Last_Advantage
- Matches_Last_24H_Advantage, Is_First_Match_Advantage, Daily_Fatigue_Advantage

---

## NEW MOMENTUM FEATURES (Not Currently Implemented)

### Category 1: Elo-Based Momentum Features

**Source:** [Tennis-match-prediction-model](https://github.com/nrccstat/Tennis-match-prediction-model), [Tennis-Betting-ML](https://github.com/BrandoPolistirolo/Tennis-Betting-ML)

#### 1.1 Dynamic Elo Rating
- **Definition:** Real-time skill rating updated after every match
- **Formula:** `Elo_new = Elo_old + K × (Actual - Expected)`
- **K-Factor Options:**
  - Static: K=32 (standard)
  - Dynamic: K varies by match importance
- **Impact:** Tennis models using Elo achieve ~66-70% accuracy
- **Priority:** HIGH - Core skill metric missing from current model

#### 1.2 Rolling Elo Momentum (5-match)
- **Definition:** Rolling average of player's Elo over last 5 matches
- **Formula:** `mean(Elo[-5:])`
- **Purpose:** Captures momentum/form better than static Elo
- **Priority:** HIGH

#### 1.3 Elo Velocity (Elo Change Rate)
- **Definition:** Rate of Elo change over recent matches
- **Formula:** `(Current_Elo - Elo_N_matches_ago) / N`
- **Purpose:** Captures improving/declining trajectory
- **Priority:** MEDIUM

#### 1.4 Weighted Elo (WElo) - Hot Hand Effect
- **Definition:** Elo with extra weight on most recent match scoreline
- **Formula:** `WElo = Elo + α × (margin_of_victory_last_match)`
- **Source:** Research shows this captures "hot-hand" phenomenon
- **Priority:** MEDIUM

#### 1.5 Surface/Venue-Specific Elo
- **Definition:** Separate Elo ratings per playing surface/venue
- **Application:** For table tennis, could track venue-specific performance
- **Priority:** LOW (may not apply well to TT)

---

### Category 2: Exponentially Weighted Moving Averages (EWMA)

**Source:** [deepshot NBA Predictor](https://github.com/saccofrancesco/deepshot), Stanford EWMM Research

#### 2.1 EWMA Win Rate
- **Definition:** Exponentially weighted win rate (recent games matter more)
- **Formula:** `EWMA = α × current_result + (1-α) × previous_EWMA`
- **Half-life Options:** 3, 5, 10, 20 matches
- **Advantage over Rolling:** Responds faster to form changes, uses ALL history
- **Priority:** HIGH - Better than simple rolling averages

#### 2.2 EWMA Points Dominance Ratio
- **Definition:** EWMA applied to PDR instead of simple rolling mean
- **Formula:** `EWMA_PDR with halflife=10 matches`
- **Priority:** HIGH - Direct upgrade to existing PDR_Advantage

#### 2.3 EWMA Clutch Performance
- **Definition:** EWMA of close-set win rate
- **Priority:** MEDIUM

---

### Category 3: Streak-Based Features

**Source:** [tennispredictor](https://github.com/jdlamstein/tennispredictor), [nba-prediction](https://github.com/cmunch1/nba-prediction)

#### 3.1 Current Winning Streak
- **Definition:** Consecutive wins count
- **Formula:** Count consecutive W's from most recent match backward
- **Research Finding:** "Winning streaks and losing streaks were CRITICAL across ALL classifiers"
- **Priority:** HIGH - Top predictor in multiple models

#### 3.2 Current Losing Streak
- **Definition:** Consecutive losses count
- **Formula:** Count consecutive L's from most recent match backward
- **Priority:** HIGH

#### 3.3 Streak Advantage
- **Definition:** Combined streak differential
- **Formula:** `P1_WinStreak - P1_LoseStreak - (P2_WinStreak - P2_LoseStreak)`
- **Priority:** HIGH

#### 3.4 Maximum Recent Streak
- **Definition:** Longest win streak in last 20 matches
- **Purpose:** Captures peak momentum potential
- **Priority:** MEDIUM

---

### Category 4: Recency-Weighted Momentum Score

**Source:** [momentum-football-bets](https://github.com/DidierRLopes/momentum-football-bets)

#### 4.1 Weighted Momentum Score (WMS)
- **Definition:** Linearly weighted score based on last N results
- **Formula (6-match version):**
  ```
  WMS = Σ(weight_i × result_i) for i in [1,6]
  weight = [6,5,4,3,2,1] (newest to oldest)
  result = +1 (win), 0 (draw), -1 (loss)
  ```
- **Range:** -21 to +21
- **Priority:** HIGH - Novel approach not in current model

#### 4.2 Momentum Gap
- **Definition:** Difference in momentum scores between players
- **Formula:** `P1_WMS - P2_WMS`
- **Range:** -42 to +42 (higher = bigger form advantage)
- **Use Case:** Confidence metric for bet sizing
- **Priority:** HIGH

---

### Category 5: Psychological Momentum Features

**Source:** Journal of Big Data (2025), ACM Conference on Big Data & AI (2024)

#### 5.1 Entropy-Weighted Psychological Momentum (IEW-TOPSIS)
- **Definition:** Multi-factor momentum using information entropy weighting
- **Components:**
  1. Scoring momentum (recent point trends)
  2. Technical momentum (serve/return efficiency trends)
  3. Psychological momentum (performance under pressure)
- **Method:** Entropy Weight Method assigns weights based on information content
- **Research Result:** CatBoost + RF with this feature achieved 97.5% accuracy
- **Priority:** HIGH - State-of-the-art approach

#### 5.2 Pressure Performance Index
- **Definition:** Win rate in high-pressure situations
- **Formula:** `Points_Won_When_Behind / Total_Points_When_Behind`
- **Similar to:** Close_Set_Win_Rate but more granular
- **Priority:** MEDIUM

#### 5.3 Comeback Momentum
- **Definition:** Points won after falling behind in games
- **Granularity:** Point-level (not just set-level like current feature)
- **Priority:** MEDIUM

#### 5.4 Break Point Conversion Rate
- **Definition:** Success rate on break point opportunities
- **Formula:** `Break_Points_Won / Break_Points_Faced`
- **Analogy for TT:** Performance when opponent is serving (receiving efficiency)
- **Priority:** MEDIUM

#### 5.5 Break Point Save Rate
- **Definition:** Success rate saving break points against
- **Analogy for TT:** Performance defending own serve under pressure
- **Priority:** MEDIUM

---

### Category 6: Serve/Receive Efficiency Features

**Source:** arXiv papers on tennis prediction, ATP statistical analysis

#### 6.1 First Serve Win Rate Advantage
- **Definition:** Win rate on points where first serve is in
- **TT Analogy:** Win rate on service points (serves 1-2 of rotation)
- **Research:** "FirstWonFirstIn markedly decreases accuracy when removed"
- **Priority:** HIGH - Key predictor in tennis models

#### 6.2 Second Serve Win Rate Advantage
- **Definition:** Win rate when forced to second serve
- **TT Analogy:** Performance on weaker service opportunities
- **Priority:** MEDIUM

#### 6.3 Return Win Rate Advantage
- **Definition:** Win rate when receiving serve
- **Formula:** `P1_Return_Win_Rate - P2_Return_Win_Rate`
- **Priority:** HIGH - Complements serve metrics

#### 6.4 Service Games Won Percentage
- **Definition:** Percentage of service games held
- **TT Analogy:** Win rate during service rotation
- **Priority:** MEDIUM

#### 6.5 Mutual Point-Winning Probability (MPW)
- **Definition:** Player's chances of winning a point vs specific opponent as server AND receiver
- **Source:** Table tennis specific research
- **Formula:** Two separate probabilities: `MPW_serve` and `MPW_receive`
- **Priority:** HIGH - Designed specifically for table tennis

---

### Category 7: Form Decay & Recovery Features

**Source:** [NBA_RANKINGS](https://github.com/klarsen1/NBA_RANKINGS), Tennis prediction research

#### 7.1 Inactivity Decay Index
- **Definition:** Form degradation during absence from competition
- **Formula:** `Form_Current = Form_Last × decay^days_inactive`
- **Decay Rate:** Typically 0.95-0.99 per day
- **Purpose:** Players returning from breaks perform worse initially
- **Priority:** MEDIUM

#### 7.2 Form Recovery Rate
- **Definition:** Speed of returning to baseline performance after break
- **Calculation:** Track matches needed to return to pre-break performance
- **Priority:** LOW

#### 7.3 Trailing 90-Day Performance
- **Definition:** Win rate over past 90 days with early-season weighting
- **Formula:** Season progress weights results (later = more important)
- **Priority:** MEDIUM

---

### Category 8: Opponent-Adjusted Features

**Source:** [nba-prediction](https://github.com/cmunch1/nba-prediction), [NBA_RANKINGS](https://github.com/klarsen1/NBA_RANKINGS)

#### 8.1 League-Normalized Performance
- **Definition:** Performance relative to league/tournament average
- **Formula:** `Player_Metric - League_Average_Metric`
- **Purpose:** Adjusts for overall competition quality
- **Priority:** MEDIUM

#### 8.2 Strength of Schedule (SOS)
- **Definition:** Average Elo/skill of recent opponents
- **Formula:** `mean(Opponent_Elo[-N:])`
- **Purpose:** Context for win/loss streaks
- **Priority:** MEDIUM

#### 8.3 Quality-Weighted Win Rate
- **Definition:** Wins against strong opponents count more
- **Formula:** `Σ(win × opponent_elo) / Σ(opponent_elo)`
- **Research:** "Higher importance to wins where opponent has high Elo"
- **Priority:** HIGH - Distinguishes lucky vs meaningful wins

---

### Category 9: Time-Series & Sequential Features

**Source:** arXiv "Capturing Momentum" paper, Hidden Markov Model research

#### 9.1 Hidden Markov Model Momentum State
- **Definition:** Probability of being in "hot" vs "cold" state
- **Method:** HMM trained on point sequences to identify latent momentum states
- **States:** Hot (elevated performance), Neutral, Cold (depressed)
- **Output:** `P(Hot_State)` as feature
- **Priority:** LOW (complex implementation)

#### 9.2 Momentum State Transition Probability
- **Definition:** Likelihood of transitioning between momentum states
- **Use Case:** Predict momentum shifts mid-match
- **Priority:** LOW

#### 9.3 LSTM Momentum Encoding
- **Definition:** Neural network encoding of recent match sequences
- **Note:** Already have LSTM infrastructure in codebase (unused)
- **Priority:** MEDIUM - Reactivate existing capability

---

### Category 10: Schedule & Context Features

**Source:** [NBA_RANKINGS](https://github.com/klarsen1/NBA_RANKINGS), [nfmcclure/NBA_Predictions](https://github.com/nfmcclure/NBA_Predictions)

#### 10.1 Rest Days Advantage
- **Definition:** Differential in days since last match
- **Note:** Already tested (Time_Since_Last_Advantage) - showed zero value
- **Status:** SKIP - Already proven ineffective

#### 10.2 Travel Impact (if applicable)
- **Definition:** Performance degradation from travel
- **Applicability:** May not apply to table tennis tournaments
- **Priority:** LOW

#### 10.3 Tournament Stage Momentum
- **Definition:** Performance trend within current tournament
- **Formula:** Win rate in current tournament matches
- **Source:** Tennis model uses "tournament wins" feature
- **Priority:** MEDIUM

---

## Implementation Recommendations

### Tier 1: High Priority (Likely Highest Impact)

| Feature | Complexity | Expected Impact |
|---------|------------|-----------------|
| **Current Win/Lose Streak** | LOW | HIGH - Critical in all models |
| **EWMA Win Rate** | LOW | HIGH - Better than rolling avg |
| **Weighted Momentum Score** | LOW | HIGH - Novel approach |
| **Dynamic Elo Rating** | MEDIUM | HIGH - Core skill metric |
| **Quality-Weighted Win Rate** | MEDIUM | HIGH - Context for wins |
| **Serve/Receive Win Rate** | MEDIUM | HIGH - Key in racket sports |

### Tier 2: Medium Priority

| Feature | Complexity | Expected Impact |
|---------|------------|-----------------|
| EWMA PDR | LOW | MEDIUM |
| Rolling Elo Momentum | LOW | MEDIUM |
| Elo Velocity | LOW | MEDIUM |
| Momentum Gap | LOW | MEDIUM |
| Break Point Rates | MEDIUM | MEDIUM |
| SOS (Strength of Schedule) | MEDIUM | MEDIUM |
| Tournament Stage Momentum | MEDIUM | MEDIUM |

### Tier 3: Experimental (Lower Priority)

| Feature | Complexity | Expected Impact |
|---------|------------|-----------------|
| Entropy-Weighted Momentum | HIGH | UNKNOWN |
| HMM Momentum States | HIGH | UNKNOWN |
| LSTM Reactivation | MEDIUM | MEDIUM |
| Inactivity Decay | LOW | LOW |
| Surface-Specific Elo | MEDIUM | LOW for TT |

---

## Quick Wins (Implement First)

These can be added with minimal code changes:

### 1. Win Streak / Lose Streak Features
```python
def compute_streaks(results):
    """Compute current win and lose streaks"""
    win_streak = 0
    lose_streak = 0
    for result in reversed(results):
        if result == 1:  # Win
            if lose_streak == 0:
                win_streak += 1
            else:
                break
        else:  # Loss
            if win_streak == 0:
                lose_streak += 1
            else:
                break
    return win_streak, lose_streak
```

### 2. EWMA Win Rate
```python
def ewma_win_rate(results, halflife=5):
    """Exponentially weighted moving average win rate"""
    import pandas as pd
    return pd.Series(results).ewm(halflife=halflife).mean().iloc[-1]
```

### 3. Weighted Momentum Score
```python
def weighted_momentum_score(results_last_6):
    """6-match weighted momentum score (-21 to +21)"""
    weights = [6, 5, 4, 3, 2, 1]  # newest to oldest
    score = sum(w * (1 if r == 'W' else -1 if r == 'L' else 0)
                for w, r in zip(weights, results_last_6))
    return score
```

---

## Sources

### GitHub Repositories
- [ProphitBet-Soccer-Bets-Predictor](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor)
- [Bet-on-Sibyl](https://github.com/jrbadiabo/Bet-on-Sibyl)
- [Tennis-match-prediction-model](https://github.com/nrccstat/Tennis-match-prediction-model)
- [momentum-football-bets](https://github.com/DidierRLopes/momentum-football-bets)
- [deepshot](https://github.com/saccofrancesco/deepshot)
- [NBA_RANKINGS](https://github.com/klarsen1/NBA_RANKINGS)
- [nba-prediction](https://github.com/cmunch1/nba-prediction)
- [tennispredictor](https://github.com/jdlamstein/tennispredictor)
- [Tennis-Prediction](https://github.com/VincentAuriau/Tennis-Prediction)
- [JeffSackmann/tennis_slam_pointbypoint](https://github.com/JeffSackmann/tennis_slam_pointbypoint)

### Academic Research
- "Capturing Momentum: Tennis Match Analysis Using ML and Time Series" (arXiv 2404.13300)
- "Predicting tennis match outcomes mid-game using ML on psychological data" (Journal of Big Data, 2025)
- "Entropy weight-TOPSIS and machine learning model for tennis" (TCSISR, 2024)
- "Random forest model identifies serve strength as key predictor" (J. Sports Analytics)
- "Exponentially Weighted Moving Models" (Stanford/arXiv)
- "Machine learning for sports betting: should model selection..." (arXiv 2303.06021)

---

## Next Steps

1. **Implement Tier 1 features** one at a time
2. **Run A/B tests** against baseline to measure ROI impact
3. **Track feature importance** using existing GBM infrastructure
4. **Remove features with zero impact** (per established methodology)
5. **Consider ensemble approaches** if multiple features show marginal gains
