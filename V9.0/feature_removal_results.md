# Feature Removal Analysis Results

## Objective
Systematically test removing each of the 12 baseline features to identify the minimal optimal feature set.

## Baseline Performance (12 Features)
| Metric | Value |
|--------|-------|
| ROI | 2.44% |
| Max Drawdown | 34.12% |
| Sharpe Ratio | 2.80 |
| Total Bets | 4,503 |

## Target Criteria
- ROI > 2%
- Sharpe Ratio > 2.5
- Max Drawdown < 35%

---

## Summary Table (All 12 Tests Completed)

| # | Feature | ROI | Delta | Decision |
|---|---------|-----|-------|----------|
| 1 | Set_Comebacks_Advantage | 2.43% | -0.01% | **KEEP** |
| 2 | Is_First_Match_Advantage | 2.44% | 0% | **REMOVE** |
| 3 | Matches_Last_24H_Advantage | 2.44% | 0% | **REMOVE** |
| 4 | H2H_Dominance_Score | 2.31% | -0.13% | **KEEP** |
| 5 | PDR_Slope_Advantage | 2.44% | 0% | **REMOVE** |
| 6 | Daily_Fatigue_Advantage | 2.43% | -0.01% | **REMOVE** |
| 7 | Close_Set_Win_Rate_Advantage | 2.34% | -0.10% | **KEEP** |
| 8 | Win_Rate_L5_Advantage | 2.31% | -0.13% | **KEEP** |
| 9 | Time_Since_Last_Advantage | 2.44% | 0% | **REMOVE** |
| 10 | H2H_P1_Win_Rate | 1.88% | -0.56% | **KEEP** (CRITICAL) |
| 11 | Win_Rate_Advantage | 2.47% | +0.03% | **REMOVE** |
| 12 | PDR_Advantage | 2.35% | -0.09% | **KEEP** |

---

## Final Optimal Feature Set (6 Features)

### Features to REMOVE (6):
1. `Time_Since_Last_Advantage` - 0% impact
2. `Matches_Last_24H_Advantage` - 0% impact
3. `Is_First_Match_Advantage` - 0% impact
4. `PDR_Slope_Advantage` - 0% impact
5. `Daily_Fatigue_Advantage` - -0.01% impact (negligible)
6. `Win_Rate_Advantage` - +0.03% impact (removal improves ROI!)

### Features to KEEP (6):
1. `H2H_P1_Win_Rate` - **CRITICAL** (-0.56% if removed, DD exceeds 35%)
2. `H2H_Dominance_Score` - Important (-0.13% if removed)
3. `PDR_Advantage` - Important (-0.09% if removed)
4. `Win_Rate_L5_Advantage` - Important (-0.13% if removed, DD exceeds 35%)
5. `Close_Set_Win_Rate_Advantage` - Important (-0.10% if removed)
6. `Set_Comebacks_Advantage` - Marginal (-0.01% if removed)

---

## Final Validation (6-Feature Model)

| Metric | Baseline (12) | Optimized (6) | Change |
|--------|---------------|---------------|--------|
| ROI | 2.44% | **2.45%** | +0.01% |
| Max Drawdown | 34.12% | 34.19% | +0.07% |
| Sharpe Ratio | 2.80 | 2.78 | -0.02 |
| Total Bets | 4,503 | 4,812 | +309 |
| Features | 12 | **6** | -50% |

### All Targets Met:
- ROI > 2%: **2.45%** (PASS)
- Sharpe > 2.5: **2.78** (PASS)
- Max DD < 35%: **34.19%** (PASS)

---

## Key Findings

### Most Important Features (by impact when removed):
1. **H2H_P1_Win_Rate** - Critical! Removal causes -0.56% ROI drop
2. **H2H_Dominance_Score** - -0.13% impact
3. **Win_Rate_L5_Advantage** - -0.13% impact (also affects drawdown)
4. **Close_Set_Win_Rate_Advantage** - -0.10% impact
5. **PDR_Advantage** - -0.09% impact
6. **Set_Comebacks_Advantage** - -0.01% impact (marginal)

### Features with Zero Predictive Value:
- Time_Since_Last_Advantage
- Matches_Last_24H_Advantage
- Is_First_Match_Advantage
- PDR_Slope_Advantage

### Surprising Result:
- `Win_Rate_Advantage` (L20) removal actually IMPROVED ROI by +0.03%
- This suggests L5 hot streak captures the signal better than L20 form

---

## Recommendations

1. **Deploy 6-feature model** - Same or better performance with 50% less complexity
2. **Monitor H2H_P1_Win_Rate** - Most critical feature for model performance
3. **Consider removing Set_Comebacks_Advantage** in future if further simplification needed
4. **L5 win rate preferred over L20** - Hot streak signal more valuable than longer form
