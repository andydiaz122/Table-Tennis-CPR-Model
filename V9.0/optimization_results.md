# V9.0 Optimization Results Log

## Baseline Performance (V8.0 - 12 Features)
- **ROI:** 2.44%
- **Max Drawdown:** 34.12%
- **Sharpe Ratio:** ~2.5
- **Features:** 12

---

## Experiment 1: First_Set_Win_Rate_L20_Advantage

**Date:** 2026-01-04
**Feature Added:** `First_Set_Win_Rate_L20_Advantage`
**Total Features:** 13

### Hypothesis
Players who consistently win the first set may have a psychological or physical advantage that carries through the match. This feature captures the L20 rolling first-set win rate differential.

### Results

| Metric | Baseline | With Feature | Change |
|--------|----------|--------------|--------|
| ROI | 2.44% | 1.50% | -0.94% |
| Max Drawdown | 34.12% | 34.12% | 0.00% |
| Sharpe Ratio | ~2.5 | 2.02 | -0.48 |
| Total Bets | ~4,500 | 4,509 | ~0 |

### Performance Analysis (Wide Filter Exploration)
```
Performance by Market Odds:
- Heavy Favorite (<1.5): -2.52% ROI
- Favorite (1.5-2.0): -1.54% ROI
- Underdog (2.0-3.0): -2.56% ROI
- Longshot (>3.0): -28.04% ROI

Performance by Model Edge:
- Mega Edge (50%+): -0.24% ROI
- High Edge (25-50%): -2.48% ROI
- Medium Edge (10-25%): -3.07% ROI
- Low Edge (0-10%): -12.05% ROI

Notable Profitable Categories:
- Higher PDR: +11.11% ROI (2181 bets)
- Bet on Less Rested: +7.10% ROI (1048 bets)
- Played More Matches (24H): +8.07% ROI (1821 bets)
- Opponent's First Match: +11.17% ROI (752 bets)
- Even Form (L20): +6.91% ROI (691 bets)
```

### Conclusion
**NEGATIVE IMPACT** - The `First_Set_Win_Rate_L20_Advantage` feature degraded model performance.

### Recommendation
**REMOVE** this feature from the model and revert to 12-feature baseline.

---

## Model Configuration Log

### Best GBM Parameters (GridSearchCV)
```python
{
    'learning_rate': 0.05,
    'max_depth': 2,
    'min_samples_leaf': 50,
    'n_estimators': 100,
    'subsample': 0.7
}
```

### Current Feature List (13 features - to be reverted to 12)
1. Time_Since_Last_Advantage
2. Matches_Last_24H_Advantage
3. Is_First_Match_Advantage
4. PDR_Slope_Advantage
5. H2H_P1_Win_Rate
6. H2H_Dominance_Score
7. Daily_Fatigue_Advantage
8. PDR_Advantage
9. Win_Rate_Advantage
10. Win_Rate_L5_Advantage
11. Close_Set_Win_Rate_Advantage
12. Set_Comebacks_Advantage
13. ~~First_Set_Win_Rate_L20_Advantage~~ (REMOVE)

---

## Next Steps
1. Remove `First_Set_Win_Rate_L20_Advantage` from feature list
2. Re-run pipeline with 12 features to verify baseline restoration
3. Test next candidate feature individually
