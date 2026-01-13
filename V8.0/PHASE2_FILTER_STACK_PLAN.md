# Phase 2: Filter Stack Design (Multi-Agent Architecture)

## Current State
- **Commit:** `71a7d10` (Phase 1.1-1.3 complete)
- **Branch:** `feature-research-github-scrape`
- **Baseline Metrics:** Sharpe 5.56, Max DD 9.51%, ROI 102.87%, Bets 4,509

---

## Multi-Agent Architecture

### Agent 1: `filter-logic-architect` (The Builder)
**Problem:** Dynamic Dependency - Order matters! Filter by Edge first, then cap = wrong Edge values.

**Tasks:**
1. Implement H2H Regularization as PRE-FILTER transformation
2. Recalculate Edge AFTER Model_Prob is capped
3. Write `filter_stack_v8.py` with strict separation: "Data Transformations" vs "Boolean Masks"

### Agent 2: `volume-attrition-analyst` (The Guardian)
**Problem:** Sample Size - Aggressive filters killed previous experiments.

**Tasks:**
1. Run Waterfall Analysis at every step
2. HALT if `bets_remaining < 1,200` (safety buffer)
3. Propose loosening actions if sample critical

### Agent 3: `calibration-integrity-specialist`
**Problem:** Root Cause - Model is 25pp miscalibrated on H2H=100%.

**Tasks:**
1. Validate Dead Zones (Prob > 0.75) strictly capped
2. Verify H2H logic: `H2H_Win_Rate == 1.0 AND n_h2h_games >= 2`
3. Ensure Calibration-Weighted Kelly applied to FINAL filtered set

### Agent 4: `metric-distortion-auditor`
**Problem:** False Success - Filtering often increases ROI by removing losers (overfitting).

**Tasks:**
1. Compare metrics: Unfiltered → Tier 1 → Tier 2 → Tier 3
2. Check Drawdown Duration
3. Generate CSVs proving ROI/Sharpe gains are valid

### Agent 5: `strategic-pipeline-overseer` (The Boss)
**Problem:** Tunnel Vision - Ensure team solved H2H problem, not just "made it run."

**Tasks:**
1. Final Gatekeeper: Reject if `bets < 1000` OR `Sharpe < 5.56`
2. Verify NO circuit breakers implemented
3. Write Phase 3 handoff

---

## Critical: Order of Operations

```
STEP 1: LOAD RAW DATA
         │
         ▼
STEP 2: DATA TRANSFORMATIONS (Before filtering!)
   2a. H2H Perfect Cap:
       IF H2H_Win_Rate == 1.0 AND H2H_games >= 2:
           Model_Prob_Adjusted = min(Model_Prob, 0.70)

   2b. Dead Zone Cap:
       IF Model_Prob > 0.75:
           Model_Prob_Adjusted = 0.75

   2c. RECALCULATE EDGE (Critical!):
       Edge_Adjusted = Model_Prob_Adjusted - (1 / Market_Odds)
         │
         ▼
STEP 3: BOOLEAN MASK FILTERS
   TIER 1 (Hard): Edge 0.03-0.25, Min_Games >= 12, Odds 1.10-4.00
   TIER 2 (Calibration): Model_Prob 0.35-0.75, Divergence 0.03-0.20
   TIER 3 (Dynamic): Kelly adjustment only (no filter)
         │
         ▼
STEP 4: APPLY CALIBRATION-WEIGHTED KELLY
         │
         ▼
STEP 5: OUTPUT METRICS & WATERFALL
```

---

## Hard Rules (DO NOT IMPLEMENT)

| Forbidden | Reason |
|-----------|--------|
| Circuit Breakers | Reduces profit $1,600→$978 |
| Late Night Filter | Late night outperforms 2.10%>1.50% |
| Loss Streak Halts | Streaks normal, model recovers |
| Platt Scaling | Band-aid that hides feature rot |

---

## Success Criteria

| Metric | Requirement | Baseline |
|--------|-------------|----------|
| Bets Remaining | ≥ 1,000 | 4,509 |
| Sharpe Ratio | > 5.56 | 5.56 |
| Max Drawdown | < 9.51% | 9.51% |
| Dead Zone Bets | 0 (after cap) | 1,047 |

---

## Failure Recovery

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Over-filtering | Bets < 1,000 | Loosen Min_Games 12→10, Edge 3%→2.5% |
| Wrong order | Edge before H2H cap | Refactor: Transform → Recalc → Filter |
| H2H false positives | 1-win/1-game capped | Add `n_h2h_games >= 2` |
| Sharpe regression | Sharpe < 5.56 | Review which filter caused it |

---

## Deliverables

- [ ] `filter_stack_v8.py` with correct order-of-operations
- [ ] Waterfall analysis at each stage
- [ ] Final bet count ≥ 1,000
- [ ] Sharpe > 5.56 baseline
- [ ] No circuit breakers
- [ ] Metrics comparison CSV
- [ ] Git commit and push
- [ ] Phase 3 handoff
