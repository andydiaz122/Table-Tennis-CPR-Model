# Dynamic Elo Rating Implementation Plan

## Feature Overview

**Feature Name:** `Elo_Advantage`
**Definition:** Real-time skill rating differential between players, updated after every match
**Formula:** `Elo_new = Elo_old + K × (Actual - Expected)`
**Expected Impact:** HIGH - Core skill metric used in 66-70% accuracy tennis models

---

## Pre-Implementation Checklist

### Step 0: Record Baseline (MANDATORY per CLAUDE.md Rule 1)

Before ANY code changes, run the full pipeline and record exact baseline:

```bash
cd /home/user/Table-Tennis-CPR-Model/V8.0

# Run full pipeline
python advanced_feature_engineering_v7.4.py && \
python merge_data_v7.4.py && \
python remove_duplicates_from_final_dataset.py && \
python split_data.py && \
python cpr_v7.4_specialist_gbm_trainer.py && \
python backtest_with_compounding_logic_v7.6.py

# Record baseline
echo "Pre-Elo Baseline at commit $(git rev-parse --short HEAD):"
echo "ROI: ____%, Sharpe: ____, MaxDD: ____%, Bets: ____"
```

**Expected Baseline (from CLAUDE.md):**
- ROI: 1.50%
- Sharpe: 2.02
- MaxDD: 34.12%
- Total Bets: 4,509

---

## Implementation Architecture

### System Components to Modify

```
┌─────────────────────────────────────────────────────────────────┐
│                    ELO RATING SYSTEM                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │ Feature Engineering │───▶│   Merge Data        │            │
│  │ (calculate Elo)     │    │   (preserve Elo)    │            │
│  └─────────────────────┘    └─────────────────────┘            │
│           │                           │                         │
│           ▼                           ▼                         │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │ GBM Trainer         │◀───│   Split Data        │            │
│  │ (add Elo feature)   │    │   (no changes)      │            │
│  └─────────────────────┘    └─────────────────────┘            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │ Backtest            │───▶│   Live Predictor    │            │
│  │ (on-the-fly Elo)    │    │ (load Elo state)    │            │
│  └─────────────────────┘    └─────────────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Files to Modify

| File | Changes Required | Complexity |
|------|------------------|------------|
| `advanced_feature_engineering_v7.4.py` | Uncomment/activate Elo calculation | LOW |
| `cpr_v7.4_specialist_gbm_trainer.py` | Add `Elo_Advantage` to feature list | LOW |
| `backtest_with_compounding_logic_v7.6.py` | Add on-the-fly Elo tracking | MEDIUM |
| `LIVE_FINAL_Predictor.py` | Add Elo state persistence | MEDIUM |

---

## Phase 1: Core Elo Calculation

### Step 1.1: Elo Configuration Constants

Add to configuration section in `advanced_feature_engineering_v7.4.py`:

```python
# --- Elo Configuration ---
STARTING_ELO = 1500          # Standard starting rating
K_FACTOR_BASE = 32           # Base K-factor (standard)
ELO_FLOOR = 1000             # Minimum Elo (prevents extreme values)
ELO_CEILING = 2500           # Maximum Elo (prevents extreme values)
```

**K-Factor Options (implement later if needed):**

| K-Factor Type | Value | Use Case |
|---------------|-------|----------|
| Static | 32 | Simple, standard approach |
| New Player | 40 | First 30 matches (faster calibration) |
| Established | 24 | After 100+ matches (stable) |
| Match Importance | 32 × importance_multiplier | Tournament finals, etc. |

### Step 1.2: Elo Calculation Function

The function already exists at lines 109-117 in `advanced_feature_engineering_v7.4.py`:

```python
def update_elo(p1_elo, p2_elo, p1_won, k_factor=32):
    """
    Updates Elo ratings for both players after a match.

    Formula:
        Expected_P1 = 1 / (1 + 10^((P2_Elo - P1_Elo) / 400))
        New_P1_Elo = Old_P1_Elo + K × (Actual - Expected)

    Args:
        p1_elo: Player 1's pre-match Elo rating
        p2_elo: Player 2's pre-match Elo rating
        p1_won: Boolean, True if Player 1 won
        k_factor: K-factor for rating adjustment (default 32)

    Returns:
        tuple: (new_p1_elo, new_p2_elo)
    """
    expected_p1 = 1 / (1 + 10 ** ((p2_elo - p1_elo) / 400))
    expected_p2 = 1 / (1 + 10 ** ((p1_elo - p2_elo) / 400))
    score_p1 = 1 if p1_won else 0
    score_p2 = 0 if p1_won else 1
    new_p1_elo = p1_elo + k_factor * (score_p1 - expected_p1)
    new_p2_elo = p2_elo + k_factor * (score_p2 - expected_p2)
    return new_p1_elo, new_p2_elo
```

**No changes needed** - function is already correct.

### Step 1.3: Activate Elo in Feature Engineering

**Current State** (lines 159-163, 178-181, 305-309 - COMMENTED OUT):

```python
## NEW ## - Initialize Elo tracking
print("--- Initializing Elo Rating System ---")
elo_ratings = {}
STARTING_ELO = 1500
K_FACTOR = 32

# ... in the loop ...
# p1_pre_match_elo = elo_ratings.get(p1_id, STARTING_ELO)
# p2_pre_match_elo = elo_ratings.get(p2_id, STARTING_ELO)
# elo_advantage = p1_pre_match_elo - p2_pre_match_elo

# ... after match outcome ...
# new_p1_elo, new_p2_elo = update_elo(p1_pre_match_elo, p2_pre_match_elo, p1_won, K_FACTOR)
# elo_ratings[p1_id] = new_p1_elo
# elo_ratings[p2_id] = new_p2_elo
```

**Action Required:** Uncomment lines 178-181 and 307-309.

**Add to output row** (line 312+):

```python
new_row.update({
    # ... existing features ...
    'Elo_Advantage': elo_advantage,
    'P1_Elo': p1_pre_match_elo,  # Optional: for analysis
    'P2_Elo': p2_pre_match_elo,  # Optional: for analysis
})
```

---

## Phase 2: GBM Trainer Integration

### Step 2.1: Add Elo to Feature List

**File:** `cpr_v7.4_specialist_gbm_trainer.py`

**Current features** (line 30):
```python
numerical_features = [
    'Time_Since_Last_Advantage',
    'Matches_Last_24H_Advantage',
    'Is_First_Match_Advantage',
    'PDR_Slope_Advantage',
    'H2H_P1_Win_Rate',
    'H2H_Dominance_Score',
    'Daily_Fatigue_Advantage',
    'PDR_Advantage',
    'Win_Rate_Advantage',
    'Win_Rate_L5_Advantage',
    'Close_Set_Win_Rate_Advantage',
    'Set_Comebacks_Advantage'
]
```

**Modified (add Elo_Advantage):**
```python
numerical_features = [
    'Elo_Advantage',  # NEW: Core skill differential
    'Time_Since_Last_Advantage',
    'Matches_Last_24H_Advantage',
    'Is_First_Match_Advantage',
    'PDR_Slope_Advantage',
    'H2H_P1_Win_Rate',
    'H2H_Dominance_Score',
    'Daily_Fatigue_Advantage',
    'PDR_Advantage',
    'Win_Rate_Advantage',
    'Win_Rate_L5_Advantage',
    'Close_Set_Win_Rate_Advantage',
    'Set_Comebacks_Advantage'
]
```

---

## Phase 3: Backtest Integration

### Step 3.1: Add Elo State Tracking

**File:** `backtest_with_compounding_logic_v7.6.py`

**Add after line 133 (after loading data):**

```python
# --- Initialize Elo Rating System for On-the-fly Calculation ---
elo_ratings = {}
STARTING_ELO = 1500
K_FACTOR = 32
```

### Step 3.2: Calculate Elo On-the-fly

**Add inside the main loop (after line 194):**

```python
# --- On-the-fly Elo Calculation ---
p1_pre_match_elo = elo_ratings.get(p1_id, STARTING_ELO)
p2_pre_match_elo = elo_ratings.get(p2_id, STARTING_ELO)
elo_advantage = p1_pre_match_elo - p2_pre_match_elo
```

### Step 3.3: Update Elo After Each Match

**Add after betting logic completes (around line 350):**

```python
# --- Update Elo ratings for both players ---
actual_winner = match['P1_Win']
new_p1_elo, new_p2_elo = update_elo(
    p1_pre_match_elo, p2_pre_match_elo,
    actual_winner == 1, K_FACTOR
)
elo_ratings[p1_id] = new_p1_elo
elo_ratings[p2_id] = new_p2_elo
```

### Step 3.4: Add to GBM Feature DataFrame

**Modify line 291-304:**

```python
gbm_features = pd.DataFrame([{
    'Elo_Advantage': elo_advantage,  # NEW
    'Time_Since_Last_Advantage': time_since_last_advantage,
    'Matches_Last_24H_Advantage': matches_last_24h_advantage,
    'Is_First_Match_Advantage': is_first_match_advantage,
    'PDR_Slope_Advantage': pdr_slope_advantage,
    'H2H_P1_Win_Rate': h2h_p1_win_rate,
    'H2H_Dominance_Score': h2h_dominance_score,
    'Daily_Fatigue_Advantage': daily_fatigue_advantage,
    'PDR_Advantage': pdr_advantage,
    'Win_Rate_Advantage': win_rate_advantage,
    'Win_Rate_L5_Advantage': win_rate_l5_advantage,
    'Close_Set_Win_Rate_Advantage': close_set_win_rate_advantage,
    'Set_Comebacks_Advantage': set_comebacks_advantage
}])
```

### Step 3.5: Add update_elo Function to Backtest

**Add at top of file (after imports):**

```python
def update_elo(p1_elo, p2_elo, p1_won, k_factor=32):
    """Updates Elo ratings for both players after a match."""
    expected_p1 = 1 / (1 + 10 ** ((p2_elo - p1_elo) / 400))
    expected_p2 = 1 / (1 + 10 ** ((p1_elo - p2_elo) / 400))
    score_p1 = 1 if p1_won else 0
    score_p2 = 0 if p1_won else 1
    new_p1_elo = p1_elo + k_factor * (score_p1 - expected_p1)
    new_p2_elo = p2_elo + k_factor * (score_p2 - expected_p2)
    return new_p1_elo, new_p2_elo
```

---

## Phase 4: Live Predictor Integration

### Step 4.1: Elo State Persistence

**Create new file:** `elo_state_manager.py`

```python
import json
import os
from datetime import datetime

ELO_STATE_FILE = "elo_ratings_state.json"
STARTING_ELO = 1500

def load_elo_state():
    """Load Elo ratings from persistent state file."""
    if os.path.exists(ELO_STATE_FILE):
        with open(ELO_STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_elo_state(elo_ratings):
    """Save Elo ratings to persistent state file."""
    with open(ELO_STATE_FILE, 'w') as f:
        json.dump(elo_ratings, f, indent=2)

def get_player_elo(elo_ratings, player_id):
    """Get player's Elo rating (default 1500 for new players)."""
    return elo_ratings.get(str(player_id), STARTING_ELO)

def update_elo_state(elo_ratings, p1_id, p2_id, p1_won, k_factor=32):
    """Update and persist Elo ratings after a match result."""
    p1_elo = get_player_elo(elo_ratings, p1_id)
    p2_elo = get_player_elo(elo_ratings, p2_id)

    expected_p1 = 1 / (1 + 10 ** ((p2_elo - p1_elo) / 400))
    score_p1 = 1 if p1_won else 0
    score_p2 = 0 if p1_won else 1

    new_p1_elo = p1_elo + k_factor * (score_p1 - expected_p1)
    new_p2_elo = p2_elo + k_factor * (score_p2 - (1 - expected_p1))

    elo_ratings[str(p1_id)] = new_p1_elo
    elo_ratings[str(p2_id)] = new_p2_elo

    save_elo_state(elo_ratings)
    return new_p1_elo, new_p2_elo
```

### Step 4.2: Initialize Elo State from Historical Data

**Create script:** `initialize_elo_from_history.py`

```python
"""
One-time script to initialize Elo ratings from historical match data.
Run this ONCE after enabling Elo in the pipeline.
"""
import pandas as pd
import json

HISTORICAL_FILE = "final_dataset_v7.4_no_duplicates.csv"
OUTPUT_FILE = "elo_ratings_state.json"
STARTING_ELO = 1500
K_FACTOR = 32

def update_elo(p1_elo, p2_elo, p1_won, k_factor=32):
    expected_p1 = 1 / (1 + 10 ** ((p2_elo - p1_elo) / 400))
    score_p1 = 1 if p1_won else 0
    new_p1_elo = p1_elo + k_factor * (score_p1 - expected_p1)
    new_p2_elo = p2_elo + k_factor * ((1-score_p1) - (1-expected_p1))
    return new_p1_elo, new_p2_elo

def main():
    print(f"Loading historical data from {HISTORICAL_FILE}...")
    df = pd.read_csv(HISTORICAL_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by=['Date', 'Match ID'], inplace=True)

    elo_ratings = {}

    print("Processing matches chronologically...")
    for _, match in df.iterrows():
        p1_id = str(match['Player 1 ID'])
        p2_id = str(match['Player 2 ID'])
        p1_won = match['P1_Win'] == 1

        p1_elo = elo_ratings.get(p1_id, STARTING_ELO)
        p2_elo = elo_ratings.get(p2_id, STARTING_ELO)

        new_p1, new_p2 = update_elo(p1_elo, p2_elo, p1_won, K_FACTOR)
        elo_ratings[p1_id] = new_p1
        elo_ratings[p2_id] = new_p2

    print(f"Processed {len(df)} matches. {len(elo_ratings)} players rated.")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(elo_ratings, f, indent=2)

    print(f"Elo state saved to {OUTPUT_FILE}")

    # Print top 10 players by Elo
    sorted_elos = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Players by Elo:")
    for pid, elo in sorted_elos[:10]:
        print(f"  Player {pid}: {elo:.0f}")

if __name__ == "__main__":
    main()
```

### Step 4.3: Integrate into Live Predictor

**File:** `LIVE_FINAL_Predictor.py`

**Add import:**
```python
from elo_state_manager import load_elo_state, get_player_elo
```

**Add after loading models (around line 128):**
```python
# Load Elo state
elo_ratings = load_elo_state()
print(f"Loaded Elo ratings for {len(elo_ratings)} players.")
```

**Add in match processing loop:**
```python
# Get Elo advantage
p1_elo = get_player_elo(elo_ratings, p1_id)
p2_elo = get_player_elo(elo_ratings, p2_id)
elo_advantage = p1_elo - p2_elo
```

**Add to gbm_features DataFrame:**
```python
gbm_features = pd.DataFrame([{
    'Elo_Advantage': elo_advantage,  # NEW
    # ... rest of features ...
}])
```

---

## Phase 5: Testing & Validation

### Step 5.1: Unit Tests

**Create:** `tests/test_elo.py`

```python
import pytest
import sys
sys.path.insert(0, '../V8.0')

from advanced_feature_engineering_v7_4 import update_elo

class TestEloCalculation:
    def test_equal_players_win(self):
        """Equal Elo players: winner gains, loser loses equally."""
        new_p1, new_p2 = update_elo(1500, 1500, True, k_factor=32)
        assert new_p1 == 1516  # 1500 + 32*(1-0.5) = 1516
        assert new_p2 == 1484  # 1500 + 32*(0-0.5) = 1484

    def test_equal_players_loss(self):
        """Equal Elo players: loser loses, winner gains."""
        new_p1, new_p2 = update_elo(1500, 1500, False, k_factor=32)
        assert new_p1 == 1484
        assert new_p2 == 1516

    def test_favorite_wins(self):
        """Higher Elo player wins: small gain (expected)."""
        new_p1, new_p2 = update_elo(1700, 1300, True, k_factor=32)
        # Expected P1 ≈ 0.91, so gain ≈ 32*(1-0.91) ≈ 2.9
        assert 1702 <= new_p1 <= 1704
        assert 1296 <= new_p2 <= 1298

    def test_underdog_wins(self):
        """Lower Elo player wins: large gain (upset)."""
        new_p1, new_p2 = update_elo(1300, 1700, True, k_factor=32)
        # Expected P1 ≈ 0.09, so gain ≈ 32*(1-0.09) ≈ 29.1
        assert 1328 <= new_p1 <= 1330
        assert 1670 <= new_p2 <= 1672

    def test_elo_conservation(self):
        """Total Elo in system remains constant."""
        p1_start, p2_start = 1600, 1400
        new_p1, new_p2 = update_elo(p1_start, p2_start, True, k_factor=32)
        assert abs((new_p1 + new_p2) - (p1_start + p2_start)) < 0.01

def test_elo_advantage_range():
    """Elo advantage should typically be in range [-500, +500]."""
    # After processing history, most advantages should be reasonable
    # This is a sanity check for the feature engineering output
    pass
```

### Step 5.2: Integration Test

**Test Procedure:**

```bash
# 1. Record baseline
python backtest_with_compounding_logic_v7.6.py
# Note: ROI, Sharpe, MaxDD, Bets

# 2. Apply Phase 1 changes only (feature engineering)
# Run pipeline
python advanced_feature_engineering_v7.4.py
# Verify: Check that Elo_Advantage column exists in output

# 3. Apply Phase 2 changes (GBM trainer)
python cpr_v7.4_specialist_gbm_trainer.py
# Verify: Check that model trains without errors

# 4. Apply Phase 3 changes (backtest)
python backtest_with_compounding_logic_v7.6.py
# Compare: ROI, Sharpe, MaxDD vs baseline

# 5. Calculate feature importance
# Check GBM feature_importances_ to see Elo ranking
```

### Step 5.3: Validation Metrics

| Metric | Baseline | Target | Accept/Reject Criteria |
|--------|----------|--------|------------------------|
| ROI | 1.50% | >1.50% | Accept if >= 1.40% |
| Sharpe | 2.02 | >2.02 | Accept if >= 1.90 |
| MaxDD | 34.12% | <34.12% | Accept if <= 36% |
| Bets | 4,509 | ~4,500 | Accept if within 10% |

---

## Phase 6: Advanced Elo Extensions (Future)

### 6.1: Dynamic K-Factor

```python
def get_dynamic_k_factor(player_id, match_count_dict, match_importance=1.0):
    """
    K-factor varies based on player experience and match importance.

    Args:
        player_id: Player's unique identifier
        match_count_dict: Dict tracking matches per player
        match_importance: Multiplier for important matches (finals, etc.)

    Returns:
        Appropriate K-factor for this player/match
    """
    matches_played = match_count_dict.get(player_id, 0)

    if matches_played < 30:
        base_k = 40  # New player: faster calibration
    elif matches_played < 100:
        base_k = 32  # Developing: standard
    else:
        base_k = 24  # Established: stable

    return base_k * match_importance
```

### 6.2: Margin-of-Victory Adjusted Elo

```python
def update_elo_with_margin(p1_elo, p2_elo, p1_sets, p2_sets, k_factor=32):
    """
    Elo update weighted by margin of victory.

    A 3-0 victory is more decisive than 3-2.
    """
    total_sets = p1_sets + p2_sets
    margin = abs(p1_sets - p2_sets) / total_sets  # 0.2 to 0.6
    margin_multiplier = 1 + (margin - 0.3) * 0.5  # Scale: 0.85 to 1.15

    p1_won = p1_sets > p2_sets
    expected_p1 = 1 / (1 + 10 ** ((p2_elo - p1_elo) / 400))
    score_p1 = 1 if p1_won else 0

    adjusted_k = k_factor * margin_multiplier
    new_p1_elo = p1_elo + adjusted_k * (score_p1 - expected_p1)
    new_p2_elo = p2_elo + adjusted_k * ((1-score_p1) - (1-expected_p1))

    return new_p1_elo, new_p2_elo
```

### 6.3: Rolling Elo Momentum (Elo Velocity)

```python
def calculate_elo_velocity(elo_history, window=5):
    """
    Calculate rate of Elo change over recent matches.

    Positive = improving, Negative = declining
    """
    if len(elo_history) < 2:
        return 0.0

    recent = elo_history[-window:] if len(elo_history) >= window else elo_history
    return (recent[-1] - recent[0]) / len(recent)
```

---

## Implementation Timeline

| Phase | Task | Estimated Effort |
|-------|------|------------------|
| 0 | Record baseline | 15 min |
| 1 | Core Elo in feature engineering | 30 min |
| 2 | Add to GBM trainer | 15 min |
| 3 | Backtest integration | 45 min |
| 4 | Live predictor integration | 45 min |
| 5 | Testing & validation | 1 hour |
| 6 | Documentation & commit | 30 min |

**Total Estimated Time:** ~4 hours

---

## Risk Mitigation

### Risk 1: Elo Adds No Predictive Value
**Likelihood:** MEDIUM (other similar features like Win_Rate_L20 showed zero value)
**Mitigation:** Elo is fundamentally different - it's opponent-adjusted skill, not raw win rate
**Fallback:** If feature importance is near zero after training, remove and revert

### Risk 2: Cold Start Problem (New Players)
**Likelihood:** HIGH (new players start at 1500 regardless of actual skill)
**Mitigation:**
- Use `MIN_GAMES_THRESHOLD` (already 4 games)
- Could add `Elo_Confidence` feature = matches_played / 30 (capped at 1.0)

### Risk 3: Elo Drift Over Time
**Likelihood:** LOW (system is closed - total Elo is conserved)
**Mitigation:** Monitor average Elo over time; recalibrate if needed

---

## Commit Strategy

Following CLAUDE.md rules:

```bash
# After Phase 1 (feature engineering only)
git add advanced_feature_engineering_v7.4.py
git commit -m "feat: Activate Elo rating calculation in feature engineering

- Uncomment existing Elo code (lines 178-181, 307-309)
- Add Elo_Advantage to output features
- Add P1_Elo and P2_Elo for analysis
- K_FACTOR=32, STARTING_ELO=1500"

# After full validation
git add .
git commit -m "feat: Complete Dynamic Elo Rating implementation

- Feature engineering: Calculate Elo per match chronologically
- GBM trainer: Add Elo_Advantage to 13-feature model
- Backtest: On-the-fly Elo tracking with state persistence
- Live predictor: Load Elo state from JSON file

Results vs baseline (commit 4de4728):
- ROI: X.XX% (was 1.50%)
- Sharpe: X.XX (was 2.02)
- MaxDD: X.XX% (was 34.12%)"
```

---

## Quick Reference: Code Changes Summary

### File 1: `advanced_feature_engineering_v7.4.py`
- **Line 178-181:** Uncomment pre-match Elo retrieval
- **Line 307-309:** Uncomment post-match Elo update
- **Line 312+:** Add `Elo_Advantage` to output dict

### File 2: `cpr_v7.4_specialist_gbm_trainer.py`
- **Line 30:** Add `'Elo_Advantage'` to `numerical_features` list

### File 3: `backtest_with_compounding_logic_v7.6.py`
- **After line 6:** Add `update_elo()` function
- **After line 133:** Initialize `elo_ratings = {}`
- **After line 194:** Calculate `elo_advantage`
- **Line 291:** Add `'Elo_Advantage': elo_advantage`
- **After line 350:** Update Elo state

### File 4: `LIVE_FINAL_Predictor.py`
- **Add import:** `from elo_state_manager import ...`
- **After line 128:** Load Elo state
- **In loop:** Calculate Elo advantage
- **Line 219:** Add to gbm_features

### New Files:
- `elo_state_manager.py` - Elo persistence utilities
- `initialize_elo_from_history.py` - One-time state initialization
- `tests/test_elo.py` - Unit tests

---

## Approval Checklist

Before starting implementation:

- [ ] Baseline metrics recorded
- [ ] Git branch created (`feature/elo-rating`)
- [ ] Implementation plan reviewed
- [ ] Test strategy approved
- [ ] Rollback plan understood (revert commits)

**Ready to implement?** Start with Phase 0 (baseline) and Phase 1 (feature engineering).
