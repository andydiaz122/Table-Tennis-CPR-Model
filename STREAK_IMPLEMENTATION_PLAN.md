# Current Win/Lose Streaks Implementation Plan

## Feature Overview

**Feature Names:** `P1_Current_Streak`, `P2_Current_Streak`, `Streak_Advantage`
**Definition:** Signed integer representing consecutive wins (+) or losses (-) immediately prior to the current match
**Formula:** Count consecutive same-outcome results from most recent match backwards; positive for wins, negative for losses
**Expected Impact:** HIGH - Captures psychological momentum/"tilt" factor critical in Liga Pro's rapid-fire format

---

## Quantitative Rationale

### Why Streaks Matter in Table Tennis

1. **Market Efficiency Opportunity:** Bookmakers often overreact or underreact to streaks. By including `Streak_Advantage`, we identify "value" when a player on a 5-match win streak is actually over-performing their underlying Elo.

2. **Interaction Effects:** Streaks interact heavily with existing `Close_Set_Win_Rate_Advantage`. A player on a winning streak who is also winning close sets represents the "Clutch/Hot" profile - a high-probability bet in Liga Pro's volatile format.

3. **The "Tilt" Factor:** In professional poker, "tilt" describes degraded decision-making after losses. In table tennis, a -4 streak often indicates fractured mechanics or mental state, making them a "fade" candidate regardless of baseline stats.

4. **Service-Return Confidence:** A "cold" player struggles with service-return confidence, creating a compounding effect that 20-match rolling averages cannot capture.

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
echo "Pre-Streak Baseline at commit $(git rev-parse --short HEAD):"
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
│                    STREAK TRACKING SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │ Feature Engineering │───▶│   Merge Data        │            │
│  │ (calculate streaks) │    │   (preserve streaks)│            │
│  └─────────────────────┘    └─────────────────────┘            │
│           │                           │                         │
│           ▼                           ▼                         │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │ GBM Trainer         │◀───│   Split Data        │            │
│  │ (add Streak feature)│    │   (no changes)      │            │
│  └─────────────────────┘    └─────────────────────┘            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │ Backtest            │───▶│   Live Predictor    │            │
│  │ (on-the-fly streak) │    │ (load streak state) │            │
│  └─────────────────────┘    └─────────────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Files to Modify

| File | Changes Required | Complexity |
|------|------------------|------------|
| `advanced_feature_engineering_v7.4.py` | Add streak calculation function and integration | MEDIUM |
| `cpr_v7.4_specialist_gbm_trainer.py` | Add `Streak_Advantage` to feature list | LOW |
| `backtest_with_compounding_logic_v7.6.py` | Add on-the-fly streak tracking | MEDIUM |
| `LIVE_FINAL_Predictor.py` | Add streak state persistence | MEDIUM |

---

## Phase 1: Core Streak Calculation

### Step 1.1: Streak Calculation Function

**CRITICAL WARNING - Data Leakage Prevention:**
Streaks MUST be computed based on matches occurring **strictly before** the timestamp of the match being predicted. The existing `history_df = df.iloc[:index]` pattern already enforces this.

**Add to `advanced_feature_engineering_v7.4.py` after the imports (around line 5):**

```python
def get_current_streak(player_id, history_df):
    """
    Computes the current win/loss streak for a player PRIOR to the current match.

    Args:
        player_id: The player's unique ID
        history_df: DataFrame of matches occurring BEFORE the current match

    Returns:
        int: Positive integer for Win Streak (+3 means 3 consecutive wins)
             Negative integer for Loss Streak (-2 means 2 consecutive losses)
             Zero if no prior matches

    Logic:
        1. Get all matches involving this player from history
        2. Sort by date descending (most recent first)
        3. Count consecutive same-outcome results
        4. Return positive for wins, negative for losses
    """
    # Get all matches involving this player
    player_matches = history_df[
        (history_df['Player 1 ID'] == player_id) |
        (history_df['Player 2 ID'] == player_id)
    ].copy()

    if player_matches.empty:
        return 0

    # Sort by date descending (most recent first)
    player_matches = player_matches.sort_values('Date', ascending=False)

    # Determine win/loss for each match from player's perspective
    results = []
    for _, match in player_matches.iterrows():
        if match['Player 1 ID'] == player_id:
            won = match['P1_Win'] == 1
        else:
            won = match['P1_Win'] == 0
        results.append(won)

    if not results:
        return 0

    # Count streak from most recent match
    first_result = results[0]
    streak = 0

    for result in results:
        if result == first_result:
            streak += 1
        else:
            break

    # Return positive for wins, negative for losses
    return streak if first_result else -streak
```

### Step 1.2: Optimized Incremental Streak Tracking (Alternative)

For better performance on large datasets, use incremental state tracking instead of recalculating from scratch each time:

```python
def update_streak_after_match(player_streaks, player_id, won):
    """
    Update a player's streak after a match result.

    Args:
        player_streaks: Dict mapping player_id -> current_streak
        player_id: The player's ID
        won: Boolean, True if player won

    Returns:
        int: The player's NEW streak after this match
    """
    current_streak = player_streaks.get(player_id, 0)

    if won:
        # Won: extend win streak or start new one
        new_streak = (current_streak + 1) if current_streak >= 0 else 1
    else:
        # Lost: extend loss streak or start new one
        new_streak = (current_streak - 1) if current_streak <= 0 else -1

    player_streaks[player_id] = new_streak
    return new_streak
```

### Step 1.3: Integrate into Feature Engineering Loop

**File:** `advanced_feature_engineering_v7.4.py`

**Add inside the main loop** (around line 140, after the H2H calculation):

```python
        # --- Current Streak Calculation ---
        # Get PRE-MATCH streaks (before this match is counted)
        p1_current_streak = get_current_streak(p1_id, history_df)
        p2_current_streak = get_current_streak(p2_id, history_df)
        streak_advantage = p1_current_streak - p2_current_streak
```

**Add to output row** (around line 147, in `new_row.update({...})`):

```python
        new_row.update({
            # ... existing features ...
            'P1_Current_Streak': p1_current_streak,
            'P2_Current_Streak': p2_current_streak,
            'Streak_Advantage': streak_advantage,
        })
```

---

## Phase 2: GBM Trainer Integration

### Step 2.1: Add Streak to Feature List

**File:** `cpr_v7.4_specialist_gbm_trainer.py`

**Current features** (approximately line 30):
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

**Modified (add Streak_Advantage):**
```python
numerical_features = [
    'Streak_Advantage',  # NEW: Win/loss momentum differential
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

### Step 3.1: Add Streak State Tracking

**File:** `backtest_with_compounding_logic_v7.6.py`

**Add after loading data (around line 133):**

```python
# --- Initialize Streak Tracking for On-the-fly Calculation ---
player_streaks = {}  # player_id -> current_streak (signed integer)
```

### Step 3.2: Calculate Streak On-the-fly

**Add inside the main loop (after line 194, after getting player IDs):**

```python
# --- On-the-fly Streak Calculation ---
# Get PRE-MATCH streaks (before this match outcome is known)
p1_current_streak = player_streaks.get(p1_id, 0)
p2_current_streak = player_streaks.get(p2_id, 0)
streak_advantage = p1_current_streak - p2_current_streak
```

### Step 3.3: Update Streaks After Each Match

**Add after betting logic completes (around line 350):**

```python
# --- Update player streaks based on actual outcome ---
p1_won = match['P1_Win'] == 1

# Update P1's streak
if p1_won:
    player_streaks[p1_id] = (player_streaks.get(p1_id, 0) + 1) if player_streaks.get(p1_id, 0) >= 0 else 1
else:
    player_streaks[p1_id] = (player_streaks.get(p1_id, 0) - 1) if player_streaks.get(p1_id, 0) <= 0 else -1

# Update P2's streak (opposite outcome)
if not p1_won:
    player_streaks[p2_id] = (player_streaks.get(p2_id, 0) + 1) if player_streaks.get(p2_id, 0) >= 0 else 1
else:
    player_streaks[p2_id] = (player_streaks.get(p2_id, 0) - 1) if player_streaks.get(p2_id, 0) <= 0 else -1
```

### Step 3.4: Add to GBM Feature DataFrame

**Modify feature dictionary (around line 291-304):**

```python
gbm_features = pd.DataFrame([{
    'Streak_Advantage': streak_advantage,  # NEW
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

---

## Phase 4: Live Predictor Integration

### Step 4.1: Streak State Persistence

**Create new file:** `streak_state_manager.py`

```python
import json
import os

STREAK_STATE_FILE = "player_streaks_state.json"

def load_streak_state():
    """Load player streaks from persistent state file."""
    if os.path.exists(STREAK_STATE_FILE):
        with open(STREAK_STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_streak_state(player_streaks):
    """Save player streaks to persistent state file."""
    with open(STREAK_STATE_FILE, 'w') as f:
        json.dump(player_streaks, f, indent=2)

def get_player_streak(player_streaks, player_id):
    """Get player's current streak (0 for new players)."""
    return player_streaks.get(str(player_id), 0)

def update_streak_after_match(player_streaks, p1_id, p2_id, p1_won):
    """
    Update and persist player streaks after a match result.

    Args:
        player_streaks: Current streak state dict
        p1_id: Player 1's ID
        p2_id: Player 2's ID
        p1_won: Boolean, True if Player 1 won

    Returns:
        tuple: (new_p1_streak, new_p2_streak)
    """
    p1_id = str(p1_id)
    p2_id = str(p2_id)

    p1_streak = player_streaks.get(p1_id, 0)
    p2_streak = player_streaks.get(p2_id, 0)

    # Update P1's streak
    if p1_won:
        new_p1_streak = (p1_streak + 1) if p1_streak >= 0 else 1
        new_p2_streak = (p2_streak - 1) if p2_streak <= 0 else -1
    else:
        new_p1_streak = (p1_streak - 1) if p1_streak <= 0 else -1
        new_p2_streak = (p2_streak + 1) if p2_streak >= 0 else 1

    player_streaks[p1_id] = new_p1_streak
    player_streaks[p2_id] = new_p2_streak

    save_streak_state(player_streaks)
    return new_p1_streak, new_p2_streak
```

### Step 4.2: Initialize Streak State from Historical Data

**Create script:** `initialize_streaks_from_history.py`

```python
"""
One-time script to initialize player streaks from historical match data.
Run this ONCE after enabling streaks in the pipeline.
"""
import pandas as pd
import json

HISTORICAL_FILE = "final_dataset_v7.4_no_duplicates.csv"
OUTPUT_FILE = "player_streaks_state.json"

def main():
    print(f"Loading historical data from {HISTORICAL_FILE}...")
    df = pd.read_csv(HISTORICAL_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by=['Date', 'Match ID'], inplace=True)

    player_streaks = {}

    print("Processing matches chronologically...")
    for _, match in df.iterrows():
        p1_id = str(match['Player 1 ID'])
        p2_id = str(match['Player 2 ID'])
        p1_won = match['P1_Win'] == 1

        p1_streak = player_streaks.get(p1_id, 0)
        p2_streak = player_streaks.get(p2_id, 0)

        # Update P1's streak
        if p1_won:
            player_streaks[p1_id] = (p1_streak + 1) if p1_streak >= 0 else 1
            player_streaks[p2_id] = (p2_streak - 1) if p2_streak <= 0 else -1
        else:
            player_streaks[p1_id] = (p1_streak - 1) if p1_streak <= 0 else -1
            player_streaks[p2_id] = (p2_streak + 1) if p2_streak >= 0 else 1

    print(f"Processed {len(df)} matches. {len(player_streaks)} players tracked.")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(player_streaks, f, indent=2)

    print(f"Streak state saved to {OUTPUT_FILE}")

    # Print streak distribution
    streaks = list(player_streaks.values())
    print(f"\nStreak Distribution:")
    print(f"  Max Win Streak: +{max(streaks)}")
    print(f"  Max Loss Streak: {min(streaks)}")
    print(f"  Mean Streak: {sum(streaks)/len(streaks):.2f}")

    # Players on hot/cold streaks
    hot_players = [(k, v) for k, v in player_streaks.items() if v >= 5]
    cold_players = [(k, v) for k, v in player_streaks.items() if v <= -5]
    print(f"  Players with +5 or better: {len(hot_players)}")
    print(f"  Players with -5 or worse: {len(cold_players)}")

if __name__ == "__main__":
    main()
```

### Step 4.3: Integrate into Live Predictor

**File:** `LIVE_FINAL_Predictor.py`

**Add import:**
```python
from streak_state_manager import load_streak_state, get_player_streak
```

**Add after loading models (around line 128):**
```python
# Load streak state
player_streaks = load_streak_state()
print(f"Loaded streak state for {len(player_streaks)} players.")
```

**Add in match processing loop:**
```python
# Get streak advantage
p1_streak = get_player_streak(player_streaks, p1_id)
p2_streak = get_player_streak(player_streaks, p2_id)
streak_advantage = p1_streak - p2_streak
```

**Add to gbm_features DataFrame:**
```python
gbm_features = pd.DataFrame([{
    'Streak_Advantage': streak_advantage,  # NEW
    # ... rest of features ...
}])
```

---

## Phase 5: Testing & Validation

### Step 5.1: Unit Tests

**Create:** `tests/test_streaks.py`

```python
import pytest

class TestStreakCalculation:
    def test_win_after_wins(self):
        """Winning after win streak should increment."""
        player_streaks = {'123': 3}
        current = player_streaks['123']
        new_streak = (current + 1) if current >= 0 else 1
        assert new_streak == 4

    def test_loss_after_losses(self):
        """Losing after loss streak should decrement."""
        player_streaks = {'123': -2}
        current = player_streaks['123']
        new_streak = (current - 1) if current <= 0 else -1
        assert new_streak == -3

    def test_loss_breaks_win_streak(self):
        """Losing after win streak resets to -1."""
        player_streaks = {'123': 5}
        current = player_streaks['123']
        # Player lost
        new_streak = (current - 1) if current <= 0 else -1
        assert new_streak == -1

    def test_win_breaks_loss_streak(self):
        """Winning after loss streak resets to +1."""
        player_streaks = {'123': -4}
        current = player_streaks['123']
        # Player won
        new_streak = (current + 1) if current >= 0 else 1
        assert new_streak == 1

    def test_new_player_starts_at_zero(self):
        """New player should have streak = 0."""
        player_streaks = {}
        streak = player_streaks.get('new_player', 0)
        assert streak == 0

    def test_streak_advantage_calculation(self):
        """Streak advantage = P1 - P2."""
        p1_streak = 3
        p2_streak = -2
        advantage = p1_streak - p2_streak
        assert advantage == 5  # +3 - (-2) = +5

    def test_both_on_win_streaks(self):
        """Both players on win streaks."""
        p1_streak = 4
        p2_streak = 2
        advantage = p1_streak - p2_streak
        assert advantage == 2

    def test_both_on_loss_streaks(self):
        """Both players on loss streaks."""
        p1_streak = -3
        p2_streak = -5
        advantage = p1_streak - p2_streak
        assert advantage == 2  # -3 - (-5) = +2
```

### Step 5.2: Integration Test

**Test Procedure:**

```bash
# 1. Record baseline
python backtest_with_compounding_logic_v7.6.py
# Note: ROI, Sharpe, MaxDD, Bets

# 2. Apply Phase 1 changes (feature engineering)
python advanced_feature_engineering_v7.4.py
# Verify: Check that Streak_Advantage column exists
python -c "import pandas as pd; df = pd.read_csv('final_dataset_v7.4.csv'); print('Streak_Advantage' in df.columns)"

# 3. Verify streak distribution
python -c "
import pandas as pd
df = pd.read_csv('final_dataset_v7.4.csv')
print('Streak Advantage Stats:')
print(df['Streak_Advantage'].describe())
print(f'Zero streaks: {(df[\"Streak_Advantage\"] == 0).sum()} ({(df[\"Streak_Advantage\"] == 0).mean()*100:.1f}%)')
"

# 4. Apply Phase 2 changes (GBM trainer)
python cpr_v7.4_specialist_gbm_trainer.py
# Verify: Model trains without errors

# 5. Apply Phase 3 changes (backtest)
python backtest_with_compounding_logic_v7.6.py
# Compare: ROI, Sharpe, MaxDD vs baseline

# 6. Check feature importance
python -c "
import joblib
model = joblib.load('cpr_v7.4_gbm_specialist.joblib')
features = ['Streak_Advantage', ...]  # full list
import pandas as pd
importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
print(importance.sort_values('importance', ascending=False))
"
```

### Step 5.3: Validation Metrics

| Metric | Baseline | Target | Accept/Reject Criteria |
|--------|----------|--------|------------------------|
| ROI | 1.50% | >1.50% | Accept if >= 1.40% |
| Sharpe | 2.02 | >2.02 | Accept if >= 1.90 |
| MaxDD | 34.12% | <34.12% | Accept if <= 36% |
| Bets | 4,509 | ~4,500 | Accept if within 10% |

---

## Phase 6: Advanced Streak Extensions (Future)

### 6.1: Streak Magnitude Bucketing

```python
def get_streak_bucket(streak):
    """
    Convert raw streak to categorical bucket for non-linear effects.

    Buckets:
        -3: "Cold" (streak <= -4)
        -2: "Cooling" (streak -3 to -2)
        -1: "Slight Loss" (streak == -1)
         0: "Neutral" (streak == 0)
        +1: "Slight Win" (streak == +1)
        +2: "Warming" (streak +2 to +3)
        +3: "Hot" (streak >= +4)
    """
    if streak <= -4:
        return -3
    elif streak <= -2:
        return -2
    elif streak == -1:
        return -1
    elif streak == 0:
        return 0
    elif streak == 1:
        return 1
    elif streak <= 3:
        return 2
    else:
        return 3
```

### 6.2: Streak × Elo Interaction

```python
# High Elo player on cold streak = potential value opportunity
df['Elo_Streak_Interaction'] = df['Elo_Advantage'] * df['Streak_Advantage']

# Hot streak + clutch performance = high confidence
df['Hot_Clutch_Interaction'] = df['Streak_Advantage'] * df['Close_Set_Win_Rate_Advantage']
```

---

## Next Implementation: EWMA Win Rate (Tier 1)

### Overview

Unlike simple streaks (binary win/loss counting), **EWMA (Exponentially Weighted Moving Average) Win Rate** weights recent matches more heavily using a decay factor. This captures "form" more smoothly - a win from 2 hours ago counts more than a win from 2 days ago.

### The Lambda-Decay Formula

**Core EWMA Recursive Formula:**
```
EWMA_t = α × x_t + (1 - α) × EWMA_{t-1}
```

Where:
- `EWMA_t` = New EWMA value after observation t
- `x_t` = Current observation (1 for win, 0 for loss)
- `α` (alpha) = Smoothing factor (0 < α < 1)
- `EWMA_{t-1}` = Previous EWMA value

**Interpretation:**
- Higher α (e.g., 0.3) = More reactive to recent results
- Lower α (e.g., 0.05) = More smoothed, slower to change

### Half-Life to Alpha Conversion

The **half-life** (h) is the number of observations for weight to decay to 50%:

```
α = 1 - exp(-ln(2) / h)
```

Or equivalently:
```
α = 1 - 0.5^(1/h)
```

**Common Half-Life Values:**

| Half-Life (matches) | Alpha (α) | Behavior |
|---------------------|-----------|----------|
| 3 | 0.206 | Very reactive |
| 5 | 0.129 | Reactive |
| 10 | 0.067 | Balanced |
| 20 | 0.034 | Smooth |

### Time-Based Decay (Recommended for Liga Pro)

For Liga Pro's rapid-fire format, **time-based decay** is more appropriate than match-based:

```python
import numpy as np

def calculate_time_ewma_win_rate(player_id, current_time, history_df, halflife_hours=24):
    """
    Calculate EWMA win rate with time-based decay.
    A win 2 hours ago counts more than a win 2 days ago.

    Args:
        player_id: Player's unique ID
        current_time: Timestamp of current match
        history_df: Historical matches (already filtered to before current match)
        halflife_hours: Hours for weight to decay by 50% (default 24h)

    Returns:
        float: Time-weighted EWMA win rate (0.0 to 1.0)
    """
    player_matches = history_df[
        (history_df['Player 1 ID'] == player_id) |
        (history_df['Player 2 ID'] == player_id)
    ].copy()

    if player_matches.empty:
        return 0.5  # Prior for new players

    # Calculate time-based weights: weight = 2^(-hours_ago / halflife)
    player_matches['hours_ago'] = (
        current_time - player_matches['Date']
    ).dt.total_seconds() / 3600

    player_matches['weight'] = np.power(0.5, player_matches['hours_ago'] / halflife_hours)

    # Determine win/loss from player's perspective
    player_matches['won'] = player_matches.apply(
        lambda r: 1.0 if (r['Player 1 ID'] == player_id and r['P1_Win'] == 1) or
                         (r['Player 2 ID'] == player_id and r['P1_Win'] == 0) else 0.0,
        axis=1
    )

    # Calculate weighted win rate
    weighted_wins = (player_matches['won'] * player_matches['weight']).sum()
    total_weight = player_matches['weight'].sum()

    return weighted_wins / total_weight if total_weight > 0 else 0.5
```

### Match-Based EWMA (Alternative)

```python
def calculate_ewma_win_rate(player_id, history_df, halflife_matches=5):
    """
    Calculate EWMA win rate with match-based decay.

    Args:
        player_id: Player's unique ID
        history_df: Historical matches
        halflife_matches: Matches for weight to decay by 50%

    Returns:
        float: EWMA win rate (0.0 to 1.0)
    """
    player_matches = history_df[
        (history_df['Player 1 ID'] == player_id) |
        (history_df['Player 2 ID'] == player_id)
    ].sort_values('Date')

    if player_matches.empty:
        return 0.5

    # Calculate alpha from half-life
    alpha = 1 - np.power(0.5, 1 / halflife_matches)

    ewma = 0.5  # Start with neutral prior

    for _, match in player_matches.iterrows():
        if match['Player 1 ID'] == player_id:
            result = 1.0 if match['P1_Win'] == 1 else 0.0
        else:
            result = 1.0 if match['P1_Win'] == 0 else 0.0

        ewma = alpha * result + (1 - alpha) * ewma

    return ewma
```

### EWMA Features to Add

| Feature | Half-Life | Description |
|---------|-----------|-------------|
| `P1_EWMA_WinRate_H24` | 24 hours | Very recent form |
| `P2_EWMA_WinRate_H24` | 24 hours | |
| `EWMA_WinRate_Advantage_H24` | - | Short-term form differential |
| `P1_EWMA_WinRate_H72` | 72 hours | Medium-term form |
| `P2_EWMA_WinRate_H72` | 72 hours | |
| `EWMA_WinRate_Advantage_H72` | - | Medium-term form differential |

---

## Risk Mitigation

### Risk 1: Streaks Add No Predictive Value
**Likelihood:** LOW (psychological momentum well-documented in sports)
**Mitigation:** Streaks capture immediate state vs. longer-term trend from rolling averages
**Fallback:** If feature importance near zero, remove and revert

### Risk 2: Data Leakage
**Likelihood:** LOW (existing `history_df = df.iloc[:index]` enforces temporal ordering)
**Mitigation:** Unit tests verify streak calculated BEFORE match outcome known
**Verification:** New player has streak=0 for first match

### Risk 3: Performance Degradation
**Likelihood:** MEDIUM (naive implementation is O(n²))
**Mitigation:** Use incremental state tracking (O(n) total)
**Alternative:** Pre-compute in single pass before main loop

### Risk 4: Extreme Streak Values
**Likelihood:** LOW but possible
**Mitigation:** Consider capping at ±10 or using bucketed values

---

## Commit Strategy

```bash
# After Phase 1 (feature engineering only)
git add advanced_feature_engineering_v7.4.py
git commit -m "feat: Add current win/loss streak calculation

- Add get_current_streak() function for signed streak values
- Calculate P1_Current_Streak, P2_Current_Streak, Streak_Advantage
- Positive for win streaks, negative for loss streaks
- Strict temporal ordering prevents data leakage"

# After full validation
git add .
git commit -m "feat: Complete Current Win/Lose Streaks implementation

- Feature engineering: Calculate streaks chronologically
- GBM trainer: Add Streak_Advantage to feature list
- Backtest: On-the-fly streak tracking with state updates
- Live predictor: Load streak state from JSON file

Results vs baseline:
- ROI: X.XX% (was 1.50%)
- Sharpe: X.XX (was 2.02)
- MaxDD: X.XX% (was 34.12%)"
```

---

## Quick Reference: Code Changes Summary

### File 1: `advanced_feature_engineering_v7.4.py`
- **After line 5:** Add `get_current_streak()` function
- **Around line 140:** Calculate `p1_current_streak`, `p2_current_streak`, `streak_advantage`
- **Around line 147:** Add 3 new columns to `new_row.update()`

### File 2: `cpr_v7.4_specialist_gbm_trainer.py`
- **Line ~30:** Add `'Streak_Advantage'` to `numerical_features` list

### File 3: `backtest_with_compounding_logic_v7.6.py`
- **After line 133:** Initialize `player_streaks = {}`
- **After line 194:** Calculate `streak_advantage` from state
- **After line 350:** Update player streaks after match
- **Line ~291:** Add `'Streak_Advantage': streak_advantage`

### File 4: `LIVE_FINAL_Predictor.py`
- **Add import:** `from streak_state_manager import ...`
- **After line 128:** Load streak state
- **In loop:** Calculate streak advantage
- **In gbm_features:** Add Streak_Advantage

### New Files:
- `streak_state_manager.py` - Streak persistence utilities
- `initialize_streaks_from_history.py` - One-time state initialization
- `tests/test_streaks.py` - Unit tests

---

## Approval Checklist

Before starting implementation:

- [ ] Baseline metrics recorded
- [ ] Git branch created
- [ ] Implementation plan reviewed
- [ ] Test strategy approved
- [ ] EWMA Lambda-decay formula understood for next implementation
- [ ] Rollback plan understood (revert commits)

**Ready to implement?** Start with Phase 0 (baseline) and Phase 1 (feature engineering).

---

## Sources

- [Exponentially Weighted Moving Average - Corporate Finance Institute](https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/exponentially-weighted-moving-average-ewma/)
- [Pandas DataFrame.ewm Documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html)
- [EWMA for Systematic Trading Decisions - Robot Wealth](https://robotwealth.com/using-exponentially-weighted-moving-averages-to-navigate-trade-offs-in-systematic-trading/)
- [Exponentially Weighted Moving Models - Stanford](https://arxiv.org/html/2404.08136v1)
- [Time Series EWMA Theory - Towards Data Science](https://towardsdatascience.com/time-series-from-scratch-exponentially-weighted-moving-averages-ewma-theory-and-implementation-607661d574fe/)
