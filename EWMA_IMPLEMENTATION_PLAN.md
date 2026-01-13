# EWMA (Exponentially Weighted Moving Averages) Implementation Plan

## Feature Overview

**Feature Names:**
- `EWMA_WinRate_Advantage` (PRIMARY - halflife=10 matches)
- `EWMA_PDR_Advantage` (halflife=10 matches)
- `EWMA_Clutch_Advantage` (halflife=10 matches)

**Definition:** Exponentially weighted metrics where recent performance counts more heavily than older performance, using the formula:

```
EWMA_t = alpha * current_result + (1 - alpha) * EWMA_{t-1}
```

**Conversion:** `alpha = 1 - 0.5^(1/halflife)` where halflife is number of matches for weight to decay 50%

**Expected Impact:** HIGH - Captures "form" more smoothly than simple rolling averages; uses ALL history while naturally weighting recent matches more heavily

---

## Quantitative Rationale

### Why EWMA > Rolling Averages for Liga Pro

1. **Full History Utilization:** Unlike `tail(20)` which discards all but the last 20 matches, EWMA uses the complete match history with exponential decay. A win 25 matches ago still contributes, just less than a win 5 matches ago.

2. **Smoother Form Transitions:** Simple rolling averages create "cliff effects" when old matches drop out of the window. EWMA provides continuous, smooth transitions as recent performance naturally gets weighted more heavily.

3. **Faster Response to Form Changes:** With halflife=10, recent wins/losses have 2x the impact of matches 10 games ago. This captures rapid form changes in Liga Pro's high-frequency format.

4. **Proven in Sports Analytics:** DeepShot NBA predictor and Stanford EWMM research demonstrate EWMA's superiority over simple moving averages for sports prediction.

5. **Direct Upgrade Path:**
   - `EWMA_WinRate_Advantage` → replaces `Win_Rate_L5_Advantage` (more stable, uses all data)
   - `EWMA_PDR_Advantage` → replaces `PDR_Advantage` (captures form trajectory)
   - `EWMA_Clutch_Advantage` → extends `Close_Set_Win_Rate_Advantage` (weights recent clutch performance)

### Halflife Selection Rationale

| Halflife (matches) | Alpha | Weight after 5 matches | Weight after 10 matches | Use Case |
|-------------------|-------|----------------------|------------------------|----------|
| 3 | 0.206 | 18% | 3% | Very short-term "hot hand" |
| 5 | 0.129 | 50% | 25% | Short-term form |
| **10** | **0.067** | **71%** | **50%** | **Balanced (RECOMMENDED)** |
| 20 | 0.034 | 84% | 71% | Long-term trend |

**Recommendation:** Start with halflife=10 for all EWMA features. This balances responsiveness (50% weight decays after 10 matches) with stability (uses full history).

---

## Pre-Implementation Checklist

### Step 0: Record Baseline (MANDATORY per CLAUDE.md Rule 1)

Before ANY code changes, run the full pipeline and record exact baseline:

```bash
cd /home/user/Table-Tennis-CPR-Model/V8.0

# Run full pipeline (shortened version for ROI verification)
python advanced_feature_engineering_v7.4.py && \
python merge_data_v7.4.py && \
python remove_duplicates_from_final_dataset.py && \
python split_data.py && \
python cpr_v7.4_specialist_gbm_trainer.py && \
python backtest_final_v7.4.py

# Record baseline with commit hash
echo "Pre-EWMA Baseline at commit $(git rev-parse --short HEAD):"
echo "ROI: ____%, Sharpe: ____, MaxDD: ____%, Bets: ____"
```

**Expected Baseline (from CLAUDE.md V8.0+):**
- ROI: 1.50%
- Sharpe: 2.02
- MaxDD: 34.12%
- Total Bets: 4,509

---

## Implementation Architecture

### System Components to Modify

```
+------------------------------------------------------------------+
|                     EWMA TRACKING SYSTEM                          |
+------------------------------------------------------------------+
|                                                                    |
|  +----------------------+     +----------------------+             |
|  | Feature Engineering  |---->|   Merge Data         |             |
|  | (calculate EWMA)     |     |   (preserve EWMA)    |             |
|  +----------------------+     +----------------------+             |
|           |                            |                           |
|           v                            v                           |
|  +----------------------+     +----------------------+             |
|  | GBM Trainer          |<----|   Split Data         |             |
|  | (add EWMA features)  |     |   (no changes)       |             |
|  +----------------------+     +----------------------+             |
|           |                                                        |
|           v                                                        |
|  +----------------------+     +----------------------+             |
|  | Backtest             |---->|   Live Predictor     |             |
|  | (on-the-fly EWMA)    |     | (load EWMA state)    |             |
|  +----------------------+     +----------------------+             |
|                                                                    |
+------------------------------------------------------------------+
```

### Files to Modify

| File | Changes Required | Complexity |
|------|------------------|------------|
| `advanced_feature_engineering_v7.4.py` | Add EWMA state tracking and calculation | MEDIUM |
| `cpr_v7.4_specialist_gbm_trainer.py` | Add EWMA features to feature list | LOW |
| `backtest_with_compounding_logic_v7.6.py` | Add on-the-fly EWMA tracking | MEDIUM |
| `backtest_final_v7.4.py` | Add on-the-fly EWMA tracking | MEDIUM |
| `LIVE_FINAL_Predictor.py` | Add EWMA state persistence | MEDIUM |

---

## EWMA Mathematics

### Core Formula

The recursive EWMA formula:
```
EWMA_t = alpha * x_t + (1 - alpha) * EWMA_{t-1}
```

Where:
- `EWMA_t` = New EWMA value after observation t
- `x_t` = Current observation (1 for win, 0 for loss; or PDR value; or clutch win)
- `alpha` = Smoothing factor derived from halflife
- `EWMA_{t-1}` = Previous EWMA value (default: 0.5 for new players)

### Halflife to Alpha Conversion

```python
import math
alpha = 1 - math.pow(0.5, 1 / halflife)
```

**Pre-computed alphas for common halflives:**
```python
EWMA_ALPHA = {
    3:  0.2063,   # 1 - 0.5^(1/3)
    5:  0.1294,   # 1 - 0.5^(1/5)
    10: 0.0670,   # 1 - 0.5^(1/10) <- RECOMMENDED
    20: 0.0341,   # 1 - 0.5^(1/20)
}
```

### Weight Decay Visualization

For halflife=10:
```
Match   | Weight  | Cumulative
--------|---------|------------
Latest  | 0.067   | 0.067
-1      | 0.063   | 0.130
-2      | 0.059   | 0.189
-5      | 0.048   | 0.293 (71% from last 5)
-10     | 0.033   | 0.500 (50% from last 10)
-20     | 0.018   | 0.706
-30     | 0.010   | 0.826
```

---

## Phase 1: Core EWMA State Tracking

### Step 1.1: Add EWMA Configuration (line ~142)

**File:** `advanced_feature_engineering_v7.4.py`

**Add after H2H_DECAY_FACTOR:**

```python
# --- EWMA Configuration ---
EWMA_HALFLIFE = 10  # Number of matches for weight to decay by 50%
EWMA_ALPHA = 1 - math.pow(0.5, 1 / EWMA_HALFLIFE)  # ~0.067 for halflife=10
```

### Step 1.2: Initialize EWMA State Trackers (line ~186)

**Add after streak_tracker initialization:**

```python
# --- NEW: Initialize EWMA State Trackers (O(n) single-pass) ---
# Each tracker: player_id (str) -> current EWMA value (float, 0-1 range)
ewma_winrate = {}      # EWMA of win rate
ewma_pdr = {}          # EWMA of points dominance ratio
ewma_clutch = {}       # EWMA of close-set win rate
```

### Step 1.3: EWMA Update Helper Function (after line 134)

**Add new helper function:**

```python
def update_ewma(current_ewma, new_value, alpha):
    """
    Update EWMA with a new observation.

    Args:
        current_ewma: Current EWMA value (0.5 for new players)
        new_value: New observation (0 or 1 for win/loss, or continuous for PDR)
        alpha: Smoothing factor (derived from halflife)

    Returns:
        float: Updated EWMA value
    """
    return alpha * new_value + (1 - alpha) * current_ewma
```

---

## Phase 2: EWMA Feature Calculation

### Step 2.1: Retrieve Pre-Match EWMA State (line ~233, after streak retrieval)

**Add after streak calculation:**

```python
# --- NEW: Retrieve PRE-MATCH EWMA state (O(n) single-pass) ---
# Default to 0.5 (neutral) for new players
p1_ewma_winrate = ewma_winrate.get(str(p1_id), 0.5)
p2_ewma_winrate = ewma_winrate.get(str(p2_id), 0.5)
ewma_winrate_advantage = p1_ewma_winrate - p2_ewma_winrate

p1_ewma_pdr = ewma_pdr.get(str(p1_id), 0.5)
p2_ewma_pdr = ewma_pdr.get(str(p2_id), 0.5)
ewma_pdr_advantage = p1_ewma_pdr - p2_ewma_pdr

p1_ewma_clutch = ewma_clutch.get(str(p1_id), 0.5)
p2_ewma_clutch = ewma_clutch.get(str(p2_id), 0.5)
ewma_clutch_advantage = p1_ewma_clutch - p2_ewma_clutch
```

### Step 2.2: Calculate Per-Match Metrics for EWMA Update

We need three metrics per player per match:
1. **Win result:** 1 if player won, 0 if lost
2. **Match PDR:** Points won / (Points won + Points lost) for this single match
3. **Clutch result:** Close sets won / Close sets played in this match

**For P1 metrics (around line 305, after p1_close_set_win_rate):**

```python
# --- Calculate P1's single-match metrics for EWMA update ---
# These will be used AFTER the loop iteration to update state
# (We record pre-match EWMA, then update with this match's result)
```

**Note:** The actual EWMA update happens AFTER recording pre-match values, similar to the streak pattern.

### Step 2.3: Update EWMA State After Match (line ~413, after streak update)

**Add after streak update:**

```python
# --- NEW: Update EWMA state AFTER recording pre-match values ---
p1_id_str = str(p1_id)
p2_id_str = str(p2_id)

# Update EWMA Win Rate
p1_won_result = 1.0 if p1_won else 0.0
p2_won_result = 0.0 if p1_won else 1.0
ewma_winrate[p1_id_str] = update_ewma(ewma_winrate.get(p1_id_str, 0.5), p1_won_result, EWMA_ALPHA)
ewma_winrate[p2_id_str] = update_ewma(ewma_winrate.get(p2_id_str, 0.5), p2_won_result, EWMA_ALPHA)

# Update EWMA PDR (using this match's PDR)
p1_total_points = match['P1 Total Points']
p2_total_points = match['P2 Total Points']
total_points = p1_total_points + p2_total_points
if total_points > 0:
    p1_match_pdr = p1_total_points / total_points
    p2_match_pdr = p2_total_points / total_points
else:
    p1_match_pdr = 0.5
    p2_match_pdr = 0.5
ewma_pdr[p1_id_str] = update_ewma(ewma_pdr.get(p1_id_str, 0.5), p1_match_pdr, EWMA_ALPHA)
ewma_pdr[p2_id_str] = update_ewma(ewma_pdr.get(p2_id_str, 0.5), p2_match_pdr, EWMA_ALPHA)

# Update EWMA Clutch (close-set performance this match)
# Parse set scores to determine close sets won/lost
p1_close_won, p1_close_total = 0, 0
set_scores_str = match.get('Set Scores')
if pd.notna(set_scores_str):
    for set_score in str(set_scores_str).split(','):
        try:
            s1, s2 = map(int, set_score.strip().split('-'))
            if abs(s1 - s2) == 2:  # Close set
                p1_close_total += 1
                if s1 > s2:  # P1 won this close set
                    p1_close_won += 1
        except (ValueError, IndexError):
            continue

if p1_close_total > 0:
    p1_clutch_result = p1_close_won / p1_close_total
    p2_clutch_result = 1 - p1_clutch_result
    ewma_clutch[p1_id_str] = update_ewma(ewma_clutch.get(p1_id_str, 0.5), p1_clutch_result, EWMA_ALPHA)
    ewma_clutch[p2_id_str] = update_ewma(ewma_clutch.get(p2_id_str, 0.5), p2_clutch_result, EWMA_ALPHA)
# If no close sets in this match, don't update clutch EWMA (it stays at current value)
```

---

## Phase 3: Output Integration

### Step 3.1: Add EWMA Features to Output Row (line ~428, in new_row.update)

**Add to the new_row.update() dictionary:**

```python
new_row.update({
    # Elo features (6 features)
    'Elo_Advantage': elo_advantage,
    'P1_Elo': p1_pre_match_elo,
    'P2_Elo': p2_pre_match_elo,
    'Elo_Sum': elo_sum,
    'P1_Elo_Confidence': p1_elo_confidence,
    'P2_Elo_Confidence': p2_elo_confidence,
    # Streak features
    'P1_Current_Streak': p1_current_streak,
    'P2_Current_Streak': p2_current_streak,
    'Streak_Advantage': streak_advantage,
    # EWMA features (NEW - 3 advantage features)
    'EWMA_WinRate_Advantage': ewma_winrate_advantage,
    'EWMA_PDR_Advantage': ewma_pdr_advantage,
    'EWMA_Clutch_Advantage': ewma_clutch_advantage,
    # Existing features...
    'PDR_Slope_Advantage': pdr_slope_advantage,
    # ... rest of existing features ...
})
```

---

## Phase 4: GBM Trainer Integration

### Step 4.1: Add EWMA Features to Feature List

**File:** `cpr_v7.4_specialist_gbm_trainer.py`

**Modified numerical_features list:**

```python
numerical_features = [
    # EWMA features (NEW - replace less effective rolling features)
    'EWMA_WinRate_Advantage',    # Replaces Win_Rate_L5_Advantage
    'EWMA_PDR_Advantage',        # Complements PDR_Advantage
    'EWMA_Clutch_Advantage',     # Complements Close_Set_Win_Rate_Advantage
    # Existing high-value features
    'H2H_P1_Win_Rate',
    'H2H_Dominance_Score',
    'PDR_Advantage',
    'Close_Set_Win_Rate_Advantage',
    'Set_Comebacks_Advantage',
]
```

**Option A (Conservative):** Add EWMA features alongside existing features (15 total)
**Option B (Replacement):** Replace `Win_Rate_L5_Advantage` with `EWMA_WinRate_Advantage` (6 total)

**Recommendation:** Start with Option A (additive), then test removing redundant features if EWMA proves valuable.

---

## Phase 5: Backtest Integration

### Step 5.1: Add EWMA State Tracking (backtest_with_compounding_logic_v7.6.py)

**Add after loading data:**

```python
# --- Initialize EWMA State Tracking ---
EWMA_HALFLIFE = 10
EWMA_ALPHA = 1 - math.pow(0.5, 1 / EWMA_HALFLIFE)
ewma_winrate = {}
ewma_pdr = {}
ewma_clutch = {}
```

### Step 5.2: Calculate EWMA On-the-fly

**Add inside the main loop (after getting player IDs):**

```python
# --- On-the-fly EWMA Calculation ---
# Get PRE-MATCH EWMA values
p1_ewma_winrate = ewma_winrate.get(str(p1_id), 0.5)
p2_ewma_winrate = ewma_winrate.get(str(p2_id), 0.5)
ewma_winrate_advantage = p1_ewma_winrate - p2_ewma_winrate

p1_ewma_pdr = ewma_pdr.get(str(p1_id), 0.5)
p2_ewma_pdr = ewma_pdr.get(str(p2_id), 0.5)
ewma_pdr_advantage = p1_ewma_pdr - p2_ewma_pdr

p1_ewma_clutch = ewma_clutch.get(str(p1_id), 0.5)
p2_ewma_clutch = ewma_clutch.get(str(p2_id), 0.5)
ewma_clutch_advantage = p1_ewma_clutch - p2_ewma_clutch
```

### Step 5.3: Update EWMA After Each Match

**Add after betting logic completes:**

```python
# --- Update EWMA state based on actual outcome ---
p1_id_str = str(p1_id)
p2_id_str = str(p2_id)
p1_won = match['P1_Win'] == 1

# Update EWMA Win Rate
ewma_winrate[p1_id_str] = EWMA_ALPHA * (1.0 if p1_won else 0.0) + (1 - EWMA_ALPHA) * ewma_winrate.get(p1_id_str, 0.5)
ewma_winrate[p2_id_str] = EWMA_ALPHA * (0.0 if p1_won else 1.0) + (1 - EWMA_ALPHA) * ewma_winrate.get(p2_id_str, 0.5)

# Update EWMA PDR
p1_pts = match['P1 Total Points']
p2_pts = match['P2 Total Points']
total_pts = p1_pts + p2_pts
if total_pts > 0:
    ewma_pdr[p1_id_str] = EWMA_ALPHA * (p1_pts / total_pts) + (1 - EWMA_ALPHA) * ewma_pdr.get(p1_id_str, 0.5)
    ewma_pdr[p2_id_str] = EWMA_ALPHA * (p2_pts / total_pts) + (1 - EWMA_ALPHA) * ewma_pdr.get(p2_id_str, 0.5)

# Update EWMA Clutch (only if close sets in this match)
# [Include set score parsing logic from Phase 2.3]
```

### Step 5.4: Add to GBM Feature DataFrame

**Modify feature dictionary:**

```python
gbm_features = pd.DataFrame([{
    'EWMA_WinRate_Advantage': ewma_winrate_advantage,
    'EWMA_PDR_Advantage': ewma_pdr_advantage,
    'EWMA_Clutch_Advantage': ewma_clutch_advantage,
    # ... existing features ...
}])
```

---

## Phase 6: Live Predictor Integration

### Step 6.1: Create EWMA State Manager

**Create new file:** `ewma_state_manager.py`

```python
import json
import os
import math

EWMA_STATE_FILE = "player_ewma_state.json"
EWMA_HALFLIFE = 10
EWMA_ALPHA = 1 - math.pow(0.5, 1 / EWMA_HALFLIFE)

def load_ewma_state():
    """Load player EWMA values from persistent state file."""
    if os.path.exists(EWMA_STATE_FILE):
        with open(EWMA_STATE_FILE, 'r') as f:
            return json.load(f)
    return {'winrate': {}, 'pdr': {}, 'clutch': {}}

def save_ewma_state(ewma_state):
    """Save player EWMA values to persistent state file."""
    with open(EWMA_STATE_FILE, 'w') as f:
        json.dump(ewma_state, f, indent=2)

def get_player_ewma(ewma_state, player_id, metric='winrate'):
    """Get player's current EWMA value (0.5 for new players)."""
    return ewma_state.get(metric, {}).get(str(player_id), 0.5)

def update_ewma_after_match(ewma_state, p1_id, p2_id, match_data):
    """
    Update EWMA values for both players after a match.

    Args:
        ewma_state: Current EWMA state dict
        p1_id, p2_id: Player IDs
        match_data: Dict with P1_Win, P1_Total_Points, P2_Total_Points, Set_Scores
    """
    p1_id = str(p1_id)
    p2_id = str(p2_id)
    p1_won = match_data.get('P1_Win', 0) == 1

    # Ensure sub-dicts exist
    for metric in ['winrate', 'pdr', 'clutch']:
        if metric not in ewma_state:
            ewma_state[metric] = {}

    # Update Win Rate EWMA
    p1_wr = ewma_state['winrate'].get(p1_id, 0.5)
    p2_wr = ewma_state['winrate'].get(p2_id, 0.5)
    ewma_state['winrate'][p1_id] = EWMA_ALPHA * (1.0 if p1_won else 0.0) + (1 - EWMA_ALPHA) * p1_wr
    ewma_state['winrate'][p2_id] = EWMA_ALPHA * (0.0 if p1_won else 1.0) + (1 - EWMA_ALPHA) * p2_wr

    # Update PDR EWMA
    p1_pts = match_data.get('P1_Total_Points', 0)
    p2_pts = match_data.get('P2_Total_Points', 0)
    total = p1_pts + p2_pts
    if total > 0:
        p1_pdr = ewma_state['pdr'].get(p1_id, 0.5)
        p2_pdr = ewma_state['pdr'].get(p2_id, 0.5)
        ewma_state['pdr'][p1_id] = EWMA_ALPHA * (p1_pts / total) + (1 - EWMA_ALPHA) * p1_pdr
        ewma_state['pdr'][p2_id] = EWMA_ALPHA * (p2_pts / total) + (1 - EWMA_ALPHA) * p2_pdr

    # Update Clutch EWMA (parse set scores for close sets)
    set_scores = match_data.get('Set_Scores', '')
    if set_scores:
        p1_close_won, p1_close_total = 0, 0
        for set_score in str(set_scores).split(','):
            try:
                s1, s2 = map(int, set_score.strip().split('-'))
                if abs(s1 - s2) == 2:
                    p1_close_total += 1
                    if s1 > s2:
                        p1_close_won += 1
            except:
                continue

        if p1_close_total > 0:
            p1_clutch = ewma_state['clutch'].get(p1_id, 0.5)
            p2_clutch = ewma_state['clutch'].get(p2_id, 0.5)
            ewma_state['clutch'][p1_id] = EWMA_ALPHA * (p1_close_won / p1_close_total) + (1 - EWMA_ALPHA) * p1_clutch
            ewma_state['clutch'][p2_id] = EWMA_ALPHA * (1 - p1_close_won / p1_close_total) + (1 - EWMA_ALPHA) * p2_clutch

    save_ewma_state(ewma_state)
    return ewma_state
```

### Step 6.2: Initialize EWMA State from Historical Data

**Create script:** `initialize_ewma_from_history.py`

```python
"""
One-time script to initialize player EWMA values from historical match data.
Run this ONCE after enabling EWMA in the pipeline.
"""
import pandas as pd
import json
import math

HISTORICAL_FILE = "final_dataset_v7.4_no_duplicates.csv"
OUTPUT_FILE = "player_ewma_state.json"
EWMA_HALFLIFE = 10
EWMA_ALPHA = 1 - math.pow(0.5, 1 / EWMA_HALFLIFE)

def main():
    print(f"Loading historical data from {HISTORICAL_FILE}...")
    df = pd.read_csv(HISTORICAL_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by=['Date', 'Time'], inplace=True)

    ewma_state = {'winrate': {}, 'pdr': {}, 'clutch': {}}

    print("Processing matches chronologically...")
    for _, match in df.iterrows():
        p1_id = str(match['Player 1 ID'])
        p2_id = str(match['Player 2 ID'])
        p1_won = match['P1_Win'] == 1

        # Update Win Rate EWMA
        p1_wr = ewma_state['winrate'].get(p1_id, 0.5)
        p2_wr = ewma_state['winrate'].get(p2_id, 0.5)
        ewma_state['winrate'][p1_id] = EWMA_ALPHA * (1.0 if p1_won else 0.0) + (1 - EWMA_ALPHA) * p1_wr
        ewma_state['winrate'][p2_id] = EWMA_ALPHA * (0.0 if p1_won else 1.0) + (1 - EWMA_ALPHA) * p2_wr

        # Update PDR EWMA
        p1_pts = match['P1 Total Points']
        p2_pts = match['P2 Total Points']
        total = p1_pts + p2_pts
        if total > 0:
            p1_pdr = ewma_state['pdr'].get(p1_id, 0.5)
            p2_pdr = ewma_state['pdr'].get(p2_id, 0.5)
            ewma_state['pdr'][p1_id] = EWMA_ALPHA * (p1_pts / total) + (1 - EWMA_ALPHA) * p1_pdr
            ewma_state['pdr'][p2_id] = EWMA_ALPHA * (p2_pts / total) + (1 - EWMA_ALPHA) * p2_pdr

        # Update Clutch EWMA
        set_scores = match.get('Set Scores', '')
        if pd.notna(set_scores):
            p1_close_won, p1_close_total = 0, 0
            for set_score in str(set_scores).split(','):
                try:
                    s1, s2 = map(int, set_score.strip().split('-'))
                    if abs(s1 - s2) == 2:
                        p1_close_total += 1
                        if s1 > s2:
                            p1_close_won += 1
                except:
                    continue

            if p1_close_total > 0:
                p1_clutch = ewma_state['clutch'].get(p1_id, 0.5)
                p2_clutch = ewma_state['clutch'].get(p2_id, 0.5)
                ewma_state['clutch'][p1_id] = EWMA_ALPHA * (p1_close_won / p1_close_total) + (1 - EWMA_ALPHA) * p1_clutch
                ewma_state['clutch'][p2_id] = EWMA_ALPHA * (1 - p1_close_won / p1_close_total) + (1 - EWMA_ALPHA) * p2_clutch

    print(f"Processed {len(df)} matches.")
    print(f"Tracked {len(ewma_state['winrate'])} players (winrate)")
    print(f"Tracked {len(ewma_state['pdr'])} players (pdr)")
    print(f"Tracked {len(ewma_state['clutch'])} players (clutch)")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(ewma_state, f, indent=2)

    print(f"\nEWMA state saved to {OUTPUT_FILE}")

    # Distribution stats
    for metric in ['winrate', 'pdr', 'clutch']:
        values = list(ewma_state[metric].values())
        if values:
            print(f"\n{metric.upper()} Distribution:")
            print(f"  Min: {min(values):.4f}")
            print(f"  Max: {max(values):.4f}")
            print(f"  Mean: {sum(values)/len(values):.4f}")

if __name__ == "__main__":
    main()
```

---

## Phase 7: Testing & Validation

### Step 7.1: Unit Tests

**Create:** `tests/test_ewma.py`

```python
import pytest
import math

EWMA_HALFLIFE = 10
EWMA_ALPHA = 1 - math.pow(0.5, 1 / EWMA_HALFLIFE)

def update_ewma(current_ewma, new_value, alpha):
    return alpha * new_value + (1 - alpha) * current_ewma

class TestEWMACalculation:
    def test_alpha_calculation(self):
        """Alpha should be ~0.067 for halflife=10."""
        assert abs(EWMA_ALPHA - 0.0670) < 0.001

    def test_ewma_win_after_loss(self):
        """Win after neutral should increase EWMA."""
        current = 0.5
        new_ewma = update_ewma(current, 1.0, EWMA_ALPHA)
        assert new_ewma > 0.5
        assert abs(new_ewma - 0.5335) < 0.001  # 0.067*1 + 0.933*0.5

    def test_ewma_loss_after_win(self):
        """Loss after high EWMA should decrease it."""
        current = 0.7
        new_ewma = update_ewma(current, 0.0, EWMA_ALPHA)
        assert new_ewma < 0.7

    def test_ewma_converges_to_streak(self):
        """10 consecutive wins should push EWMA toward 1.0."""
        ewma = 0.5
        for _ in range(10):
            ewma = update_ewma(ewma, 1.0, EWMA_ALPHA)
        assert ewma > 0.75  # Should be ~0.84 after 10 wins from 0.5

    def test_ewma_new_player_default(self):
        """New player should default to 0.5."""
        ewma_state = {}
        player_ewma = ewma_state.get('new_player', 0.5)
        assert player_ewma == 0.5

    def test_ewma_advantage_calculation(self):
        """Advantage should be P1 - P2."""
        p1_ewma = 0.65
        p2_ewma = 0.45
        advantage = p1_ewma - p2_ewma
        assert advantage == 0.20

    def test_halflife_decay_property(self):
        """After 10 observations, original weight should be ~50%."""
        # Start with prior=0.5, observe 10 wins
        ewma_all_wins = 0.5
        for _ in range(10):
            ewma_all_wins = update_ewma(ewma_all_wins, 1.0, EWMA_ALPHA)

        # The weight of the original 0.5 prior after 10 updates
        # should be (1-alpha)^10 = 0.933^10 = ~0.5
        prior_weight = (1 - EWMA_ALPHA) ** 10
        assert abs(prior_weight - 0.5) < 0.02
```

### Step 7.2: Integration Test

```bash
# 1. Record baseline
python backtest_final_v7.4.py
# Note: ROI, Sharpe, MaxDD, Bets

# 2. Apply Phase 1-3 changes (feature engineering)
python advanced_feature_engineering_v7.4.py
# Verify: Check that EWMA columns exist
python -c "import pandas as pd; df = pd.read_csv('final_engineered_features_v7.4.csv'); print('EWMA_WinRate_Advantage' in df.columns)"

# 3. Verify EWMA distribution
python -c "
import pandas as pd
df = pd.read_csv('final_engineered_features_v7.4.csv')
for col in ['EWMA_WinRate_Advantage', 'EWMA_PDR_Advantage', 'EWMA_Clutch_Advantage']:
    if col in df.columns:
        print(f'{col}:')
        print(df[col].describe())
        print()
"

# 4. Run full pipeline with EWMA
python merge_data_v7.4.py && \
python remove_duplicates_from_final_dataset.py && \
python split_data.py && \
python cpr_v7.4_specialist_gbm_trainer.py && \
python backtest_final_v7.4.py

# 5. Compare metrics vs baseline
```

### Step 7.3: Validation Metrics

| Metric | Baseline | Target | Accept/Reject Criteria |
|--------|----------|--------|------------------------|
| ROI | 1.50% | >1.50% | Accept if >= 1.40% |
| Sharpe | 2.02 | >2.02 | Accept if >= 1.90 |
| MaxDD | 34.12% | <34.12% | Accept if <= 36% |
| Bets | 4,509 | ~4,500 | Accept if within 10% |

---

## Risk Mitigation

### Risk 1: EWMA Adds No Predictive Value
**Likelihood:** MEDIUM (streak showed no alpha; EWMA may be similar)
**Mitigation:** EWMA captures different information than streaks (magnitude, not just direction)
**Fallback:** If feature importance near zero after testing, remove and revert

### Risk 2: Correlation with Existing Features
**Likelihood:** HIGH (EWMA_WinRate may correlate with Win_Rate_L5)
**Mitigation:** Check correlation matrix; if >0.8, consider removing redundant feature
**Analysis:**
```python
import pandas as pd
df = pd.read_csv('final_engineered_features_v7.4.csv')
correlation = df[['EWMA_WinRate_Advantage', 'Win_Rate_L5_Advantage', 'Win_Rate_Advantage']].corr()
print(correlation)
```

### Risk 3: Data Leakage
**Likelihood:** LOW (pattern matches streak implementation)
**Mitigation:** EWMA state retrieved BEFORE match, updated AFTER
**Verification:** New player has EWMA=0.5 for first match

### Risk 4: Performance Degradation
**Likelihood:** LOW (O(n) single-pass like streaks)
**Mitigation:** Uses incremental state tracking, not full recalculation

### Risk 5: Cold Start Problem
**Likelihood:** LOW (default 0.5 is neutral)
**Mitigation:** New players get neutral EWMA which won't bias predictions

---

## Commit Strategy

```bash
# After Phase 1-3 (feature engineering only)
git add advanced_feature_engineering_v7.4.py
git commit -m "feat: Add EWMA state tracking to feature engineering

- Add update_ewma() helper function
- Initialize ewma_winrate, ewma_pdr, ewma_clutch state trackers
- Calculate EWMA advantages using O(n) single-pass pattern
- halflife=10 matches (alpha=0.067)
- Prevents data leakage: read pre-match EWMA, update after match"

# After full validation
git add .
git commit -m "feat: Complete EWMA feature implementation

- Feature engineering: EWMA Win Rate, PDR, Clutch with halflife=10
- GBM trainer: Add 3 EWMA advantage features
- Backtest: On-the-fly EWMA tracking with state updates
- Live predictor: EWMA state persistence via JSON

Results vs baseline:
- ROI: X.XX% (was 1.50%)
- Sharpe: X.XX (was 2.02)
- MaxDD: X.XX% (was 34.12%)"
```

---

## Quick Reference: Code Changes Summary

### File 1: `advanced_feature_engineering_v7.4.py`
- **Line ~142:** Add EWMA configuration constants
- **Line ~134:** Add `update_ewma()` helper function
- **Line ~186:** Initialize ewma_winrate, ewma_pdr, ewma_clutch dictionaries
- **Line ~233:** Retrieve pre-match EWMA values and calculate advantages
- **Line ~413:** Update EWMA state after match outcome
- **Line ~428:** Add 3 EWMA advantage columns to output

### File 2: `cpr_v7.4_specialist_gbm_trainer.py`
- **Line ~31:** Add `EWMA_WinRate_Advantage`, `EWMA_PDR_Advantage`, `EWMA_Clutch_Advantage` to `numerical_features`

### File 3: `backtest_with_compounding_logic_v7.6.py`
- **After loading:** Initialize EWMA config and state dicts
- **In main loop:** Calculate EWMA advantages from state
- **After betting:** Update EWMA state based on outcome
- **In gbm_features:** Add 3 EWMA columns

### File 4: `backtest_final_v7.4.py`
- Same changes as File 3

### File 5: `LIVE_FINAL_Predictor.py`
- **Add import:** `from ewma_state_manager import ...`
- **After loading models:** Load EWMA state
- **In loop:** Calculate EWMA advantages
- **In gbm_features:** Add EWMA columns

### New Files:
- `ewma_state_manager.py` - EWMA persistence utilities
- `initialize_ewma_from_history.py` - One-time state initialization
- `tests/test_ewma.py` - Unit tests

---

## Feature Comparison: EWMA vs Existing Features

| Feature | Type | Window | Weighting | History Used |
|---------|------|--------|-----------|--------------|
| `Win_Rate_L5_Advantage` | Rolling | 5 matches | Uniform | Last 5 only |
| `Win_Rate_Advantage` (L20) | Rolling | 20 matches | Uniform | Last 20 only |
| `EWMA_WinRate_Advantage` | EWMA | All history | Exponential | All (decayed) |
| `PDR_Advantage` | Rolling | 20 matches | Uniform | Last 20 only |
| `EWMA_PDR_Advantage` | EWMA | All history | Exponential | All (decayed) |
| `Close_Set_Win_Rate_Advantage` | Rolling | 20 matches | Uniform | Last 20 only |
| `EWMA_Clutch_Advantage` | EWMA | All history | Exponential | All (decayed) |

---

## Multiple Halflife Options (Future Enhancement)

If the initial halflife=10 shows promise, consider testing multiple halflives:

```python
# Add to feature engineering for comprehensive testing
EWMA_HALFLIVES = [3, 5, 10, 20]

for hl in EWMA_HALFLIVES:
    alpha = 1 - math.pow(0.5, 1 / hl)
    # Calculate EWMA with this halflife
    # Add feature: EWMA_WinRate_Advantage_H{hl}
```

This allows the GBM to learn which halflife captures the most signal for different match contexts.

---

## Expected Outcomes

### Optimistic Scenario
- EWMA features capture "form" better than rolling averages
- ROI increases by 0.2-0.5% to 1.7-2.0%
- Sharpe improves due to more stable predictions
- Can remove redundant `Win_Rate_L5_Advantage` feature

### Neutral Scenario (Like Streak)
- EWMA adds no additional alpha
- ROI stays at ~1.50%
- Features have low importance scores
- Remove EWMA features and revert

### Pessimistic Scenario
- EWMA correlates too heavily with existing features
- Model overfit increases
- ROI drops below 1.4%
- Immediate revert required

---

## Approval Checklist

Before starting implementation:

- [ ] Baseline metrics recorded (ROI: 1.50%, Sharpe: 2.02, MaxDD: 34.12%)
- [ ] Git branch created
- [ ] Implementation plan reviewed
- [ ] EWMA mathematics understood (halflife=10, alpha=0.067)
- [ ] Test strategy approved
- [ ] Rollback plan understood (git revert)
- [ ] Correlation analysis planned for post-implementation

**Ready to implement?** Start with Phase 0 (baseline) and Phase 1 (configuration + state tracking).

---

## Sources

- [DeepShot NBA Predictor - EWMA Win Rate](https://deepshot.ai/)
- [Stanford EWMM Research - Exponentially Weighted Moving Models](https://arxiv.org/html/2404.08136v1)
- [Pandas DataFrame.ewm Documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html)
- [EWMA for Systematic Trading - Robot Wealth](https://robotwealth.com/using-exponentially-weighted-moving-averages-to-navigate-trade-offs-in-systematic-trading/)
- [Corporate Finance Institute - EWMA Explained](https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/exponentially-weighted-moving-average-ewma/)
