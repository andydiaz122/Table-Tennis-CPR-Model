# Pipeline Speed Optimization Plan

**Branch**: `speed-bottleneck-v1`
**Baseline**: V9.0 with 2.44% ROI
**Goal**: Optimize pipeline speed without affecting model outputs

---

## Executive Summary

| Script | Bottlenecks | Estimated Speedup |
|--------|-------------|-------------------|
| advanced_feature_engineering_v7.4.py | 7 | 5-10x |
| merge_data_v7.4.py | 7 | 2.5-3.5x |
| backtest_final_v7.4.py | 15 | 50-100x |

**Total estimated improvement**: Pipeline from ~30+ minutes to ~2-5 minutes

---

## Priority 1: CRITICAL Bottlenecks (Highest Impact)

### 1.1 advanced_feature_engineering_v7.4.py

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Main loop `iterrows()` | 261-412 | 50K row-by-row iterations | Vectorize with groupby().transform() |
| Nested `iterrows()` in helpers | 18, 56, 93 | O(N²) complexity | Pre-compute all rolling stats |
| Repeated list filtering | 307-350 | Same filters run 50K times | Use binary search (bisect) |

**Solution**: Replace match-by-match processing with player-history vectorized approach:
1. Create player-match view (long format)
2. Pre-compute ALL rolling stats using `groupby().transform()`
3. Merge features back to match level
4. Eliminate per-match loops entirely

### 1.2 backtest_final_v7.4.py

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Main loop with `df.iloc[:index]` | 114-130 | Creates new DataFrame every iteration | Pre-build player index once |
| Helper functions `iterrows()` | 9-55 | 30K+ slow iterations | Vectorize with numpy |
| `apply(axis=1)` patterns | 156-159, 201-202, 214-216 | Row-by-row Python | Use boolean masking |
| H2H linear search | 188-189 | Full scan per iteration | Build hash index |

**Solution**:
1. Pre-build player games index at load time
2. Replace all `iterrows()` with vectorized NumPy operations
3. Replace `apply(lambda, axis=1)` with boolean masking
4. Cache H2H lookups in dictionary

---

## Priority 2: HIGH Bottlenecks (Significant Impact)

### 2.1 merge_data_v7.4.py

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| No dtype specification | 13, 17 | Type inference on 980MB file | Specify dtypes explicitly |
| Unnecessary `.copy()` | 31 | Doubles memory usage | Remove .copy() |
| Multiple merge passes | 35, 54 | Redundant data passes | Single merge strategy |

### 2.2 All Scripts - I/O Optimization

| Issue | Fix |
|-------|-----|
| CSV without compression | Use gzip or switch to Parquet |
| float64 everywhere | Use float32 for odds/advantages |
| Datetime parsing | Specify format explicitly |

---

## Priority 3: MEDIUM Bottlenecks (Cumulative Impact)

| Script | Issue | Fix |
|--------|-------|-----|
| feature_engineering | `list.pop(0)` O(N) | Use `collections.deque(maxlen=N)` |
| backtest | Duplicate H2H calculations | Compute once, reuse |
| backtest | String parsing in loop | Pre-parse set scores at load |
| all | `inplace=True` anti-pattern | Chain operations instead |

---

## Implementation Plan

### Phase 1: Feature Engineering Optimization
**File**: `V9.0/advanced_feature_engineering_v7.4.py`

1. Add dtype specification to CSV read
2. Replace main `iterrows()` loop with vectorized groupby approach
3. Vectorize `calculate_close_set_win_rate()` - pre-parse set scores
4. Vectorize `calculate_h2h_dominance()` - use NumPy
5. Replace `list.pop(0)` with `deque(maxlen=N)`
6. Vectorize `apply(axis=1)` patterns

### Phase 2: Merge Data Optimization
**File**: `V9.0/merge_data_v7.4.py`

1. Add dtype specification for 980MB odds file
2. Remove unnecessary `.copy()` after filtering
3. Specify datetime format explicitly
4. Restructure to single merge pass
5. Add compression to output

### Phase 3: Backtest Optimization
**File**: `V9.0/backtest_final_v7.4.py`

1. Add dtype specification to CSV read
2. Pre-build player games index at load time
3. Pre-build H2H lookup dictionary
4. Pre-parse set scores at load time
5. Vectorize all helper functions:
   - `calculate_pdr()` -> `calculate_pdr_vectorized()`
   - `calculate_close_set_win_rate()` -> vectorized version
   - `calculate_h2h_dominance()` -> vectorized version
6. Replace all `apply(axis=1)` with boolean masking
7. Remove duplicate calculations (H2H filtering, win rates)
8. Remove unnecessary `.copy()` operations

### Phase 4: Validation
1. Run optimized pipeline end-to-end
2. Compare outputs to baseline (must be identical)
3. Measure timing improvements
4. Verify ROI/Sharpe/Drawdown unchanged

---

## Key Code Patterns to Apply

### Pattern 1: Vectorized Win Rate
```python
# BEFORE (slow)
df.apply(lambda r: 1 if (r['Player 1 ID'] == pid and r['P1_Win'] == 1) else 0, axis=1)

# AFTER (fast)
is_p1 = df['Player 1 ID'] == pid
((is_p1 & (df['P1_Win'] == 1)) | (~is_p1 & (df['P1_Win'] == 0))).astype(int)
```

### Pattern 2: Pre-built Index
```python
# BEFORE (slow) - inside loop
p1_games = history_df[(history_df['Player 1 ID'] == p1_id) | (history_df['Player 2 ID'] == p1_id)]

# AFTER (fast) - build once, lookup O(1)
player_index = {pid: (df['Player 1 ID'] == pid) | (df['Player 2 ID'] == pid)
                for pid in all_player_ids}
p1_games = df[player_index[p1_id] & (df.index < current_index)]
```

### Pattern 3: Deque for Sliding Window
```python
# BEFORE (slow)
history_list.append(val)
if len(history_list) > WINDOW:
    history_list.pop(0)  # O(N)

# AFTER (fast)
from collections import deque
history_deque = deque(maxlen=WINDOW)
history_deque.append(val)  # O(1), auto-evicts oldest
```

---

## Success Criteria

- [ ] All three scripts run without errors
- [ ] Output files are byte-identical to baseline
- [ ] ROI remains 2.44%
- [ ] Total pipeline time reduced by 10x+

---

## Files to Modify

1. `V9.0/advanced_feature_engineering_v7.4.py`
2. `V9.0/merge_data_v7.4.py`
3. `V9.0/backtest_final_v7.4.py`
