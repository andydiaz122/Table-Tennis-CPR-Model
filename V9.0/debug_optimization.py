"""
Debug script to understand why pre-computation optimization fails.
Compare the baseline history_df approach vs pre-computed indices.
"""
import pandas as pd
import bisect
from collections import defaultdict

# Load data
INPUT_FILE = 'czech_liga_pro_advanced_stats_FIXED.csv'
df = pd.read_csv(INPUT_FILE)

# Data prep (same as baseline)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by=['Date', 'Match ID'], inplace=True)
df.reset_index(drop=True, inplace=True)
df['Player 1 ID'] = df['Player 1 ID'].astype(int)
df['Player 2 ID'] = df['Player 2 ID'].astype(int)

# Create P1_Win column
import numpy as np
def get_winner(score_str):
    try:
        cleaned_score = str(score_str).strip('="')
        p1_score, p2_score = map(int, cleaned_score.split('-'))
        return 1 if p1_score > p2_score else 0
    except (ValueError, IndexError):
        return np.nan
df['P1_Win'] = df['Final Score'].apply(get_winner)
df.dropna(subset=['P1_Win'], inplace=True)
df.reset_index(drop=True, inplace=True)  # Critical: reset index after dropna

print(f"DataFrame shape: {df.shape}")
print(f"Index range: {df.index.min()} to {df.index.max()}")

# Build pre-computed indices
player_match_indices = defaultdict(list)
h2h_pair_indices = defaultdict(list)

for idx in range(len(df)):
    p1_id = df.at[idx, 'Player 1 ID']
    p2_id = df.at[idx, 'Player 2 ID']
    player_match_indices[p1_id].append(idx)
    player_match_indices[p2_id].append(idx)
    pair_key = (min(p1_id, p2_id), max(p1_id, p2_id))
    h2h_pair_indices[pair_key].append(idx)

print(f"\nPre-computed {len(player_match_indices)} players, {len(h2h_pair_indices)} H2H pairs")

# Test at row 25000
test_index = 25000
match = df.iloc[test_index]
p1_id = match['Player 1 ID']
p2_id = match['Player 2 ID']

print(f"\n{'='*60}")
print(f"Testing at row {test_index}")
print(f"P1 ID: {p1_id}, P2 ID: {p2_id}")
print(f"Date: {match['Date']}")
print(f"{'='*60}")

# BASELINE approach
history_df_baseline = df.iloc[:test_index]
p1_games_baseline = history_df_baseline[(history_df_baseline['Player 1 ID'] == p1_id) |
                                         (history_df_baseline['Player 2 ID'] == p1_id)]
p2_games_baseline = history_df_baseline[(history_df_baseline['Player 1 ID'] == p2_id) |
                                         (history_df_baseline['Player 2 ID'] == p2_id)]
h2h_baseline = history_df_baseline[((history_df_baseline['Player 1 ID'] == p1_id) &
                                    (history_df_baseline['Player 2 ID'] == p2_id)) |
                                   ((history_df_baseline['Player 1 ID'] == p2_id) &
                                    (history_df_baseline['Player 2 ID'] == p1_id))]

print(f"\nBASELINE:")
print(f"  history_df rows: {len(history_df_baseline)}")
print(f"  P1 games: {len(p1_games_baseline)}")
print(f"  P2 games: {len(p2_games_baseline)}")
print(f"  H2H games: {len(h2h_baseline)}")
print(f"  P1 last game date: {p1_games_baseline['Date'].max() if len(p1_games_baseline) > 0 else 'None'}")
print(f"  P2 last game date: {p2_games_baseline['Date'].max() if len(p2_games_baseline) > 0 else 'None'}")

# OPTIMIZED approach
p1_all_indices = player_match_indices[p1_id]
p1_history_end = bisect.bisect_left(p1_all_indices, test_index)
p1_history_indices = p1_all_indices[:p1_history_end]
p1_games_opt = df.iloc[p1_history_indices] if p1_history_indices else df.iloc[0:0]

p2_all_indices = player_match_indices[p2_id]
p2_history_end = bisect.bisect_left(p2_all_indices, test_index)
p2_history_indices = p2_all_indices[:p2_history_end]
p2_games_opt = df.iloc[p2_history_indices] if p2_history_indices else df.iloc[0:0]

h2h_pair_key = (min(p1_id, p2_id), max(p1_id, p2_id))
h2h_all_indices = h2h_pair_indices[h2h_pair_key]
h2h_history_end = bisect.bisect_left(h2h_all_indices, test_index)
h2h_history_indices = h2h_all_indices[:h2h_history_end]
h2h_opt = df.iloc[h2h_history_indices] if h2h_history_indices else df.iloc[0:0]

print(f"\nOPTIMIZED:")
print(f"  P1 all indices count: {len(p1_all_indices)}")
print(f"  P1 history indices count: {len(p1_history_indices)}")
print(f"  P1 games: {len(p1_games_opt)}")
print(f"  P2 games: {len(p2_games_opt)}")
print(f"  H2H games: {len(h2h_opt)}")
print(f"  P1 last game date: {p1_games_opt['Date'].max() if len(p1_games_opt) > 0 else 'None'}")
print(f"  P2 last game date: {p2_games_opt['Date'].max() if len(p2_games_opt) > 0 else 'None'}")

# Compare
print(f"\n{'='*60}")
print("COMPARISON:")
p1_match = len(p1_games_baseline) == len(p1_games_opt)
p2_match = len(p2_games_baseline) == len(p2_games_opt)
h2h_match = len(h2h_baseline) == len(h2h_opt)

print(f"  P1 games match: {p1_match} ({len(p1_games_baseline)} vs {len(p1_games_opt)})")
print(f"  P2 games match: {p2_match} ({len(p2_games_baseline)} vs {len(p2_games_opt)})")
print(f"  H2H games match: {h2h_match} ({len(h2h_baseline)} vs {len(h2h_opt)})")

if not p1_match or not p2_match or not h2h_match:
    print("\n[MISMATCH DETECTED] Investigating differences...")

    # Check P1 games differences
    baseline_indices = set(p1_games_baseline.index)
    opt_indices = set(p1_history_indices)

    in_baseline_not_opt = baseline_indices - opt_indices
    in_opt_not_baseline = opt_indices - baseline_indices

    print(f"\n  P1 games in baseline but not optimized: {len(in_baseline_not_opt)}")
    if in_baseline_not_opt:
        print(f"    Sample indices: {list(in_baseline_not_opt)[:5]}")
        for idx in list(in_baseline_not_opt)[:3]:
            row = df.iloc[idx]
            print(f"      idx={idx}: P1={row['Player 1 ID']}, P2={row['Player 2 ID']}, target p1_id={p1_id}")

    print(f"\n  P1 games in optimized but not baseline: {len(in_opt_not_baseline)}")
    if in_opt_not_baseline:
        print(f"    Sample indices: {list(in_opt_not_baseline)[:5]}")
else:
    print("\n[OK] All counts match!")

# Even if counts match, verify actual indices are identical
baseline_p1_indices = set(p1_games_baseline.index)
opt_p1_indices = set(p1_history_indices)
indices_match = baseline_p1_indices == opt_p1_indices
print(f"\n  P1 indices identical: {indices_match}")

# Calculate Time_Since_Last for comparison
current_date = match['Date']

p1_last_baseline = p1_games_baseline['Date'].max()
p2_last_baseline = p2_games_baseline['Date'].max()
if pd.notna(p1_last_baseline):
    p1_hours_baseline = (current_date - p1_last_baseline).total_seconds() / 3600
else:
    p1_hours_baseline = 72
if pd.notna(p2_last_baseline):
    p2_hours_baseline = (current_date - p2_last_baseline).total_seconds() / 3600
else:
    p2_hours_baseline = 72
time_advantage_baseline = p1_hours_baseline - p2_hours_baseline

p1_last_opt = p1_games_opt['Date'].max()
p2_last_opt = p2_games_opt['Date'].max()
if pd.notna(p1_last_opt):
    p1_hours_opt = (current_date - p1_last_opt).total_seconds() / 3600
else:
    p1_hours_opt = 72
if pd.notna(p2_last_opt):
    p2_hours_opt = (current_date - p2_last_opt).total_seconds() / 3600
else:
    p2_hours_opt = 72
time_advantage_opt = p1_hours_opt - p2_hours_opt

print(f"\n  Time_Since_Last_Advantage:")
print(f"    Baseline: {time_advantage_baseline:.2f} hours")
print(f"    Optimized: {time_advantage_opt:.2f} hours")
print(f"    Match: {abs(time_advantage_baseline - time_advantage_opt) < 0.01}")
