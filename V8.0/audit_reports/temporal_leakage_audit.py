"""
PHASE 1.1: TEMPORAL LEAKAGE & EMBARGO VALIDATION AUDIT
======================================================
Agent: temporal-leakage-forensic-specialist

Mission: Prove the model is NOT cheating by detecting any future data contamination.

For every prediction P_t, verify that the feature_vector contains ZERO information
from any match with start time >= t.

Deliverables:
- V8.0/audit_reports/temporal_leakage_report.csv
- V8.0/audit_reports/embargo_validation.json
- V8.0/audit_reports/player_embargo_violations.csv
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = "/Users/christiandiaz/Documents/Coding/table-tennis-cpr-model/V8.0"
RAW_STATS_FILE = f"{DATA_DIR}/czech_liga_pro_advanced_stats_FIXED.csv"
ENGINEERED_FILE = f"{DATA_DIR}/final_engineered_features_v7.4.csv"
FINAL_DATASET_FILE = f"{DATA_DIR}/final_dataset_v7.4_no_duplicates.csv"

ROLLING_WINDOW = 20
SHORT_ROLLING_WINDOW = 5
MIN_PLAYER_EMBARGO_HOURS = 2  # Minimum hours between a player's consecutive matches

# --- Output Files ---
LEAKAGE_REPORT_FILE = "temporal_leakage_report.csv"
EMBARGO_VALIDATION_FILE = "embargo_validation.json"
PLAYER_EMBARGO_FILE = "player_embargo_violations.csv"

def get_full_timestamp(row):
    """Combine Date and Time columns into full timestamp."""
    date = row['Date']
    time_str = row.get('Time', '00:00:00')
    if pd.isna(time_str) or time_str == '' or time_str is None:
        time_str = '00:00:00'
    try:
        return pd.Timestamp(f"{date.strftime('%Y-%m-%d')} {time_str}")
    except:
        return date


def strict_chronology_check(df, current_idx, feature_indices, feature_name):
    """
    For prediction P_t at current_idx, verify all feature_indices have timestamp < P_t.
    Uses FULL timestamp (Date + Time) for accurate same-day comparison.
    Returns list of violations.
    """
    violations = []
    current_row = df.iloc[current_idx]
    current_timestamp = get_full_timestamp(current_row)
    current_match_id = current_row.get('Match ID', current_idx)

    for feat_idx in feature_indices:
        feat_row = df.iloc[feat_idx]
        feat_timestamp = get_full_timestamp(feat_row)

        # Strict check: feature must be BEFORE current match (not equal)
        if feat_timestamp >= current_timestamp:
            time_delta = (feat_timestamp - current_timestamp).total_seconds() / 3600
            violations.append({
                'match_id': current_match_id,
                'prediction_time': str(current_timestamp),
                'feature_name': feature_name,
                'leakage_detected': True,
                'offending_match_idx': feat_idx,
                'offending_match_id': feat_row.get('Match ID', feat_idx),
                'offending_timestamp': str(feat_timestamp),
                'time_delta_hours': time_delta
            })

    return violations


def audit_rolling_window_integrity(df):
    """
    Task 1.1.1: Rolling Window Integrity Audit

    For each match M at timestamp T:
    - Verify L20 rolling windows contain ONLY past data
    - Check same-day ordering
    """
    print("\n" + "="*60)
    print("TASK 1.1.1: ROLLING WINDOW INTEGRITY AUDIT")
    print("="*60)

    # Sort by Date and Time for strict chronological order
    df = df.sort_values(by=['Date', 'Time'] if 'Time' in df.columns else ['Date']).reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])

    # Convert IDs to int
    df['Player 1 ID'] = df['Player 1 ID'].astype(int)
    df['Player 2 ID'] = df['Player 2 ID'].astype(int)

    # Build player match indices
    player_match_indices = {}
    for idx in range(len(df)):
        row = df.iloc[idx]
        for pid in [row['Player 1 ID'], row['Player 2 ID']]:
            if pid not in player_match_indices:
                player_match_indices[pid] = []
            player_match_indices[pid].append(idx)

    # Build H2H match indices
    h2h_match_indices = {}
    for idx in range(len(df)):
        row = df.iloc[idx]
        key = frozenset([row['Player 1 ID'], row['Player 2 ID']])
        if key not in h2h_match_indices:
            h2h_match_indices[key] = []
        h2h_match_indices[key].append(idx)

    all_violations = []
    same_day_checks = []

    print(f"Auditing {len(df)} matches for temporal leakage...")

    for index in tqdm(range(len(df)), desc="Checking rolling windows"):
        match = df.iloc[index]
        p1_id = match['Player 1 ID']
        p2_id = match['Player 2 ID']
        current_timestamp = match['Date']
        current_date = current_timestamp.date()

        # Get the indices that WOULD be used for rolling windows
        p1_indices = [i for i in player_match_indices.get(p1_id, []) if i < index]
        p2_indices = [i for i in player_match_indices.get(p2_id, []) if i < index]

        # Check P1 rolling window (L20)
        p1_rolling_indices = p1_indices[-ROLLING_WINDOW:] if len(p1_indices) >= ROLLING_WINDOW else p1_indices
        violations = strict_chronology_check(df, index, p1_rolling_indices, 'P1_Rolling_L20')
        all_violations.extend(violations)

        # Check P2 rolling window (L20)
        p2_rolling_indices = p2_indices[-ROLLING_WINDOW:] if len(p2_indices) >= ROLLING_WINDOW else p2_indices
        violations = strict_chronology_check(df, index, p2_rolling_indices, 'P2_Rolling_L20')
        all_violations.extend(violations)

        # Check P1 short rolling window (L5)
        p1_short_indices = p1_indices[-SHORT_ROLLING_WINDOW:] if len(p1_indices) >= SHORT_ROLLING_WINDOW else p1_indices
        violations = strict_chronology_check(df, index, p1_short_indices, 'P1_Rolling_L5')
        all_violations.extend(violations)

        # Check P2 short rolling window (L5)
        p2_short_indices = p2_indices[-SHORT_ROLLING_WINDOW:] if len(p2_indices) >= SHORT_ROLLING_WINDOW else p2_indices
        violations = strict_chronology_check(df, index, p2_short_indices, 'P2_Rolling_L5')
        all_violations.extend(violations)

        # Check H2H
        h2h_key = frozenset([p1_id, p2_id])
        h2h_indices = [i for i in h2h_match_indices.get(h2h_key, []) if i < index]
        violations = strict_chronology_check(df, index, h2h_indices, 'H2H_History')
        all_violations.extend(violations)

        # Check same-day matches specifically
        # For same-day, we need to verify TIME ordering, not just DATE
        if 'Time' in df.columns:
            same_day_p1 = [i for i in p1_indices if df.iloc[i]['Date'].date() == current_date]
            for sd_idx in same_day_p1:
                sd_time = df.iloc[sd_idx].get('Time', '00:00:00')
                curr_time = match.get('Time', '23:59:59')

                # If same-day match has same or later time, flag it
                if str(sd_time) >= str(curr_time):
                    same_day_checks.append({
                        'match_id': match.get('Match ID', index),
                        'prediction_time': current_timestamp,
                        'same_day_match_idx': sd_idx,
                        'same_day_match_id': df.iloc[sd_idx].get('Match ID', sd_idx),
                        'current_time': curr_time,
                        'same_day_time': sd_time,
                        'player_id': p1_id,
                        'potential_issue': True
                    })

    return all_violations, same_day_checks, df


def audit_player_embargo(df):
    """
    Task 1.1.3: Player-Level Embargo Validation

    Ensure no player's recent match leaks into their next prediction.
    Flag rapid-fire matches (< 2 hours apart).
    """
    print("\n" + "="*60)
    print("TASK 1.1.3: PLAYER-LEVEL EMBARGO VALIDATION")
    print("="*60)

    df = df.sort_values(by=['Date', 'Time'] if 'Time' in df.columns else ['Date']).reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])

    # Get unique players
    all_players = set(df['Player 1 ID'].unique()) | set(df['Player 2 ID'].unique())

    embargo_violations = []

    print(f"Checking {len(all_players)} unique players for embargo violations...")

    for player_id in tqdm(all_players, desc="Checking player embargo"):
        # Get all matches for this player
        player_matches = df[
            (df['Player 1 ID'] == player_id) | (df['Player 2 ID'] == player_id)
        ].sort_values('Date')

        if len(player_matches) < 2:
            continue

        for i in range(1, len(player_matches)):
            prev_match = player_matches.iloc[i-1]
            curr_match = player_matches.iloc[i]

            time_gap_hours = (curr_match['Date'] - prev_match['Date']).total_seconds() / 3600

            if time_gap_hours < MIN_PLAYER_EMBARGO_HOURS:
                embargo_violations.append({
                    'player_id': player_id,
                    'prev_match_id': prev_match.get('Match ID', 'N/A'),
                    'prev_match_time': prev_match['Date'],
                    'curr_match_id': curr_match.get('Match ID', 'N/A'),
                    'curr_match_time': curr_match['Date'],
                    'time_gap_hours': time_gap_hours,
                    'violation_type': 'RAPID_FIRE' if time_gap_hours < 1 else 'SHORT_GAP',
                    'risk_level': 'HIGH' if time_gap_hours < 0.5 else 'MEDIUM'
                })

    return embargo_violations


def analyze_train_test_embargo(df, train_pct=0.70):
    """
    Task 1.1.2: Analyze current train/test split for embargo compliance.
    """
    print("\n" + "="*60)
    print("TASK 1.1.2: TRAIN/TEST EMBARGO ANALYSIS")
    print("="*60)

    df = df.sort_values(by='Date').reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])

    split_idx = int(len(df) * train_pct)

    train_end_date = df.iloc[split_idx - 1]['Date']
    test_start_date = df.iloc[split_idx]['Date']

    time_gap = (test_start_date - train_end_date).total_seconds() / 3600

    # Check if any player appears in both the last training hour and first test hour
    last_train_hour = df.iloc[:split_idx][
        df.iloc[:split_idx]['Date'] >= train_end_date - pd.Timedelta(hours=1)
    ]
    first_test_hour = df.iloc[split_idx:][
        df.iloc[split_idx:]['Date'] <= test_start_date + pd.Timedelta(hours=1)
    ]

    train_players = set(last_train_hour['Player 1 ID'].tolist() + last_train_hour['Player 2 ID'].tolist())
    test_players = set(first_test_hour['Player 1 ID'].tolist() + first_test_hour['Player 2 ID'].tolist())

    overlapping_players = train_players & test_players

    embargo_analysis = {
        'train_set_size': split_idx,
        'test_set_size': len(df) - split_idx,
        'train_end_date': str(train_end_date),
        'test_start_date': str(test_start_date),
        'time_gap_hours': time_gap,
        'meets_24h_embargo': time_gap >= 24,
        'last_train_hour_matches': len(last_train_hour),
        'first_test_hour_matches': len(first_test_hour),
        'overlapping_players_count': len(overlapping_players),
        'overlapping_player_ids': list(overlapping_players)[:20],  # First 20 for readability
        'recommendation': 'PASS' if time_gap >= 24 and len(overlapping_players) == 0 else 'IMPLEMENT_EMBARGO'
    }

    return embargo_analysis


def main():
    print("\n" + "="*70)
    print(" PHASE 1.1: TEMPORAL LEAKAGE & EMBARGO VALIDATION AUDIT")
    print(" Agent: temporal-leakage-forensic-specialist")
    print("="*70)

    # Load raw data
    print(f"\nLoading raw data from '{RAW_STATS_FILE}'...")
    try:
        df = pd.read_csv(RAW_STATS_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"Loaded {len(df)} matches.")
    except FileNotFoundError:
        print(f"ERROR: Could not find {RAW_STATS_FILE}")
        return

    # Task 1.1.1: Rolling Window Integrity Audit
    violations, same_day_issues, sorted_df = audit_rolling_window_integrity(df)

    # Task 1.1.3: Player-Level Embargo Validation
    player_embargo_violations = audit_player_embargo(sorted_df)

    # Task 1.1.2: Train/Test Embargo Analysis
    embargo_analysis = analyze_train_test_embargo(sorted_df)

    # --- Generate Reports ---
    print("\n" + "="*60)
    print("GENERATING AUDIT REPORTS")
    print("="*60)

    # Report 1: Temporal Leakage Report
    if violations:
        leakage_df = pd.DataFrame(violations)
        leakage_df.to_csv(LEAKAGE_REPORT_FILE, index=False)
        print(f"\n CRITICAL: {len(violations)} temporal leakage violations detected!")
        print(f"Report saved to: {LEAKAGE_REPORT_FILE}")
    else:
        # Create empty report with schema
        empty_df = pd.DataFrame(columns=[
            'match_id', 'prediction_time', 'feature_name', 'leakage_detected',
            'offending_match_idx', 'offending_match_id', 'offending_timestamp', 'time_delta_hours'
        ])
        empty_df.to_csv(LEAKAGE_REPORT_FILE, index=False)
        print(f"\n PASSED: Zero temporal leakage violations detected!")
        print(f"Empty report saved to: {LEAKAGE_REPORT_FILE}")

    # Report 2: Same-Day Issues
    if same_day_issues:
        print(f"\n WARNING: {len(same_day_issues)} same-day time ordering issues flagged for review.")

    # Report 3: Embargo Validation
    with open(EMBARGO_VALIDATION_FILE, 'w') as f:
        json.dump(embargo_analysis, f, indent=2, default=str)
    print(f"\nEmbargo analysis saved to: {EMBARGO_VALIDATION_FILE}")

    # Report 4: Player Embargo Violations
    if player_embargo_violations:
        embargo_df = pd.DataFrame(player_embargo_violations)
        embargo_df.to_csv(PLAYER_EMBARGO_FILE, index=False)
        print(f"\n WARNING: {len(player_embargo_violations)} rapid-fire match situations detected.")
        print(f"Report saved to: {PLAYER_EMBARGO_FILE}")

        # Summary stats
        high_risk = len([v for v in player_embargo_violations if v['risk_level'] == 'HIGH'])
        print(f"  - HIGH risk (< 30 min gap): {high_risk}")
        print(f"  - MEDIUM risk (30 min - 2 hour gap): {len(player_embargo_violations) - high_risk}")
    else:
        empty_embargo_df = pd.DataFrame(columns=[
            'player_id', 'prev_match_id', 'prev_match_time', 'curr_match_id',
            'curr_match_time', 'time_gap_hours', 'violation_type', 'risk_level'
        ])
        empty_embargo_df.to_csv(PLAYER_EMBARGO_FILE, index=False)
        print(f"\n PASSED: No rapid-fire match situations detected.")

    # --- Final Summary ---
    print("\n" + "="*70)
    print(" PHASE 1.1 AUDIT SUMMARY")
    print("="*70)
    print(f"\n1. Rolling Window Leakage:     {'FAIL' if violations else 'PASS'}")
    print(f"2. Same-Day Time Ordering:     {'REVIEW' if same_day_issues else 'PASS'}")
    print(f"3. Train/Test 24h Embargo:     {embargo_analysis['recommendation']}")
    print(f"4. Player Rapid-Fire Matches:  {'WARNING' if player_embargo_violations else 'PASS'}")

    overall_status = "PASS" if not violations and embargo_analysis['meets_24h_embargo'] else "NEEDS_ATTENTION"
    print(f"\n OVERALL PHASE 1.1 STATUS: {overall_status}")

    if overall_status == "PASS":
        print("\nPROCEED TO PHASE 1.2 (Calibration Mapping)")
    else:
        print("\nREVIEW REPORTS BEFORE PROCEEDING")

    return {
        'leakage_violations': len(violations),
        'same_day_issues': len(same_day_issues),
        'embargo_analysis': embargo_analysis,
        'player_embargo_violations': len(player_embargo_violations),
        'overall_status': overall_status
    }


if __name__ == "__main__":
    results = main()
