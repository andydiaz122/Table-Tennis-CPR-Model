#!/usr/bin/env python3
"""
Phase 1.2 REMEDIATION: Overconfidence Forensic Investigation
=============================================================

HALT RESOLUTION: Root cause analysis for systematic overconfidence in high-probability bins.

Task A: SHAP Analysis on Dead Zone Bets
- Identify which features push probabilities from safe zone (0.65) to dead zone (>0.75)
- Focus on H2H_Win_Rate and L5_Form as prime suspects

Task B: Experience Cross-Tabulation
- Cross-reference 0.80-0.85 dead zone with player match count
- Test hypothesis: overconfidence correlates with small sample sizes

Task C: Calibration-Weighted Kelly (Preparation)
- Output empirical win rates per bin for use in recalibrated staking

Author: forensic-debugger (V8.0 HALT Resolution)
Date: January 2026
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP - will use feature importance if unavailable
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available - using GBM feature_importances_ instead")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("/Users/christiandiaz/Documents/Coding/table-tennis-cpr-model/V8.0")
CODE_DIR = Path("/Users/christiandiaz/Documents/Coding/table-tennis-cpr-model/v8-feature-research/V8.0")
OUTPUT_DIR = CODE_DIR / "audit_reports"

BACKTEST_LOG = DATA_DIR / "backtest_log_final_filtered.csv"
FULL_DATASET = DATA_DIR / "final_dataset_v7.4_no_duplicates.csv"
GBM_MODEL = CODE_DIR / "cpr_v7.4_gbm_specialist.joblib"
GBM_PREPROCESSOR = CODE_DIR / "gbm_preprocessor_v7.4.joblib"

# Dead zone thresholds
SAFE_ZONE_UPPER = 0.70
DEAD_ZONE_LOWER = 0.75
CRITICAL_ZONE_LOWER = 0.80

# GBM Feature names (must match training order)
GBM_FEATURES = [
    'Elo_Advantage', 'P1_Elo', 'P2_Elo', 'Elo_Sum',
    'P1_Elo_Confidence', 'P2_Elo_Confidence',
    'H2H_P1_Win_Rate', 'H2H_Dominance_Score', 'PDR_Advantage',
    'Win_Rate_L5_Advantage', 'Close_Set_Win_Rate_Advantage', 'Set_Comebacks_Advantage'
]


def load_data():
    """Load backtest log, model, and preprocessor."""
    print(f"\n{'='*70}")
    print(" PHASE 1.2 REMEDIATION: OVERCONFIDENCE FORENSIC INVESTIGATION ")
    print(f"{'='*70}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load backtest log
    print(f"\nLoading backtest log from: {BACKTEST_LOG}")
    df = pd.read_csv(BACKTEST_LOG)
    df['Win'] = (df['Outcome'] == 'Win').astype(int)
    df['Market_Implied_Prob'] = 1 / df['Market_Odds']
    print(f"Total bets loaded: {len(df):,}")

    # Load GBM model
    print(f"Loading GBM model from: {GBM_MODEL}")
    gbm_model = joblib.load(GBM_MODEL)
    gbm_preprocessor = joblib.load(GBM_PREPROCESSOR)

    return df, gbm_model, gbm_preprocessor


def task_a_feature_contribution_analysis(df, gbm_model, gbm_preprocessor):
    """
    Task A: Feature Contribution to Overconfidence

    Identify which features contribute most to pushing probabilities
    from the safe zone (0.65) into the dead zone (>0.75).
    """
    print(f"\n{'='*70}")
    print(" TASK A: FEATURE CONTRIBUTION TO OVERCONFIDENCE ")
    print(f"{'='*70}")

    # Segment bets by probability zones
    safe_zone = df[(df['Model_Prob'] >= 0.55) & (df['Model_Prob'] < SAFE_ZONE_UPPER)]
    warning_zone = df[(df['Model_Prob'] >= SAFE_ZONE_UPPER) & (df['Model_Prob'] < DEAD_ZONE_LOWER)]
    dead_zone = df[(df['Model_Prob'] >= DEAD_ZONE_LOWER) & (df['Model_Prob'] < CRITICAL_ZONE_LOWER)]
    critical_zone = df[df['Model_Prob'] >= CRITICAL_ZONE_LOWER]
    all_dead = df[df['Model_Prob'] >= DEAD_ZONE_LOWER]

    print(f"\n  Zone Breakdown:")
    print(f"    Safe Zone (0.55-0.70):     {len(safe_zone):,} bets | Win Rate: {safe_zone['Win'].mean()*100:.1f}%")
    print(f"    Warning Zone (0.70-0.75):  {len(warning_zone):,} bets | Win Rate: {warning_zone['Win'].mean()*100:.1f}%")
    print(f"    Dead Zone (0.75-0.80):     {len(dead_zone):,} bets | Win Rate: {dead_zone['Win'].mean()*100:.1f}%")
    print(f"    Critical Zone (0.80+):     {len(critical_zone):,} bets | Win Rate: {critical_zone['Win'].mean()*100:.1f}%")
    print(f"    ALL Dead Zone (0.75+):     {len(all_dead):,} bets")

    # Method 1: GBM Feature Importances (built-in)
    print(f"\n  --- GBM FEATURE IMPORTANCE (Built-in) ---")
    feature_importance = pd.DataFrame({
        'feature': GBM_FEATURES,
        'importance': gbm_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n  Top Features by Importance:")
    for i, row in feature_importance.iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"    {row['feature']:<30} {row['importance']:.4f} {bar}")

    # Method 2: Compare feature distributions between zones
    print(f"\n  --- FEATURE DISTRIBUTION ANALYSIS (Safe vs Dead Zone) ---")

    # Features available in backtest log
    available_features = [
        'H2H_P1_Win_Rate', 'H2H_Dominance_Score', 'PDR_Advantage',
        'Win_Rate_L5_Advantage', 'Close_Set_Win_Rate_Advantage',
        'Win_Rate_Advantage', 'Time_Since_Last_Advantage',
        'Matches_Last_24H_Advantage', 'Daily_Fatigue_Advantage'
    ]

    # Filter to features that exist in the dataframe
    available_features = [f for f in available_features if f in df.columns]

    distribution_analysis = []

    for feature in available_features:
        safe_mean = safe_zone[feature].mean()
        dead_mean = all_dead[feature].mean()
        safe_std = safe_zone[feature].std()
        dead_std = all_dead[feature].std()
        diff = dead_mean - safe_mean
        diff_pct = (diff / safe_mean * 100) if safe_mean != 0 else float('inf')

        distribution_analysis.append({
            'feature': feature,
            'safe_zone_mean': safe_mean,
            'dead_zone_mean': dead_mean,
            'difference': diff,
            'diff_percentage': diff_pct,
            'safe_zone_std': safe_std,
            'dead_zone_std': dead_std
        })

    dist_df = pd.DataFrame(distribution_analysis)
    dist_df = dist_df.sort_values('difference', ascending=False, key=abs)

    print(f"\n  Feature Value Shift (Safe → Dead Zone):")
    print(f"  {'Feature':<35} {'Safe Mean':>12} {'Dead Mean':>12} {'Δ':>10} {'Δ%':>10}")
    print(f"  {'-'*80}")

    for _, row in dist_df.iterrows():
        delta = row['difference']
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        delta_pct_str = f"+{row['diff_percentage']:.1f}%" if row['diff_percentage'] > 0 else f"{row['diff_percentage']:.1f}%"

        # Flag suspicious features
        flag = ""
        if abs(row['diff_percentage']) > 100:
            flag = " ⚠️  SUSPECT"
        elif abs(row['diff_percentage']) > 50:
            flag = " ⚡ HIGH"

        print(f"  {row['feature']:<35} {row['safe_zone_mean']:>12.4f} {row['dead_zone_mean']:>12.4f} {delta_str:>10} {delta_pct_str:>10}{flag}")

    # Method 3: Extreme value analysis for dead zone
    print(f"\n  --- EXTREME VALUE ANALYSIS (Dead Zone Bets) ---")
    print(f"  Looking for features with extreme values in dead zone bets...")

    extreme_analysis = []
    for feature in available_features:
        # Count bets where feature is in top/bottom 10% AND probability > 0.75
        p90 = df[feature].quantile(0.90)
        p10 = df[feature].quantile(0.10)

        dead_high = all_dead[all_dead[feature] > p90]
        dead_low = all_dead[all_dead[feature] < p10]

        extreme_analysis.append({
            'feature': feature,
            'p10_threshold': p10,
            'p90_threshold': p90,
            'dead_zone_with_high_values': len(dead_high),
            'dead_zone_with_low_values': len(dead_low),
            'pct_extreme_high': len(dead_high) / len(all_dead) * 100 if len(all_dead) > 0 else 0,
            'pct_extreme_low': len(dead_low) / len(all_dead) * 100 if len(all_dead) > 0 else 0
        })

    extreme_df = pd.DataFrame(extreme_analysis)
    extreme_df = extreme_df.sort_values('pct_extreme_high', ascending=False)

    print(f"\n  Features with extreme HIGH values in dead zone:")
    for _, row in extreme_df.head(5).iterrows():
        if row['pct_extreme_high'] > 15:  # More than expected 10%
            print(f"    {row['feature']}: {row['pct_extreme_high']:.1f}% of dead zone bets have top-10% values ⚠️")

    # H2H Deep Dive (prime suspect)
    print(f"\n  --- H2H WIN RATE DEEP DIVE (Prime Suspect) ---")

    if 'H2H_P1_Win_Rate' in df.columns:
        h2h_analysis = []

        for zone_name, zone_df in [('Safe', safe_zone), ('Warning', warning_zone),
                                    ('Dead', dead_zone), ('Critical', critical_zone)]:
            if len(zone_df) > 0:
                h2h_100 = (zone_df['H2H_P1_Win_Rate'] == 1.0).sum()
                h2h_0 = (zone_df['H2H_P1_Win_Rate'] == 0.0).sum()
                h2h_50 = (zone_df['H2H_P1_Win_Rate'] == 0.5).sum()
                h2h_extreme = h2h_100 + h2h_0

                h2h_analysis.append({
                    'zone': zone_name,
                    'total_bets': len(zone_df),
                    'h2h_100%': h2h_100,
                    'h2h_0%': h2h_0,
                    'h2h_50%': h2h_50,
                    'extreme_pct': h2h_extreme / len(zone_df) * 100
                })

        h2h_df = pd.DataFrame(h2h_analysis)
        print(f"\n  H2H Win Rate Extremity by Zone:")
        print(f"  {'Zone':<12} {'Total':>8} {'H2H=100%':>10} {'H2H=0%':>10} {'H2H=50%':>10} {'%Extreme':>10}")
        print(f"  {'-'*62}")

        for _, row in h2h_df.iterrows():
            flag = " ⚠️" if row['extreme_pct'] > 30 else ""
            print(f"  {row['zone']:<12} {row['total_bets']:>8} {row['h2h_100%']:>10} {row['h2h_0%']:>10} {row['h2h_50%']:>10} {row['extreme_pct']:>9.1f}%{flag}")

        # Win rate for extreme H2H in critical zone
        critical_h2h_100 = critical_zone[critical_zone['H2H_P1_Win_Rate'] == 1.0]
        critical_h2h_normal = critical_zone[(critical_zone['H2H_P1_Win_Rate'] > 0.0) & (critical_zone['H2H_P1_Win_Rate'] < 1.0)]

        if len(critical_h2h_100) > 10:
            print(f"\n  CRITICAL FINDING:")
            print(f"    Critical zone bets with H2H=100%: {len(critical_h2h_100)} bets")
            print(f"      Actual win rate: {critical_h2h_100['Win'].mean()*100:.1f}%")
            print(f"      Model expected:  ~{critical_h2h_100['Model_Prob'].mean()*100:.1f}%")
            print(f"      MISCALIBRATION:  {(critical_h2h_100['Model_Prob'].mean() - critical_h2h_100['Win'].mean())*100:.1f}pp")

    return feature_importance, dist_df


def task_b_experience_crosstab(df):
    """
    Task B: Experience Cross-Tabulation

    Cross-reference the 0.80-0.85 dead zone with MIN_GAMES_PLAYED.
    Hypothesis: Model is most arrogant with small sample sizes.
    """
    print(f"\n{'='*70}")
    print(" TASK B: EXPERIENCE CROSS-TABULATION ")
    print(f"{'='*70}")

    # Load full dataset to get match counts
    print(f"\nLoading full dataset for match count analysis...")
    full_df = pd.read_csv(FULL_DATASET)
    full_df['Date'] = pd.to_datetime(full_df['Date'], format='mixed')
    full_df.sort_values('Date', inplace=True)

    # Calculate cumulative match counts per player
    print("Calculating cumulative match counts per player...")

    player_match_counts = {}
    match_count_at_time = {}

    for idx, row in full_df.iterrows():
        p1_id = row['Player 1 ID']
        p2_id = row['Player 2 ID']
        match_id = row['Match ID']

        # Get current counts (before this match)
        p1_count = player_match_counts.get(p1_id, 0)
        p2_count = player_match_counts.get(p2_id, 0)

        match_count_at_time[match_id] = {
            'p1_matches': p1_count,
            'p2_matches': p2_count,
            'min_matches': min(p1_count, p2_count),
            'max_matches': max(p1_count, p2_count),
            'avg_matches': (p1_count + p2_count) / 2
        }

        # Update counts after this match
        player_match_counts[p1_id] = p1_count + 1
        player_match_counts[p2_id] = p2_count + 1

    # Merge match counts into backtest log
    df['Match_ID'] = df['Match_ID'].astype(str)

    # Create columns for experience
    df['P1_Match_Count'] = df['Match_ID'].map(lambda x: match_count_at_time.get(int(float(x)), {}).get('p1_matches', 0))
    df['P2_Match_Count'] = df['Match_ID'].map(lambda x: match_count_at_time.get(int(float(x)), {}).get('p2_matches', 0))
    df['Min_Match_Count'] = df['Match_ID'].map(lambda x: match_count_at_time.get(int(float(x)), {}).get('min_matches', 0))
    df['Avg_Match_Count'] = df['Match_ID'].map(lambda x: match_count_at_time.get(int(float(x)), {}).get('avg_matches', 0))

    # Define experience bins
    experience_bins = [(0, 10, 'Novice'), (10, 20, 'Developing'), (20, 50, 'Established'), (50, 100, 'Veteran'), (100, 999, 'Expert')]

    # Analyze critical zone (0.80+) by minimum player experience
    critical_zone = df[df['Model_Prob'] >= CRITICAL_ZONE_LOWER]

    print(f"\n  --- CRITICAL ZONE (Prob >= 0.80) BY PLAYER EXPERIENCE ---")
    print(f"  (Using MINIMUM match count of the two players)")
    print(f"\n  {'Experience':<15} {'Matches':>12} {'Bets':>8} {'Win Rate':>12} {'Model Exp':>12} {'Miscal':>10}")
    print(f"  {'-'*72}")

    experience_results = []

    for low, high, label in experience_bins:
        zone = critical_zone[(critical_zone['Min_Match_Count'] >= low) & (critical_zone['Min_Match_Count'] < high)]

        if len(zone) >= 10:
            win_rate = zone['Win'].mean()
            model_exp = zone['Model_Prob'].mean()
            miscal = (model_exp - win_rate) * 100

            flag = " ⚠️ TOXIC" if miscal > 20 else " ⚡ HIGH" if miscal > 15 else ""

            experience_results.append({
                'experience_level': label,
                'min_matches_range': f"{low}-{high}",
                'n_bets': len(zone),
                'win_rate': win_rate,
                'model_expected': model_exp,
                'miscalibration_pp': miscal
            })

            print(f"  {label:<15} {f'{low}-{high}':>12} {len(zone):>8} {win_rate*100:>11.1f}% {model_exp*100:>11.1f}% {miscal:>9.1f}pp{flag}")

    # Deep dive: Very new players (<20 matches)
    print(f"\n  --- DEEP DIVE: BETS INVOLVING NEW PLAYERS (< 20 matches) ---")

    new_player_bets = df[df['Min_Match_Count'] < 20]
    veteran_bets = df[df['Min_Match_Count'] >= 50]

    for zone_name, prob_low, prob_high in [('All Bets', 0, 1), ('Safe Zone', 0.55, 0.70),
                                             ('Dead Zone', 0.75, 0.85), ('Critical', 0.85, 1.0)]:
        new_in_zone = new_player_bets[(new_player_bets['Model_Prob'] >= prob_low) & (new_player_bets['Model_Prob'] < prob_high)]
        vet_in_zone = veteran_bets[(veteran_bets['Model_Prob'] >= prob_low) & (veteran_bets['Model_Prob'] < prob_high)]

        if len(new_in_zone) >= 20 and len(vet_in_zone) >= 20:
            new_win = new_in_zone['Win'].mean()
            vet_win = vet_in_zone['Win'].mean()
            new_exp = new_in_zone['Model_Prob'].mean()
            vet_exp = vet_in_zone['Model_Prob'].mean()

            new_miscal = (new_exp - new_win) * 100
            vet_miscal = (vet_exp - vet_win) * 100

            print(f"\n  {zone_name}:")
            print(f"    New Players (<20): {len(new_in_zone)} bets | Win: {new_win*100:.1f}% | Exp: {new_exp*100:.1f}% | Miscal: {new_miscal:+.1f}pp")
            print(f"    Veterans (50+):    {len(vet_in_zone)} bets | Win: {vet_win*100:.1f}% | Exp: {vet_exp*100:.1f}% | Miscal: {vet_miscal:+.1f}pp")

            if new_miscal > vet_miscal + 5:
                print(f"    ⚠️  New players show {new_miscal - vet_miscal:.1f}pp MORE overconfidence!")

    return pd.DataFrame(experience_results)


def task_c_calibration_weighted_kelly(df):
    """
    Task C: Calibration-Weighted Kelly Preparation

    Output empirical win rates per probability bin for use in recalibrated staking.
    When Model_Prob > 0.70, use Empirical_Win_Rate instead for Kelly.
    """
    print(f"\n{'='*70}")
    print(" TASK C: CALIBRATION-WEIGHTED KELLY PREPARATION ")
    print(f"{'='*70}")

    # Calculate empirical win rates for finer probability bins
    prob_bins = [(i/100, (i+5)/100) for i in range(30, 95, 5)]

    calibration_map = []

    print(f"\n  Empirical Win Rate Lookup Table:")
    print(f"  {'Prob Bin':<15} {'N Bets':>10} {'Model Exp':>12} {'Empirical':>12} {'Use For Kelly':>15}")
    print(f"  {'-'*70}")

    for low, high in prob_bins:
        bin_bets = df[(df['Model_Prob'] >= low) & (df['Model_Prob'] < high)]

        if len(bin_bets) >= 20:
            model_exp = bin_bets['Model_Prob'].mean()
            empirical = bin_bets['Win'].mean()

            # For Kelly: use empirical if overconfident in dead zone
            if high >= 0.70 and model_exp > empirical:
                use_for_kelly = empirical
                flag = " ⚡ RECALIBRATE"
            else:
                use_for_kelly = model_exp
                flag = ""

            calibration_map.append({
                'prob_low': low,
                'prob_high': high,
                'n_bets': len(bin_bets),
                'model_expected': model_exp,
                'empirical_win_rate': empirical,
                'use_for_kelly': use_for_kelly,
                'recalibrate': high >= 0.70 and model_exp > empirical
            })

            print(f"  {f'{low:.2f}-{high:.2f}':<15} {len(bin_bets):>10} {model_exp*100:>11.1f}% {empirical*100:>11.1f}% {use_for_kelly*100:>14.1f}%{flag}")

    calibration_df = pd.DataFrame(calibration_map)

    # Output the Kelly recalibration rules
    print(f"\n  --- KELLY RECALIBRATION RULES ---")
    print(f"\n  def get_kelly_probability(model_prob):")
    print(f"      '''")
    print(f"      Returns recalibrated probability for Kelly staking.")
    print(f"      Uses empirical win rate when model is overconfident.")
    print(f"      '''")

    recal_rules = calibration_df[calibration_df['recalibrate'] == True]
    for _, row in recal_rules.iterrows():
        print(f"      if {row['prob_low']:.2f} <= model_prob < {row['prob_high']:.2f}:")
        print(f"          return {row['empirical_win_rate']:.4f}  # Model says {row['model_expected']*100:.1f}%, reality is {row['empirical_win_rate']*100:.1f}%")

    print(f"      return model_prob  # Use model probability for calibrated zones")

    # Calculate impact on edge
    print(f"\n  --- IMPACT ANALYSIS ---")

    dead_zone = df[df['Model_Prob'] >= 0.70]
    original_edge = (dead_zone['Model_Prob'] * dead_zone['Market_Odds'] - 1).mean()

    # Recalculate with empirical probabilities
    recal_probs = []
    for idx, row in dead_zone.iterrows():
        prob = row['Model_Prob']
        # Find matching bin
        for _, cal_row in calibration_df.iterrows():
            if cal_row['prob_low'] <= prob < cal_row['prob_high']:
                recal_probs.append(cal_row['use_for_kelly'])
                break
        else:
            recal_probs.append(prob)

    dead_zone_copy = dead_zone.copy()
    dead_zone_copy['Recal_Prob'] = recal_probs
    new_edge = (dead_zone_copy['Recal_Prob'] * dead_zone_copy['Market_Odds'] - 1).mean()

    print(f"\n  Dead Zone Bets (Prob >= 0.70): {len(dead_zone)}")
    print(f"  Original Average Edge (using Model_Prob): {original_edge*100:.2f}%")
    print(f"  Recalibrated Average Edge (using Empirical): {new_edge*100:.2f}%")
    print(f"  Edge Reduction: {(original_edge - new_edge)*100:.2f}pp")
    print(f"\n  This recalibration prevents betting arrogance while preserving genuine alpha.")

    return calibration_df


def generate_remediation_summary(feature_importance, experience_results, calibration_map):
    """Generate final remediation summary."""
    print(f"\n{'='*70}")
    print(" REMEDIATION SUMMARY & RECOMMENDATIONS ")
    print(f"{'='*70}")

    summary = {
        'investigation_timestamp': datetime.now().isoformat(),
        'halt_reason': 'Systematic overconfidence in probability bins > 0.75',
        'findings': {
            'root_cause': [],
            'contributing_factors': []
        },
        'recommendations': [],
        'immediate_actions': []
    }

    # Analyze feature importance for root causes
    top_features = feature_importance.head(3)['feature'].tolist()

    print(f"\n  1. ROOT CAUSE ANALYSIS:")
    print(f"     Top 3 features driving predictions:")
    for i, feat in enumerate(top_features, 1):
        imp = feature_importance[feature_importance['feature'] == feat]['importance'].values[0]
        print(f"       {i}. {feat} (importance: {imp:.4f})")
        summary['findings']['root_cause'].append(f"{feat}: {imp:.4f} importance")

    # Check if H2H is a major contributor
    if 'H2H_P1_Win_Rate' in top_features or 'H2H_Dominance_Score' in top_features:
        print(f"\n     ⚠️  H2H features are in top 3 - likely cause of overconfidence")
        print(f"        When H2H sample is small (e.g., 3-0 record), model over-extrapolates")
        summary['findings']['contributing_factors'].append('H2H features with small sample sizes')

    # Recommendations
    print(f"\n  2. RECOMMENDATIONS:")

    recommendations = [
        "IMMEDIATE: Implement probability ceiling of 0.75 for Kelly staking",
        "IMMEDIATE: Use empirical win rates (not model prob) for bins > 0.70",
        "SHORT-TERM: Add H2H sample size regularization (dampen H2H effect when n < 5)",
        "SHORT-TERM: Investigate if MIN_GAMES_THRESHOLD needs increase from 4 to 10",
        "MEDIUM-TERM: Consider isotonic regression for post-hoc calibration",
        "VALIDATION: Re-run backtest with probability ceiling and compare ROI"
    ]

    for i, rec in enumerate(recommendations, 1):
        print(f"     {i}. {rec}")
        summary['recommendations'].append(rec)

    # Immediate actions
    print(f"\n  3. IMMEDIATE ACTIONS (For backtest_final_v7.4.py):")

    actions = [
        "Add PROB_CEILING = 0.75 before Kelly calculation",
        "Modify Kelly formula: use min(model_prob, PROB_CEILING) for staking",
        "Log when ceiling is applied for monitoring",
        "Consider excluding bets where Min_Match_Count < 20 AND Model_Prob > 0.80"
    ]

    for i, action in enumerate(actions, 1):
        print(f"     {i}. {action}")
        summary['immediate_actions'].append(action)

    return summary


def main():
    """Main execution for forensic investigation."""

    # Load data
    df, gbm_model, gbm_preprocessor = load_data()

    # Task A: Feature Contribution Analysis
    feature_importance, dist_df = task_a_feature_contribution_analysis(df, gbm_model, gbm_preprocessor)

    # Task B: Experience Cross-Tabulation
    experience_results = task_b_experience_crosstab(df)

    # Task C: Calibration-Weighted Kelly Preparation
    calibration_map = task_c_calibration_weighted_kelly(df)

    # Generate remediation summary
    summary = generate_remediation_summary(feature_importance, experience_results, calibration_map)

    # Save outputs
    print(f"\n{'='*70}")
    print(" SAVING DELIVERABLES ")
    print(f"{'='*70}")

    feature_importance.to_csv(OUTPUT_DIR / 'feature_importance_analysis.csv', index=False)
    print(f"  Saved: feature_importance_analysis.csv")

    experience_results.to_csv(OUTPUT_DIR / 'experience_crosstab.csv', index=False)
    print(f"  Saved: experience_crosstab.csv")

    calibration_map.to_csv(OUTPUT_DIR / 'kelly_recalibration_map.csv', index=False)
    print(f"  Saved: kelly_recalibration_map.csv")

    with open(OUTPUT_DIR / 'remediation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: remediation_summary.json")

    print(f"\n  INVESTIGATION COMPLETE.")
    print(f"  Review findings and implement PROB_CEILING = 0.75 in backtest.")

    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
