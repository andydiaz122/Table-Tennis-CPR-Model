#!/usr/bin/env python3
"""
Phase 1.2: Probability Calibration Audit
=========================================

Mission: Map model probability outputs to empirical win rates and identify "Dead Zones."

Deliverables:
- calibration_by_bin.csv: Calibration analysis per probability bin
- dead_zones.json: Dead zone identification with recommendations
- calibration_drift.csv: Rolling calibration check for regime drift
- calibration_curve.png: Visual calibration curve
- MODEL_ARROGANCE_FLIPS.csv: Bets where model diverges >20% from market

HALT TRIGGERS:
- Any bin with miscalibration >15%
- Outputs Forensic Report if HALT triggered

Hard Constraints:
- 3% EDGE_FLOOR (non-negotiable)
- 15% miscalibration threshold = HALT
- Minimum 30 bets per bin for statistical validity

Author: probability-calibration-auditor (V8.0 Implementation)
Date: January 2026
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
EDGE_FLOOR = 0.03  # 3% minimum edge (non-negotiable)
MISCALIBRATION_HALT_THRESHOLD = 0.15  # 15% = HALT trigger
DEAD_ZONE_WARNING_THRESHOLD = 0.10  # 10% = dead zone warning
MODEL_ARROGANCE_THRESHOLD = 0.20  # 20% divergence from market
MIN_BETS_PER_BIN = 30  # Minimum sample for statistical validity
ROLLING_WINDOW_BETS = 200  # Rolling window for drift analysis
BRIER_DEGRADATION_THRESHOLD = 1.5  # 50% increase from baseline = degradation

# Probability bins (0.30 to 0.90 in 5% increments)
PROB_BINS = [
    (0.30, 0.35), (0.35, 0.40), (0.40, 0.45), (0.45, 0.50),
    (0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
    (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90)
]

# Data paths
DATA_DIR = Path("/Users/christiandiaz/Documents/Coding/table-tennis-cpr-model/V8.0")
OUTPUT_DIR = Path(__file__).parent
BACKTEST_LOG = DATA_DIR / "backtest_log_final_filtered.csv"


def load_backtest_data():
    """Load and prepare backtest data for calibration analysis."""
    print(f"\n{'='*60}")
    print("PHASE 1.2: PROBABILITY CALIBRATION AUDIT")
    print(f"{'='*60}")
    print(f"\nLoading backtest data from: {BACKTEST_LOG}")

    df = pd.read_csv(BACKTEST_LOG)
    print(f"Total bets loaded: {len(df):,}")

    # Convert Outcome to binary (Win=1, Loss=0)
    df['Win'] = (df['Outcome'] == 'Win').astype(int)

    # Calculate implied market probability from odds
    df['Market_Implied_Prob'] = 1 / df['Market_Odds']

    # Calculate model-market divergence
    df['Model_Market_Divergence'] = abs(df['Model_Prob'] - df['Market_Implied_Prob'])

    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'])

    # Validate edge floor
    bets_below_edge_floor = (df['Edge'] < EDGE_FLOOR).sum()
    if bets_below_edge_floor > 0:
        print(f"WARNING: {bets_below_edge_floor} bets have edge < {EDGE_FLOOR*100}% (should be 0)")

    return df


def calculate_brier_score(predictions, outcomes):
    """Calculate Brier score for probability calibration."""
    return ((predictions - outcomes) ** 2).mean()


def task_1_2_1_probability_bin_analysis(df):
    """
    Task 1.2.1: Probability Bin Analysis
    Calculate empirical win rate for each probability bin.
    """
    print(f"\n{'='*60}")
    print("TASK 1.2.1: PROBABILITY BIN ANALYSIS")
    print(f"{'='*60}")

    calibration_results = []
    halt_triggered = False
    halt_reasons = []

    for bin_low, bin_high in PROB_BINS:
        bin_bets = df[(df['Model_Prob'] >= bin_low) & (df['Model_Prob'] < bin_high)]
        n_bets = len(bin_bets)

        if n_bets < MIN_BETS_PER_BIN:
            print(f"  [{bin_low:.2f}-{bin_high:.2f}]: SKIPPED (only {n_bets} bets, need {MIN_BETS_PER_BIN})")
            continue

        expected_prob = bin_bets['Model_Prob'].mean()
        empirical_win_rate = bin_bets['Win'].mean()
        calibration_error = empirical_win_rate - expected_prob
        abs_calibration_error = abs(calibration_error)

        # Brier score for this bin
        brier_score = calculate_brier_score(bin_bets['Model_Prob'], bin_bets['Win'])

        # Reliability Score = 1 - abs(empirical - predicted)
        reliability_score = 1 - abs_calibration_error

        # Dead zone detection
        is_dead_zone = abs_calibration_error > DEAD_ZONE_WARNING_THRESHOLD
        is_halt_zone = abs_calibration_error > MISCALIBRATION_HALT_THRESHOLD

        # Classify zone type
        if calibration_error > DEAD_ZONE_WARNING_THRESHOLD:
            zone_type = "underconfident"  # Model predicts lower than empirical
        elif calibration_error < -DEAD_ZONE_WARNING_THRESHOLD:
            zone_type = "overconfident"   # Model predicts higher than empirical
        elif abs(expected_prob - 0.50) < 0.05:
            zone_type = "noise_zone"       # Near coin-flip
        else:
            zone_type = "calibrated"

        result = {
            'bin': f"{bin_low:.2f}-{bin_high:.2f}",
            'bin_low': bin_low,
            'bin_high': bin_high,
            'n_bets': n_bets,
            'expected_prob': expected_prob,
            'empirical_win_rate': empirical_win_rate,
            'calibration_error': calibration_error,
            'abs_calibration_error': abs_calibration_error,
            'reliability_score': reliability_score,
            'brier_score': brier_score,
            'zone_type': zone_type,
            'is_dead_zone': is_dead_zone,
            'is_halt_zone': is_halt_zone
        }
        calibration_results.append(result)

        # Status output
        status = "OK"
        if is_halt_zone:
            status = "HALT TRIGGER"
            halt_triggered = True
            halt_reasons.append(f"Bin {bin_low:.2f}-{bin_high:.2f}: {abs_calibration_error*100:.1f}% miscalibration")
        elif is_dead_zone:
            status = "DEAD ZONE"

        direction = "+" if calibration_error > 0 else ""
        print(f"  [{bin_low:.2f}-{bin_high:.2f}]: n={n_bets:4d} | "
              f"Exp={expected_prob:.3f} | Emp={empirical_win_rate:.3f} | "
              f"Err={direction}{calibration_error*100:.1f}% | "
              f"Rel={reliability_score:.3f} | {status}")

    # Create DataFrame
    calibration_df = pd.DataFrame(calibration_results)

    # Overall Brier score
    overall_brier = calculate_brier_score(df['Model_Prob'], df['Win'])
    print(f"\nOverall Brier Score: {overall_brier:.4f}")
    print(f"Brier Score Threshold (useful model): < 0.25")
    print(f"Status: {'PASS' if overall_brier < 0.25 else 'FAIL'}")

    return calibration_df, halt_triggered, halt_reasons, overall_brier


def task_1_2_2_dead_zone_identification(calibration_df):
    """
    Task 1.2.2: Dead Zone Identification & Mapping
    Identify probability ranges where model systematically over/under-estimates.
    """
    print(f"\n{'='*60}")
    print("TASK 1.2.2: DEAD ZONE IDENTIFICATION")
    print(f"{'='*60}")

    dead_zones = []
    safe_zones = []

    for _, row in calibration_df.iterrows():
        zone_info = {
            'range': row['bin'],
            'n_bets': row['n_bets'],
            'expected': row['expected_prob'],
            'empirical': row['empirical_win_rate'],
            'calibration_error': row['calibration_error'],
            'reliability_score': row['reliability_score']
        }

        if row['is_halt_zone']:
            zone_info['type'] = row['zone_type']
            zone_info['severity'] = 'CRITICAL'
            zone_info['recommendation'] = 'EXCLUDE - Miscalibration exceeds HALT threshold (>15%)'
            dead_zones.append(zone_info)
            print(f"  CRITICAL DEAD ZONE: {row['bin']} - {row['zone_type'].upper()}")
            print(f"    Expected: {row['expected_prob']:.3f}, Empirical: {row['empirical_win_rate']:.3f}")
            print(f"    Recommendation: EXCLUDE or apply calibration correction")

        elif row['is_dead_zone']:
            zone_info['type'] = row['zone_type']
            zone_info['severity'] = 'WARNING'
            zone_info['recommendation'] = 'MONITOR - Apply probability correction or reduce stake'
            dead_zones.append(zone_info)
            print(f"  WARNING DEAD ZONE: {row['bin']} - {row['zone_type']}")
            print(f"    Expected: {row['expected_prob']:.3f}, Empirical: {row['empirical_win_rate']:.3f}")

        else:
            zone_info['recommendation'] = 'SAFE - Calibration within acceptable range'
            safe_zones.append(zone_info)

    # Identify primary betting zone (best calibrated bins with good sample size)
    safe_df = pd.DataFrame(safe_zones) if safe_zones else pd.DataFrame()
    if not safe_df.empty:
        best_calibrated = safe_df.nsmallest(3, 'calibration_error')
        print(f"\n  SAFE ZONES (well-calibrated):")
        for _, zone in safe_df.iterrows():
            print(f"    {zone['range']}: Error={zone['calibration_error']*100:.1f}%, n={zone['n_bets']}")

    return {
        'dead_zones': dead_zones,
        'safe_zones': safe_zones,
        'total_dead_zones': len(dead_zones),
        'total_safe_zones': len(safe_zones)
    }


def task_1_2_3_temporal_calibration_drift(df, baseline_brier):
    """
    Task 1.2.3: Temporal Calibration Drift Analysis
    Check if calibration degrades over time (regime drift).
    """
    print(f"\n{'='*60}")
    print("TASK 1.2.3: TEMPORAL CALIBRATION DRIFT ANALYSIS")
    print(f"{'='*60}")

    # Sort by date
    df_sorted = df.sort_values('Date').reset_index(drop=True)

    rolling_calibration = []
    degradation_detected = False
    degradation_periods = []

    for i in range(ROLLING_WINDOW_BETS, len(df_sorted)):
        window = df_sorted.iloc[i-ROLLING_WINDOW_BETS:i]

        brier_score = calculate_brier_score(window['Model_Prob'], window['Win'])
        calibration_error = window['Win'].mean() - window['Model_Prob'].mean()

        rolling_calibration.append({
            'end_idx': i,
            'end_date': window.iloc[-1]['Date'],
            'brier_score': brier_score,
            'calibration_error': calibration_error,
            'window_win_rate': window['Win'].mean(),
            'window_expected': window['Model_Prob'].mean()
        })

        # Flag degradation
        if brier_score > baseline_brier * BRIER_DEGRADATION_THRESHOLD:
            degradation_detected = True
            degradation_periods.append({
                'end_date': window.iloc[-1]['Date'],
                'brier_score': brier_score,
                'baseline_ratio': brier_score / baseline_brier
            })

    drift_df = pd.DataFrame(rolling_calibration)

    # Statistics
    print(f"  Rolling window size: {ROLLING_WINDOW_BETS} bets")
    print(f"  Windows analyzed: {len(drift_df)}")
    print(f"  Baseline Brier score: {baseline_brier:.4f}")
    print(f"  Degradation threshold: {baseline_brier * BRIER_DEGRADATION_THRESHOLD:.4f}")
    print(f"\n  Brier Score Statistics:")
    print(f"    Min: {drift_df['brier_score'].min():.4f}")
    print(f"    Max: {drift_df['brier_score'].max():.4f}")
    print(f"    Mean: {drift_df['brier_score'].mean():.4f}")
    print(f"    Std: {drift_df['brier_score'].std():.4f}")

    print(f"\n  Calibration Error Statistics:")
    print(f"    Min: {drift_df['calibration_error'].min()*100:.2f}%")
    print(f"    Max: {drift_df['calibration_error'].max()*100:.2f}%")
    print(f"    Mean: {drift_df['calibration_error'].mean()*100:.2f}%")

    # Check if calibration error stays within ±5%
    within_bounds = (drift_df['calibration_error'].abs() <= 0.05).mean()
    print(f"\n  % of windows with calibration error within +/-5%: {within_bounds*100:.1f}%")
    print(f"  Status: {'PASS' if within_bounds >= 0.80 else 'INVESTIGATE'}")

    if degradation_detected:
        print(f"\n  WARNING: {len(degradation_periods)} periods with Brier > 1.5x baseline detected")
    else:
        print(f"\n  No significant calibration degradation detected")

    return drift_df, degradation_detected, degradation_periods


def identify_model_arrogance_flips(df):
    """
    Identify MODEL_ARROGANCE_FLIPS: bets where model diverges >20% from market odds.
    """
    print(f"\n{'='*60}")
    print("MODEL ARROGANCE FLIP ANALYSIS")
    print(f"{'='*60}")

    arrogance_flips = df[df['Model_Market_Divergence'] > MODEL_ARROGANCE_THRESHOLD].copy()

    print(f"  Total bets: {len(df):,}")
    print(f"  Arrogance flips (>{MODEL_ARROGANCE_THRESHOLD*100}% divergence): {len(arrogance_flips):,}")
    print(f"  Percentage: {len(arrogance_flips)/len(df)*100:.2f}%")

    if len(arrogance_flips) > 0:
        arrogance_win_rate = arrogance_flips['Win'].mean()
        regular_bets = df[df['Model_Market_Divergence'] <= MODEL_ARROGANCE_THRESHOLD]
        regular_win_rate = regular_bets['Win'].mean() if len(regular_bets) > 0 else 0

        print(f"\n  Arrogance flip win rate: {arrogance_win_rate*100:.2f}%")
        print(f"  Regular bet win rate: {regular_win_rate*100:.2f}%")
        print(f"  Delta: {(arrogance_win_rate - regular_win_rate)*100:.2f}pp")

        # When model is more confident than market
        model_higher = arrogance_flips[arrogance_flips['Model_Prob'] > arrogance_flips['Market_Implied_Prob']]
        model_lower = arrogance_flips[arrogance_flips['Model_Prob'] < arrogance_flips['Market_Implied_Prob']]

        if len(model_higher) > 0:
            print(f"\n  Model MORE confident than market: {len(model_higher)} bets")
            print(f"    Win rate: {model_higher['Win'].mean()*100:.2f}%")

        if len(model_lower) > 0:
            print(f"  Model LESS confident than market: {len(model_lower)} bets")
            print(f"    Win rate: {model_lower['Win'].mean()*100:.2f}%")

    return arrogance_flips


def create_calibration_curve(calibration_df, output_path):
    """Create calibration curve visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Calibration Curve
    ax1 = axes[0]

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)

    # Add shaded regions for dead zones
    ax1.fill_between([0, 1], [0-0.15, 1-0.15], [0+0.15, 1+0.15],
                     alpha=0.2, color='green', label='Acceptable (+/-15%)')
    ax1.fill_between([0, 1], [0-0.10, 1-0.10], [0+0.10, 1+0.10],
                     alpha=0.2, color='yellow', label='Warning (+/-10%)')

    # Actual calibration points
    colors = []
    for _, row in calibration_df.iterrows():
        if row['is_halt_zone']:
            colors.append('red')
        elif row['is_dead_zone']:
            colors.append('orange')
        else:
            colors.append('blue')

    ax1.scatter(calibration_df['expected_prob'], calibration_df['empirical_win_rate'],
               c=colors, s=calibration_df['n_bets']/10, alpha=0.7, edgecolors='black')

    ax1.set_xlabel('Model Predicted Probability', fontsize=12)
    ax1.set_ylabel('Empirical Win Rate', fontsize=12)
    ax1.set_title('Calibration Curve by Probability Bin', fontsize=14)
    ax1.set_xlim(0.25, 0.95)
    ax1.set_ylim(0.25, 0.95)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Add bin labels
    for _, row in calibration_df.iterrows():
        ax1.annotate(row['bin'], (row['expected_prob'], row['empirical_win_rate']),
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # Plot 2: Reliability Score by Bin
    ax2 = axes[1]
    colors = ['red' if row['is_halt_zone'] else 'orange' if row['is_dead_zone'] else 'green'
              for _, row in calibration_df.iterrows()]

    bars = ax2.bar(calibration_df['bin'], calibration_df['reliability_score'], color=colors, alpha=0.7)
    ax2.axhline(y=0.85, color='green', linestyle='--', label='Good (>0.85)')
    ax2.axhline(y=0.90, color='red', linestyle='--', label='Excellent (>0.90)')

    ax2.set_xlabel('Probability Bin', fontsize=12)
    ax2.set_ylabel('Reliability Score', fontsize=12)
    ax2.set_title('Reliability Score by Bin (Higher = Better Calibrated)', fontsize=14)
    ax2.set_ylim(0.5, 1.0)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Calibration curve saved to: {output_path}")


def create_drift_plot(drift_df, baseline_brier, output_path):
    """Create temporal calibration drift visualization."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Plot 1: Rolling Brier Score
    ax1 = axes[0]
    ax1.plot(drift_df['end_date'], drift_df['brier_score'], 'b-', alpha=0.7, linewidth=1)
    ax1.axhline(y=baseline_brier, color='green', linestyle='--',
                label=f'Baseline ({baseline_brier:.4f})', linewidth=2)
    ax1.axhline(y=baseline_brier * BRIER_DEGRADATION_THRESHOLD, color='red', linestyle='--',
                label=f'Degradation Threshold ({baseline_brier * BRIER_DEGRADATION_THRESHOLD:.4f})', linewidth=2)

    ax1.set_ylabel('Rolling Brier Score', fontsize=12)
    ax1.set_title(f'Temporal Calibration Drift (Rolling {ROLLING_WINDOW_BETS}-bet window)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Rolling Calibration Error
    ax2 = axes[1]
    ax2.plot(drift_df['end_date'], drift_df['calibration_error']*100, 'b-', alpha=0.7, linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=5, color='orange', linestyle='--', label='+5% threshold')
    ax2.axhline(y=-5, color='orange', linestyle='--', label='-5% threshold')
    ax2.fill_between(drift_df['end_date'], -5, 5, alpha=0.2, color='green')

    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Calibration Error (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Calibration drift plot saved to: {output_path}")


def generate_forensic_report(halt_reasons, calibration_df, dead_zone_info, overall_brier):
    """Generate forensic report when HALT is triggered."""
    report = {
        'report_type': 'CALIBRATION_FORENSIC_REPORT',
        'generated_at': datetime.now().isoformat(),
        'status': 'HALT_TRIGGERED',
        'halt_reasons': halt_reasons,
        'summary': {
            'total_bins_analyzed': len(calibration_df),
            'dead_zones_critical': len([z for z in dead_zone_info['dead_zones'] if z['severity'] == 'CRITICAL']),
            'dead_zones_warning': len([z for z in dead_zone_info['dead_zones'] if z['severity'] == 'WARNING']),
            'safe_zones': dead_zone_info['total_safe_zones'],
            'overall_brier_score': overall_brier
        },
        'recommendations': [
            'IMMEDIATE: Do not proceed to Phase 2 until calibration issues resolved',
            'ACTION: Investigate root cause of miscalibration in flagged bins',
            'OPTION 1: Implement probability recalibration using Platt scaling or isotonic regression',
            'OPTION 2: Exclude bets in critically miscalibrated zones (may reduce sample size)',
            'OPTION 3: Investigate if specific player types or match conditions cause miscalibration'
        ],
        'dead_zones_detail': dead_zone_info['dead_zones']
    }
    return report


def main():
    """Main execution for Phase 1.2 Calibration Audit."""
    print("\n" + "="*70)
    print(" CPR V8.0 - PHASE 1.2: PROBABILITY CALIBRATION AUDIT ")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"HALT Threshold: {MISCALIBRATION_HALT_THRESHOLD*100}% miscalibration")
    print(f"Edge Floor: {EDGE_FLOOR*100}% (non-negotiable)")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_backtest_data()

    # Task 1.2.1: Probability Bin Analysis
    calibration_df, halt_triggered, halt_reasons, overall_brier = task_1_2_1_probability_bin_analysis(df)

    # Task 1.2.2: Dead Zone Identification
    dead_zone_info = task_1_2_2_dead_zone_identification(calibration_df)

    # Task 1.2.3: Temporal Calibration Drift
    drift_df, degradation_detected, degradation_periods = task_1_2_3_temporal_calibration_drift(df, overall_brier)

    # Model Arrogance Flip Analysis
    arrogance_flips = identify_model_arrogance_flips(df)

    # Save outputs
    print(f"\n{'='*60}")
    print("SAVING DELIVERABLES")
    print(f"{'='*60}")

    # 1. Calibration by bin
    calibration_df.to_csv(OUTPUT_DIR / 'calibration_by_bin.csv', index=False)
    print(f"  Saved: calibration_by_bin.csv")

    # 2. Dead zones JSON
    with open(OUTPUT_DIR / 'dead_zones.json', 'w') as f:
        json.dump(dead_zone_info, f, indent=2, default=str)
    print(f"  Saved: dead_zones.json")

    # 3. Calibration drift
    drift_df.to_csv(OUTPUT_DIR / 'calibration_drift.csv', index=False)
    print(f"  Saved: calibration_drift.csv")

    # 4. Model arrogance flips
    if len(arrogance_flips) > 0:
        arrogance_flips.to_csv(OUTPUT_DIR / 'MODEL_ARROGANCE_FLIPS.csv', index=False)
        print(f"  Saved: MODEL_ARROGANCE_FLIPS.csv ({len(arrogance_flips)} bets)")

    # 5. Calibration curve plot
    create_calibration_curve(calibration_df, OUTPUT_DIR / 'calibration_curve.png')

    # 6. Drift plot
    create_drift_plot(drift_df, overall_brier, OUTPUT_DIR / 'calibration_drift.png')

    # Final Status
    print(f"\n{'='*70}")
    print(" PHASE 1.2 CALIBRATION AUDIT SUMMARY ")
    print(f"{'='*70}")

    print(f"\n  Overall Brier Score: {overall_brier:.4f} {'(PASS)' if overall_brier < 0.25 else '(FAIL)'}")
    print(f"  Total Dead Zones: {dead_zone_info['total_dead_zones']}")
    print(f"  Total Safe Zones: {dead_zone_info['total_safe_zones']}")
    print(f"  Calibration Degradation: {'DETECTED' if degradation_detected else 'NONE'}")
    print(f"  Model Arrogance Flips: {len(arrogance_flips)}")

    if halt_triggered:
        print(f"\n  {'!'*60}")
        print(f"  HALT TRIGGERED - MISCALIBRATION EXCEEDS {MISCALIBRATION_HALT_THRESHOLD*100}% THRESHOLD")
        print(f"  {'!'*60}")
        for reason in halt_reasons:
            print(f"    - {reason}")

        # Generate forensic report
        forensic_report = generate_forensic_report(halt_reasons, calibration_df, dead_zone_info, overall_brier)
        with open(OUTPUT_DIR / 'FORENSIC_REPORT_HALT.json', 'w') as f:
            json.dump(forensic_report, f, indent=2, default=str)
        print(f"\n  FORENSIC REPORT generated: FORENSIC_REPORT_HALT.json")
        print(f"\n  DO NOT PROCEED TO PHASE 2 UNTIL RESOLVED")

        return False  # Halt
    else:
        print(f"\n  STATUS: PASS - No HALT triggers detected")
        print(f"  Proceed to Phase 1.3 (Residual Analysis)")

        # Create pass summary
        summary = {
            'phase': '1.2',
            'status': 'PASS',
            'timestamp': datetime.now().isoformat(),
            'overall_brier_score': overall_brier,
            'total_bins_analyzed': len(calibration_df),
            'dead_zones': dead_zone_info['total_dead_zones'],
            'safe_zones': dead_zone_info['total_safe_zones'],
            'calibration_degradation': degradation_detected,
            'model_arrogance_flips': len(arrogance_flips),
            'recommendations': [
                f"Monitor {dead_zone_info['total_dead_zones']} dead zones during Phase 2 filter design",
                f"Consider probability recalibration if dead zones impact ROI"
            ] if dead_zone_info['total_dead_zones'] > 0 else ['Calibration is healthy']
        }
        with open(OUTPUT_DIR / 'phase_1_2_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Phase 1.2 summary saved: phase_1_2_summary.json")

        return True  # Pass


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
