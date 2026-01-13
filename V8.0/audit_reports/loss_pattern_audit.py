#!/usr/bin/env python3
"""
Phase 1.3: Loss Pattern Detection Audit
=========================================
Agent: residual-loss-pattern-detective

Mission: Forensically diagnose WHERE and WHEN the model loses money.

Tasks:
- 1.3.1: Hour-of-Day Regime Analysis (Late Night Window 22:00-06:00)
- 1.3.2: H2H Sample Size / Extreme Value Analysis
- 1.3.3: Consecutive Loss Streak Analysis with Circuit Breaker State Machine

Outputs:
- hourly_performance.csv
- h2h_extreme_analysis.csv
- streak_analysis.csv
- loss_pattern_report.json
- hourly_regime_plot.png
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

# === CONFIGURATION ===
DATA_DIR = Path('/Users/christiandiaz/Documents/Coding/table-tennis-cpr-model/V8.0')
OUTPUT_DIR = Path('/Users/christiandiaz/Documents/Coding/table-tennis-cpr-model/v8-feature-research/V8.0/audit_reports')

# Backtest log (filtered results)
BACKTEST_LOG = DATA_DIR / 'backtest_log_final_filtered.csv'
# Raw data with Time column
RAW_DATA = DATA_DIR / 'czech_liga_pro_advanced_stats_FIXED.csv'

# Circuit Breaker Configuration - Calibrated based on observed max drawdown of 36%
CB_CONFIG_AGGRESSIVE = {
    'DRAWDOWN_CAUTION': 0.10,      # 10% drawdown -> CAUTION (half stake)
    'DRAWDOWN_HALT': 0.15,          # 15% drawdown -> HALT (stop betting)
    'CONSECUTIVE_LOSS_CAUTION': 5,  # 5 losses -> CAUTION
    'CONSECUTIVE_LOSS_HALT': 8,     # 8 losses -> HALT
}

CB_CONFIG_MODERATE = {
    'DRAWDOWN_CAUTION': 0.20,      # 20% drawdown -> CAUTION (half stake)
    'DRAWDOWN_HALT': 0.30,          # 30% drawdown -> HALT (stop betting)
    'CONSECUTIVE_LOSS_CAUTION': 7,  # 7 losses -> CAUTION
    'CONSECUTIVE_LOSS_HALT': 10,    # 10 losses -> HALT
}

# Use moderate config by default (based on observed 36% max DD)
CB_CONFIG = CB_CONFIG_MODERATE

# Late Night Regime (22:00 - 06:00)
LATE_NIGHT_HOURS = list(range(22, 24)) + list(range(0, 7))


def load_data():
    """Load backtest log and raw data, merge to get Time column."""
    print("Loading data...")

    # Load backtest log
    bt = pd.read_csv(BACKTEST_LOG)
    print(f"  Backtest log: {len(bt)} bets")

    # Load raw data (only columns we need for merge)
    raw = pd.read_csv(RAW_DATA, usecols=['Match ID', 'Time'])
    raw = raw.rename(columns={'Match ID': 'Match_ID'})
    print(f"  Raw data: {len(raw)} matches")

    # Merge to get Time
    bt = bt.merge(raw, on='Match_ID', how='left')

    # Parse Time column
    bt['Hour'] = pd.to_datetime(bt['Time'], format='%H:%M:%S', errors='coerce').dt.hour

    missing_time = bt['Hour'].isna().sum()
    print(f"  Bets with missing time: {missing_time} ({missing_time/len(bt)*100:.1f}%)")

    # Mark Late Night regime
    bt['Is_Late_Night'] = bt['Hour'].isin(LATE_NIGHT_HOURS)

    return bt


def analyze_hourly_performance(df):
    """Task 1.3.1: Hour-of-Day Regime Analysis."""
    print("\n" + "="*60)
    print("TASK 1.3.1: HOUR-OF-DAY REGIME ANALYSIS")
    print("="*60)

    # Filter out missing hours
    df_valid = df[df['Hour'].notna()].copy()

    # Calculate baseline metrics
    baseline_win_rate = (df_valid['Outcome'] == 'Win').mean()
    baseline_roi = df_valid['Profit'].sum() / df_valid['Stake'].sum()

    print(f"\nBaseline (all hours):")
    print(f"  Win Rate: {baseline_win_rate:.1%}")
    print(f"  ROI: {baseline_roi:.2%}")

    # Group by hour
    hourly_stats = []

    for hour in range(24):
        hour_df = df_valid[df_valid['Hour'] == hour]

        if len(hour_df) < 10:  # Skip hours with insufficient data
            continue

        win_rate = (hour_df['Outcome'] == 'Win').mean()
        roi = hour_df['Profit'].sum() / hour_df['Stake'].sum() if hour_df['Stake'].sum() > 0 else 0
        avg_edge = hour_df['Edge'].mean()
        avg_model_prob = hour_df['Model_Prob'].mean()

        # Chi-square test for win rate difference
        n_wins = (hour_df['Outcome'] == 'Win').sum()
        n_total = len(hour_df)
        expected_wins = n_total * baseline_win_rate

        chi2_stat = (n_wins - expected_wins)**2 / expected_wins if expected_wins > 0 else 0

        hourly_stats.append({
            'hour': hour,
            'n_bets': len(hour_df),
            'win_rate': win_rate,
            'roi': roi,
            'avg_edge': avg_edge,
            'avg_model_prob': avg_model_prob,
            'win_rate_vs_baseline': win_rate - baseline_win_rate,
            'roi_vs_baseline': roi - baseline_roi,
            'is_late_night': hour in LATE_NIGHT_HOURS,
            'chi2_stat': chi2_stat
        })

    hourly_df = pd.DataFrame(hourly_stats)

    # Identify toxic hours (loss rate > baseline + 10%)
    toxic_hours = hourly_df[hourly_df['win_rate_vs_baseline'] < -0.10]

    # Identify profitable hours
    profitable_hours = hourly_df[hourly_df['roi'] > 0.05]

    print(f"\n=== HOURLY PERFORMANCE SUMMARY ===")
    print(hourly_df.to_string(index=False))

    print(f"\n=== LATE NIGHT REGIME (22:00-06:00) ===")
    late_night_df = df_valid[df_valid['Is_Late_Night']]
    if len(late_night_df) > 0:
        ln_win_rate = (late_night_df['Outcome'] == 'Win').mean()
        ln_roi = late_night_df['Profit'].sum() / late_night_df['Stake'].sum()
        print(f"  Bets: {len(late_night_df)} ({len(late_night_df)/len(df_valid)*100:.1f}%)")
        print(f"  Win Rate: {ln_win_rate:.1%} (vs baseline {baseline_win_rate:.1%})")
        print(f"  ROI: {ln_roi:.2%} (vs baseline {baseline_roi:.2%})")
        print(f"  Delta: {(ln_win_rate - baseline_win_rate)*100:+.1f}pp win rate, {(ln_roi - baseline_roi)*100:+.1f}pp ROI")

    if len(toxic_hours) > 0:
        print(f"\n⚠️  TOXIC HOURS (win rate < baseline - 10%):")
        print(toxic_hours[['hour', 'n_bets', 'win_rate', 'roi']].to_string(index=False))
    else:
        print(f"\n✅ No toxic hours detected (all within 10% of baseline)")

    # Save results
    hourly_df.to_csv(OUTPUT_DIR / 'hourly_performance.csv', index=False)
    print(f"\nSaved: hourly_performance.csv")

    return hourly_df


def analyze_h2h_extremes(df):
    """Task 1.3.2: H2H Sample Size / Extreme Value Analysis.

    Since we don't have n_h2h_games directly, we analyze H2H extremes (0.0, 1.0)
    as proxies for potential small-sample issues.
    """
    print("\n" + "="*60)
    print("TASK 1.3.2: H2H EXTREME VALUE ANALYSIS")
    print("="*60)

    # Create H2H bins
    def classify_h2h(rate):
        if pd.isna(rate):
            return 'unknown'
        elif rate == 0.0:
            return 'h2h_0pct'
        elif rate == 1.0:
            return 'h2h_100pct'
        elif rate < 0.35:
            return 'h2h_low (1-35%)'
        elif rate > 0.65:
            return 'h2h_high (65-99%)'
        else:
            return 'h2h_moderate (35-65%)'

    df = df.copy()
    df['H2H_Category'] = df['H2H_P1_Win_Rate'].apply(classify_h2h)

    # Baseline metrics
    baseline_win_rate = (df['Outcome'] == 'Win').mean()

    # Analyze by H2H category
    h2h_stats = []

    for category in df['H2H_Category'].unique():
        cat_df = df[df['H2H_Category'] == category]

        if len(cat_df) < 10:
            continue

        win_rate = (cat_df['Outcome'] == 'Win').mean()
        avg_model_prob = cat_df['Model_Prob'].mean()
        calibration_error = win_rate - avg_model_prob  # Positive = underconfident, Negative = overconfident

        roi = cat_df['Profit'].sum() / cat_df['Stake'].sum() if cat_df['Stake'].sum() > 0 else 0

        h2h_stats.append({
            'category': category,
            'n_bets': len(cat_df),
            'pct_of_bets': len(cat_df) / len(df) * 100,
            'empirical_win_rate': win_rate,
            'avg_model_prob': avg_model_prob,
            'calibration_error': calibration_error,
            'roi': roi,
            'is_miscalibrated': abs(calibration_error) > 0.10,
        })

    h2h_df = pd.DataFrame(h2h_stats).sort_values('n_bets', ascending=False)

    print("\n=== H2H CATEGORY ANALYSIS ===")
    print(h2h_df.to_string(index=False))

    # Specifically analyze H2H = 100% cases (known toxic from Phase 1.2)
    h2h_100 = df[df['H2H_P1_Win_Rate'] == 1.0]
    if len(h2h_100) > 0:
        print(f"\n=== H2H = 100% DEEP DIVE (Root Cause from Phase 1.2) ===")
        print(f"  Total bets: {len(h2h_100)}")
        print(f"  Model predicts: {h2h_100['Model_Prob'].mean():.1%} average")
        print(f"  Actual wins: {(h2h_100['Outcome'] == 'Win').sum()} of {len(h2h_100)} ({(h2h_100['Outcome'] == 'Win').mean():.1%})")
        print(f"  Miscalibration: {((h2h_100['Outcome'] == 'Win').mean() - h2h_100['Model_Prob'].mean())*100:.1f} percentage points")
        print(f"  ROI: {h2h_100['Profit'].sum() / h2h_100['Stake'].sum():.2%}")

        # Betting direction analysis
        bet_on_h2h_favorite = (h2h_100['Bet_On_Player'] == h2h_100['Player_1']).sum()
        bet_on_h2h_underdog = len(h2h_100) - bet_on_h2h_favorite
        print(f"  Betting on H2H favorite (P1): {bet_on_h2h_favorite} bets")
        print(f"  Betting on H2H underdog (P2): {bet_on_h2h_underdog} bets")

    # H2H = 0% analysis
    h2h_0 = df[df['H2H_P1_Win_Rate'] == 0.0]
    if len(h2h_0) > 0:
        print(f"\n=== H2H = 0% ANALYSIS ===")
        print(f"  Total bets: {len(h2h_0)}")
        print(f"  Model predicts: {h2h_0['Model_Prob'].mean():.1%} average")
        print(f"  Actual wins: {(h2h_0['Outcome'] == 'Win').sum()} of {len(h2h_0)} ({(h2h_0['Outcome'] == 'Win').mean():.1%})")
        print(f"  Miscalibration: {((h2h_0['Outcome'] == 'Win').mean() - h2h_0['Model_Prob'].mean())*100:.1f} percentage points")

    # Moderate H2H (35-65%) - expected to be best calibrated
    h2h_mod = df[df['H2H_Category'] == 'h2h_moderate (35-65%)']
    if len(h2h_mod) > 0:
        print(f"\n=== H2H MODERATE (35-65%) - CONTROL GROUP ===")
        print(f"  Total bets: {len(h2h_mod)}")
        print(f"  Model predicts: {h2h_mod['Model_Prob'].mean():.1%} average")
        print(f"  Actual wins: {(h2h_mod['Outcome'] == 'Win').mean():.1%}")
        print(f"  Calibration error: {((h2h_mod['Outcome'] == 'Win').mean() - h2h_mod['Model_Prob'].mean())*100:.1f}pp")

    # Save results
    h2h_df.to_csv(OUTPUT_DIR / 'h2h_extreme_analysis.csv', index=False)
    print(f"\nSaved: h2h_extreme_analysis.csv")

    return h2h_df


def analyze_loss_streaks(df):
    """Task 1.3.3: Consecutive Loss Streak Analysis with Circuit Breaker."""
    print("\n" + "="*60)
    print("TASK 1.3.3: CONSECUTIVE LOSS STREAK & CIRCUIT BREAKER ANALYSIS")
    print("="*60)

    # Sort by date to ensure chronological order
    df = df.sort_values('Date').reset_index(drop=True)

    # Calculate cumulative metrics for circuit breaker analysis
    df = df.copy()
    df['Cumulative_Profit'] = df['Profit'].cumsum()
    df['Cumulative_Stake'] = df['Stake'].cumsum()

    # Peak bankroll tracking (starting at 1000)
    INITIAL_BANKROLL = 1000
    df['Bankroll'] = INITIAL_BANKROLL + df['Cumulative_Profit']
    df['Peak_Bankroll'] = df['Bankroll'].cummax()
    df['Drawdown'] = (df['Peak_Bankroll'] - df['Bankroll']) / df['Peak_Bankroll']

    # Find consecutive loss streaks
    df['Is_Loss'] = (df['Outcome'] == 'Loss').astype(int)

    # Create streak counter
    streak_counter = []
    current_streak = 0

    for is_loss in df['Is_Loss']:
        if is_loss:
            current_streak += 1
        else:
            current_streak = 0
        streak_counter.append(current_streak)

    df['Consecutive_Losses'] = streak_counter

    # Find all significant streaks (>= 5 losses)
    streaks = []
    in_streak = False
    streak_start = 0

    for i, row in df.iterrows():
        if row['Is_Loss'] and not in_streak:
            in_streak = True
            streak_start = i
        elif not row['Is_Loss'] and in_streak:
            in_streak = False
            streak_len = i - streak_start
            if streak_len >= 5:
                streaks.append({
                    'start_idx': streak_start,
                    'end_idx': i - 1,
                    'length': streak_len,
                    'date_start': df.loc[streak_start, 'Date'],
                    'date_end': df.loc[i - 1, 'Date'],
                    'loss_amount': df.loc[streak_start:i-1, 'Profit'].sum(),
                    'max_drawdown_during': df.loc[streak_start:i-1, 'Drawdown'].max(),
                })

    # Check for streak at end
    if in_streak:
        streak_len = len(df) - streak_start
        if streak_len >= 5:
            streaks.append({
                'start_idx': streak_start,
                'end_idx': len(df) - 1,
                'length': streak_len,
                'date_start': df.loc[streak_start, 'Date'],
                'date_end': df.iloc[-1]['Date'],
                'loss_amount': df.loc[streak_start:, 'Profit'].sum(),
                'max_drawdown_during': df.loc[streak_start:, 'Drawdown'].max(),
            })

    streaks_df = pd.DataFrame(streaks)

    # Calculate streak statistics
    max_streak = df['Consecutive_Losses'].max()
    avg_streak_in_streaks = streaks_df['length'].mean() if len(streaks_df) > 0 else 0

    print(f"\n=== LOSS STREAK STATISTICS ===")
    print(f"  Total bets: {len(df)}")
    print(f"  Total losses: {(df['Outcome'] == 'Loss').sum()}")
    print(f"  Win rate: {(df['Outcome'] == 'Win').mean():.1%}")
    print(f"  Maximum consecutive losses: {max_streak}")
    print(f"  Streaks >= 5 losses: {len(streaks_df)}")

    if len(streaks_df) > 0:
        print(f"\n=== SIGNIFICANT LOSS STREAKS (>= 5 consecutive) ===")
        print(streaks_df.to_string(index=False))

    # Expected max streak under random model (binomial)
    loss_rate = (df['Outcome'] == 'Loss').mean()
    n_bets = len(df)
    # Approximation: E[max streak] ≈ log(n) / log(1/p)
    expected_max_streak = np.log(n_bets) / np.log(1/loss_rate) if loss_rate < 1 else n_bets

    print(f"\n=== STREAK RANDOMNESS TEST ===")
    print(f"  Observed max streak: {max_streak}")
    print(f"  Expected max streak (binomial): {expected_max_streak:.1f}")
    print(f"  Ratio: {max_streak / expected_max_streak:.2f}x")

    if max_streak > expected_max_streak * 1.5:
        print(f"  ⚠️  WARNING: Observed streaks significantly exceed random expectation")
        print(f"     This may indicate regime breakdown or model instability")
    else:
        print(f"  ✅ Streak behavior appears consistent with random variation")

    # === CIRCUIT BREAKER SIMULATION ===
    print(f"\n=== CIRCUIT BREAKER SIMULATION ===")

    # Simulate with both configurations
    cb_aggressive = simulate_circuit_breaker(df, CB_CONFIG_AGGRESSIVE)
    cb_moderate = simulate_circuit_breaker(df, CB_CONFIG_MODERATE)

    # Compare
    original_final_bankroll = df['Bankroll'].iloc[-1]
    original_max_dd = df['Drawdown'].max()

    print(f"\n  Without Circuit Breaker:")
    print(f"    Final Bankroll: ${original_final_bankroll:.2f}")
    print(f"    Max Drawdown: {original_max_dd:.1%}")

    print(f"\n  AGGRESSIVE Circuit Breaker (DD={CB_CONFIG_AGGRESSIVE['DRAWDOWN_CAUTION']:.0%}→CAUTION, {CB_CONFIG_AGGRESSIVE['DRAWDOWN_HALT']:.0%}→HALT):")
    print(f"    Final Bankroll: ${cb_aggressive['final_bankroll']:.2f}")
    print(f"    Max Drawdown: {cb_aggressive['max_drawdown']:.1%}")
    print(f"    Times CAUTION triggered: {cb_aggressive['caution_count']}")
    print(f"    Times HALT triggered: {cb_aggressive['halt_count']}")
    print(f"    Bets skipped: {cb_aggressive['bets_skipped']}")

    print(f"\n  MODERATE Circuit Breaker (DD={CB_CONFIG_MODERATE['DRAWDOWN_CAUTION']:.0%}→CAUTION, {CB_CONFIG_MODERATE['DRAWDOWN_HALT']:.0%}→HALT):")
    print(f"    Final Bankroll: ${cb_moderate['final_bankroll']:.2f}")
    print(f"    Max Drawdown: {cb_moderate['max_drawdown']:.1%}")
    print(f"    Times CAUTION triggered: {cb_moderate['caution_count']}")
    print(f"    Times HALT triggered: {cb_moderate['halt_count']}")
    print(f"    Bets skipped: {cb_moderate['bets_skipped']}")

    # Determine best configuration
    if cb_moderate['final_bankroll'] > original_final_bankroll and cb_moderate['max_drawdown'] < original_max_dd:
        best_cb = 'MODERATE'
        cb_results = cb_moderate
    elif cb_aggressive['final_bankroll'] > original_final_bankroll and cb_aggressive['max_drawdown'] < original_max_dd:
        best_cb = 'AGGRESSIVE'
        cb_results = cb_aggressive
    else:
        best_cb = 'NONE (original outperforms)'
        cb_results = cb_moderate  # Use moderate for reporting

    print(f"\n  RECOMMENDATION: {best_cb}")

    # Save streak analysis
    if len(streaks_df) > 0:
        streaks_df.to_csv(OUTPUT_DIR / 'streak_analysis.csv', index=False)
        print(f"\nSaved: streak_analysis.csv")

    return {
        'max_streak': max_streak,
        'expected_max_streak': expected_max_streak,
        'streaks_gte_5': len(streaks_df),
        'circuit_breaker': cb_results,
    }


def simulate_circuit_breaker(df, config=None):
    """Simulate betting with circuit breaker state machine."""
    if config is None:
        config = CB_CONFIG

    INITIAL_BANKROLL = 1000
    bankroll = INITIAL_BANKROLL
    peak_bankroll = INITIAL_BANKROLL

    state = 'NORMAL'  # NORMAL, CAUTION, HALT
    consecutive_losses = 0

    caution_count = 0
    halt_count = 0
    bets_skipped = 0

    bankroll_history = []

    for _, row in df.iterrows():
        # Calculate current drawdown
        drawdown = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0

        # Recovery logic - allow state to improve if conditions improve
        if state == 'HALT':
            if drawdown < config['DRAWDOWN_CAUTION'] * 0.5 and consecutive_losses < 3:
                state = 'NORMAL'
            elif drawdown < config['DRAWDOWN_HALT'] * 0.7:
                state = 'CAUTION'
        elif state == 'CAUTION':
            if drawdown < config['DRAWDOWN_CAUTION'] * 0.5 and consecutive_losses < 3:
                state = 'NORMAL'

        # State transitions based on drawdown (can get worse)
        if drawdown >= config['DRAWDOWN_HALT'] or consecutive_losses >= config['CONSECUTIVE_LOSS_HALT']:
            if state != 'HALT':
                state = 'HALT'
                halt_count += 1
        elif drawdown >= config['DRAWDOWN_CAUTION'] or consecutive_losses >= config['CONSECUTIVE_LOSS_CAUTION']:
            if state == 'NORMAL':
                state = 'CAUTION'
                caution_count += 1

        # Execute bet based on state
        if state == 'HALT':
            bets_skipped += 1
            # Still track what would have happened for analysis
            if row['Outcome'] == 'Win':
                consecutive_losses = 0
            else:
                consecutive_losses += 1
        elif state == 'CAUTION':
            # Half stake
            stake = row['Stake'] / 2
            if row['Outcome'] == 'Win':
                profit = stake * (row['Market_Odds'] - 1)
                consecutive_losses = 0
            else:
                profit = -stake
                consecutive_losses += 1
            bankroll += profit
        else:  # NORMAL
            if row['Outcome'] == 'Win':
                bankroll += row['Profit']
                consecutive_losses = 0
            else:
                bankroll += row['Profit']
                consecutive_losses += 1

        # Update peak
        if bankroll > peak_bankroll:
            peak_bankroll = bankroll

        bankroll_history.append(bankroll)

    max_dd = max((peak_bankroll - min(bankroll_history)) / peak_bankroll if peak_bankroll > 0 else 0, 0)

    return {
        'final_bankroll': bankroll,
        'max_drawdown': max_dd,
        'caution_count': caution_count,
        'halt_count': halt_count,
        'bets_skipped': bets_skipped,
    }


def create_visualizations(hourly_df, df):
    """Create visualization plots."""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Hourly ROI
    ax1 = axes[0, 0]
    colors = ['red' if h in LATE_NIGHT_HOURS else 'steelblue' for h in hourly_df['hour']]
    ax1.bar(hourly_df['hour'], hourly_df['roi'] * 100, color=colors, edgecolor='black', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('ROI (%)')
    ax1.set_title('ROI by Hour of Day (Red = Late Night 22:00-06:00)')
    ax1.set_xticks(range(24))

    # 2. Hourly Win Rate
    ax2 = axes[0, 1]
    baseline_wr = (df['Outcome'] == 'Win').mean()
    ax2.bar(hourly_df['hour'], hourly_df['win_rate'] * 100, color=colors, edgecolor='black', alpha=0.7)
    ax2.axhline(y=baseline_wr * 100, color='green', linestyle='--', linewidth=2, label=f'Baseline ({baseline_wr:.1%})')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Win Rate by Hour of Day')
    ax2.set_xticks(range(24))
    ax2.legend()

    # 3. Cumulative ROI over time (equity curve)
    ax3 = axes[1, 0]
    df_sorted = df.sort_values('Date')
    cumulative_profit = df_sorted['Profit'].cumsum()
    cumulative_stake = df_sorted['Stake'].cumsum()
    cumulative_roi = cumulative_profit / cumulative_stake * 100
    ax3.plot(range(len(cumulative_roi)), cumulative_roi, color='steelblue', linewidth=1)
    ax3.fill_between(range(len(cumulative_roi)), cumulative_roi, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Bet Number')
    ax3.set_ylabel('Cumulative ROI (%)')
    ax3.set_title('Cumulative ROI Over Time (Equity Curve)')

    # 4. Drawdown over time
    ax4 = axes[1, 1]
    INITIAL_BANKROLL = 1000
    bankroll = INITIAL_BANKROLL + cumulative_profit
    peak = bankroll.cummax()
    drawdown = (peak - bankroll) / peak * 100
    ax4.fill_between(range(len(drawdown)), drawdown, color='red', alpha=0.5)
    ax4.axhline(y=10, color='orange', linestyle='--', label='CAUTION (10%)')
    ax4.axhline(y=15, color='red', linestyle='--', label='HALT (15%)')
    ax4.set_xlabel('Bet Number')
    ax4.set_ylabel('Drawdown (%)')
    ax4.set_title('Drawdown Over Time with Circuit Breaker Thresholds')
    ax4.legend()
    ax4.invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hourly_regime_plot.png', dpi=150)
    plt.close()
    print(f"Saved: hourly_regime_plot.png")


def generate_report(hourly_df, h2h_df, streak_results, df):
    """Generate final JSON report."""
    print("\n" + "="*60)
    print("GENERATING FINAL REPORT")
    print("="*60)

    # Late night analysis
    df_valid = df[df['Hour'].notna()]
    late_night = df_valid[df_valid['Is_Late_Night']]

    late_night_roi = late_night['Profit'].sum() / late_night['Stake'].sum() if len(late_night) > 0 else 0
    late_night_wr = (late_night['Outcome'] == 'Win').mean() if len(late_night) > 0 else 0

    baseline_roi = df['Profit'].sum() / df['Stake'].sum()
    baseline_wr = (df['Outcome'] == 'Win').mean()

    # Find worst hours
    worst_hours = hourly_df.nsmallest(3, 'roi')[['hour', 'n_bets', 'win_rate', 'roi']].to_dict('records')
    best_hours = hourly_df.nlargest(3, 'roi')[['hour', 'n_bets', 'win_rate', 'roi']].to_dict('records')

    # H2H extremes
    h2h_100_row = h2h_df[h2h_df['category'] == 'h2h_100pct']
    h2h_100_miscal = h2h_100_row['calibration_error'].values[0] if len(h2h_100_row) > 0 else None

    report = {
        'audit_timestamp': datetime.now().isoformat(),
        'phase': '1.3 - Loss Pattern Detection',
        'total_bets_analyzed': len(df),

        'hour_of_day_analysis': {
            'late_night_regime': {
                'hours': '22:00-06:00',
                'n_bets': len(late_night),
                'pct_of_total': len(late_night) / len(df_valid) * 100 if len(df_valid) > 0 else 0,
                'roi': late_night_roi,
                'win_rate': late_night_wr,
                'roi_vs_baseline': late_night_roi - baseline_roi,
                'win_rate_vs_baseline': late_night_wr - baseline_wr,
                'recommendation': 'EXCLUDE' if late_night_roi < -0.05 else 'MONITOR' if late_night_roi < baseline_roi * 0.5 else 'SAFE'
            },
            'worst_hours': worst_hours,
            'best_hours': best_hours,
        },

        'h2h_extreme_analysis': {
            'h2h_100pct_miscalibration': h2h_100_miscal,
            'recommendation': 'Implement H2H sample size regularization when H2H=100% or H2H=0%',
            'categories': h2h_df.to_dict('records') if len(h2h_df) > 0 else [],
        },

        'loss_streak_analysis': {
            'max_consecutive_losses': streak_results['max_streak'],
            'expected_max_streak_random': streak_results['expected_max_streak'],
            'ratio_observed_vs_expected': streak_results['max_streak'] / streak_results['expected_max_streak'],
            'streaks_gte_5_losses': streak_results['streaks_gte_5'],
            'streak_behavior': 'NORMAL' if streak_results['max_streak'] <= streak_results['expected_max_streak'] * 1.5 else 'CONCERNING',
        },

        'circuit_breaker_simulation': {
            'config': CB_CONFIG,
            'results': streak_results['circuit_breaker'],
            'recommendation': 'IMPLEMENT' if streak_results['circuit_breaker']['halt_count'] > 0 else 'OPTIONAL',
        },

        'recommendations': [],
    }

    # Add recommendations based on findings
    if late_night_roi < baseline_roi * 0.5:
        report['recommendations'].append({
            'priority': 'HIGH',
            'action': 'Consider excluding or reducing stakes during late night hours (22:00-06:00)',
            'reason': f'Late night ROI ({late_night_roi:.2%}) significantly underperforms baseline ({baseline_roi:.2%})',
        })

    if h2h_100_miscal is not None and h2h_100_miscal < -0.15:
        report['recommendations'].append({
            'priority': 'CRITICAL',
            'action': 'Implement H2H sample size regularization',
            'reason': f'H2H=100% bets show {abs(h2h_100_miscal)*100:.1f}pp overconfidence miscalibration',
        })

    if streak_results['max_streak'] > streak_results['expected_max_streak'] * 1.5:
        report['recommendations'].append({
            'priority': 'MEDIUM',
            'action': 'Implement circuit breaker state machine',
            'reason': f'Loss streaks ({streak_results["max_streak"]}) exceed random expectation ({streak_results["expected_max_streak"]:.1f})',
        })

    if streak_results['circuit_breaker']['halt_count'] > 0:
        report['recommendations'].append({
            'priority': 'HIGH',
            'action': 'Circuit breaker would have triggered HALT state',
            'reason': f"Would have skipped {streak_results['circuit_breaker']['bets_skipped']} bets during drawdown events",
        })

    # Save report
    with open(OUTPUT_DIR / 'loss_pattern_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Saved: loss_pattern_report.json")

    # Print summary
    print("\n" + "="*60)
    print("PHASE 1.3 SUMMARY")
    print("="*60)
    print(f"\n📊 HOUR-OF-DAY FINDINGS:")
    print(f"   Late Night ({len(late_night)} bets): ROI {late_night_roi:.2%} vs baseline {baseline_roi:.2%}")

    print(f"\n📊 H2H EXTREME FINDINGS:")
    if h2h_100_miscal is not None:
        print(f"   H2H=100%: {abs(h2h_100_miscal)*100:.1f}pp {'over' if h2h_100_miscal < 0 else 'under'}confidence")

    print(f"\n📊 LOSS STREAK FINDINGS:")
    print(f"   Max streak: {streak_results['max_streak']} (expected: {streak_results['expected_max_streak']:.1f})")
    print(f"   Circuit breaker would skip: {streak_results['circuit_breaker']['bets_skipped']} bets")

    print(f"\n📋 RECOMMENDATIONS ({len(report['recommendations'])} items):")
    for rec in report['recommendations']:
        print(f"   [{rec['priority']}] {rec['action']}")

    return report


def main():
    """Main execution."""
    print("="*60)
    print("PHASE 1.3: LOSS PATTERN DETECTION AUDIT")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Load data
    df = load_data()

    # Task 1.3.1: Hour-of-Day Analysis
    hourly_df = analyze_hourly_performance(df)

    # Task 1.3.2: H2H Extreme Analysis
    h2h_df = analyze_h2h_extremes(df)

    # Task 1.3.3: Loss Streak & Circuit Breaker
    streak_results = analyze_loss_streaks(df)

    # Generate visualizations
    create_visualizations(hourly_df, df)

    # Generate final report
    report = generate_report(hourly_df, h2h_df, streak_results, df)

    print("\n" + "="*60)
    print("PHASE 1.3 COMPLETE")
    print("="*60)

    return report


if __name__ == '__main__':
    main()
