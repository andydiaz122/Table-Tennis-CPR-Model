"""
Filter Stack V8.0 - Phase 2 Implementation (OPTIMIZED)
======================================================

Correct Order of Operations:
1. LOAD RAW DATA
2. DATA TRANSFORMATIONS (Before filtering!)
   2a. H2H Perfect Cap
   2b. Dead Zone Cap  
   2c. RECALCULATE EDGE
3. BOOLEAN MASK FILTERS
4. WATERFALL ANALYSIS
5. OUTPUT METRICS

Author: CPR Quant Team / Strategic Pipeline Overseer
Date: 2026-01-13
Optimized: Edge_Adj >= 3%, Dominance >= 4
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_FILE = "/Users/christiandiaz/Documents/Coding/table-tennis-cpr-model/V8.0/backtest_log_final_filtered.csv"
OUTPUT_FILE = "/Users/christiandiaz/Documents/Coding/table-tennis-cpr-model/v8-feature-research/V8.0/backtest_log_filtered_v8.csv"
REPORT_FILE = "/Users/christiandiaz/Documents/Coding/table-tennis-cpr-model/v8-feature-research/V8.0/filter_stack_v8_report.txt"

INITIAL_BANKROLL = 1000

# HARD CONSTRAINTS
MIN_BETS_THRESHOLD = 1000  # Final minimum (hard constraint)

# TRANSFORMATION PARAMETERS
H2H_PERFECT_CAP = 0.70           # Cap Model_Prob for H2H perfect records
DEAD_ZONE_CAP = 0.75             # Max Model_Prob allowed

# OPTIMIZED FILTER PARAMETERS (From grid search)
EDGE_MIN = 0.03                  # 3% minimum edge (adjusted)
EDGE_MAX = 0.25                  # 25% maximum edge
DOM_MIN = 4                      # Minimum H2H dominance score
ODDS_MIN = 1.10                  # Min odds
ODDS_MAX = 4.00                  # Max odds
MODEL_PROB_MIN = 0.35            # Min probability
MODEL_PROB_MAX = 0.75            # Max probability (after cap)
DIVERGENCE_MIN = 0.03            # Min divergence
DIVERGENCE_MAX = 0.20            # Max divergence


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def calculate_metrics(df, initial_bankroll=1000):
    """Calculate performance metrics using backtest_final_v7.4.py methodology."""
    if len(df) == 0:
        return {'bets': 0, 'roi': 0, 'sharpe': 0, 'max_dd': 0, 'final_bankroll': initial_bankroll}
    
    df = df.copy()
    df['Cumulative_Profit'] = df['Profit'].cumsum()
    df['Bankroll'] = initial_bankroll + df['Cumulative_Profit']
    df['Date'] = pd.to_datetime(df['Date'])
    
    # ROI
    total_staked = df['Stake'].sum()
    total_profit = df['Profit'].sum()
    roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
    
    # Daily resampling for Sharpe and Drawdown
    history_df = df[['Date', 'Bankroll']].copy()
    daily_history_df = history_df.set_index('Date').resample('D').last().ffill().reset_index()
    daily_history_df['Peak'] = daily_history_df['Bankroll'].cummax()
    daily_history_df['Drawdown'] = (daily_history_df['Peak'] - daily_history_df['Bankroll']) / daily_history_df['Peak']
    max_dd = daily_history_df['Drawdown'].max() * 100
    
    # Sharpe Ratio
    daily_history_df['Daily_Return'] = daily_history_df['Bankroll'].pct_change().fillna(0)
    if daily_history_df['Daily_Return'].std() > 0:
        sharpe = (daily_history_df['Daily_Return'].mean() / daily_history_df['Daily_Return'].std()) * np.sqrt(365)
    else:
        sharpe = 0
    
    final_bankroll = df['Bankroll'].iloc[-1]
    
    return {
        'bets': len(df),
        'roi': roi,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'final_bankroll': final_bankroll
    }


def waterfall_report(stage_name, metrics, report_lines):
    """Generate waterfall report entry."""
    line = f"  {stage_name:.<45} Bets: {metrics['bets']:,d}  |  ROI: {metrics['roi']:.2f}%  |  Sharpe: {metrics['sharpe']:.2f}  |  MaxDD: {metrics['max_dd']:.2f}%"
    print(line)
    report_lines.append(line)
    return metrics['bets']


# ==============================================================================
# MAIN FILTER STACK
# ==============================================================================

def main():
    print("=" * 80)
    print("FILTER STACK V8.0 - Phase 2 Implementation (OPTIMIZED)")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    report_lines = []
    report_lines.append("FILTER STACK V8.0 - WATERFALL ANALYSIS (OPTIMIZED)")
    report_lines.append("=" * 80)
    report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # =========================================================================
    # STEP 1: LOAD RAW DATA
    # =========================================================================
    print("STEP 1: LOADING RAW DATA")
    print("-" * 40)
    
    df = pd.read_csv(INPUT_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    baseline_metrics = calculate_metrics(df)
    print(f"  Input file: {INPUT_FILE}")
    report_lines.append(f"Input file: {INPUT_FILE}")
    report_lines.append("")
    
    print("\n[WATERFALL ANALYSIS]")
    report_lines.append("[WATERFALL ANALYSIS]")
    remaining_bets = waterfall_report("BASELINE (Raw Data)", baseline_metrics, report_lines)
    
    # =========================================================================
    # STEP 2: DATA TRANSFORMATIONS (Before filtering!)
    # =========================================================================
    print("\n" + "-" * 40)
    print("STEP 2: DATA TRANSFORMATIONS")
    print("-" * 40)
    report_lines.append("")
    report_lines.append("STEP 2: DATA TRANSFORMATIONS")
    report_lines.append("-" * 40)
    
    # Create adjusted columns
    df['Model_Prob_Original'] = df['Model_Prob'].copy()
    df['Edge_Original'] = df['Edge'].copy()
    df['Model_Prob_Adjusted'] = df['Model_Prob'].copy()
    
    # -------------------------------------------------------------------------
    # STEP 2a: H2H Perfect Cap
    # -------------------------------------------------------------------------
    print("\n  2a. H2H Perfect Record Cap")
    
    h2h_perfect_mask = (df['H2H_P1_Win_Rate'] == 1.0) & (df['H2H_Dominance_Score'].abs() >= 2)
    h2h_capped_count = ((df['Model_Prob_Adjusted'] > H2H_PERFECT_CAP) & h2h_perfect_mask).sum()
    df.loc[h2h_perfect_mask & (df['Model_Prob_Adjusted'] > H2H_PERFECT_CAP), 'Model_Prob_Adjusted'] = H2H_PERFECT_CAP
    
    print(f"      - H2H perfect records identified: {h2h_perfect_mask.sum()}")
    print(f"      - Bets with Model_Prob capped to {H2H_PERFECT_CAP}: {h2h_capped_count}")
    report_lines.append(f"  2a. H2H Perfect Cap: {h2h_capped_count} bets capped to {H2H_PERFECT_CAP}")
    
    # -------------------------------------------------------------------------
    # STEP 2b: Dead Zone Cap
    # -------------------------------------------------------------------------
    print("\n  2b. Dead Zone Cap (Model_Prob > 0.75)")
    
    dead_zone_mask = df['Model_Prob_Adjusted'] > DEAD_ZONE_CAP
    dead_zone_count = dead_zone_mask.sum()
    df.loc[dead_zone_mask, 'Model_Prob_Adjusted'] = DEAD_ZONE_CAP
    
    print(f"      - Bets in dead zone (before): {dead_zone_count}")
    print(f"      - All capped to: {DEAD_ZONE_CAP}")
    report_lines.append(f"  2b. Dead Zone Cap: {dead_zone_count} bets capped to {DEAD_ZONE_CAP}")
    
    # -------------------------------------------------------------------------
    # STEP 2c: RECALCULATE EDGE
    # -------------------------------------------------------------------------
    print("\n  2c. Recalculating Edge with Adjusted Model_Prob")
    
    df['Implied_Prob'] = 1 / df['Market_Odds']
    df['Edge_Adjusted'] = df['Model_Prob_Adjusted'] - df['Implied_Prob']
    df['Divergence'] = df['Model_Prob_Adjusted'] - df['Implied_Prob']
    
    edge_decreased = (df['Edge_Adjusted'] < df['Edge_Original']).sum()
    print(f"      - Edge decreased for {edge_decreased} bets (due to probability caps)")
    report_lines.append(f"  2c. Edge recalculated: {edge_decreased} bets affected")
    
    post_transform_metrics = calculate_metrics(df)
    remaining_bets = waterfall_report("After Transformations", post_transform_metrics, report_lines)
    
    # =========================================================================
    # STEP 3: BOOLEAN MASK FILTERS (OPTIMIZED)
    # =========================================================================
    print("\n" + "-" * 40)
    print("STEP 3: BOOLEAN MASK FILTERS (OPTIMIZED)")
    print("-" * 40)
    report_lines.append("")
    report_lines.append("STEP 3: BOOLEAN MASK FILTERS (OPTIMIZED)")
    report_lines.append("-" * 40)
    
    # Build combined optimized filter mask
    print(f"\n  Optimized Parameters:")
    print(f"      - Edge_Adjusted: {EDGE_MIN:.0%} to {EDGE_MAX:.0%}")
    print(f"      - H2H Dominance >= {DOM_MIN}")
    print(f"      - Model_Prob_Adjusted: {MODEL_PROB_MIN:.0%} to {MODEL_PROB_MAX:.0%}")
    print(f"      - Divergence: {DIVERGENCE_MIN:.0%} to {DIVERGENCE_MAX:.0%}")
    print(f"      - Odds: {ODDS_MIN} to {ODDS_MAX}")
    
    filter_mask = (
        (df['Edge_Adjusted'] >= EDGE_MIN) & 
        (df['Edge_Adjusted'] <= EDGE_MAX) &
        (df['H2H_Dominance_Score'].abs() >= DOM_MIN) &
        (df['Model_Prob_Adjusted'] >= MODEL_PROB_MIN) &
        (df['Model_Prob_Adjusted'] <= MODEL_PROB_MAX) &
        (df['Divergence'] >= DIVERGENCE_MIN) &
        (df['Divergence'] <= DIVERGENCE_MAX) &
        (df['Market_Odds'] >= ODDS_MIN) & 
        (df['Market_Odds'] <= ODDS_MAX)
    )
    
    # Show individual filter impacts
    edge_pass = ((df['Edge_Adjusted'] >= EDGE_MIN) & (df['Edge_Adjusted'] <= EDGE_MAX)).sum()
    dom_pass = (df['H2H_Dominance_Score'].abs() >= DOM_MIN).sum()
    prob_pass = ((df['Model_Prob_Adjusted'] >= MODEL_PROB_MIN) & (df['Model_Prob_Adjusted'] <= MODEL_PROB_MAX)).sum()
    div_pass = ((df['Divergence'] >= DIVERGENCE_MIN) & (df['Divergence'] <= DIVERGENCE_MAX)).sum()
    odds_pass = ((df['Market_Odds'] >= ODDS_MIN) & (df['Market_Odds'] <= ODDS_MAX)).sum()
    
    print(f"\n  Individual Filter Pass Counts:")
    print(f"      - Edge filter: {edge_pass} pass")
    print(f"      - Dominance filter: {dom_pass} pass")
    print(f"      - Probability filter: {prob_pass} pass")
    print(f"      - Divergence filter: {div_pass} pass")
    print(f"      - Odds filter: {odds_pass} pass")
    
    df_filtered = df[filter_mask].copy()
    filtered_metrics = calculate_metrics(df_filtered)
    
    remaining_bets = waterfall_report("After COMBINED FILTERS", filtered_metrics, report_lines)
    
    # =========================================================================
    # STEP 4: FINAL OUTPUT
    # =========================================================================
    print("\n" + "-" * 40)
    print("STEP 4: FINAL OUTPUT")
    print("-" * 40)
    
    df_final = df_filtered.copy()
    final_metrics = filtered_metrics
    
    # =========================================================================
    # STEP 5: COMPARISON REPORT
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL COMPARISON: BASELINE vs FILTERED")
    print("=" * 80)
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("FINAL COMPARISON: BASELINE vs FILTERED")
    report_lines.append("=" * 80)
    
    comparison = f"""
    Metric          |  BASELINE  |  FILTERED  |  DELTA     |  CHANGE
    ----------------+------------+------------+------------+---------
    Bets            |  {baseline_metrics['bets']:,d}      |  {final_metrics['bets']:,d}      |  {final_metrics['bets'] - baseline_metrics['bets']:+,d}      |  {(final_metrics['bets'] - baseline_metrics['bets'])/baseline_metrics['bets']*100:+.1f}%
    ROI             |  {baseline_metrics['roi']:.2f}%     |  {final_metrics['roi']:.2f}%     |  {final_metrics['roi'] - baseline_metrics['roi']:+.2f}%    |  {(final_metrics['roi'] - baseline_metrics['roi'])/max(abs(baseline_metrics['roi']), 0.01)*100:+.1f}%
    Sharpe Ratio    |  {baseline_metrics['sharpe']:.2f}       |  {final_metrics['sharpe']:.2f}       |  {final_metrics['sharpe'] - baseline_metrics['sharpe']:+.2f}      |  {(final_metrics['sharpe'] - baseline_metrics['sharpe'])/baseline_metrics['sharpe']*100:+.1f}%
    Max Drawdown    |  {baseline_metrics['max_dd']:.2f}%    |  {final_metrics['max_dd']:.2f}%    |  {final_metrics['max_dd'] - baseline_metrics['max_dd']:+.2f}%   |  {(final_metrics['max_dd'] - baseline_metrics['max_dd'])/baseline_metrics['max_dd']*100:+.1f}%
    Final Bankroll  |  ${baseline_metrics['final_bankroll']:.2f}   |  ${final_metrics['final_bankroll']:.2f}   |  ${final_metrics['final_bankroll'] - baseline_metrics['final_bankroll']:+.2f}    |  {(final_metrics['final_bankroll'] - baseline_metrics['final_bankroll'])/baseline_metrics['final_bankroll']*100:+.1f}%
    """
    print(comparison)
    report_lines.append(comparison)
    
    # Validation checks
    print("\n[VALIDATION CHECKS]")
    report_lines.append("\n[VALIDATION CHECKS]")
    
    passed_all = True
    
    # Check 1: Bets >= 1,000
    if final_metrics['bets'] >= 1000:
        print(f"  [PASS] Bets ({final_metrics['bets']:,d}) >= 1,000")
        report_lines.append(f"  [PASS] Bets >= 1,000")
    else:
        print(f"  [FAIL] Bets ({final_metrics['bets']:,d}) < 1,000")
        report_lines.append(f"  [FAIL] Bets < 1,000")
        passed_all = False
    
    # Check 2: Sharpe improved or maintained
    if final_metrics['sharpe'] >= baseline_metrics['sharpe']:
        print(f"  [PASS] Sharpe ({final_metrics['sharpe']:.2f}) >= Baseline ({baseline_metrics['sharpe']:.2f})")
        report_lines.append(f"  [PASS] Sharpe maintained/improved")
    else:
        print(f"  [WARN] Sharpe ({final_metrics['sharpe']:.2f}) < Baseline ({baseline_metrics['sharpe']:.2f})")
        report_lines.append(f"  [WARN] Sharpe regressed")
    
    # Check 3: Max Drawdown <= 35%
    if final_metrics['max_dd'] <= 35:
        print(f"  [PASS] Max Drawdown ({final_metrics['max_dd']:.2f}%) <= 35%")
        report_lines.append(f"  [PASS] Max Drawdown within limits")
    else:
        print(f"  [WARN] Max Drawdown ({final_metrics['max_dd']:.2f}%) > 35%")
        report_lines.append(f"  [WARN] Max Drawdown exceeded 35%")
    
    # Check 4: ROI > 2%
    if final_metrics['roi'] >= 2.0:
        print(f"  [PASS] ROI ({final_metrics['roi']:.2f}%) >= 2%")
        report_lines.append(f"  [PASS] ROI meets target")
    else:
        print(f"  [WARN] ROI ({final_metrics['roi']:.2f}%) < 2%")
        report_lines.append(f"  [WARN] ROI below target")
    
    # Check 5: No circuit breakers (verify loss streaks exist)
    df_final['is_loss'] = df_final['Outcome'] == 'Loss'
    df_final['loss_streak'] = df_final['is_loss'].groupby((~df_final['is_loss']).cumsum()).cumsum()
    max_loss_streak = df_final['loss_streak'].max()
    print(f"  [INFO] Max loss streak in filtered data: {max_loss_streak} (no circuit breakers)")
    report_lines.append(f"  [INFO] Max loss streak: {max_loss_streak} (circuit breakers NOT applied)")
    
    # Save outputs
    print(f"\n  Saving filtered data to: {OUTPUT_FILE}")
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"  [DONE] Saved {len(df_final):,d} bets")
    report_lines.append(f"\nOutput file: {OUTPUT_FILE}")
    report_lines.append(f"Total bets saved: {len(df_final):,d}")
    
    with open(REPORT_FILE, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\n  Report saved to: {REPORT_FILE}")
    
    print("\n" + "=" * 80)
    if passed_all:
        print("FILTER STACK V8.0 COMPLETE - ALL CHECKS PASSED")
    else:
        print("FILTER STACK V8.0 COMPLETE - SOME CHECKS FAILED")
    print("=" * 80)
    
    return df_final, final_metrics


if __name__ == "__main__":
    df_result, metrics = main()
