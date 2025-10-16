import pandas as pd
import numpy as np

# --- 1. Configuration ---
ANALYSIS_LOG_FILE = "backtest_log_final_filtered.csv"

# --- DEFINE REGIMES (Based on the equity curve) ---
ALPHA_REGIME_START = 0
ALPHA_REGIME_END = 1250
RISK_REGIME_START = 1251
RISK_REGIME_END = 2000

# --- 2. Analysis Functions (Copied from your script) ---
def get_bet_perspective(row, col_name):
    if row['Bet_On_Player'] == row['Player_1']:
        return row[col_name]
    else:
        return -row[col_name]

def analyze_regime(df, regime_name):
    """Performs a full forensic analysis on a given dataframe subset."""
    print("\n" + "="*80)
    print(f"--- Forensic Report for: {regime_name} ---")
    print("="*80)

    if df.empty:
        print("No bets in this regime.")
        return

    # --- Binning and Grouping ---
    df['Odds_Category'] = pd.cut(df['Market_Odds'], bins=[0, 1.5, 2.0, 3.0, 100], labels=['Heavy Favorite (<1.5)', 'Favorite (1.5-2.0)', 'Underdog (2.0-3.0)', 'Longshot (>3.0)'])
    df['Edge_Category'] = pd.cut(df['Edge'], bins=[0, 0.10, 0.25, 0.50, 2.0], labels=['Low Edge (0-10%)', 'Medium Edge (10-25%)', 'High Edge (25-50%)', 'Mega Edge (50%+)'])
    df['Bet_PDR_Advantage'] = df.apply(get_bet_perspective, axis=1, col_name='PDR_Advantage')
    df['PDR_Category'] = pd.cut(df['Bet_PDR_Advantage'], bins=[-1, -0.01, 0.01, 1], labels=['Lower PDR', 'Even PDR', 'Higher PDR'])
    df['Bet_PDR_Slope_Advantage'] = df.apply(get_bet_perspective, axis=1, col_name='PDR_Slope_Advantage')
    df['PDR_Slope_Category'] = pd.cut(df['Bet_PDR_Slope_Advantage'], bins=[-1, -0.005, 0.005, 1], labels=['Declining PDR Form', 'Stable PDR Form', 'Improving PDR Form'])
    # Add any other categories you want to compare...

    def calculate_performance(df, category):
        df_filtered = df.dropna(subset=[category])
        if df_filtered.empty: return pd.DataFrame()
        grouped = df_filtered.groupby(category).agg(
            Total_Bets=('Match_ID', 'count'),
            Total_Profit=('Profit', 'sum'),
            Total_Staked=('Stake', 'sum')
        ).reset_index()
        grouped['ROI'] = (grouped['Total_Profit'] / grouped['Total_Staked']) * 100
        return grouped.sort_values(by='ROI', ascending=False)

    report_titles = {
        'Odds_Category': "Performance by Market Odds",
        'Edge_Category': "Performance by Model's Perceived Edge",
        'PDR_Category': "Performance by Points Dominance Ratio (PDR)",
        'PDR_Slope_Category': "Performance by PDR Momentum (Slope)"
    }

    for cat, title in report_titles.items():
        print(f"\n>>> {title} <<<")
        report_df = calculate_performance(df.copy(), cat)
        if report_df.empty:
            print("No bets found for this category.")
        else:
            print(report_df.to_string(formatters={'ROI': '{:,.2f}%'.format, 'Total_Profit': '${:,.2f}'.format, 'Total_Staked': '${:,.2f}'.format}))
        print("-" * 50)

# --- 3. Main Script ---
try:
    log_df = pd.read_csv(ANALYSIS_LOG_FILE)
    print(f"Loaded {len(log_df)} total filtered bets for regime analysis.")

    # Isolate the data for each regime
    alpha_df = log_df.iloc[ALPHA_REGIME_START:ALPHA_REGIME_END + 1].copy()
    risk_df = log_df.iloc[RISK_REGIME_START:RISK_REGIME_END + 1].copy()

    # Analyze both regimes
    analyze_regime(alpha_df, f"ALPHA REGIME (Bets {ALPHA_REGIME_START}-{ALPHA_REGIME_END})")
    analyze_regime(risk_df, f"RISK REGIME (Bets {RISK_REGIME_START}-{RISK_REGIME_END})")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    import traceback
    traceback.print_exc()
