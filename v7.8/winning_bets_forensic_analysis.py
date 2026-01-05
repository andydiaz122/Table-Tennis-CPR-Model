import pandas as pd
import numpy as np

# --- 1. Configuration ---
LOG_FILE = "backtest_log_for_analysis_v7.4.csv"
INITIAL_BANKROLL = 1000
# Define the start of the profitable period from the equity curve
UPSWING_START_BET_INDEX = 2000
DOWNSWING_END_BET_INDEX = 1500 


# --- 2. Load and Prepare Data ---
try:
    log_df = pd.read_csv(LOG_FILE)
    log_df['Bankroll'] = INITIAL_BANKROLL + log_df['Profit'].cumsum()
    print(f"Loaded {len(log_df)} total bets for analysis.")

    # --- 3. Isolate the Upswing Period ---
    # Filter the DataFrame to the upswing period
    upswing_df = log_df.iloc[UPSWING_START_BET_INDEX:].copy()

    start_value = log_df.loc[UPSWING_START_BET_INDEX]['Bankroll']
    end_value = log_df.iloc[-1]['Bankroll']

    print(f"\nAnalyzing profitable upswing from bet #{UPSWING_START_BET_INDEX} to #{len(log_df)-1}")
    print(f"Bankroll at start of period: ${start_value:,.2f}")
    print(f"Bankroll at end of period:   ${end_value:,.2f}")
    
    # --- 4. Forensic Analysis of the Upswing ---
    print(f"\n--- Forensic Analysis of All {len(upswing_df)} Bets During Upswing ---")

    # Helper function to create categorized performance reports
    def analyze_by_category(df, column_to_bin, bins, labels):
        """Groups data and calculates key performance metrics."""
        df['Category'] = pd.cut(df[column_to_bin], bins=bins, labels=labels, right=False)
        analysis = df.groupby('Category', observed=False).agg(
            Total_Bets=('Outcome', 'count'),
            Total_Profit=('Profit', 'sum'),
            Total_Staked=('Stake', 'sum')
        )
        # Avoid division by zero if a category has no stakes
        analysis['ROI'] = np.where(analysis['Total_Staked'] > 0, (analysis['Total_Profit'] / analysis['Total_Staked']) * 100, 0)
        return analysis.round(2)

    # Analysis 1: Performance by Market Odds
    print("\n>>> Performance by Odds Categories during Upswing:")
    odds_bins = [1, 1.5, 2.0, 3.0, 10.0]
    odds_labels = ['Heavy Favorite (<1.5)', 'Favorite (1.5-2.0)', 'Underdog (2.0-3.0)', 'Longshot (>3.0)']
    odds_analysis = analyze_by_category(upswing_df.copy(), 'Market_Odds', odds_bins, odds_labels)
    print(odds_analysis.to_string())

    # Analysis 2: Performance by Model's Perceived Edge
    print("\n>>> Performance by Model Edge Categories during Upswing:")
    edge_bins = [0, 0.10, 0.25, 0.50, 1.0]
    edge_labels = ['Low Edge (0-10%)', 'Medium Edge (10-25%)', 'High Edge (25-50%)', 'Mega Edge (50%+)']
    edge_analysis = analyze_by_category(upswing_df.copy(), 'Edge', edge_bins, edge_labels)
    print(edge_analysis.to_string())

    # Analysis 3: Performance by Recent Form (Win Rate) Advantage
    print("\n>>> Performance by Form Advantage during Upswing:")
    upswing_df['Form_Category'] = 'Even Form'
    upswing_df.loc[upswing_df['Win_Rate_Advantage'] > 0, 'Form_Category'] = 'Better Form'
    upswing_df.loc[upswing_df['Win_Rate_Advantage'] < 0, 'Form_Category'] = 'Worse Form'
    
    form_analysis = upswing_df.groupby('Form_Category').agg(
        Total_Bets=('Outcome', 'count'),
        Total_Profit=('Profit', 'sum'),
        Total_Staked=('Stake', 'sum')
    )
    form_analysis['ROI'] = np.where(form_analysis['Total_Staked'] > 0, (form_analysis['Total_Profit'] / form_analysis['Total_Staked']) * 100, 0)
    print(form_analysis.round(2).to_string())

    # Analysis 4: Performance by H2H Advantage
    print("\n>>> Performance by H2H Advantage during Upswing:")
    # Define conditions for H2H categories
    conditions = [
        upswing_df['H2H_P1_Win_Rate'] > 0.5,
        upswing_df['H2H_P1_Win_Rate'] == 0.5,
        upswing_df['H2H_P1_Win_Rate'] < 0.5
    ]
    choices = ['H2H Advantage', 'Even H2H', 'H2H Disadvantage']
    upswing_df['H2H_Category'] = np.select(conditions, choices, default='N/A')

    h2h_analysis = upswing_df.groupby('H2H_Category').agg(
        Total_Bets=('Outcome', 'count'),
        Total_Profit=('Profit', 'sum'),
        Total_Staked=('Stake', 'sum')
    )
    h2h_analysis['ROI'] = np.where(h2h_analysis['Total_Staked'] > 0, (h2h_analysis['Total_Profit'] / h2h_analysis['Total_Staked']) * 100, 0)
    # Reorder the index to be consistent with previous reports
    h2h_analysis = h2h_analysis.reindex(['H2H Disadvantage', 'Even H2H', 'H2H Advantage'])
    print(h2h_analysis.round(2).to_string())


except FileNotFoundError:
    print(f"Error: The log file '{LOG_FILE}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
