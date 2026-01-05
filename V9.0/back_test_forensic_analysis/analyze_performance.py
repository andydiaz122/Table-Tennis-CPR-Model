import pandas as pd
import numpy as np

# --- 1. Configuration ---
ANALYSIS_LOG_FILE = "backtest_analysis_log.csv"

# --- 2. Main Analysis Logic ---
try:
    print(f"--- Loading Bet Log from '{ANALYSIS_LOG_FILE}' for Analysis ---")
    df = pd.read_csv(ANALYSIS_LOG_FILE)
    
    if df.empty:
        print("The bet log is empty. No analysis to perform.")
        exit()

    print(f"Successfully loaded {len(df)} bets to analyze.")

    # --- 3. Binning and Grouping ---
    # Create categories (bins) for different odds levels
    odds_bins = [0, 1.5, 2.0, 3.0, 100]
    odds_labels = ['Heavy Favorite (<1.5)', 'Favorite (1.5-2.0)', 'Underdog (2.0-3.0)', 'Longshot (>3.0)']
    df['Odds_Category'] = pd.cut(df['Market_Odds'], bins=odds_bins, labels=odds_labels)

    # Create categories for the model's perceived edge
    edge_bins = [0.02, 0.04, 0.06, 0.10, 0.99]
    edge_labels = ['Low Edge (2-4%)', 'Medium Edge (4-6%)', 'High Edge (6-10%)', 'Mega Edge (10–99%)']
    df['Edge_Category'] = pd.cut(df['Edge'], bins=edge_bins, labels=edge_labels)

    # --- 4. Performance Aggregation ---
    def analyze_group(df, group_by_col):
        """Helper function to calculate performance for a given category."""
        grouped = df.groupby(group_by_col).agg(
            Total_Bets=('Match_ID', 'count'),
            Total_Profit=('Profit', 'sum'),
            Total_Staked=('Stake', 'sum')
        )
        grouped['ROI'] = (grouped['Total_Profit'] / grouped['Total_Staked']) * 100
        return grouped

    # Analyze performance by each category
    odds_performance = analyze_group(df, 'Odds_Category')
    edge_performance = analyze_group(df, 'Edge_Category')
    
    # Analyze performance by key features
    # Positive advantage means the model favored P1 based on this feature
    h2h_performance = analyze_group(df[df['H2H_Advantage'] != 0.5], pd.cut(df['H2H_Advantage'], bins=3))
    win_rate_performance = analyze_group(df, pd.cut(df['Win_Rate_Advantage'], bins=3))


    # --- 5. Print the Report ---
    print("\n--- Forensic Performance Report ---")
    
    print("\n>>> Performance by Market Odds <<<")
    print(odds_performance.to_string(formatters={'ROI': '{:,.2f}%'.format}))
    
    print("\n>>> Performance by Model's Perceived Edge <<<")
    print(edge_performance.to_string(formatters={'ROI': '{:,.2f}%'.format}))
    
    print("\n>>> Performance by H2H Advantage <<<")
    print("(A positive value means Player 1 had the historical H2H edge)")
    print(h2h_performance.to_string(formatters={'ROI': '{:,.2f}%'.format}))

    print("\n>>> Performance by Recent Form (Rolling Win Rate) Advantage <<<")
    print("(A positive value means Player 1 was in better recent form)")
    print(win_rate_performance.to_string(formatters={'ROI': '{:,.2f}%'.format}))
    print("\n------------------------------------")


except FileNotFoundError:
    print(f"Error: The input file '{ANALYSIS_LOG_FILE}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")