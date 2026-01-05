import pandas as pd
import numpy as np

# --- 1. Configuration ---
ANALYSIS_LOG_FILE = "backtest_log_for_analysis_v7.3.csv"

# --- 2. Main Analysis Logic ---
try:
    print(f"--- Loading Bet Log from '{ANALYSIS_LOG_FILE}' for Analysis ---")
    df = pd.read_csv(ANALYSIS_LOG_FILE)
    
    if df.empty:
        print("The bet log is empty. No analysis to perform.")
        exit()

    print(f"Successfully loaded {len(df)} bets to analyze.")
    
    # --- NEW: Updated robustness check with correct column names ---
    required_columns = ['Match_ID', 'Profit', 'Stake', 'Market_Odds', 'Edge', 
                        'H2H_P1_Win_Rate', 'Win_Rate_Advantage', 'P1_History', 'P2_History']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"The log file is missing required columns: {missing}")

    # --- 3. Binning and Grouping ---
    # Create categories (bins) for different odds levels
    odds_bins = [0, 1.5, 2.0, 3.0, 100]
    odds_labels = ['Heavy Favorite (<1.5)', 'Favorite (1.5-2.0)', 'Underdog (2.0-3.0)', 'Longshot (>3.0)']
    df['Odds_Category'] = pd.cut(df['Market_Odds'], bins=odds_bins, labels=odds_labels)

    # Create categories for the model's perceived edge
    edge_bins = [0, 0.10, 0.25, 0.50, 2.0] # 0-10%, 10-25%, 25-50%, 50%+
    edge_labels = ['Low Edge (0-10%)', 'Medium Edge (10-25%)', 'High Edge (25-50%)', 'Mega Edge (50%+)']
    df['Edge_Category'] = pd.cut(df['Edge'], bins=edge_bins, labels=edge_labels)

    # --- NEW: Updated player history logic ---
    # Determine the minimum number of matches for either player
    df['Min_History'] = df[['P1_History', 'P2_History']].min(axis=1)
    history_bins = [-1, 9, 24, 49, 1000] # <10, 10-25, 25-50, 50+
    history_labels = ['Very New Player (<10)', 'New Player (10-25)', 'Established (25-50)', 'Veteran (50+)']
    df['History_Category'] = pd.cut(df['Min_History'], bins=history_bins, labels=history_labels)

    # --- NEW: Updated H2H advantage logic ---
    # Define H2H based on the player who was bet on
    def h2h_perspective(row):
        if row['Bet_On_Player'] == row['Player_1']:
            return row['H2H_P1_Win_Rate']
        else:
            return 1 - row['H2H_P1_Win_Rate']
    df['Bet_H2H_Win_Rate'] = df.apply(h2h_perspective, axis=1)
    h2h_bins = [-0.1, 0.4, 0.6, 1.1] # <40% (Disadvantage), 40-60% (Even), >60% (Advantage)
    h2h_labels = ['H2H Disadvantage', 'Even H2H', 'H2H Advantage']
    df['H2H_Category'] = pd.cut(df['Bet_H2H_Win_Rate'], bins=h2h_bins, labels=h2h_labels)

    # --- NEW: Updated Form advantage logic ---
    def form_perspective(row):
        if row['Bet_On_Player'] == row['Player_1']:
            return row['Win_Rate_Advantage']
        else:
            return -row['Win_Rate_Advantage'] # Invert for P2
    df['Bet_Form_Advantage'] = df.apply(form_perspective, axis=1)
    form_bins = [-1, -0.01, 0.01, 1] # Negative, Even, Positive
    form_labels = ['Worse Form', 'Even Form', 'Better Form']
    df['Form_Category'] = pd.cut(df['Bet_Form_Advantage'], bins=form_bins, labels=form_labels)

    # --- 4. Performance Calculation ---
    def calculate_performance(df, category):
        grouped = df.groupby(category).agg(
            Total_Bets=('Match_ID', 'count'),
            Total_Profit=('Profit', 'sum'),
            Total_Staked=('Stake', 'sum')
        ).reset_index()
        grouped['ROI'] = (grouped['Total_Profit'] / grouped['Total_Staked']) * 100
        return grouped

    odds_performance = calculate_performance(df, 'Odds_Category')
    edge_performance = calculate_performance(df, 'Edge_Category')
    history_performance = calculate_performance(df, 'History_Category')
    h2h_performance = calculate_performance(df, 'H2H_Category')
    form_performance = calculate_performance(df, 'Form_Category')

    # --- 5. Print the Report ---
    print("\n--- Forensic Performance Report v7.3 (Corrected Log) ---")
    
    print("\n>>> Performance by Market Odds <<<")
    print(odds_performance.to_string(formatters={'ROI': '{:,.2f}%'.format, 'Total_Profit': '${:,.2f}'.format, 'Total_Staked': '${:,.2f}'.format}))
    
    print("\n>>> Performance by Model's Perceived Edge <<<")
    print(edge_performance.to_string(formatters={'ROI': '{:,.2f}%'.format, 'Total_Profit': '${:,.2f}'.format, 'Total_Staked': '${:,.2f}'.format}))

    print("\n>>> Performance by Player History <<<")
    print("(Based on the player with fewer historical matches)")
    print(history_performance.to_string(formatters={'ROI': '{:,.2f}%'.format, 'Total_Profit': '${:,.2f}'.format, 'Total_Staked': '${:,.2f}'.format}))
    
    print("\n>>> Performance by H2H Advantage <<<")
    print("(From the perspective of the player bet on)")
    print(h2h_performance.to_string(formatters={'ROI': '{:,.2f}%'.format, 'Total_Profit': '${:,.2f}'.format, 'Total_Staked': '${:,.2f}'.format}))

    print("\n>>> Performance by Recent Form (Rolling Win Rate) Advantage <<<")
    print("(From the perspective of the player bet on)")
    print(form_performance.to_string(formatters={'ROI': '{:,.2f}%'.format, 'Total_Profit': '${:,.2f}'.format, 'Total_Staked': '${:,.2f}'.format}))
    
    print("\n------------------------------------")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
