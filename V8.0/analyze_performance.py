import pandas as pd
import numpy as np

# --- 1. Configuration ---
ANALYSIS_LOG_FILE = "backtest_log_for_analysis_v7.4.csv"

# --- 2. Main Analysis Logic ---
try:
    print(f"--- Loading Bet Log from '{ANALYSIS_LOG_FILE}' for Analysis ---")
    df = pd.read_csv(ANALYSIS_LOG_FILE)
    
    if df.empty:
        print("The bet log is empty. No analysis to perform.")
        exit()

    print(f"Successfully loaded {len(df)} bets to analyze.")
    
    # --- MODIFIED: Added all new features to the robustness check ---
    required_columns = [
        'Match_ID', 'Profit', 'Stake', 'Market_Odds', 'Edge', 'H2H_P1_Win_Rate',
        'Win_Rate_Advantage', 'Time_Since_Last_Advantage', 'Matches_Last_24H_Advantage',
        'Is_First_Match_Advantage', 'H2H_Dominance_Score', 'Win_Rate_L5_Advantage',
        'PDR_Advantage', 'PDR_Slope_Advantage', 'Daily_Fatigue_Advantage', 
        'Close_Set_Win_Rate_Advantage'
    ]
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"The log file is missing required columns: {missing}")

    # --- 3. Binning and Grouping ---
    
    # --- Helper function for perspective ---
    def get_bet_perspective(row, col_name):
        # If we bet on Player 1, the advantage is as-is.
        # If we bet on Player 2, the advantage is inverted.
        if row['Bet_On_Player'] == row['Player_1']:
            return row[col_name]
        else:
            return -row[col_name]

    # --- Standard Odds and Edge Categories ---
    df['Odds_Category'] = pd.cut(df['Market_Odds'], bins=[0, 1.5, 2.0, 3.0, 100], labels=['Heavy Favorite (<1.5)', 'Favorite (1.5-2.0)', 'Underdog (2.0-3.0)', 'Longshot (>3.0)'])
    df['Edge_Category'] = pd.cut(df['Edge'], bins=[0, 0.10, 0.25, 0.50, 2.0], labels=['Low Edge (0-10%)', 'Medium Edge (10-25%)', 'High Edge (25-50%)', 'Mega Edge (50%+)'])

    # --- Form Categories (L20 and L5) ---
    df['Bet_Form_Advantage'] = df.apply(get_bet_perspective, axis=1, col_name='Win_Rate_Advantage')
    df['Form_Category'] = pd.cut(df['Bet_Form_Advantage'], bins=[-1, -0.01, 0.01, 1], labels=['Worse Form (L20)', 'Even Form (L20)', 'Better Form (L20)'])
    df['Bet_Form_L5_Advantage'] = df.apply(get_bet_perspective, axis=1, col_name='Win_Rate_L5_Advantage')
    df['Form_L5_Category'] = pd.cut(df['Bet_Form_L5_Advantage'], bins=[-1.1, -0.1, 0.1, 1.1], labels=['Colder Player (L5)', 'Even Form (L5)', 'Hotter Player (L5)'])

    # --- PDR Categories ---
    df['Bet_PDR_Advantage'] = df.apply(get_bet_perspective, axis=1, col_name='PDR_Advantage')
    df['PDR_Category'] = pd.cut(df['Bet_PDR_Advantage'], bins=[-1, -0.01, 0.01, 1], labels=['Lower PDR', 'Even PDR', 'Higher PDR'])
    df['Bet_PDR_Slope_Advantage'] = df.apply(get_bet_perspective, axis=1, col_name='PDR_Slope_Advantage')
    df['PDR_Slope_Category'] = pd.cut(df['Bet_PDR_Slope_Advantage'], bins=[-1, -0.005, 0.005, 1], labels=['Declining PDR Form', 'Stable PDR Form', 'Improving PDR Form'])

    # --- Pressure / Close Set Categories ---
    df['Bet_Close_Set_Advantage'] = df.apply(get_bet_perspective, axis=1, col_name='Close_Set_Win_Rate_Advantage')
    df['Close_Set_Category'] = pd.cut(df['Bet_Close_Set_Advantage'], bins=[-1.1, -0.1, 0.1, 1.1], labels=['Worse in Close Sets', 'Even in Close Sets', 'Better in Close Sets'])

    # --- H2H Categories ---
    df['Bet_H2H_Dominance'] = df.apply(get_bet_perspective, axis=1, col_name='H2H_Dominance_Score')
    df['H2H_Dominance_Category'] = pd.cut(df['Bet_H2H_Dominance'], bins=[-1000, -10, 10, 1000], labels=['H2H Dominated', 'H2H Even', 'H2H Dominant'])
    
    # --- Fatigue and Recency Categories ---
    df['Bet_Fatigue_Advantage'] = df.apply(get_bet_perspective, axis=1, col_name='Daily_Fatigue_Advantage')
    df['Fatigue_Category'] = pd.cut(df['Bet_Fatigue_Advantage'], bins=[-500, -1, 1, 500], labels=['Bet on Fresher Player', 'Even Workload', 'Bet on More Fatigued Player'])
    df['Bet_Rest_Advantage'] = df.apply(get_bet_perspective, axis=1, col_name='Time_Since_Last_Advantage')
    df['Rest_Category'] = pd.cut(df['Bet_Rest_Advantage'], bins=[-500, -12, 12, 500], labels=['Bet on Less Rested', 'Even Rest', 'Bet on More Rested'])
    df['Bet_Recent_Matches_Adv'] = df.apply(get_bet_perspective, axis=1, col_name='Matches_Last_24H_Advantage')
    df['Recent_Matches_Category'] = pd.cut(df['Bet_Recent_Matches_Adv'], bins=[-5, -0.5, 0.5, 5], labels=['Played Fewer Matches', 'Played Same Matches', 'Played More Matches'])
    df['Bet_First_Match_Adv'] = df.apply(get_bet_perspective, axis=1, col_name='Is_First_Match_Advantage')
    df['First_Match_Category'] = df['Bet_First_Match_Adv'].map({-1: 'Opponent\'s First Match', 0: 'Same Status', 1: 'Player\'s First Match'})


    # --- 4. Performance Calculation ---
    def calculate_performance(df, category):
        # Drop rows where the category could not be assigned
        df_filtered = df.dropna(subset=[category])
        if df_filtered.empty:
            return pd.DataFrame() # Return empty if no data for this category
            
        grouped = df_filtered.groupby(category).agg(
            Total_Bets=('Match_ID', 'count'),
            Total_Profit=('Profit', 'sum'),
            Total_Staked=('Stake', 'sum')
        ).reset_index()
        grouped['ROI'] = (grouped['Total_Profit'] / grouped['Total_Staked']) * 100
        return grouped.sort_values(by='ROI', ascending=False)

    all_categories = [
        'Odds_Category', 'Edge_Category', 'Form_Category', 'Form_L5_Category', 'PDR_Category',
        'PDR_Slope_Category', 'Close_Set_Category', 'H2H_Dominance_Category', 'Fatigue_Category',
        'Rest_Category', 'Recent_Matches_Category', 'First_Match_Category'
    ]
    
    performance_reports = {cat: calculate_performance(df, cat) for cat in all_categories}

    # --- 5. Print the Report ---
    print("\n--- Forensic Performance Report v7.4 ---")
    
    report_titles = {
        'Odds_Category': "Performance by Market Odds",
        'Edge_Category': "Performance by Model's Perceived Edge",
        'Form_Category': "Performance by General Form (L20 Win Rate)",
        'Form_L5_Category': "Performance by Recent Form / 'Hot Streak' (L5 Win Rate)",
        'PDR_Category': "Performance by Points Dominance Ratio (PDR)",
        'PDR_Slope_Category': "Performance by PDR Momentum (Slope)",
        'Close_Set_Category': "Performance in Close Sets",
        'H2H_Dominance_Category': "Performance by H2H Dominance Score",
        'Fatigue_Category': "Performance by Daily Fatigue (Points Played Today)",
        'Rest_Category': "Performance by Rest (Hours Since Last Match)",
        'Recent_Matches_Category': "Performance by Matches Played in Last 24H",
        'First_Match_Category': "Performance by First Match of the Day Status"
    }

    for cat, report_df in performance_reports.items():
        print(f"\n>>> {report_titles.get(cat, cat)} <<<")
        if report_df.empty:
            print("No bets found for this category.")
        else:
            print(report_df.to_string(formatters={'ROI': '{:,.2f}%'.format, 'Total_Profit': '${:,.2f}'.format, 'Total_Staked': '${:,.2f}'.format}))
        print("-" * 50)

except Exception as e:
    print(f"\nAn error occurred: {e}")
    import traceback
    traceback.print_exc()