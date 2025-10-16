import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# --- 1. Configuration ---
LOG_FILE = "backtest_log_for_analysis_v7.4.csv"

# --- 2. Main Analysis Script ---
try:
    print(f"--- Loading Bet Log from '{LOG_FILE}' for Feature Analysis ---")
    df = pd.read_csv(LOG_FILE)
    
    if df.empty:
        raise ValueError("The bet log is empty. No analysis to perform.")
    
    print(f"Successfully loaded {len(df)} bet records.")

    # --- Data Integrity Check ---
    required_cols = ['H2H_Dominance_Score', 'H2H_P1_Win_Rate', 'Outcome', 'Bet_On_Player', 'Player_1', 'Player_2']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Log file is missing required columns for analysis: {missing}")

    # --- Reconstruct the Target Variable (P1_Win) ---
    # We need the ground truth outcome from Player 1's perspective for correlation.
    p1_win_condition = ((df['Bet_On_Player'] == df['Player_1']) & (df['Outcome'] == 'Win')) | \
                       ((df['Bet_On_Player'] == df['Player_2']) & (df['Outcome'] == 'Loss'))
    df['P1_Win'] = np.where(p1_win_condition, 1, 0)
    
    print("\n--- Step 1: Analyzing the Distribution of H2H_Dominance_Score ---")
    
    # Generate descriptive statistics
    print("\nDescriptive Statistics:")
    print(df['H2H_Dominance_Score'].describe())

    # Plot the distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['H2H_Dominance_Score'], kde=True, bins=50)
    plt.title('Distribution of H2H_Dominance_Score')
    plt.xlabel('H2H Dominance Score (Player 1 Perspective)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    distribution_plot_file = 'h2h_dominance_distribution.png'
    plt.savefig(distribution_plot_file)
    print(f"\n✅ Distribution plot saved to '{distribution_plot_file}'")
    
    print("\n--- Step 2: Checking Correlation with the Target Variable (P1_Win) ---")

    # --- Correlation for the NEW Feature ---
    # Calculate Pearson correlation between the new feature and the outcome
    dominance_corr, dominance_p_value = pearsonr(df['H2H_Dominance_Score'].fillna(0), df['P1_Win'])
    
    print(f"\nCorrelation (H2H_Dominance_Score vs. P1_Win):")
    print(f"  - Pearson Correlation Coefficient: {dominance_corr:.4f}")
    print(f"  - P-value: {dominance_p_value:.4f}")
    if dominance_p_value < 0.05:
        print("  - Interpretation: The correlation is statistically significant.")
    else:
        print("  - Interpretation: The correlation is NOT statistically significant.")

    # --- Correlation for the OLD Feature (for comparison) ---
    # Calculate Pearson correlation for the old H2H feature to establish a benchmark
    win_rate_corr, win_rate_p_value = pearsonr(df['H2H_P1_Win_Rate'].fillna(0.5), df['P1_Win'])

    print(f"\nCorrelation (H2H_P1_Win_Rate vs. P1_Win):")
    print(f"  - Pearson Correlation Coefficient: {win_rate_corr:.4f}")
    print(f"  - P-value: {win_rate_p_value:.4f}")
    if win_rate_p_value < 0.05:
        print("  - Interpretation: The correlation is statistically significant.")
    else:
        print("  - Interpretation: The correlation is NOT statistically significant.")

    # --- Final Verdict ---
    print("\n--- Raw Predictive Power Verdict ---")
    if abs(dominance_corr) > abs(win_rate_corr):
        print(f"Conclusion: The new H2H_Dominance_Score ({abs(dominance_corr):.4f}) shows stronger raw predictive power than the old H2H_P1_Win_Rate ({abs(win_rate_corr):.4f}).")
    else:
        print(f"Conclusion: The new H2H_Dominance_Score ({abs(dominance_corr):.4f}) does NOT show stronger raw predictive power than the old H2H_P1_Win_Rate ({abs(win_rate_corr):.4f}).")


except FileNotFoundError:
    print(f"Error: The log file '{LOG_FILE}' was not found. Please ensure the backtest has been run.")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

