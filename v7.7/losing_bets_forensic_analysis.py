import pandas as pd

# Load the detailed log of all bets from our most stable backtest
log_df = pd.read_csv("backtest_log_for_analysis_v7.4.csv")

# Calculate the running bankroll after each bet
initial_bankroll = 1000
log_df['Bankroll'] = initial_bankroll + log_df['Profit'].cumsum()

print(f"Loaded {len(log_df)} bets for analysis.")

# Find the peak bankroll value before the largest drop
peak = log_df['Bankroll'].cummax()
drawdown = (peak - log_df['Bankroll']) / peak
max_dd_end_index = drawdown.idxmax() # Index of the trough (worst point)

# Find the peak from which this drawdown started
peak_before_max_dd = peak.iloc[max_dd_end_index]
max_dd_start_index = log_df[log_df['Bankroll'] == peak_before_max_dd].index[0]

# Filter the DataFrame to contain only the bets made during the drawdown
drawdown_df = log_df.iloc[max_dd_start_index:max_dd_end_index + 1]

peak_value = log_df.loc[max_dd_start_index]['Bankroll']
trough_value = log_df.loc[max_dd_end_index]['Bankroll']

print(f"Drawdown started at bet #{max_dd_start_index} (Peak: ${peak_value:,.2f})")
print(f"Drawdown ended at bet #{max_dd_end_index} (Trough: ${trough_value:,.2f})")
print(f"Analyzing {len(drawdown_df)} bets within this period...")

# Filter for only the losing bets within the drawdown period
losing_bets_df = drawdown_df[drawdown_df['Outcome'] == 'Loss']

print(f"\n--- Analysis of {len(losing_bets_df)} Losing Bets During Drawdown ---")

# 1. What were the odds of the losing bets?
print("\n>>> Odds Categories of Losing Bets:")
bins = [1, 1.5, 2.0, 3.0, 10.0]
labels = ['Heavy Favorite (<1.5)', 'Favorite (1.5-2.0)', 'Underdog (2.0-3.0)', 'Longshot (>3.0)']
odds_dist = pd.cut(losing_bets_df['Market_Odds'], bins=bins, labels=labels, right=False).value_counts()
print(odds_dist)

# 2. How confident was the model on these losing bets? (High Edge = High Confidence)
print("\n>>> Model Edge Categories of Losing Bets:")
bins = [0, 0.10, 0.25, 0.50, 1.0]
labels = ['Low Edge (0-10%)', 'Medium Edge (10-25%)', 'High Edge (25-50%)', 'Mega Edge (50%+)']
edge_dist = pd.cut(losing_bets_df['Edge'], bins=bins, labels=labels, right=False).value_counts()
print(edge_dist)

# 3. What was the recent form of players we (incorrectly) bet on?
# (A negative Win_Rate_Advantage means we bet on the player with worse recent form)
print("\n>>> Form Advantage of Losing Bets:")
losing_bets_df['Form_Category'] = 'Worse Form'
losing_bets_df.loc[losing_bets_df['Win_Rate_Advantage'] > 0, 'Form_Category'] = 'Better Form'
form_dist = losing_bets_df['Form_Category'].value_counts()
print(form_dist)