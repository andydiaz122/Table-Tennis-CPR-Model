import pandas as pd
import numpy as np

# --- 1. Configuration ---
BET_HISTORY_FILE = "real_world_bets.csv"

# --- 2. Helper Function to Convert Odds ---
def moneyline_to_decimal(moneyline_odds):
    """
    Converts American Moneyline odds to decimal odds.
    """
    moneyline_odds = float(moneyline_odds)
    if moneyline_odds >= 100:
        return (moneyline_odds / 100) + 1
    elif moneyline_odds < 0:
        return (100 / abs(moneyline_odds)) + 1
    else:
        return np.nan # Return Not a Number for invalid odds

# --- 3. Main Analysis Logic ---
try:
    df = pd.read_csv(BET_HISTORY_FILE)

    # --- NEW: Convert Moneyline to Decimal ---
    # Apply the conversion function to the 'Odds' column.
    df['Decimal_Odds'] = df['Odds'].apply(moneyline_to_decimal)
    
    # Drop any rows where the odds conversion might have failed
    df.dropna(subset=['Decimal_Odds'], inplace=True)

    # Calculate the profit or loss for each bet using the converted decimal odds
    df['Profit'] = np.where(df['Outcome'] == 1, (df['Stake'] * df['Decimal_Odds']) - df['Stake'], -df['Stake'])

    # --- 4. Calculate Key Metrics ---
    total_bets = len(df)
    total_profit = df['Profit'].sum()
    
    # Standard Deviation of the P/L of each bet
    std_dev_per_bet = df['Profit'].std()
    
    # Average Profit Per Bet (your "edge")
    avg_profit_per_bet = df['Profit'].mean()

    # --- 5. Sharpe Ratio (The Professional's Metric) ---
    # Measures risk-adjusted return. Higher is better.
    sharpe_ratio = (avg_profit_per_bet / std_dev_per_bet) if std_dev_per_bet > 0 else 0


    print("--- Real-World Performance Analysis ---")
    print(f"Total Bets: {total_bets}")
    print(f"Total Profit: {total_profit:.2f} units")
    print(f"\nAverage Profit per Bet: {avg_profit_per_bet:.4f} units")
    print(f"Standard Deviation per Bet: {std_dev_per_bet:.4f} units")
    print(f"\nSharpe Ratio: {sharpe_ratio:.4f}")
    print("---------------------------------------")
    
    if sharpe_ratio > 0.1:
        print("Interpretation: This indicates a strong risk-adjusted return. A Sharpe Ratio > 0.1 on a per-bet basis is excellent.")
    elif sharpe_ratio > 0:
        print("Interpretation: The strategy is profitable, but the returns are more volatile relative to the profit.")
    else:
        print("Interpretation: The strategy is not currently demonstrating a profitable risk-adjusted return.")


except FileNotFoundError:
    print(f"Error: The file '{BET_HISTORY_FILE}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")