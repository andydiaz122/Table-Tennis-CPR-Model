import pandas as pd
import numpy as np

# --- 1. Configuration ---
PAPER_TRADE_LOG_FILE = "paper_trade_log.csv"
FULL_DATASET_FILE = "final_dataset_v7.1.csv"
INITIAL_BANKROLL = 1000  # Use the same constant for Kelly calculation
KELLY_FRACTION = 0.25

# --- 2. Main Analysis Logic ---
try:
    print(f"--- Analyzing Paper-Trade Log: '{PAPER_TRADE_LOG_FILE}' ---")
    
    # Load the recommendations made by the predictor
    rec_df = pd.read_csv(PAPER_TRADE_LOG_FILE)
    if rec_df.empty:
        print("Log file is empty. No bets to analyze.")
        exit()

    # Load the full dataset to find the actual results of the matches
    results_df = pd.read_csv(FULL_DATASET_FILE)
    
    # Merge the recommendations with the results based on Match ID
    # This combines what the model *said* to do with what *actually* happened
    merged_df = pd.merge(rec_df, results_df[['Match ID', 'Player 1 ID', 'Player 2 ID', 'P1_Win']], 
                         left_on='Match_ID', right_on='Match ID', how='left')

    if merged_df.isnull().any().any():
        print("Warning: Some matches in the log could not be found in the results file.")
        merged_df.dropna(inplace=True)

    # --- 3. Calculate Profit and Loss for Each Bet ---
    total_profit = 0
    total_staked = 0
    wins = 0
    losses = 0

    for index, row in merged_df.iterrows():
        # Determine if the recommended bet was a winner
        is_win = False
        if row['Bet_On_Player_ID'] == row['Player 1 ID'] and row['P1_Win'] == 1:
            is_win = True
        elif row['Bet_On_Player_ID'] == row['Player 2 ID'] and row['P1_Win'] == 0:
            is_win = True

        # Calculate stake using the same normalized Kelly Criterion as the back-test
        market_odds = row['Market_Odds']
        edge = row['Edge']
        if (market_odds - 1) <= 0: continue # Avoid invalid odds

        kelly_stake_fraction = edge / (market_odds - 1)
        stake = INITIAL_BANKROLL * kelly_stake_fraction * KELLY_FRACTION
        total_staked += stake

        if is_win:
            profit = stake * (market_odds - 1)
            wins += 1
        else:
            profit = -stake
            losses += 1
        
        total_profit += profit

    # --- 4. Print Performance Summary ---
    bet_count = len(merged_df)
    roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
    win_rate = (wins / bet_count) * 100 if bet_count > 0 else 0

    print("\n--- Paper-Trade Performance Summary ---")
    print(f"Total Bets Placed: {bet_count}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Staked: ${total_staked:,.2f}")
    print(f"Total Profit: ${total_profit:,.2f}")
    print(f"Return on Investment (ROI): {roi:.2f}%")
    print("---------------------------------------")

except FileNotFoundError:
    print(f"Error: Make sure both '{PAPER_TRADE_LOG_FILE}' and '{FULL_DATASET_FILE}' exist.")
except Exception as e:
    print(f"An error occurred: {e}")
