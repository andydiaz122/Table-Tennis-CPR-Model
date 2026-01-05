# FILE: 3_analyze_paper_trade_results.py (Final Version)
import pandas as pd

# --- Configuration ---
PAPER_TRADE_LOG_FILE = "paper_trade_log.csv"
FULL_DATASET_FILE = "final_dataset_v7.4.csv"
INITIAL_BANKROLL = 1000
KELLY_FRACTION = 0.25

# --- Main Analysis Logic ---
try:
    print(f"--- Analyzing Paper-Trade Log: '{PAPER_TRADE_LOG_FILE}' ---")
    rec_df = pd.read_csv(PAPER_TRADE_LOG_FILE)
    if rec_df.empty:
        raise SystemExit("Log file is empty. No bets to analyze.")

    results_df = pd.read_csv(FULL_DATASET_FILE)
    
    # Enforce matching data types for all relevant columns
    rec_df['Match_ID'] = rec_df['Match_ID'].astype('int64')
    rec_df['Bet_On_Player_ID'] = rec_df['Bet_On_Player_ID'].astype('int64')
    results_df['Match ID'] = results_df['Match ID'].astype('int64')
    results_df['Player 1 ID'] = results_df['Player 1 ID'].astype('int64')
    results_df['Player 2 ID'] = results_df['Player 2 ID'].astype('int64')
    results_df['P1_Win'] = results_df['P1_Win'].astype('int64')

    # Merge recommendations with actual results
    merged_df = pd.merge(rec_df, results_df[['Match ID', 'Player 1 ID', 'Player 2 ID', 'P1_Win']], 
                         left_on='Match_ID', right_on='Match ID', how='left')

    if merged_df['P1_Win'].isnull().values.any():
        print("Warning: Some matches in the log could not be found in the results file and will be skipped.")
        merged_df.dropna(subset=['P1_Win'], inplace=True)

    # --- Calculate Profit and Loss ---
    total_profit, total_staked, wins, losses = 0, 0, 0, 0
    bets_with_valid_odds = 0

    for _, row in merged_df.iterrows():
        market_odds = row['Market_Odds']
        
        # Skip any bet that was logged without valid market odds
        if pd.isna(market_odds) or (market_odds - 1) <= 0:
            continue
        
        bets_with_valid_odds += 1
        is_win = (row['Bet_On_Player_ID'] == row['Player 1 ID'] and row['P1_Win'] == 1) or \
                 (row['Bet_On_Player_ID'] == row['Player 2 ID'] and row['P1_Win'] == 0)

        edge = row['Edge']
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

    # --- Print Performance Summary ---
    total_bets_in_log = len(merged_df)
    roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
    win_rate = (wins / bets_with_valid_odds) * 100 if bets_with_valid_odds > 0 else 0

    print("\n--- 📊 Paper-Trade Performance Summary ---")
    print(f"Total Bets in Log: {total_bets_in_log}")
    print(f"Bets with Valid Odds Analyzed: {bets_with_valid_odds}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate (on analyzed bets): {win_rate:.2f}%")
    print(f"Total Staked: ${total_staked:,.2f}")
    print(f"Total Profit: ${total_profit:,.2f}")
    print(f"Return on Investment (ROI): {roi:.2f}%")
    print("------------------------------------------")

except FileNotFoundError:
    print(f"Error: Make sure '{PAPER_TRADE_LOG_FILE}' and '{FULL_DATASET_FILE}' exist.")
except Exception as e:
    print(f"An error occurred: {e}")