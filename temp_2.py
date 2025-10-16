import pandas as pd

# Load your complete, up-to-date dataset again
df = pd.read_csv("final_dataset_v7.1.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Isolate the matches from the last 5 days
paper_trade_matches = df[(df['Date'] >= '2025-09-19') & (df['Date'] <= '2025-09-22')]

# This will print the matches in the exact format you need
# for the `upcoming_matches` list in your predictor script.
print("Copy and paste the relevant matches into your predictor:\n")
for _, row in paper_trade_matches.iterrows():
    print("{")
    print(f"    'Player 1 ID': {row['Player 1 ID']}, 'Player 1 Name': '{row['Player 1']}',")
    print(f"    'Player 2 ID': {row['Player 2 ID']}, 'Player 2 Name': '{row['Player 2']}',")
    # NOTE: Moneyline odds need to be calculated from decimal if not present.
    # This assumes your final_dataset has P1_ML and P2_ML columns.
    # If not, you'd use Kickoff_P1_Odds and Kickoff_P2_Odds.
    print(f"    'P1_ML': {row.get('P1_ML', 'None')}, # Actual Decimal: {row['Kickoff_P1_Odds']:.2f}, Winner: {'P1' if row['P1_Win']==1 else 'P2'}")
    print(f"    'P2_ML': {row.get('P2_ML', 'None')}  # Actual Decimal: {row['Kickoff_P2_Odds']:.2f}")
    print("},")