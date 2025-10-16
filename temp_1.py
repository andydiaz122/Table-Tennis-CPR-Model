import pandas as pd

# Load your complete, up-to-date dataset
df = pd.read_csv("final_dataset_v7.1.csv")
df['Date'] = pd.to_datetime(df['Date'])

# --- THIS IS THE CRUCIAL LINE ---
# We are pretending today is Sept 19. The model can only know about things before this date.
historical_df = df[df['Date'] < '2025-09-19']

# Save this as the temporary historical file for your test
historical_df.to_csv("paper_trade_history.csv", index=False)
print("Created 'paper_trade_history.csv' with all data prior to 2025-09-22.")