import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

print("Training CPR v6.1 'Momentum Analyst' (LSTM)...")

# --- Configuration ---
RAW_DATA_FILE = 'czech_liga_pro_advanced_stats_FIXED.csv'
MODEL_OUTPUT_FILE = 'cpr_v6.1_lstm_specialist.h5'
SCALER_OUTPUT_FILE = 'cpr_v6.1_lstm_scaler.pkl'  # New file for the scaler
SEQUENCE_LENGTH = 15 # Look back at the last 15 matches

# --- Load and Prepare Data ---
try:
    df = pd.read_csv(RAW_DATA_FILE)
except FileNotFoundError:
    print(f"FATAL ERROR: The data file '{RAW_DATA_FILE}' was not found.")
    exit()

df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.sort_values(by='Date', inplace=True)
df.dropna(subset=['Final Score'], inplace=True)

# Create a simplified performance metric for the time series
def get_performance_score(row, is_p1):
    try:
        p1_sets, p2_sets = map(int, row['Final Score'].replace('"', '').replace('=', '').split('-'))
        if is_p1:
            return (p1_sets - p2_sets) * 10 # Simple score
        else:
            return (p2_sets - p1_sets) * 10
    except:
        return 0

# Create long format data
p1_perf = df.apply(lambda row: get_performance_score(row, True), axis=1)
p2_perf = df.apply(lambda row: get_performance_score(row, False), axis=1)
df['p1_perf_score'] = p1_perf
df['p2_perf_score'] = p2_perf

p1_ts = df[['Player 1', 'p1_perf_score']].rename(columns={'Player 1': 'Player', 'p1_perf_score': 'Perf_Score'})
p2_ts = df[['Player 2', 'p2_perf_score']].rename(columns={'Player 2': 'Player', 'p2_perf_score': 'Perf_Score'})
time_series_df = pd.concat([p1_ts, p2_ts])

# --- Create Sequences ---
X, y = [], []
scaler = MinMaxScaler(feature_range=(-1, 1))

all_players = time_series_df['Player'].unique()
for player in all_players:
    player_data = time_series_df[time_series_df['Player'] == player]['Perf_Score'].values.reshape(-1, 1)
    if len(player_data) < SEQUENCE_LENGTH + 1:
        continue
    
    scaled_data = scaler.fit_transform(player_data)
    
    for i in range(len(scaled_data) - SEQUENCE_LENGTH):
        X.append(scaled_data[i:(i + SEQUENCE_LENGTH), 0])
        y.append(scaled_data[i + SEQUENCE_LENGTH, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# --- Build and Train LSTM Model ---
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=64, verbose=2)

# --- Save Model and Scaler ---
model.save(MODEL_OUTPUT_FILE)
print(f"LSTM Specialist model trained and saved to '{MODEL_OUTPUT_FILE}'")

# 💡 NEW: Save the fitted scaler
joblib.dump(scaler, SCALER_OUTPUT_FILE)
print(f"MinMaxScaler saved to '{SCALER_OUTPUT_FILE}'")