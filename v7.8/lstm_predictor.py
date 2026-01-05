import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --- Configuration ---
MODEL_FILE = 'cpr_v6.1_lstm_specialist.h5'
RAW_DATA_FILE = 'czech_liga_pro_advanced_stats_FIXED.csv'
SEQUENCE_LENGTH = 15

# --- Load Data and Scaler ---
# Note: The scaler needs to be fitted on the same data the model was trained on
# To make this robust, you would save and load the scaler too. For now, we'll
# re-create it on the full dataset as a close approximation.
df = pd.read_csv(RAW_DATA_FILE)
df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.sort_values(by='Date', inplace=True)
df.dropna(subset=['Final Score'], inplace=True)

def get_performance_score(row, is_p1):
    try:
        p1_sets, p2_sets = map(int, row['Final Score'].replace('"', '').replace('=', '').split('-'))
        if is_p1:
            return (p1_sets - p2_sets) * 10
        else:
            return (p2_sets - p1_sets) * 10
    except:
        return 0

p1_perf = df.apply(lambda row: get_performance_score(row, True), axis=1)
p2_perf = df.apply(lambda row: get_performance_score(row, False), axis=1)
df['p1_perf_score'] = p1_perf
df['p2_perf_score'] = p2_perf

p1_ts = df[['Player 1', 'p1_perf_score']].rename(columns={'Player 1': 'Player', 'p1_perf_score': 'Perf_Score'})
p2_ts = df[['Player 2', 'p2_perf_score']].rename(columns={'Player 2': 'Player', 'p2_perf_score': 'Perf_Score'})
time_series_df = pd.concat([p1_ts, p2_ts])
scaler = MinMaxScaler(feature_range=(-1, 1))
all_player_data = time_series_df['Perf_Score'].values.reshape(-1, 1)
scaler.fit(all_player_data)

# --- The Prediction Function ---
def get_lstm_prediction(player_name, latest_match_date, model, history_df):
    """
    Predicts a player's next performance using their recent match history.
    """
    player_data = history_df[history_df['Player'] == player_name]
    
    # Filter for matches that occurred before the latest match date
    recent_matches = player_data[player_data['Date'] < latest_match_date].tail(SEQUENCE_LENGTH)

    if len(recent_matches) < SEQUENCE_LENGTH:
        return 0.5 # Default to a neutral prediction if not enough history

    # Get the performance scores and scale them
    scores = recent_matches['Perf_Score'].values.reshape(-1, 1)
    scaled_scores = scaler.transform(scores)
    
    # Reshape for LSTM model input
    input_sequence = scaled_scores.reshape(1, SEQUENCE_LENGTH, 1)

    # Make prediction and inverse transform to original scale
    predicted_score = model.predict(input_sequence, verbose=0)[0][0]
    
    # We need to convert this score to a win probability.
    # A simple sigmoid mapping is a quick way to do this.
    win_prob = 1 / (1 + np.exp(-predicted_score))
    return win_prob