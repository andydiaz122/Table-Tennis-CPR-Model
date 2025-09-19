import pandas as pd
import numpy as np
import xgboost as xgb
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- Configuration & File Paths ---
RAW_DATA_FILE = 'czech_liga_pro_advanced_stats_FIXED.csv'
FEATURES_FILE = 'full_feature_database_v6.1.csv'
GBM_MODEL_FILE = 'cpr_v6.1_gbm_specialist.json'
LSTM_MODEL_FILE = 'cpr_v6.1_lstm_specialist.h5'
META_MODEL_FILE = 'cpr_v6.1_meta_model.pkl'
SCALER_FILE = 'cpr_v6.1_lstm_scaler.pkl'  # New file path for the scaler
SEQUENCE_LENGTH = 15

# --- Back-testing Parameters ---
STARTING_BANKROLL = 10.00
KELLY_FRACTION = 0.12  # Fractional Kelly
VIG = 0.045  # Standard 4.5% vig on market odds
MIN_ODDS = 1.05  # Minimum odds to consider a bet
MAX_BETS_WINDOW = 5
WINDOW_TIME_MINUTES = 30

# --- Load Models & Data ---
try:
    gbm = xgb.XGBClassifier()
    gbm.load_model(GBM_MODEL_FILE)
    lstm = load_model(LSTM_MODEL_FILE)
    meta_model = joblib.load(META_MODEL_FILE)
    scaler = joblib.load(SCALER_FILE) # 💡 New: Load the pre-fitted scaler

    features_db = pd.read_csv(FEATURES_FILE).set_index('Player')
    matches_df = pd.read_csv(RAW_DATA_FILE)

    # 💡 CORRECTED CODE: Add the data cleaning and preparation steps
    matches_df['Date'] = pd.to_datetime(matches_df['Date'] + ' ' + matches_df['Time'])
    matches_df.sort_values(by='Date', inplace=True)
    matches_df.dropna(subset=['Final Score', 'Set Scores'], inplace=True)

    def parse_scores(row):
        try:
            p1_sets = int(row['Final Score'].split('-')[0].replace('"', '').replace('=', ''))
            p2_sets = int(row['Final Score'].split('-')[1].replace('"', '').replace('=', ''))
            return p1_sets, p2_sets
        except:
            return np.nan, np.nan

    matches_df['P1_Sets'], matches_df['P2_Sets'] = zip(*matches_df.apply(parse_scores, axis=1))
    matches_df.dropna(subset=['P1_Sets', 'P2_Sets'], inplace=True)

    # Prepare data for LSTM predictions
    def get_performance_score(row, is_p1):
        try:
            p1_sets, p2_sets = map(int, row['Final Score'].replace('"', '').replace('=', '').split('-'))
            if is_p1:
                return (p1_sets - p2_sets) * 10
            else:
                return (p2_sets - p1_sets) * 10
        except:
            return 0

    p1_perf = matches_df.apply(lambda row: get_performance_score(row, True), axis=1)
    p2_perf = matches_df.apply(lambda row: get_performance_score(row, False), axis=1)
    matches_df['p1_perf_score'] = p1_perf
    matches_df['p2_perf_score'] = p2_perf
    
    p1_ts = matches_df[['Date', 'Player 1', 'p1_perf_score']].rename(columns={'Player 1': 'Player', 'p1_perf_score': 'Perf_Score'})
    p2_ts = matches_df[['Date', 'Player 2', 'p2_perf_score']].rename(columns={'Player 2': 'Player', 'p2_perf_score': 'Perf_Score'})
    time_series_df = pd.concat([p1_ts, p2_ts])
    
    print("Successfully loaded all models and data for back-test.")
except Exception as e:
    print(f"FATAL ERROR: Could not load required files. Make sure all trainers were run. Error: {e}")
    exit()

# --- Utility Functions ---
def get_dynamic_market_odds(p1_win_prob_true, vig):
    p1_prob_vig = p1_win_prob_true + (vig / 2)
    p2_prob_vig = (1 - p1_win_prob_true) + (vig / 2)
    
    p1_odds = 1 / p1_prob_vig
    p2_odds = 1 / p2_prob_vig
    
    return p1_odds, p2_odds

def kelly_bet_size(model_prob, market_odds, bankroll):
    q = 1 / market_odds
    b = market_odds - 1
    
    kelly_fraction = (b * model_prob - (1 - model_prob)) / b
    
    if kelly_fraction > 0:
        return kelly_fraction * KELLY_FRACTION * bankroll
    else:
        return 0

def get_lstm_prediction(player_name, latest_match_date, model, history_df, scaler):
    player_data = history_df[history_df['Player'] == player_name].sort_values(by='Date')
    recent_matches = player_data[player_data['Date'] < latest_match_date].tail(SEQUENCE_LENGTH)

    if len(recent_matches) < SEQUENCE_LENGTH:
        return 0.5

    scores = recent_matches['Perf_Score'].values.reshape(-1, 1)
    scaled_scores = scaler.transform(scores)
    
    input_sequence = scaled_scores.reshape(1, SEQUENCE_LENGTH, 1)

    predicted_score = model.predict(input_sequence, verbose=0)[0][0]
    win_prob = 1 / (1 + np.exp(-predicted_score))
    return win_prob

# --- Back-testing Simulation ---
bankroll = STARTING_BANKROLL
bet_history = []
recent_bets = []

print("\n--- Starting Back-Test Simulation ---")
print(f"Initial Bankroll: {bankroll:.2f} units")

for index, row in matches_df.iterrows():
    p1 = row['Player 1']
    p2 = row['Player 2']
    match_date = row['Date']

    recent_bets = [bet for bet in recent_bets if (match_date - bet['time']).total_seconds() / 60 <= WINDOW_TIME_MINUTES]

    if p1 not in features_db.index or p2 not in features_db.index:
        continue

    # --- 1. Get Model's Prediction ---
    feature_diff = features_db.loc[p1] - features_db.loc[p2]
    gbm_win_prob = gbm.predict_proba(pd.DataFrame([feature_diff]))[0][0]
    
    lstm_p1_prob = get_lstm_prediction(p1, match_date, lstm, time_series_df, scaler)
    lstm_p2_prob = get_lstm_prediction(p2, match_date, lstm, time_series_df, scaler)
    
    lstm_momentum_p1 = lstm_p1_prob - lstm_p2_prob
    
    specialist_predictions = np.array([[gbm_win_prob, lstm_momentum_p1]])
    final_win_prob = meta_model.predict_proba(specialist_predictions)[0][0]

    # 💡 NEW DEBUGGING CODE
    # Determine the actual winner based on the data
    actual_winner = "P1" if row['P1_Sets'] > row['P2_Sets'] else "P2"
    # Print the model's prediction and the actual outcome
    print(f"Match: {p1} vs. {p2}")
    print(f"  -> Model Predicted P1 Win Prob: {final_win_prob:.2%}")
    print(f"  -> Actual Winner: {actual_winner}")
    print("-" * 30)

    # --- 2. Get Dynamic Market Odds ---
    p1_win_true_prob = 1 if row['P1_Sets'] > row['P2_Sets'] else 0
    p2_win_true_prob = 1 - p1_win_true_prob
    p1_market_odds, p2_market_odds = get_dynamic_market_odds(p1_win_true_prob, VIG)

    # --- 3. Calculate Wagers (CORRECTED) ---
    wager = 0
    side = None
    market_odds = None

    p1_implied_prob = 1 / p1_market_odds
    p2_implied_prob = 1 / p2_market_odds

    # Check for a positive edge on P1
    if final_win_prob > p1_implied_prob:
        wager = kelly_bet_size(final_win_prob, p1_market_odds, bankroll)
        side = 'P1'
        market_odds = p1_market_odds
    # Check for a positive edge on P2
    elif (1 - final_win_prob) > p2_implied_prob:
        wager = kelly_bet_size(1 - final_win_prob, p2_market_odds, bankroll)
        side = 'P2'
        market_odds = p2_market_odds
    # else: no value bet is found

    # --- 4. Enforce Constraints & Simulate Bet ---
    if wager > 0 and len(recent_bets) < MAX_BETS_WINDOW:
        if wager > bankroll:
            wager = bankroll

        bankroll -= wager

        #is_win = (side == 'P1' and row['P1_Sets'] > row['P2_Sets']) or \
                 #(side == 'P2' and row['P2_Sets'] > row['P1_Sets'])
        # 💡 FIX: Simplified and corrected win logic
        is_win = False
        if (side == 'P1' and row['P1_Sets'] > row['P2_Sets']):
            is_win = True
        elif (side == 'P2' and row['P2_Sets'] > row['P1_Sets']):
            is_win = True         

        if is_win:
            payout = wager * market_odds
            bankroll += payout
        else:
            payout = 0

        bet_history.append({'date': match_date, 'wager': wager, 'payout': payout, 'win': is_win, 'new_bankroll': bankroll})
        recent_bets.append({'time': match_date})
        
        print(f"Bet on {side} at {match_date.date()}, Wager: {wager:.2f}, Outcome: {'WIN' if is_win else 'LOSS'}, New Bankroll: {bankroll:.2f}")

# --- Final Report ---
print("\n--- Back-Test Results ---")
print(f"Final Bankroll: {bankroll:.2f} units")
total_wagered = sum(b['wager'] for b in bet_history)
total_payout = sum(b['payout'] for b in bet_history)
roi = ((total_payout - total_wagered) / total_wagered) * 100 if total_wagered > 0 else 0
win_rate = sum(b['win'] for b in bet_history) / len(bet_history) if len(bet_history) > 0 else 0

print(f"Total Bets: {len(bet_history)}")
print(f"Total Wagered: {total_wagered:.2f} units")
print(f"Win Rate: {win_rate:.2%}")
print(f"Return on Investment (ROI): {roi:.2f}%")