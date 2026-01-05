import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

# --- SYNC: New Helper Functions from back-tester ---
def calculate_close_set_win_rate(player_id, rolling_games_df):
    if rolling_games_df.empty: return 0.5
    total_close_sets_played, total_close_sets_won = 0, 0
    for _, game in rolling_games_df.iterrows():
        set_scores_str = game.get('Set Scores')
        if pd.isna(set_scores_str): continue
        for set_score in str(set_scores_str).split(','):
            try:
                p1_points, p2_points = map(int, set_score.split('-'))
                if abs(p1_points - p2_points) == 2:
                    total_close_sets_played += 1
                    is_p1 = (game['Player 1 ID'] == player_id)
                    p1_won_set = p1_points > p2_points
                    if (is_p1 and p1_won_set) or (not is_p1 and not p1_won_set):
                        total_close_sets_won += 1
            except (ValueError, IndexError): continue
    if total_close_sets_played == 0: return 0.5
    return total_close_sets_won / total_close_sets_played

def calculate_h2h_dominance(p1_id, h2h_df, current_match_date, decay_factor):
    if h2h_df.empty: return 0.0
    total_weighted_score = 0
    for _, game in h2h_df.iterrows():
        days_ago = (current_match_date - game['Date']).days
        weight = decay_factor ** days_ago
        point_diff = (game['P1 Total Points'] - game['P2 Total Points']) if game['Player 1 ID'] == p1_id else (game['P2 Total Points'] - game['P1 Total Points'])
        total_weighted_score += (point_diff * weight)
    return total_weighted_score

def calculate_performance_slope(performance_history):
    if len(performance_history) < 2: return 0.0
    y = np.array(performance_history)
    x = np.arange(len(y))
    return np.polyfit(x, y, 1)[0]

def calculate_pdr(player_id, rolling_games_df):
    if rolling_games_df.empty: return 0.5
    total_points_won, total_points_played = 0, 0
    for _, game in rolling_games_df.iterrows():
        if 'P1 Total Points' not in game or 'P2 Total Points' not in game: continue
        points_won = game['P1 Total Points'] if game['Player 1 ID'] == player_id else game['P2 Total Points']
        points_lost = game['P2 Total Points'] if game['Player 1 ID'] == player_id else game['P1 Total Points']
        total_points_won += points_won
        total_points_played += (points_won + points_lost)
    if total_points_played == 0: return 0.5
    return total_points_won / total_points_played

def moneyline_to_decimal(moneyline_odds):
    try:
        moneyline_odds = float(moneyline_odds)
        if moneyline_odds >= 100: return (moneyline_odds / 100) + 1
        elif moneyline_odds < 0: return (100 / abs(moneyline_odds)) + 1
        else: return np.nan
    except (ValueError, TypeError): return np.nan

# --- 1. Configuration ---
# ---!!!--- MANUALLY EDIT THIS SECTION FOR DAILY MATCHES ---!!!---
upcoming_matches = [
    
{
    'Player 1 ID': '', 'Player 1 Name': 'Jan Cervenka',
    'Player 2 ID': '', 'Player 2 Name': 'Tomas Paldus',
    'P1_ML': -120,
    'P2_ML': -120,
    'Daily_Fatigue_Advantage': 0
},
{
    'Player 1 ID': 380220, 'Player 1 Name': 'Patrik Pycha',
    'Player 2 ID': 917524, 'Player 2 Name': 'Jakub Tazler',
    'P1_ML': 110,
    'P2_ML': -155,
    'Daily_Fatigue_Advantage': 0
},
{
    'Player 1 ID': 553241, 'Player 1 Name': 'Martin Sobisek',
    'Player 2 ID': 954785, 'Player 2 Name': 'Martin Sychra',
    'P1_ML': -140,
    'P2_ML': 100,
    'Daily_Fatigue_Advantage': 0
},
{
    'Player 1 ID': 1015337, 'Player 1 Name': 'Miroslav Novotny',
    'Player 2 ID': 1168737, 'Player 2 Name': 'Vaclav Zidek',
    'P1_ML': 190,
    'P2_ML': -300,
    'Daily_Fatigue_Advantage': 0
},

]
# ---!!!--- ---!!!--- ---!!!--- ---!!!--- ---!!!--- ---!!!---

# --- File Paths ---
# HISTORICAL_DATA_FILE = "final_dataset_v7.4_no_duplicates.csv"
HISTORICAL_DATA_FILE = "final_dataset_v7.4.csv"
GBM_MODEL_FILE = "cpr_v7.4_gbm_specialist.joblib"
GBM_PREPROCESSOR_FILE = "gbm_preprocessor_v7.4.joblib"

# --- SYNC: Configuration from Back-tester ---
KELLY_FRACTION = 0.030
ROLLING_WINDOW = 20
SHORT_ROLLING_WINDOW = 5
SLOPE_WINDOW = 10
H2H_DECAY_FACTOR = 0.98
EDGE_THRESHOLD_MIN = 0.0001
EDGE_THRESHOLD_MAX = 0.99
MIN_ODDS_DENOMINATOR = 0.10
MIN_GAMES_THRESHOLD = 4

# --- SYNC: Strategic Filters from Back-tester ---
ODDS_THRESHOLD_MAX = 3.0
FORM_L5_THRESHOLD = 0.1

# --- 2. Load Data & Models ---
try:
    print("--- Loading All Models and Historical Data for Prediction ---")
    df_history = pd.read_csv(HISTORICAL_DATA_FILE, na_values=['-'], keep_default_na=True, low_memory=False)
    df_history['Date'] = pd.to_datetime(df_history['Date'], format='mixed')
    
    gbm_model = joblib.load(GBM_MODEL_FILE)
    gbm_preprocessor = joblib.load(GBM_PREPROCESSOR_FILE)
    print("All models and filters loaded successfully.")

    # --- 3. Feature Engineering and Prediction for Each Match ---
    print("\n--- Analyzing Upcoming Matches with Synced Strategy ---")
    
    player_pdr_history = {} # NEW: Initialize dictionary for PDR slope calculation

    for match in upcoming_matches:
        p1_id, p2_id = match['Player 1 ID'], match['Player 2 ID']
        p1_name, p2_name = match['Player 1 Name'], match['Player 2 Name']
        
        print("\n---------------------------------")
        print(f"Matchup: {p1_name} vs. {p2_name}")

        # --- SYNC: Overhauled On-the-fly Feature Engineering ---
        p1_games = df_history[(df_history['Player 1 ID'] == p1_id) | (df_history['Player 2 ID'] == p1_id)]
        p2_games = df_history[(df_history['Player 1 ID'] == p2_id) | (df_history['Player 2 ID'] == p2_id)]
        p1_games_gbm = p1_games.tail(ROLLING_WINDOW)
        p2_games_gbm = p2_games.tail(ROLLING_WINDOW)
        
        if len(p1_games_gbm) < MIN_GAMES_THRESHOLD or len(p2_games_gbm) < MIN_GAMES_THRESHOLD:
            print("RECOMMENDATION: NO BET (Insufficient player history)")
            print("---------------------------------")
            continue
        
        # NEW: Fatigue and Recency Features
        # 1. Time Since Last Match
        p1_last_game_date = p1_games['Date'].max()
        p1_time_since_last = (datetime.now() - p1_last_game_date).total_seconds() / 3600 if pd.notna(p1_last_game_date) else 72
        p2_last_game_date = p2_games['Date'].max()
        p2_time_since_last = (datetime.now() - p2_last_game_date).total_seconds() / 3600 if pd.notna(p2_last_game_date) else 72
        time_since_last_advantage = p1_time_since_last - p2_time_since_last

        # 2. Matches in Last 24 Hours
        p1_matches_last_24h = len(p1_games[p1_games['Date'] > (datetime.now() - pd.Timedelta(hours=24))])
        p2_matches_last_24h = len(p2_games[p2_games['Date'] > (datetime.now() - pd.Timedelta(hours=24))])
        matches_last_24h_advantage = p1_matches_last_24h - p2_matches_last_24h
        
        # 3. Is First Match of the Day
        today_history_df = df_history[df_history['Date'].dt.date == datetime.now().date()]
        p1_is_first_match = 1 if today_history_df[(today_history_df['Player 1 ID'] == p1_id) | (today_history_df['Player 2 ID'] == p1_id)].empty else 0
        p2_is_first_match = 1 if today_history_df[(today_history_df['Player 1 ID'] == p2_id) | (today_history_df['Player 2 ID'] == p2_id)].empty else 0
        is_first_match_advantage = p1_is_first_match - p2_is_first_match

        # NEW: "Hot Streak" Win Rate (L5)
        p1_rolling_games_short = p1_games.tail(SHORT_ROLLING_WINDOW)
        p1_win_rate_l5 = p1_rolling_games_short.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p1_rolling_games_short.empty else 0.5
        p2_rolling_games_short = p2_games.tail(SHORT_ROLLING_WINDOW)
        p2_win_rate_l5 = p2_rolling_games_short.apply(lambda r: 1 if (r['Player 1 ID'] == p2_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p2_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p2_rolling_games_short.empty else 0.5
        win_rate_l5_advantage = p1_win_rate_l5 - p2_win_rate_l5

        # PDR and PDR Slope
        p1_pdr = calculate_pdr(p1_id, p1_games_gbm)
        p2_pdr = calculate_pdr(p2_id, p2_games_gbm)
        pdr_advantage = p1_pdr - p2_pdr

        daily_fatigue_advantage = match['Daily_Fatigue_Advantage']

        # Update and manage history for Player 1
        if p1_id not in player_pdr_history: player_pdr_history[p1_id] = []
        player_pdr_history[p1_id].append(p1_pdr)
        if len(player_pdr_history[p1_id]) > SLOPE_WINDOW: player_pdr_history[p1_id].pop(0)
        p1_pdr_slope = calculate_performance_slope(player_pdr_history[p1_id])

        # Update and manage history for Player 2
        if p2_id not in player_pdr_history: player_pdr_history[p2_id] = []
        player_pdr_history[p2_id].append(p2_pdr)
        if len(player_pdr_history[p2_id]) > SLOPE_WINDOW: player_pdr_history[p2_id].pop(0)
        p2_pdr_slope = calculate_performance_slope(player_pdr_history[p2_id])
        pdr_slope_advantage = p1_pdr_slope - p2_pdr_slope
        
        # NEW: H2H Features
        h2h_df = df_history[((df_history['Player 1 ID'] == p1_id) & (df_history['Player 2 ID'] == p2_id)) | ((df_history['Player 1 ID'] == p2_id) & (df_history['Player 2 ID'] == p1_id))]
        p1_h2h_wins = len(h2h_df[((h2h_df['Player 1 ID'] == p1_id) & (h2h_df['P1_Win'] == 1)) | ((h2h_df['Player 2 ID'] == p1_id) & (h2h_df['P1_Win'] == 0))])
        h2h_p1_win_rate = p1_h2h_wins / len(h2h_df) if not h2h_df.empty else 0.5
        h2h_dominance_score = calculate_h2h_dominance(p1_id, h2h_df, datetime.now(), H2H_DECAY_FACTOR)

        # Standard Form Features
        p1_win_rate = p1_games_gbm.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p1_games_gbm.empty else 0.5
        p2_win_rate = p2_games_gbm.apply(lambda r: 1 if (r['Player 1 ID'] == p2_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p2_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p2_games_gbm.empty else 0.5
        win_rate_advantage = p1_win_rate - p2_win_rate
        
        # NEW: Close Set and Comeback Features
        p1_close_set_win_rate = calculate_close_set_win_rate(p1_id, p1_games_gbm)
        p2_close_set_win_rate = calculate_close_set_win_rate(p2_id, p2_games_gbm)
        close_set_win_rate_advantage = p1_close_set_win_rate - p2_close_set_win_rate
        p1_set_comebacks = p1_games_gbm.apply(lambda r: r['P1 Set Comebacks'] if r['Player 1 ID'] == p1_id else r['P2 Set Comebacks'], axis=1).sum() if not p1_games_gbm.empty else 0.0
        p2_set_comebacks = p2_games_gbm.apply(lambda r: r['P1 Set Comebacks'] if r['Player 1 ID'] == p2_id else r['P2 Set Comebacks'], axis=1).sum() if not p2_games_gbm.empty else 0.0
        set_comebacks_advantage = p1_set_comebacks - p2_set_comebacks

        # --- SYNC: Model Prediction with all 12 Features ---
        gbm_features = pd.DataFrame([{
            'Time_Since_Last_Advantage': time_since_last_advantage,
            'Matches_Last_24H_Advantage': matches_last_24h_advantage,
            'Is_First_Match_Advantage': is_first_match_advantage,
            'PDR_Slope_Advantage': pdr_slope_advantage,
            'H2H_P1_Win_Rate': h2h_p1_win_rate,
            'H2H_Dominance_Score': h2h_dominance_score,
            'Daily_Fatigue_Advantage': daily_fatigue_advantage,
            'PDR_Advantage': pdr_advantage,
            'Win_Rate_Advantage': win_rate_advantage,
            'Win_Rate_L5_Advantage': win_rate_l5_advantage,
            'Close_Set_Win_Rate_Advantage': close_set_win_rate_advantage,
            'Set_Comebacks_Advantage': set_comebacks_advantage
        }])
        
        X_gbm_processed = gbm_preprocessor.transform(gbm_features)
        model_prob_p1 = gbm_model.predict_proba(X_gbm_processed)[0, 1]
        model_prob_p2 = 1 - model_prob_p1

        print(f"Model Prediction: {p1_name} ({model_prob_p1:.2%}) vs. {p2_name} ({model_prob_p2:.2%})")

        # --- SYNC: Final Betting Logic from Back-tester ---
        p1_market_odds = moneyline_to_decimal(match['P1_ML'])
        p2_market_odds = moneyline_to_decimal(match['P2_ML'])
        edge_p1 = model_prob_p1 * p1_market_odds - 1 if pd.notna(p1_market_odds) else -1
        edge_p2 = model_prob_p2 * p2_market_odds - 1 if pd.notna(p2_market_odds) else -1
        
        p1_filters = {
            "Has_Edge": edge_p1 > EDGE_THRESHOLD_MIN and edge_p1 < EDGE_THRESHOLD_MAX,
            "Is_Betable": pd.notna(p1_market_odds) and (p1_market_odds - 1) > MIN_ODDS_DENOMINATOR,
            "Prime_Directive": pdr_advantage > 0,
            "Risk_Off_Switch": pd.notna(p1_market_odds) and p1_market_odds <= ODDS_THRESHOLD_MAX,
            "Clarity_Mandate": abs(win_rate_l5_advantage) > FORM_L5_THRESHOLD,
        }
        
        p2_filters = {
            "Has_Edge": edge_p2 > EDGE_THRESHOLD_MIN and edge_p2 < EDGE_THRESHOLD_MAX,
            "Is_Betable": pd.notna(p2_market_odds) and (p2_market_odds - 1) > MIN_ODDS_DENOMINATOR,
            "Prime_Directive": pdr_advantage < 0,
            "Risk_Off_Switch": pd.notna(p2_market_odds) and p2_market_odds <= ODDS_THRESHOLD_MAX,
            "Clarity_Mandate": abs(win_rate_l5_advantage) > FORM_L5_THRESHOLD,
        }

        recommendation_found = False
        
        # --- Analysis for Player 1 ---
        print(f"\n--- Analysis for {p1_name} ---")
        print(f"Market Odds: {match['P1_ML']} ({p1_market_odds:.2f} dec)")
        print("Filter Checklist:")
        for name, passed in p1_filters.items():
            print(f"  - {name}: {'PASS' if passed else 'FAIL'}")

        if all(p1_filters.values()):
            kelly_fraction_rec = (edge_p1 / (p1_market_odds - 1)) * KELLY_FRACTION
            capped_fraction = min(kelly_fraction_rec, 0.05)
            if win_rate_advantage < 0: capped_fraction /= 4
            
            print(f"\nRECOMMENDATION: BET on {p1_name}")
            print(f"  - Model Edge: {edge_p1:.2%}")
            print(f"  - Recommended Bet Size (Fraction of Bankroll): {capped_fraction:.2%}")
            recommendation_found = True
        else:
            print("\nRECOMMENDATION: NO BET")

        # --- Analysis for Player 2 ---
        print(f"\n--- Analysis for {p2_name} ---")
        print(f"Market Odds: {match['P2_ML']} ({p2_market_odds:.2f} dec)")
        print("Filter Checklist:")
        for name, passed in p2_filters.items():
            print(f"  - {name}: {'PASS' if passed else 'FAIL'}")

        if all(p2_filters.values()) and not recommendation_found:
            kelly_fraction_rec = (edge_p2 / (p2_market_odds - 1)) * KELLY_FRACTION
            capped_fraction = min(kelly_fraction_rec, 0.05)
            if win_rate_advantage > 0: capped_fraction /= 4
            
            print(f"\nRECOMMENDATION: BET on {p2_name}")
            print(f"  - Model Edge: {edge_p2:.2%}")
            print(f"  - Recommended Bet Size (Fraction of Bankroll): {capped_fraction:.2%}")
        elif not recommendation_found:
            print("\nRECOMMENDATION: NO BET")

        print("---------------------------------")

except FileNotFoundError as e:
    print(f"\n--- ERROR ---")
    print(f"Missing file: {e.filename}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()