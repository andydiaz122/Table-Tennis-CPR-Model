import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

# =================================================================
# ===== NEW ROBUST LOGGING SYSTEM (v2 - With Opponent Info) =====
# =================================================================
import csv
import os

LOG_FILE = "paper_trade_log.csv"

def setup_log_file():
    """
    Deletes the old log file and creates a new one with the header row.
    """
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        # ADDED Opponent_ID and Opponent_Name to the header
        writer.writerow(['Match_ID', 'Bet_On_Player_ID', 'Bet_On_Player_Name', 
                         'Opponent_ID', 'Opponent_Name', 'Market_Odds', 'Edge'])

def log_bet(match_id, player_id, player_name, opponent_id, opponent_name, market_odds, edge):
    """
    Safely appends a single bet, including opponent info, to the log file.
    """
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        # ADDED opponent_id and opponent_name to the data row
        writer.writerow([match_id, player_id, player_name, 
                         opponent_id, opponent_name, market_odds, edge])
# =================================================================

# --- 1. Configuration ---
# ---!!!--- MANUALLY EDIT THIS SECTION FOR DAILY MATCHES ---!!!---
upcoming_matches = [
    # --- Matches for 2025-09-00 ---
{
    'Match ID': 10682221,
    'Player 1 ID': 741786, 'Player 1 Name': 'Ondrej Svacha',
    'Player 2 ID': 561176, 'Player 2 Name': 'Jiri Ruzicka',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P2
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682243,
    'Player 1 ID': 639253, 'Player 1 Name': 'Michal Zahradka',
    'Player 2 ID': 338586, 'Player 2 Name': 'Marek Fabini',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682301,
    'Player 1 ID': 561176, 'Player 1 Name': 'Jiri Ruzicka',
    'Player 2 ID': 1088851, 'Player 2 Name': 'Oleksandr Kolisnyk',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682230,
    'Player 1 ID': 373963, 'Player 1 Name': 'Tomas Janata',
    'Player 2 ID': 548705, 'Player 2 Name': 'Jiri Plachy',
    'P1_ML': -187, # Actual Decimal: 1.53, Winner: P1
    'P2_ML': 137  # Actual Decimal: 2.38
},
{
    'Match ID': 10682241,
    'Player 1 ID': 1088851, 'Player 1 Name': 'Oleksandr Kolisnyk',
    'Player 2 ID': 976107, 'Player 2 Name': 'Dan Volhejn',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682319,
    'Player 1 ID': 380220, 'Player 1 Name': 'Patrik Pycha',
    'Player 2 ID': 798958, 'Player 2 Name': 'Jiri Zuzanek',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P1
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10687618,
    'Player 1 ID': 881949, 'Player 1 Name': 'Tomas Barta',
    'Player 2 ID': 373963, 'Player 2 Name': 'Tomas Janata',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P1
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10682201,
    'Player 1 ID': 798958, 'Player 1 Name': 'Jiri Zuzanek',
    'Player 2 ID': 1098619, 'Player 2 Name': 'Jaroslav Bares',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682334,
    'Player 1 ID': 338586, 'Player 1 Name': 'Marek Fabini',
    'Player 2 ID': 387783, 'Player 2 Name': 'Tomas Postelt',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682280,
    'Player 1 ID': 1163803, 'Player 1 Name': 'David Mikula',
    'Player 2 ID': 686100, 'Player 2 Name': 'Jakub Levicky',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P2
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10682292,
    'Player 1 ID': 685221, 'Player 1 Name': 'Jiri Nesnera',
    'Player 2 ID': 639253, 'Player 2 Name': 'Michal Zahradka',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10687643,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 881949, 'Player 2 Name': 'Tomas Barta',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10689213,
    'Player 1 ID': 380220, 'Player 1 Name': 'Patrik Pycha',
    'Player 2 ID': 358665, 'Player 2 Name': 'Marek Sedlak',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P2
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10682277,
    'Player 1 ID': 358665, 'Player 1 Name': 'Marek Sedlak',
    'Player 2 ID': 380220, 'Player 2 Name': 'Patrik Pycha',
    'P1_ML': -300, # Actual Decimal: 1.33, Winner: P2
    'P2_ML': 225  # Actual Decimal: 3.25
},
{
    'Match ID': 10682857,
    'Player 1 ID': 358665, 'Player 1 Name': 'Marek Sedlak',
    'Player 2 ID': 1098619, 'Player 2 Name': 'Jaroslav Bares',
    'P1_ML': -400, # Actual Decimal: 1.25, Winner: P1
    'P2_ML': 275  # Actual Decimal: 3.75
},
{
    'Match ID': 10682264,
    'Player 1 ID': 1098619, 'Player 1 Name': 'Jaroslav Bares',
    'Player 2 ID': 380220, 'Player 2 Name': 'Patrik Pycha',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682291,
    'Player 1 ID': 685221, 'Player 1 Name': 'Jiri Nesnera',
    'Player 2 ID': 387783, 'Player 2 Name': 'Tomas Postelt',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P1
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10688322,
    'Player 1 ID': 686099, 'Player 1 Name': 'Michal Vedmoch',
    'Player 2 ID': 686972, 'Player 2 Name': 'Kyryl Darin',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P1
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10688383,
    'Player 1 ID': 639253, 'Player 1 Name': 'Michal Zahradka',
    'Player 2 ID': 1149975, 'Player 2 Name': 'Jan Mecl Jr',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688384,
    'Player 1 ID': 688067, 'Player 1 Name': 'Tadeas Zika',
    'Player 2 ID': 683470, 'Player 2 Name': 'Petr Kalias',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10689214,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 881949, 'Player 2 Name': 'Tomas Barta',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10688386,
    'Player 1 ID': 561176, 'Player 1 Name': 'Jiri Ruzicka',
    'Player 2 ID': 976107, 'Player 2 Name': 'Dan Volhejn',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682318,
    'Player 1 ID': 683470, 'Player 1 Name': 'Petr Kalias',
    'Player 2 ID': 1168056, 'Player 2 Name': 'Michal Bazalka',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P1
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10688387,
    'Player 1 ID': 1030283, 'Player 1 Name': 'Tomas Vinter',
    'Player 2 ID': 689390, 'Player 2 Name': 'Marek Kostal',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10682306,
    'Player 1 ID': 741786, 'Player 1 Name': 'Ondrej Svacha',
    'Player 2 ID': 976107, 'Player 2 Name': 'Dan Volhejn',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682199,
    'Player 1 ID': 798958, 'Player 1 Name': 'Jiri Zuzanek',
    'Player 2 ID': 358665, 'Player 2 Name': 'Marek Sedlak',
    'P1_ML': 200, # Actual Decimal: 3.00, Winner: P2
    'P2_ML': -275  # Actual Decimal: 1.36
},
{
    'Match ID': 10682254,
    'Player 1 ID': 373963, 'Player 1 Name': 'Tomas Janata',
    'Player 2 ID': 387428, 'Player 2 Name': 'Daniel Tuma',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P2
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682304,
    'Player 1 ID': 1088851, 'Player 1 Name': 'Oleksandr Kolisnyk',
    'Player 2 ID': 741786, 'Player 2 Name': 'Ondrej Svacha',
    'P1_ML': 150, # Actual Decimal: 2.50, Winner: P1
    'P2_ML': -200  # Actual Decimal: 1.50
},
{
    'Match ID': 10688319,
    'Player 1 ID': 1151479, 'Player 1 Name': 'BA Hoang Tai Nguyen',
    'Player 2 ID': 847624, 'Player 2 Name': 'Lukas Malek',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10682286,
    'Player 1 ID': 976107, 'Player 1 Name': 'Dan Volhejn',
    'Player 2 ID': 561176, 'Player 2 Name': 'Jiri Ruzicka',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682305,
    'Player 1 ID': 387783, 'Player 1 Name': 'Tomas Postelt',
    'Player 2 ID': 639253, 'Player 2 Name': 'Michal Zahradka',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682323,
    'Player 1 ID': 1159459, 'Player 1 Name': 'Roman Guliak',
    'Player 2 ID': 686100, 'Player 2 Name': 'Jakub Levicky',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P2
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10687619,
    'Player 1 ID': 548705, 'Player 1 Name': 'Jiri Plachy',
    'Player 2 ID': 881949, 'Player 2 Name': 'Tomas Barta',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682251,
    'Player 1 ID': 1163803, 'Player 1 Name': 'David Mikula',
    'Player 2 ID': 1159459, 'Player 2 Name': 'Roman Guliak',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P2
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10682282,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 548705, 'Player 2 Name': 'Jiri Plachy',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682209,
    'Player 1 ID': 338586, 'Player 1 Name': 'Marek Fabini',
    'Player 2 ID': 685221, 'Player 2 Name': 'Jiri Nesnera',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10689216,
    'Player 1 ID': 685221, 'Player 1 Name': 'Jiri Nesnera',
    'Player 2 ID': 338586, 'Player 2 Name': 'Marek Fabini',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P1
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10682294,
    'Player 1 ID': 342540, 'Player 1 Name': 'Matous Klimenta',
    'Player 2 ID': 546478, 'Player 2 Name': 'Jan Sucharda',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682231,
    'Player 1 ID': 1099292, 'Player 1 Name': 'Jiri Dedek',
    'Player 2 ID': 781022, 'Player 2 Name': 'Milan Fisera',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10689755,
    'Player 1 ID': 781022, 'Player 1 Name': 'Milan Fisera',
    'Player 2 ID': 1099292, 'Player 2 Name': 'Jiri Dedek',
    'P1_ML': -250, # Actual Decimal: 1.40, Winner: P1
    'P2_ML': 175  # Actual Decimal: 2.75
},
{
    'Match ID': 10689753,
    'Player 1 ID': 339257, 'Player 1 Name': 'Petr Bradach',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10683001,
    'Player 1 ID': 1015754, 'Player 1 Name': 'Marek Chlebecek',
    'Player 2 ID': 1122137, 'Player 2 Name': 'Karel Kapras',
    'P1_ML': 225, # Actual Decimal: 3.25, Winner: P1
    'P2_ML': -300  # Actual Decimal: 1.33
},
{
    'Match ID': 10682203,
    'Player 1 ID': 1122137, 'Player 1 Name': 'Karel Kapras',
    'Player 2 ID': 685556, 'Player 2 Name': 'Lukas Tonar',
    'P1_ML': -300, # Actual Decimal: 1.33, Winner: P1
    'P2_ML': 225  # Actual Decimal: 3.25
},
{
    'Match ID': 10682252,
    'Player 1 ID': 1149975, 'Player 1 Name': 'Jan Mecl Jr',
    'Player 2 ID': 339257, 'Player 2 Name': 'Petr Bradach',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10682266,
    'Player 1 ID': 770057, 'Player 1 Name': 'Josef Pelikan',
    'Player 2 ID': 874419, 'Player 2 Name': 'Tomas Regner',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682283,
    'Player 1 ID': 1099292, 'Player 1 Name': 'Jiri Dedek',
    'Player 2 ID': 686099, 'Player 2 Name': 'Michal Vedmoch',
    'P1_ML': -400, # Actual Decimal: 1.25, Winner: P1
    'P2_ML': 275  # Actual Decimal: 3.75
},
{
    'Match ID': 10682995,
    'Player 1 ID': 765084, 'Player 1 Name': 'Radomir Vidlicka',
    'Player 2 ID': 1162474, 'Player 2 Name': 'David Heczko',
    'P1_ML': -250, # Actual Decimal: 1.40, Winner: P1
    'P2_ML': 175  # Actual Decimal: 2.75
},
{
    'Match ID': 10682255,
    'Player 1 ID': 699464, 'Player 1 Name': 'Martin Stefek',
    'Player 2 ID': 765084, 'Player 2 Name': 'Radomir Vidlicka',
    'P1_ML': 225, # Actual Decimal: 3.25, Winner: P2
    'P2_ML': -300  # Actual Decimal: 1.33
},
{
    'Match ID': 10682234,
    'Player 1 ID': 355559, 'Player 1 Name': 'Michal Regner',
    'Player 2 ID': 342540, 'Player 2 Name': 'Matous Klimenta',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P1
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10682211,
    'Player 1 ID': 1182742, 'Player 1 Name': 'Vitezslav Bosak',
    'Player 2 ID': 699464, 'Player 2 Name': 'Martin Stefek',
    'P1_ML': -400, # Actual Decimal: 1.25, Winner: P1
    'P2_ML': 275  # Actual Decimal: 3.75
},
{
    'Match ID': 10689751,
    'Player 1 ID': 685556, 'Player 1 Name': 'Lukas Tonar',
    'Player 2 ID': 1122137, 'Player 2 Name': 'Karel Kapras',
    'P1_ML': 275, # Actual Decimal: 3.75, Winner: P1
    'P2_ML': -400  # Actual Decimal: 1.25
},
{
    'Match ID': 10682320,
    'Player 1 ID': 607888, 'Player 1 Name': 'Lubor Sulava',
    'Player 2 ID': 689521, 'Player 2 Name': 'Jaroslav Strnad 1964',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682989,
    'Player 1 ID': 781022, 'Player 1 Name': 'Milan Fisera',
    'Player 2 ID': 1107438, 'Player 2 Name': 'Pavel Vondra',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682289,
    'Player 1 ID': 689390, 'Player 1 Name': 'Marek Kostal',
    'Player 2 ID': 548705, 'Player 2 Name': 'Jiri Plachy',
    'P1_ML': -250, # Actual Decimal: 1.40, Winner: P1
    'P2_ML': 175  # Actual Decimal: 2.75
},
{
    'Match ID': 10682267,
    'Player 1 ID': 687549, 'Player 1 Name': 'Michal Vavrecka',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': 200, # Actual Decimal: 3.00, Winner: P2
    'P2_ML': -275  # Actual Decimal: 1.36
},
{
    'Match ID': 10689217,
    'Player 1 ID': 561176, 'Player 1 Name': 'Jiri Ruzicka',
    'Player 2 ID': 1088851, 'Player 2 Name': 'Oleksandr Kolisnyk',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682268,
    'Player 1 ID': 559309, 'Player 1 Name': 'Lukas Zeman',
    'Player 2 ID': 341888, 'Player 2 Name': 'Martin Huk',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10682249,
    'Player 1 ID': 888755, 'Player 1 Name': 'Vratislav Petracek',
    'Player 2 ID': 922416, 'Player 2 Name': 'Jakub Vales',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10682232,
    'Player 1 ID': 765084, 'Player 1 Name': 'Radomir Vidlicka',
    'Player 2 ID': 1182742, 'Player 2 Name': 'Vitezslav Bosak',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682310,
    'Player 1 ID': 770057, 'Player 1 Name': 'Josef Pelikan',
    'Player 2 ID': 607888, 'Player 2 Name': 'Lubor Sulava',
    'P1_ML': 137, # Actual Decimal: 2.38, Winner: P2
    'P2_ML': -187  # Actual Decimal: 1.53
},
{
    'Match ID': 10682329,
    'Player 1 ID': 1122137, 'Player 1 Name': 'Karel Kapras',
    'Player 2 ID': 797186, 'Player 2 Name': 'Simon Kadavy',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682333,
    'Player 1 ID': 1149975, 'Player 1 Name': 'Jan Mecl Jr',
    'Player 2 ID': 687549, 'Player 2 Name': 'Michal Vavrecka',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682217,
    'Player 1 ID': 1162474, 'Player 1 Name': 'David Heczko',
    'Player 2 ID': 699464, 'Player 2 Name': 'Martin Stefek',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682222,
    'Player 1 ID': 339257, 'Player 1 Name': 'Petr Bradach',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P1
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10682293,
    'Player 1 ID': 874419, 'Player 1 Name': 'Tomas Regner',
    'Player 2 ID': 689521, 'Player 2 Name': 'Jaroslav Strnad 1964',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P1
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10682987,
    'Player 1 ID': 685556, 'Player 1 Name': 'Lukas Tonar',
    'Player 2 ID': 1015754, 'Player 2 Name': 'Marek Chlebecek',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682988,
    'Player 1 ID': 686099, 'Player 1 Name': 'Michal Vedmoch',
    'Player 2 ID': 1107438, 'Player 2 Name': 'Pavel Vondra',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P1
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10682202,
    'Player 1 ID': 781022, 'Player 1 Name': 'Milan Fisera',
    'Player 2 ID': 686099, 'Player 2 Name': 'Michal Vedmoch',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682307,
    'Player 1 ID': 1098619, 'Player 1 Name': 'Jaroslav Bares',
    'Player 2 ID': 553241, 'Player 2 Name': 'Martin Sobisek',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P2
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682233,
    'Player 1 ID': 797186, 'Player 1 Name': 'Simon Kadavy',
    'Player 2 ID': 685556, 'Player 2 Name': 'Lukas Tonar',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682265,
    'Player 1 ID': 687549, 'Player 1 Name': 'Michal Vavrecka',
    'Player 2 ID': 339257, 'Player 2 Name': 'Petr Bradach',
    'P1_ML': 200, # Actual Decimal: 3.00, Winner: P1
    'P2_ML': -275  # Actual Decimal: 1.36
},
{
    'Match ID': 10683000,
    'Player 1 ID': 1182742, 'Player 1 Name': 'Vitezslav Bosak',
    'Player 2 ID': 1162474, 'Player 2 Name': 'David Heczko',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682223,
    'Player 1 ID': 689521, 'Player 1 Name': 'Jaroslav Strnad 1964',
    'Player 2 ID': 770057, 'Player 2 Name': 'Josef Pelikan',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P1
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10682296,
    'Player 1 ID': 559309, 'Player 1 Name': 'Lukas Zeman',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682295,
    'Player 1 ID': 1098619, 'Player 1 Name': 'Jaroslav Bares',
    'Player 2 ID': 380220, 'Player 2 Name': 'Patrik Pycha',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682197,
    'Player 1 ID': 355559, 'Player 1 Name': 'Michal Regner',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682321,
    'Player 1 ID': 553241, 'Player 1 Name': 'Martin Sobisek',
    'Player 2 ID': 1015754, 'Player 2 Name': 'Marek Chlebecek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682224,
    'Player 1 ID': 341888, 'Player 1 Name': 'Martin Huk',
    'Player 2 ID': 687549, 'Player 2 Name': 'Michal Vavrecka',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10683002,
    'Player 1 ID': 380220, 'Player 1 Name': 'Patrik Pycha',
    'Player 2 ID': 1133447, 'Player 2 Name': 'Petr Sebera',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P1
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10682985,
    'Player 1 ID': 1090473, 'Player 1 Name': 'Karel Schopf',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682245,
    'Player 1 ID': 607888, 'Player 1 Name': 'Lubor Sulava',
    'Player 2 ID': 874419, 'Player 2 Name': 'Tomas Regner',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P2
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682279,
    'Player 1 ID': 1149975, 'Player 1 Name': 'Jan Mecl Jr',
    'Player 2 ID': 639253, 'Player 2 Name': 'Michal Zahradka',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682210,
    'Player 1 ID': 797186, 'Player 1 Name': 'Simon Kadavy',
    'Player 2 ID': 1015754, 'Player 2 Name': 'Marek Chlebecek',
    'P1_ML': 250, # Actual Decimal: 3.50, Winner: P1
    'P2_ML': -350  # Actual Decimal: 1.28
},
{
    'Match ID': 10682262,
    'Player 1 ID': 847624, 'Player 1 Name': 'Lukas Malek',
    'Player 2 ID': 686972, 'Player 2 Name': 'Kyryl Darin',
    'P1_ML': 120, # Actual Decimal: 2.20, Winner: P2
    'P2_ML': -162  # Actual Decimal: 1.61
},
{
    'Match ID': 10686389,
    'Player 1 ID': 339048, 'Player 1 Name': 'Karel Brozhik',
    'Player 2 ID': 600584, 'Player 2 Name': 'Denis Hofman',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682248,
    'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma',
    'Player 2 ID': 1166071, 'Player 2 Name': 'Jiri Havel',
    'P1_ML': -300, # Actual Decimal: 1.33, Winner: P2
    'P2_ML': 225  # Actual Decimal: 3.25
},
{
    'Match ID': 10682281,
    'Player 1 ID': 341166, 'Player 1 Name': 'Milan Klement',
    'Player 2 ID': 355559, 'Player 2 Name': 'Michal Regner',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P2
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10682308,
    'Player 1 ID': 799638, 'Player 1 Name': 'Jindrich Vrba',
    'Player 2 ID': 1013721, 'Player 2 Name': 'Jiri Jira',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10682313,
    'Player 1 ID': 391465, 'Player 1 Name': 'Vaclav Dolezal',
    'Player 2 ID': 606056, 'Player 2 Name': 'Vaclav Hruska Sr',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682339,
    'Player 1 ID': 686799, 'Player 1 Name': 'Lukas Martinak',
    'Player 2 ID': 925949, 'Player 2 Name': 'Ivan Jemelka',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682219,
    'Player 1 ID': 1122093, 'Player 1 Name': 'Petr Franc',
    'Player 2 ID': 360808, 'Player 2 Name': 'Jiri Svec',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682220,
    'Player 1 ID': 1157770, 'Player 1 Name': 'Evzen Rychlik',
    'Player 2 ID': 1171338, 'Player 2 Name': 'Josef Cabak',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682225,
    'Player 1 ID': 635749, 'Player 1 Name': 'Cesta Havrda',
    'Player 2 ID': 650467, 'Player 2 Name': 'Vladimir Kubat',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682257,
    'Player 1 ID': 339709, 'Player 1 Name': 'Jiri Louda',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682328,
    'Player 1 ID': 346257, 'Player 1 Name': 'Petr Sudek',
    'Player 2 ID': 689521, 'Player 2 Name': 'Jaroslav Strnad 1964',
    'P1_ML': -187, # Actual Decimal: 1.53, Winner: P2
    'P2_ML': 137  # Actual Decimal: 2.38
},
{
    'Match ID': 10682287,
    'Player 1 ID': 1013721, 'Player 1 Name': 'Jiri Jira',
    'Player 2 ID': 635749, 'Player 2 Name': 'Cesta Havrda',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682299,
    'Player 1 ID': 925949, 'Player 1 Name': 'Ivan Jemelka',
    'Player 2 ID': 1157770, 'Player 2 Name': 'Evzen Rychlik',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P1
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10682314,
    'Player 1 ID': 606056, 'Player 1 Name': 'Vaclav Hruska Sr',
    'Player 2 ID': 346257, 'Player 2 Name': 'Petr Sudek',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10682330,
    'Player 1 ID': 355559, 'Player 1 Name': 'Michal Regner',
    'Player 2 ID': 339709, 'Player 2 Name': 'Jiri Louda',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682335,
    'Player 1 ID': 1166071, 'Player 1 Name': 'Jiri Havel',
    'Player 2 ID': 1122093, 'Player 2 Name': 'Petr Franc',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682263,
    'Player 1 ID': 976107, 'Player 1 Name': 'Dan Volhejn',
    'Player 2 ID': 561176, 'Player 2 Name': 'Jiri Ruzicka',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P2
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682250,
    'Player 1 ID': 689521, 'Player 1 Name': 'Jaroslav Strnad 1964',
    'Player 2 ID': 391465, 'Player 2 Name': 'Vaclav Dolezal',
    'P1_ML': 200, # Actual Decimal: 3.00, Winner: P1
    'P2_ML': -275  # Actual Decimal: 1.36
},
{
    'Match ID': 10682315,
    'Player 1 ID': 650467, 'Player 1 Name': 'Vladimir Kubat',
    'Player 2 ID': 799638, 'Player 2 Name': 'Jindrich Vrba',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10686359,
    'Player 1 ID': 1007725, 'Player 1 Name': 'Jiri Grohsgott',
    'Player 2 ID': 999484, 'Player 2 Name': 'Tomas Kucera',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10686353,
    'Player 1 ID': 1169654, 'Player 1 Name': 'Vojtech Svechota',
    'Player 2 ID': 553241, 'Player 2 Name': 'Martin Sobisek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10686373,
    'Player 1 ID': 954785, 'Player 1 Name': 'Martin Sychra',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': 550, # Actual Decimal: 6.50, Winner: P2
    'P2_ML': -999  # Actual Decimal: 1.10
},
{
    'Match ID': 10686331,
    'Player 1 ID': 680338, 'Player 1 Name': 'Matej Pycha',
    'Player 2 ID': 701362, 'Player 2 Name': 'Stanislav Mazanek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676098,
    'Player 1 ID': 1169654, 'Player 1 Name': 'Vojtech Svechota',
    'Player 2 ID': 636504, 'Player 2 Name': 'Vladimir Cermak',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676135,
    'Player 1 ID': 1078065, 'Player 1 Name': 'Martin Sopko',
    'Player 2 ID': 954785, 'Player 2 Name': 'Martin Sychra',
    'P1_ML': -6666, # Actual Decimal: 1.01, Winner: P1
    'P2_ML': 1100  # Actual Decimal: 12.00
},
{
    'Match ID': 10676174,
    'Player 1 ID': 654692, 'Player 1 Name': 'Pavel Sprynar',
    'Player 2 ID': 999484, 'Player 2 Name': 'Tomas Kucera',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676154,
    'Player 1 ID': 1007725, 'Player 1 Name': 'Jiri Grohsgott',
    'Player 2 ID': 680338, 'Player 2 Name': 'Matej Pycha',
    'P1_ML': -4000, # Actual Decimal: 1.02, Winner: P1
    'P2_ML': 950  # Actual Decimal: 10.50
},
{
    'Match ID': 10676192,
    'Player 1 ID': 339048, 'Player 1 Name': 'Karel Brozhik',
    'Player 2 ID': 600584, 'Player 2 Name': 'Denis Hofman',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676202,
    'Player 1 ID': 359407, 'Player 1 Name': 'Jaroslav Prokupek',
    'Player 2 ID': 360789, 'Player 2 Name': 'Jakub Hradecky',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676122,
    'Player 1 ID': 636504, 'Player 1 Name': 'Vladimir Cermak',
    'Player 2 ID': 553241, 'Player 2 Name': 'Martin Sobisek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676155,
    'Player 1 ID': 954785, 'Player 1 Name': 'Martin Sychra',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682984,
    'Player 1 ID': 1107438, 'Player 1 Name': 'Pavel Vondra',
    'Player 2 ID': 1099292, 'Player 2 Name': 'Jiri Dedek',
    'P1_ML': 137, # Actual Decimal: 2.38, Winner: P2
    'P2_ML': -187  # Actual Decimal: 1.53
},
{
    'Match ID': 10682340,
    'Player 1 ID': 1090473, 'Player 1 Name': 'Karel Schopf',
    'Player 2 ID': 341166, 'Player 2 Name': 'Milan Klement',
    'P1_ML': 120, # Actual Decimal: 2.20, Winner: P2
    'P2_ML': -162  # Actual Decimal: 1.61
},
{
    'Match ID': 10676866,
    'Player 1 ID': 999484, 'Player 1 Name': 'Tomas Kucera',
    'Player 2 ID': 701362, 'Player 2 Name': 'Stanislav Mazanek',
    'P1_ML': -1999, # Actual Decimal: 1.05, Winner: P1
    'P2_ML': 750  # Actual Decimal: 8.50
},
{
    'Match ID': 10676867,
    'Player 1 ID': 600584, 'Player 1 Name': 'Denis Hofman',
    'Player 2 ID': 1078065, 'Player 2 Name': 'Martin Sopko',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676868,
    'Player 1 ID': 360789, 'Player 1 Name': 'Jakub Hradecky',
    'Player 2 ID': 1169654, 'Player 2 Name': 'Vojtech Svechota',
    'P1_ML': 2500, # Actual Decimal: 26.00, Winner: P2
    'P2_ML': -100000  # Actual Decimal: 1.00
},
{
    'Match ID': 10676134,
    'Player 1 ID': 636504, 'Player 1 Name': 'Vladimir Cermak',
    'Player 2 ID': 359407, 'Player 2 Name': 'Jaroslav Prokupek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676156,
    'Player 1 ID': 999484, 'Player 1 Name': 'Tomas Kucera',
    'Player 2 ID': 1007725, 'Player 2 Name': 'Jiri Grohsgott',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676157,
    'Player 1 ID': 954785, 'Player 1 Name': 'Martin Sychra',
    'Player 2 ID': 339048, 'Player 2 Name': 'Karel Brozhik',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676079,
    'Player 1 ID': 654692, 'Player 1 Name': 'Pavel Sprynar',
    'Player 2 ID': 701362, 'Player 2 Name': 'Stanislav Mazanek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676158,
    'Player 1 ID': 1169654, 'Player 1 Name': 'Vojtech Svechota',
    'Player 2 ID': 553241, 'Player 2 Name': 'Martin Sobisek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676869,
    'Player 1 ID': 1078065, 'Player 1 Name': 'Martin Sopko',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': 500, # Actual Decimal: 6.00, Winner: P2
    'P2_ML': -900  # Actual Decimal: 1.11
},
{
    'Match ID': 10686323,
    'Player 1 ID': 636504, 'Player 1 Name': 'Vladimir Cermak',
    'Player 2 ID': 360789, 'Player 2 Name': 'Jakub Hradecky',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676142,
    'Player 1 ID': 680338, 'Player 1 Name': 'Matej Pycha',
    'Player 2 ID': 654692, 'Player 2 Name': 'Pavel Sprynar',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682341,
    'Player 1 ID': 1171338, 'Player 1 Name': 'Josef Cabak',
    'Player 2 ID': 686799, 'Player 2 Name': 'Lukas Martinak',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P2
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10682238,
    'Player 1 ID': 360808, 'Player 1 Name': 'Jiri Svec',
    'Player 2 ID': 1005699, 'Player 2 Name': 'Miroslav Tuma',
    'P1_ML': 137, # Actual Decimal: 2.38, Winner: P2
    'P2_ML': -187  # Actual Decimal: 1.53
},
{
    'Match ID': 10682215,
    'Player 1 ID': 341166, 'Player 1 Name': 'Milan Klement',
    'Player 2 ID': 339709, 'Player 2 Name': 'Jiri Louda',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682253,
    'Player 1 ID': 1030283, 'Player 1 Name': 'Tomas Vinter',
    'Player 2 ID': 548705, 'Player 2 Name': 'Jiri Plachy',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682260,
    'Player 1 ID': 678236, 'Player 1 Name': 'Tomas Turek',
    'Player 2 ID': 639253, 'Player 2 Name': 'Michal Zahradka',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682213,
    'Player 1 ID': 391465, 'Player 1 Name': 'Vaclav Dolezal',
    'Player 2 ID': 346257, 'Player 2 Name': 'Petr Sudek',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P1
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10682336,
    'Player 1 ID': 688067, 'Player 1 Name': 'Tadeas Zika',
    'Player 2 ID': 1168056, 'Player 2 Name': 'Michal Bazalka',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682240,
    'Player 1 ID': 689390, 'Player 1 Name': 'Marek Kostal',
    'Player 2 ID': 1030283, 'Player 2 Name': 'Tomas Vinter',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10682244,
    'Player 1 ID': 1149975, 'Player 1 Name': 'Jan Mecl Jr',
    'Player 2 ID': 678236, 'Player 2 Name': 'Tomas Turek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682261,
    'Player 1 ID': 976107, 'Player 1 Name': 'Dan Volhejn',
    'Player 2 ID': 1088851, 'Player 2 Name': 'Oleksandr Kolisnyk',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682332,
    'Player 1 ID': 683470, 'Player 1 Name': 'Petr Kalias',
    'Player 2 ID': 688067, 'Player 2 Name': 'Tadeas Zika',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10682278,
    'Player 1 ID': 686099, 'Player 1 Name': 'Michal Vedmoch',
    'Player 2 ID': 847624, 'Player 2 Name': 'Lukas Malek',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P1
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10682239,
    'Player 1 ID': 639253, 'Player 1 Name': 'Michal Zahradka',
    'Player 2 ID': 387783, 'Player 2 Name': 'Tomas Postelt',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10682303,
    'Player 1 ID': 548705, 'Player 1 Name': 'Jiri Plachy',
    'Player 2 ID': 387428, 'Player 2 Name': 'Daniel Tuma',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682316,
    'Player 1 ID': 561176, 'Player 1 Name': 'Jiri Ruzicka',
    'Player 2 ID': 341888, 'Player 2 Name': 'Martin Huk',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682371,
    'Player 1 ID': 1168056, 'Player 1 Name': 'Michal Bazalka',
    'Player 2 ID': 697880, 'Player 2 Name': 'David Szotek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682208,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 1030283, 'Player 2 Name': 'Tomas Vinter',
    'P1_ML': 120, # Actual Decimal: 2.20, Winner: P2
    'P2_ML': -162  # Actual Decimal: 1.61
},
{
    'Match ID': 10682214,
    'Player 1 ID': 387783, 'Player 1 Name': 'Tomas Postelt',
    'Player 2 ID': 678236, 'Player 2 Name': 'Tomas Turek',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682229,
    'Player 1 ID': 341888, 'Player 1 Name': 'Martin Huk',
    'Player 2 ID': 1088851, 'Player 2 Name': 'Oleksandr Kolisnyk',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682317,
    'Player 1 ID': 697880, 'Player 1 Name': 'David Szotek',
    'Player 2 ID': 688067, 'Player 2 Name': 'Tadeas Zika',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682216,
    'Player 1 ID': 686972, 'Player 1 Name': 'Kyryl Darin',
    'Player 2 ID': 686099, 'Player 2 Name': 'Michal Vedmoch',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P2
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10682273,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 689390, 'Player 2 Name': 'Marek Kostal',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10682309,
    'Player 1 ID': 1088851, 'Player 1 Name': 'Oleksandr Kolisnyk',
    'Player 2 ID': 561176, 'Player 2 Name': 'Jiri Ruzicka',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682258,
    'Player 1 ID': 341888, 'Player 1 Name': 'Martin Huk',
    'Player 2 ID': 976107, 'Player 2 Name': 'Dan Volhejn',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682226,
    'Player 1 ID': 799638, 'Player 1 Name': 'Jindrich Vrba',
    'Player 2 ID': 635749, 'Player 2 Name': 'Cesta Havrda',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682288,
    'Player 1 ID': 686799, 'Player 1 Name': 'Lukas Martinak',
    'Player 2 ID': 1157770, 'Player 2 Name': 'Evzen Rychlik',
    'P1_ML': 137, # Actual Decimal: 2.38, Winner: P2
    'P2_ML': -187  # Actual Decimal: 1.53
},
{
    'Match ID': 10682342,
    'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma',
    'Player 2 ID': 1122093, 'Player 2 Name': 'Petr Franc',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682198,
    'Player 1 ID': 1013721, 'Player 1 Name': 'Jiri Jira',
    'Player 2 ID': 650467, 'Player 2 Name': 'Vladimir Kubat',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P2
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682227,
    'Player 1 ID': 1166071, 'Player 1 Name': 'Jiri Havel',
    'Player 2 ID': 360808, 'Player 2 Name': 'Jiri Svec',
    'P1_ML': -275, # Actual Decimal: 1.36, Winner: P2
    'P2_ML': 200  # Actual Decimal: 3.00
},
{
    'Match ID': 10682259,
    'Player 1 ID': 387783, 'Player 1 Name': 'Tomas Postelt',
    'Player 2 ID': 1149975, 'Player 2 Name': 'Jan Mecl Jr',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682331,
    'Player 1 ID': 606056, 'Player 1 Name': 'Vaclav Hruska Sr',
    'Player 2 ID': 689521, 'Player 2 Name': 'Jaroslav Strnad 1964',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10682337,
    'Player 1 ID': 925949, 'Player 1 Name': 'Ivan Jemelka',
    'Player 2 ID': 1171338, 'Player 2 Name': 'Josef Cabak',
    'P1_ML': 150, # Actual Decimal: 2.50, Winner: P2
    'P2_ML': -200  # Actual Decimal: 1.50
},
{
    'Match ID': 10687364,
    'Player 1 ID': 1122093, 'Player 1 Name': 'Petr Franc',
    'Player 2 ID': 360808, 'Player 2 Name': 'Jiri Svec',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10687365,
    'Player 1 ID': 1090473, 'Player 1 Name': 'Karel Schopf',
    'Player 2 ID': 339709, 'Player 2 Name': 'Jiri Louda',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682228,
    'Player 1 ID': 355559, 'Player 1 Name': 'Michal Regner',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687367,
    'Player 1 ID': 1013721, 'Player 1 Name': 'Jiri Jira',
    'Player 2 ID': 635749, 'Player 2 Name': 'Cesta Havrda',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P1
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10682207,
    'Player 1 ID': 697880, 'Player 1 Name': 'David Szotek',
    'Player 2 ID': 683470, 'Player 2 Name': 'Petr Kalias',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P2
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10687423,
    'Player 1 ID': 355559, 'Player 1 Name': 'Michal Regner',
    'Player 2 ID': 341166, 'Player 2 Name': 'Milan Klement',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P2
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10682325,
    'Player 1 ID': 888755, 'Player 1 Name': 'Vratislav Petracek',
    'Player 2 ID': 1149975, 'Player 2 Name': 'Jan Mecl Jr',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P1
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10687418,
    'Player 1 ID': 925949, 'Player 1 Name': 'Ivan Jemelka',
    'Player 2 ID': 1171338, 'Player 2 Name': 'Josef Cabak',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P2
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10687417,
    'Player 1 ID': 689521, 'Player 1 Name': 'Jaroslav Strnad 1964',
    'Player 2 ID': 606056, 'Player 2 Name': 'Vaclav Hruska Sr',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P2
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10687411,
    'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma',
    'Player 2 ID': 1166071, 'Player 2 Name': 'Jiri Havel',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P1
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10687419,
    'Player 1 ID': 799638, 'Player 1 Name': 'Jindrich Vrba',
    'Player 2 ID': 650467, 'Player 2 Name': 'Vladimir Kubat',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P2
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10683001,
    'Player 1 ID': 1015754, 'Player 1 Name': 'Marek Chlebecek',
    'Player 2 ID': 1122137, 'Player 2 Name': 'Karel Kapras',
    'P1_ML': 225, # Actual Decimal: 3.25, Winner: P1
    'P2_ML': -300  # Actual Decimal: 1.33
},
{
    'Match ID': 10682245,
    'Player 1 ID': 607888, 'Player 1 Name': 'Lubor Sulava',
    'Player 2 ID': 874419, 'Player 2 Name': 'Tomas Regner',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P2
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682265,
    'Player 1 ID': 687549, 'Player 1 Name': 'Michal Vavrecka',
    'Player 2 ID': 339257, 'Player 2 Name': 'Petr Bradach',
    'P1_ML': 200, # Actual Decimal: 3.00, Winner: P1
    'P2_ML': -275  # Actual Decimal: 1.36
},
{
    'Match ID': 10683000,
    'Player 1 ID': 1182742, 'Player 1 Name': 'Vitezslav Bosak',
    'Player 2 ID': 1162474, 'Player 2 Name': 'David Heczko',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682223,
    'Player 1 ID': 689521, 'Player 1 Name': 'Jaroslav Strnad 1964',
    'Player 2 ID': 770057, 'Player 2 Name': 'Josef Pelikan',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P1
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10682255,
    'Player 1 ID': 699464, 'Player 1 Name': 'Martin Stefek',
    'Player 2 ID': 765084, 'Player 2 Name': 'Radomir Vidlicka',
    'P1_ML': 225, # Actual Decimal: 3.25, Winner: P2
    'P2_ML': -300  # Actual Decimal: 1.33
},
{
    'Match ID': 10682325,
    'Player 1 ID': 888755, 'Player 1 Name': 'Vratislav Petracek',
    'Player 2 ID': 1149975, 'Player 2 Name': 'Jan Mecl Jr',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P1
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10682995,
    'Player 1 ID': 765084, 'Player 1 Name': 'Radomir Vidlicka',
    'Player 2 ID': 1162474, 'Player 2 Name': 'David Heczko',
    'P1_ML': -250, # Actual Decimal: 1.40, Winner: P1
    'P2_ML': 175  # Actual Decimal: 2.75
},
{
    'Match ID': 10682283,
    'Player 1 ID': 1099292, 'Player 1 Name': 'Jiri Dedek',
    'Player 2 ID': 686099, 'Player 2 Name': 'Michal Vedmoch',
    'P1_ML': -400, # Actual Decimal: 1.25, Winner: P1
    'P2_ML': 275  # Actual Decimal: 3.75
},
{
    'Match ID': 10682266,
    'Player 1 ID': 770057, 'Player 1 Name': 'Josef Pelikan',
    'Player 2 ID': 874419, 'Player 2 Name': 'Tomas Regner',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682984,
    'Player 1 ID': 1107438, 'Player 1 Name': 'Pavel Vondra',
    'Player 2 ID': 1099292, 'Player 2 Name': 'Jiri Dedek',
    'P1_ML': 137, # Actual Decimal: 2.38, Winner: P2
    'P2_ML': -187  # Actual Decimal: 1.53
},
{
    'Match ID': 10682252,
    'Player 1 ID': 1149975, 'Player 1 Name': 'Jan Mecl Jr',
    'Player 2 ID': 339257, 'Player 2 Name': 'Petr Bradach',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10682233,
    'Player 1 ID': 797186, 'Player 1 Name': 'Simon Kadavy',
    'Player 2 ID': 685556, 'Player 2 Name': 'Lukas Tonar',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682203,
    'Player 1 ID': 1122137, 'Player 1 Name': 'Karel Kapras',
    'Player 2 ID': 685556, 'Player 2 Name': 'Lukas Tonar',
    'P1_ML': -300, # Actual Decimal: 1.33, Winner: P1
    'P2_ML': 225  # Actual Decimal: 3.25
},
{
    'Match ID': 10687643,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 881949, 'Player 2 Name': 'Tomas Barta',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682211,
    'Player 1 ID': 1182742, 'Player 1 Name': 'Vitezslav Bosak',
    'Player 2 ID': 699464, 'Player 2 Name': 'Martin Stefek',
    'P1_ML': -400, # Actual Decimal: 1.25, Winner: P1
    'P2_ML': 275  # Actual Decimal: 3.75
},
{
    'Match ID': 10682217,
    'Player 1 ID': 1162474, 'Player 1 Name': 'David Heczko',
    'Player 2 ID': 699464, 'Player 2 Name': 'Martin Stefek',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682333,
    'Player 1 ID': 1149975, 'Player 1 Name': 'Jan Mecl Jr',
    'Player 2 ID': 687549, 'Player 2 Name': 'Michal Vavrecka',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682329,
    'Player 1 ID': 1122137, 'Player 1 Name': 'Karel Kapras',
    'Player 2 ID': 797186, 'Player 2 Name': 'Simon Kadavy',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682310,
    'Player 1 ID': 770057, 'Player 1 Name': 'Josef Pelikan',
    'Player 2 ID': 607888, 'Player 2 Name': 'Lubor Sulava',
    'P1_ML': 137, # Actual Decimal: 2.38, Winner: P2
    'P2_ML': -187  # Actual Decimal: 1.53
},
{
    'Match ID': 10682232,
    'Player 1 ID': 765084, 'Player 1 Name': 'Radomir Vidlicka',
    'Player 2 ID': 1182742, 'Player 2 Name': 'Vitezslav Bosak',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682231,
    'Player 1 ID': 1099292, 'Player 1 Name': 'Jiri Dedek',
    'Player 2 ID': 781022, 'Player 2 Name': 'Milan Fisera',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10689217,
    'Player 1 ID': 561176, 'Player 1 Name': 'Jiri Ruzicka',
    'Player 2 ID': 1088851, 'Player 2 Name': 'Oleksandr Kolisnyk',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10689216,
    'Player 1 ID': 685221, 'Player 1 Name': 'Jiri Nesnera',
    'Player 2 ID': 338586, 'Player 2 Name': 'Marek Fabini',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P1
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10689214,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 881949, 'Player 2 Name': 'Tomas Barta',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10689213,
    'Player 1 ID': 380220, 'Player 1 Name': 'Patrik Pycha',
    'Player 2 ID': 358665, 'Player 2 Name': 'Marek Sedlak',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P2
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10682222,
    'Player 1 ID': 339257, 'Player 1 Name': 'Petr Bradach',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P1
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10682210,
    'Player 1 ID': 797186, 'Player 1 Name': 'Simon Kadavy',
    'Player 2 ID': 1015754, 'Player 2 Name': 'Marek Chlebecek',
    'P1_ML': 250, # Actual Decimal: 3.50, Winner: P1
    'P2_ML': -350  # Actual Decimal: 1.28
},
{
    'Match ID': 10682293,
    'Player 1 ID': 874419, 'Player 1 Name': 'Tomas Regner',
    'Player 2 ID': 689521, 'Player 2 Name': 'Jaroslav Strnad 1964',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P1
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10682249,
    'Player 1 ID': 888755, 'Player 1 Name': 'Vratislav Petracek',
    'Player 2 ID': 922416, 'Player 2 Name': 'Jakub Vales',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10682267,
    'Player 1 ID': 687549, 'Player 1 Name': 'Michal Vavrecka',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': 200, # Actual Decimal: 3.00, Winner: P2
    'P2_ML': -275  # Actual Decimal: 1.36
},
{
    'Match ID': 10682320,
    'Player 1 ID': 607888, 'Player 1 Name': 'Lubor Sulava',
    'Player 2 ID': 689521, 'Player 2 Name': 'Jaroslav Strnad 1964',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682989,
    'Player 1 ID': 781022, 'Player 1 Name': 'Milan Fisera',
    'Player 2 ID': 1107438, 'Player 2 Name': 'Pavel Vondra',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10689751,
    'Player 1 ID': 685556, 'Player 1 Name': 'Lukas Tonar',
    'Player 2 ID': 1122137, 'Player 2 Name': 'Karel Kapras',
    'P1_ML': 275, # Actual Decimal: 3.75, Winner: P1
    'P2_ML': -400  # Actual Decimal: 1.25
},
{
    'Match ID': 10689753,
    'Player 1 ID': 339257, 'Player 1 Name': 'Petr Bradach',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10689755,
    'Player 1 ID': 781022, 'Player 1 Name': 'Milan Fisera',
    'Player 2 ID': 1099292, 'Player 2 Name': 'Jiri Dedek',
    'P1_ML': -250, # Actual Decimal: 1.40, Winner: P1
    'P2_ML': 175  # Actual Decimal: 2.75
},
{
    'Match ID': 10682268,
    'Player 1 ID': 559309, 'Player 1 Name': 'Lukas Zeman',
    'Player 2 ID': 341888, 'Player 2 Name': 'Martin Huk',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10682307,
    'Player 1 ID': 1098619, 'Player 1 Name': 'Jaroslav Bares',
    'Player 2 ID': 553241, 'Player 2 Name': 'Martin Sobisek',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P2
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682987,
    'Player 1 ID': 685556, 'Player 1 Name': 'Lukas Tonar',
    'Player 2 ID': 1015754, 'Player 2 Name': 'Marek Chlebecek',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682234,
    'Player 1 ID': 355559, 'Player 1 Name': 'Michal Regner',
    'Player 2 ID': 342540, 'Player 2 Name': 'Matous Klimenta',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P1
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10687411,
    'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma',
    'Player 2 ID': 1166071, 'Player 2 Name': 'Jiri Havel',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P1
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10683002,
    'Player 1 ID': 380220, 'Player 1 Name': 'Patrik Pycha',
    'Player 2 ID': 1133447, 'Player 2 Name': 'Petr Sebera',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P1
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10682198,
    'Player 1 ID': 1013721, 'Player 1 Name': 'Jiri Jira',
    'Player 2 ID': 650467, 'Player 2 Name': 'Vladimir Kubat',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P2
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682342,
    'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma',
    'Player 2 ID': 1122093, 'Player 2 Name': 'Petr Franc',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682288,
    'Player 1 ID': 686799, 'Player 1 Name': 'Lukas Martinak',
    'Player 2 ID': 1157770, 'Player 2 Name': 'Evzen Rychlik',
    'P1_ML': 137, # Actual Decimal: 2.38, Winner: P2
    'P2_ML': -187  # Actual Decimal: 1.53
},
{
    'Match ID': 10682226,
    'Player 1 ID': 799638, 'Player 1 Name': 'Jindrich Vrba',
    'Player 2 ID': 635749, 'Player 2 Name': 'Cesta Havrda',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682215,
    'Player 1 ID': 341166, 'Player 1 Name': 'Milan Klement',
    'Player 2 ID': 339709, 'Player 2 Name': 'Jiri Louda',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682213,
    'Player 1 ID': 391465, 'Player 1 Name': 'Vaclav Dolezal',
    'Player 2 ID': 346257, 'Player 2 Name': 'Petr Sudek',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P1
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10682227,
    'Player 1 ID': 1166071, 'Player 1 Name': 'Jiri Havel',
    'Player 2 ID': 360808, 'Player 2 Name': 'Jiri Svec',
    'P1_ML': -275, # Actual Decimal: 1.36, Winner: P2
    'P2_ML': 200  # Actual Decimal: 3.00
},
{
    'Match ID': 10682341,
    'Player 1 ID': 1171338, 'Player 1 Name': 'Josef Cabak',
    'Player 2 ID': 686799, 'Player 2 Name': 'Lukas Martinak',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P2
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10682315,
    'Player 1 ID': 650467, 'Player 1 Name': 'Vladimir Kubat',
    'Player 2 ID': 799638, 'Player 2 Name': 'Jindrich Vrba',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10682250,
    'Player 1 ID': 689521, 'Player 1 Name': 'Jaroslav Strnad 1964',
    'Player 2 ID': 391465, 'Player 2 Name': 'Vaclav Dolezal',
    'P1_ML': 200, # Actual Decimal: 3.00, Winner: P1
    'P2_ML': -275  # Actual Decimal: 1.36
},
{
    'Match ID': 10682238,
    'Player 1 ID': 360808, 'Player 1 Name': 'Jiri Svec',
    'Player 2 ID': 1005699, 'Player 2 Name': 'Miroslav Tuma',
    'P1_ML': 137, # Actual Decimal: 2.38, Winner: P2
    'P2_ML': -187  # Actual Decimal: 1.53
},
{
    'Match ID': 10682335,
    'Player 1 ID': 1166071, 'Player 1 Name': 'Jiri Havel',
    'Player 2 ID': 1122093, 'Player 2 Name': 'Petr Franc',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682330,
    'Player 1 ID': 355559, 'Player 1 Name': 'Michal Regner',
    'Player 2 ID': 339709, 'Player 2 Name': 'Jiri Louda',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682314,
    'Player 1 ID': 606056, 'Player 1 Name': 'Vaclav Hruska Sr',
    'Player 2 ID': 346257, 'Player 2 Name': 'Petr Sudek',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10682340,
    'Player 1 ID': 1090473, 'Player 1 Name': 'Karel Schopf',
    'Player 2 ID': 341166, 'Player 2 Name': 'Milan Klement',
    'P1_ML': 120, # Actual Decimal: 2.20, Winner: P2
    'P2_ML': -162  # Actual Decimal: 1.61
},
{
    'Match ID': 10682299,
    'Player 1 ID': 925949, 'Player 1 Name': 'Ivan Jemelka',
    'Player 2 ID': 1157770, 'Player 2 Name': 'Evzen Rychlik',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P1
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10682228,
    'Player 1 ID': 355559, 'Player 1 Name': 'Michal Regner',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682337,
    'Player 1 ID': 925949, 'Player 1 Name': 'Ivan Jemelka',
    'Player 2 ID': 1171338, 'Player 2 Name': 'Josef Cabak',
    'P1_ML': 150, # Actual Decimal: 2.50, Winner: P2
    'P2_ML': -200  # Actual Decimal: 1.50
},
{
    'Match ID': 10682259,
    'Player 1 ID': 387783, 'Player 1 Name': 'Tomas Postelt',
    'Player 2 ID': 1149975, 'Player 2 Name': 'Jan Mecl Jr',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682258,
    'Player 1 ID': 341888, 'Player 1 Name': 'Martin Huk',
    'Player 2 ID': 976107, 'Player 2 Name': 'Dan Volhejn',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682207,
    'Player 1 ID': 697880, 'Player 1 Name': 'David Szotek',
    'Player 2 ID': 683470, 'Player 2 Name': 'Petr Kalias',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P2
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10687423,
    'Player 1 ID': 355559, 'Player 1 Name': 'Michal Regner',
    'Player 2 ID': 341166, 'Player 2 Name': 'Milan Klement',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P2
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10682985,
    'Player 1 ID': 1090473, 'Player 1 Name': 'Karel Schopf',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682331,
    'Player 1 ID': 606056, 'Player 1 Name': 'Vaclav Hruska Sr',
    'Player 2 ID': 689521, 'Player 2 Name': 'Jaroslav Strnad 1964',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10687418,
    'Player 1 ID': 925949, 'Player 1 Name': 'Ivan Jemelka',
    'Player 2 ID': 1171338, 'Player 2 Name': 'Josef Cabak',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P2
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10682202,
    'Player 1 ID': 781022, 'Player 1 Name': 'Milan Fisera',
    'Player 2 ID': 686099, 'Player 2 Name': 'Michal Vedmoch',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687367,
    'Player 1 ID': 1013721, 'Player 1 Name': 'Jiri Jira',
    'Player 2 ID': 635749, 'Player 2 Name': 'Cesta Havrda',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P1
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10687365,
    'Player 1 ID': 1090473, 'Player 1 Name': 'Karel Schopf',
    'Player 2 ID': 339709, 'Player 2 Name': 'Jiri Louda',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10687364,
    'Player 1 ID': 1122093, 'Player 1 Name': 'Petr Franc',
    'Player 2 ID': 360808, 'Player 2 Name': 'Jiri Svec',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10687417,
    'Player 1 ID': 689521, 'Player 1 Name': 'Jaroslav Strnad 1964',
    'Player 2 ID': 606056, 'Player 2 Name': 'Vaclav Hruska Sr',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P2
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682273,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 689390, 'Player 2 Name': 'Marek Kostal',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10682287,
    'Player 1 ID': 1013721, 'Player 1 Name': 'Jiri Jira',
    'Player 2 ID': 635749, 'Player 2 Name': 'Cesta Havrda',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682257,
    'Player 1 ID': 339709, 'Player 1 Name': 'Jiri Louda',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10676156,
    'Player 1 ID': 999484, 'Player 1 Name': 'Tomas Kucera',
    'Player 2 ID': 1007725, 'Player 2 Name': 'Jiri Grohsgott',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676134,
    'Player 1 ID': 636504, 'Player 1 Name': 'Vladimir Cermak',
    'Player 2 ID': 359407, 'Player 2 Name': 'Jaroslav Prokupek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676868,
    'Player 1 ID': 360789, 'Player 1 Name': 'Jakub Hradecky',
    'Player 2 ID': 1169654, 'Player 2 Name': 'Vojtech Svechota',
    'P1_ML': 2500, # Actual Decimal: 26.00, Winner: P2
    'P2_ML': -100000  # Actual Decimal: 1.00
},
{
    'Match ID': 10676867,
    'Player 1 ID': 600584, 'Player 1 Name': 'Denis Hofman',
    'Player 2 ID': 1078065, 'Player 2 Name': 'Martin Sopko',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676142,
    'Player 1 ID': 680338, 'Player 1 Name': 'Matej Pycha',
    'Player 2 ID': 654692, 'Player 2 Name': 'Pavel Sprynar',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676866,
    'Player 1 ID': 999484, 'Player 1 Name': 'Tomas Kucera',
    'Player 2 ID': 701362, 'Player 2 Name': 'Stanislav Mazanek',
    'P1_ML': -1999, # Actual Decimal: 1.05, Winner: P1
    'P2_ML': 750  # Actual Decimal: 8.50
},
{
    'Match ID': 10676157,
    'Player 1 ID': 954785, 'Player 1 Name': 'Martin Sychra',
    'Player 2 ID': 339048, 'Player 2 Name': 'Karel Brozhik',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676155,
    'Player 1 ID': 954785, 'Player 1 Name': 'Martin Sychra',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676202,
    'Player 1 ID': 359407, 'Player 1 Name': 'Jaroslav Prokupek',
    'Player 2 ID': 360789, 'Player 2 Name': 'Jakub Hradecky',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676192,
    'Player 1 ID': 339048, 'Player 1 Name': 'Karel Brozhik',
    'Player 2 ID': 600584, 'Player 2 Name': 'Denis Hofman',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676154,
    'Player 1 ID': 1007725, 'Player 1 Name': 'Jiri Grohsgott',
    'Player 2 ID': 680338, 'Player 2 Name': 'Matej Pycha',
    'P1_ML': -4000, # Actual Decimal: 1.02, Winner: P1
    'P2_ML': 950  # Actual Decimal: 10.50
},
{
    'Match ID': 10676174,
    'Player 1 ID': 654692, 'Player 1 Name': 'Pavel Sprynar',
    'Player 2 ID': 999484, 'Player 2 Name': 'Tomas Kucera',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676135,
    'Player 1 ID': 1078065, 'Player 1 Name': 'Martin Sopko',
    'Player 2 ID': 954785, 'Player 2 Name': 'Martin Sychra',
    'P1_ML': -6666, # Actual Decimal: 1.01, Winner: P1
    'P2_ML': 1100  # Actual Decimal: 12.00
},
{
    'Match ID': 10676098,
    'Player 1 ID': 1169654, 'Player 1 Name': 'Vojtech Svechota',
    'Player 2 ID': 636504, 'Player 2 Name': 'Vladimir Cermak',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676122,
    'Player 1 ID': 636504, 'Player 1 Name': 'Vladimir Cermak',
    'Player 2 ID': 553241, 'Player 2 Name': 'Martin Sobisek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682328,
    'Player 1 ID': 346257, 'Player 1 Name': 'Petr Sudek',
    'Player 2 ID': 689521, 'Player 2 Name': 'Jaroslav Strnad 1964',
    'P1_ML': -187, # Actual Decimal: 1.53, Winner: P2
    'P2_ML': 137  # Actual Decimal: 2.38
},
{
    'Match ID': 10676079,
    'Player 1 ID': 654692, 'Player 1 Name': 'Pavel Sprynar',
    'Player 2 ID': 701362, 'Player 2 Name': 'Stanislav Mazanek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676869,
    'Player 1 ID': 1078065, 'Player 1 Name': 'Martin Sopko',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': 500, # Actual Decimal: 6.00, Winner: P2
    'P2_ML': -900  # Actual Decimal: 1.11
},
{
    'Match ID': 10682225,
    'Player 1 ID': 635749, 'Player 1 Name': 'Cesta Havrda',
    'Player 2 ID': 650467, 'Player 2 Name': 'Vladimir Kubat',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682220,
    'Player 1 ID': 1157770, 'Player 1 Name': 'Evzen Rychlik',
    'Player 2 ID': 1171338, 'Player 2 Name': 'Josef Cabak',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682219,
    'Player 1 ID': 1122093, 'Player 1 Name': 'Petr Franc',
    'Player 2 ID': 360808, 'Player 2 Name': 'Jiri Svec',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682339,
    'Player 1 ID': 686799, 'Player 1 Name': 'Lukas Martinak',
    'Player 2 ID': 925949, 'Player 2 Name': 'Ivan Jemelka',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682313,
    'Player 1 ID': 391465, 'Player 1 Name': 'Vaclav Dolezal',
    'Player 2 ID': 606056, 'Player 2 Name': 'Vaclav Hruska Sr',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682308,
    'Player 1 ID': 799638, 'Player 1 Name': 'Jindrich Vrba',
    'Player 2 ID': 1013721, 'Player 2 Name': 'Jiri Jira',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10676158,
    'Player 1 ID': 1169654, 'Player 1 Name': 'Vojtech Svechota',
    'Player 2 ID': 553241, 'Player 2 Name': 'Martin Sobisek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682281,
    'Player 1 ID': 341166, 'Player 1 Name': 'Milan Klement',
    'Player 2 ID': 355559, 'Player 2 Name': 'Michal Regner',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P2
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10686389,
    'Player 1 ID': 339048, 'Player 1 Name': 'Karel Brozhik',
    'Player 2 ID': 600584, 'Player 2 Name': 'Denis Hofman',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10686359,
    'Player 1 ID': 1007725, 'Player 1 Name': 'Jiri Grohsgott',
    'Player 2 ID': 999484, 'Player 2 Name': 'Tomas Kucera',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10686353,
    'Player 1 ID': 1169654, 'Player 1 Name': 'Vojtech Svechota',
    'Player 2 ID': 553241, 'Player 2 Name': 'Martin Sobisek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10686373,
    'Player 1 ID': 954785, 'Player 1 Name': 'Martin Sychra',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': 550, # Actual Decimal: 6.50, Winner: P2
    'P2_ML': -999  # Actual Decimal: 1.10
},
{
    'Match ID': 10686331,
    'Player 1 ID': 680338, 'Player 1 Name': 'Matej Pycha',
    'Player 2 ID': 701362, 'Player 2 Name': 'Stanislav Mazanek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10686323,
    'Player 1 ID': 636504, 'Player 1 Name': 'Vladimir Cermak',
    'Player 2 ID': 360789, 'Player 2 Name': 'Jakub Hradecky',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682248,
    'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma',
    'Player 2 ID': 1166071, 'Player 2 Name': 'Jiri Havel',
    'P1_ML': -300, # Actual Decimal: 1.33, Winner: P2
    'P2_ML': 225  # Actual Decimal: 3.25
},
{
    'Match ID': 10682278,
    'Player 1 ID': 686099, 'Player 1 Name': 'Michal Vedmoch',
    'Player 2 ID': 847624, 'Player 2 Name': 'Lukas Malek',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P1
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10687419,
    'Player 1 ID': 799638, 'Player 1 Name': 'Jindrich Vrba',
    'Player 2 ID': 650467, 'Player 2 Name': 'Vladimir Kubat',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P2
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10682260,
    'Player 1 ID': 678236, 'Player 1 Name': 'Tomas Turek',
    'Player 2 ID': 639253, 'Player 2 Name': 'Michal Zahradka',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682243,
    'Player 1 ID': 639253, 'Player 1 Name': 'Michal Zahradka',
    'Player 2 ID': 338586, 'Player 2 Name': 'Marek Fabini',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682857,
    'Player 1 ID': 358665, 'Player 1 Name': 'Marek Sedlak',
    'Player 2 ID': 1098619, 'Player 2 Name': 'Jaroslav Bares',
    'P1_ML': -400, # Actual Decimal: 1.25, Winner: P1
    'P2_ML': 275  # Actual Decimal: 3.75
},
{
    'Match ID': 10682306,
    'Player 1 ID': 741786, 'Player 1 Name': 'Ondrej Svacha',
    'Player 2 ID': 976107, 'Player 2 Name': 'Dan Volhejn',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682291,
    'Player 1 ID': 685221, 'Player 1 Name': 'Jiri Nesnera',
    'Player 2 ID': 387783, 'Player 2 Name': 'Tomas Postelt',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P1
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10682282,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 548705, 'Player 2 Name': 'Jiri Plachy',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682301,
    'Player 1 ID': 561176, 'Player 1 Name': 'Jiri Ruzicka',
    'Player 2 ID': 1088851, 'Player 2 Name': 'Oleksandr Kolisnyk',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682251,
    'Player 1 ID': 1163803, 'Player 1 Name': 'David Mikula',
    'Player 2 ID': 1159459, 'Player 2 Name': 'Roman Guliak',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P2
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10682253,
    'Player 1 ID': 1030283, 'Player 1 Name': 'Tomas Vinter',
    'Player 2 ID': 548705, 'Player 2 Name': 'Jiri Plachy',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682305,
    'Player 1 ID': 387783, 'Player 1 Name': 'Tomas Postelt',
    'Player 2 ID': 639253, 'Player 2 Name': 'Michal Zahradka',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682286,
    'Player 1 ID': 976107, 'Player 1 Name': 'Dan Volhejn',
    'Player 2 ID': 561176, 'Player 2 Name': 'Jiri Ruzicka',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682264,
    'Player 1 ID': 1098619, 'Player 1 Name': 'Jaroslav Bares',
    'Player 2 ID': 380220, 'Player 2 Name': 'Patrik Pycha',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682304,
    'Player 1 ID': 1088851, 'Player 1 Name': 'Oleksandr Kolisnyk',
    'Player 2 ID': 741786, 'Player 2 Name': 'Ondrej Svacha',
    'P1_ML': 150, # Actual Decimal: 2.50, Winner: P1
    'P2_ML': -200  # Actual Decimal: 1.50
},
{
    'Match ID': 10687619,
    'Player 1 ID': 548705, 'Player 1 Name': 'Jiri Plachy',
    'Player 2 ID': 881949, 'Player 2 Name': 'Tomas Barta',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682254,
    'Player 1 ID': 373963, 'Player 1 Name': 'Tomas Janata',
    'Player 2 ID': 387428, 'Player 2 Name': 'Daniel Tuma',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P2
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682319,
    'Player 1 ID': 380220, 'Player 1 Name': 'Patrik Pycha',
    'Player 2 ID': 798958, 'Player 2 Name': 'Jiri Zuzanek',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P1
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10682201,
    'Player 1 ID': 798958, 'Player 1 Name': 'Jiri Zuzanek',
    'Player 2 ID': 1098619, 'Player 2 Name': 'Jaroslav Bares',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682224,
    'Player 1 ID': 341888, 'Player 1 Name': 'Martin Huk',
    'Player 2 ID': 687549, 'Player 2 Name': 'Michal Vavrecka',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682294,
    'Player 1 ID': 342540, 'Player 1 Name': 'Matous Klimenta',
    'Player 2 ID': 546478, 'Player 2 Name': 'Jan Sucharda',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682321,
    'Player 1 ID': 553241, 'Player 1 Name': 'Martin Sobisek',
    'Player 2 ID': 1015754, 'Player 2 Name': 'Marek Chlebecek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682197,
    'Player 1 ID': 355559, 'Player 1 Name': 'Michal Regner',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682295,
    'Player 1 ID': 1098619, 'Player 1 Name': 'Jaroslav Bares',
    'Player 2 ID': 380220, 'Player 2 Name': 'Patrik Pycha',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682296,
    'Player 1 ID': 559309, 'Player 1 Name': 'Lukas Zeman',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10687618,
    'Player 1 ID': 881949, 'Player 1 Name': 'Tomas Barta',
    'Player 2 ID': 373963, 'Player 2 Name': 'Tomas Janata',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P1
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10682292,
    'Player 1 ID': 685221, 'Player 1 Name': 'Jiri Nesnera',
    'Player 2 ID': 639253, 'Player 2 Name': 'Michal Zahradka',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682277,
    'Player 1 ID': 358665, 'Player 1 Name': 'Marek Sedlak',
    'Player 2 ID': 380220, 'Player 2 Name': 'Patrik Pycha',
    'P1_ML': -300, # Actual Decimal: 1.33, Winner: P2
    'P2_ML': 225  # Actual Decimal: 3.25
},
{
    'Match ID': 10682221,
    'Player 1 ID': 741786, 'Player 1 Name': 'Ondrej Svacha',
    'Player 2 ID': 561176, 'Player 2 Name': 'Jiri Ruzicka',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P2
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682334,
    'Player 1 ID': 338586, 'Player 1 Name': 'Marek Fabini',
    'Player 2 ID': 387783, 'Player 2 Name': 'Tomas Postelt',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682241,
    'Player 1 ID': 1088851, 'Player 1 Name': 'Oleksandr Kolisnyk',
    'Player 2 ID': 976107, 'Player 2 Name': 'Dan Volhejn',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682230,
    'Player 1 ID': 373963, 'Player 1 Name': 'Tomas Janata',
    'Player 2 ID': 548705, 'Player 2 Name': 'Jiri Plachy',
    'P1_ML': -187, # Actual Decimal: 1.53, Winner: P1
    'P2_ML': 137  # Actual Decimal: 2.38
},
{
    'Match ID': 10682280,
    'Player 1 ID': 1163803, 'Player 1 Name': 'David Mikula',
    'Player 2 ID': 686100, 'Player 2 Name': 'Jakub Levicky',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P2
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10682209,
    'Player 1 ID': 338586, 'Player 1 Name': 'Marek Fabini',
    'Player 2 ID': 685221, 'Player 2 Name': 'Jiri Nesnera',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682323,
    'Player 1 ID': 1159459, 'Player 1 Name': 'Roman Guliak',
    'Player 2 ID': 686100, 'Player 2 Name': 'Jakub Levicky',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P2
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10688387,
    'Player 1 ID': 1030283, 'Player 1 Name': 'Tomas Vinter',
    'Player 2 ID': 689390, 'Player 2 Name': 'Marek Kostal',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10682208,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 1030283, 'Player 2 Name': 'Tomas Vinter',
    'P1_ML': 120, # Actual Decimal: 2.20, Winner: P2
    'P2_ML': -162  # Actual Decimal: 1.61
},
{
    'Match ID': 10682371,
    'Player 1 ID': 1168056, 'Player 1 Name': 'Michal Bazalka',
    'Player 2 ID': 697880, 'Player 2 Name': 'David Szotek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682316,
    'Player 1 ID': 561176, 'Player 1 Name': 'Jiri Ruzicka',
    'Player 2 ID': 341888, 'Player 2 Name': 'Martin Huk',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682303,
    'Player 1 ID': 548705, 'Player 1 Name': 'Jiri Plachy',
    'Player 2 ID': 387428, 'Player 2 Name': 'Daniel Tuma',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682239,
    'Player 1 ID': 639253, 'Player 1 Name': 'Michal Zahradka',
    'Player 2 ID': 387783, 'Player 2 Name': 'Tomas Postelt',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10682199,
    'Player 1 ID': 798958, 'Player 1 Name': 'Jiri Zuzanek',
    'Player 2 ID': 358665, 'Player 2 Name': 'Marek Sedlak',
    'P1_ML': 200, # Actual Decimal: 3.00, Winner: P2
    'P2_ML': -275  # Actual Decimal: 1.36
},
{
    'Match ID': 10682214,
    'Player 1 ID': 387783, 'Player 1 Name': 'Tomas Postelt',
    'Player 2 ID': 678236, 'Player 2 Name': 'Tomas Turek',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682261,
    'Player 1 ID': 976107, 'Player 1 Name': 'Dan Volhejn',
    'Player 2 ID': 1088851, 'Player 2 Name': 'Oleksandr Kolisnyk',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682244,
    'Player 1 ID': 1149975, 'Player 1 Name': 'Jan Mecl Jr',
    'Player 2 ID': 678236, 'Player 2 Name': 'Tomas Turek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682240,
    'Player 1 ID': 689390, 'Player 1 Name': 'Marek Kostal',
    'Player 2 ID': 1030283, 'Player 2 Name': 'Tomas Vinter',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10682336,
    'Player 1 ID': 688067, 'Player 1 Name': 'Tadeas Zika',
    'Player 2 ID': 1168056, 'Player 2 Name': 'Michal Bazalka',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682309,
    'Player 1 ID': 1088851, 'Player 1 Name': 'Oleksandr Kolisnyk',
    'Player 2 ID': 561176, 'Player 2 Name': 'Jiri Ruzicka',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682332,
    'Player 1 ID': 683470, 'Player 1 Name': 'Petr Kalias',
    'Player 2 ID': 688067, 'Player 2 Name': 'Tadeas Zika',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10682229,
    'Player 1 ID': 341888, 'Player 1 Name': 'Martin Huk',
    'Player 2 ID': 1088851, 'Player 2 Name': 'Oleksandr Kolisnyk',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682216,
    'Player 1 ID': 686972, 'Player 1 Name': 'Kyryl Darin',
    'Player 2 ID': 686099, 'Player 2 Name': 'Michal Vedmoch',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P2
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10682317,
    'Player 1 ID': 697880, 'Player 1 Name': 'David Szotek',
    'Player 2 ID': 688067, 'Player 2 Name': 'Tadeas Zika',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10688386,
    'Player 1 ID': 561176, 'Player 1 Name': 'Jiri Ruzicka',
    'Player 2 ID': 976107, 'Player 2 Name': 'Dan Volhejn',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10688384,
    'Player 1 ID': 688067, 'Player 1 Name': 'Tadeas Zika',
    'Player 2 ID': 683470, 'Player 2 Name': 'Petr Kalias',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10688383,
    'Player 1 ID': 639253, 'Player 1 Name': 'Michal Zahradka',
    'Player 2 ID': 1149975, 'Player 2 Name': 'Jan Mecl Jr',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688322,
    'Player 1 ID': 686099, 'Player 1 Name': 'Michal Vedmoch',
    'Player 2 ID': 686972, 'Player 2 Name': 'Kyryl Darin',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P1
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10682988,
    'Player 1 ID': 686099, 'Player 1 Name': 'Michal Vedmoch',
    'Player 2 ID': 1107438, 'Player 2 Name': 'Pavel Vondra',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P1
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10688319,
    'Player 1 ID': 1151479, 'Player 1 Name': 'BA Hoang Tai Nguyen',
    'Player 2 ID': 847624, 'Player 2 Name': 'Lukas Malek',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10682318,
    'Player 1 ID': 683470, 'Player 1 Name': 'Petr Kalias',
    'Player 2 ID': 1168056, 'Player 2 Name': 'Michal Bazalka',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P1
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10682289,
    'Player 1 ID': 689390, 'Player 1 Name': 'Marek Kostal',
    'Player 2 ID': 548705, 'Player 2 Name': 'Jiri Plachy',
    'P1_ML': -250, # Actual Decimal: 1.40, Winner: P1
    'P2_ML': 175  # Actual Decimal: 2.75
},
{
    'Match ID': 10682279,
    'Player 1 ID': 1149975, 'Player 1 Name': 'Jan Mecl Jr',
    'Player 2 ID': 639253, 'Player 2 Name': 'Michal Zahradka',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682263,
    'Player 1 ID': 976107, 'Player 1 Name': 'Dan Volhejn',
    'Player 2 ID': 561176, 'Player 2 Name': 'Jiri Ruzicka',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P2
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682262,
    'Player 1 ID': 847624, 'Player 1 Name': 'Lukas Malek',
    'Player 2 ID': 686972, 'Player 2 Name': 'Kyryl Darin',
    'P1_ML': 120, # Actual Decimal: 2.20, Winner: P2
    'P2_ML': -162  # Actual Decimal: 1.61
},
{
    'Match ID': 10683001,
    'Player 1 ID': 1015754, 'Player 1 Name': 'Marek Chlebecek',
    'Player 2 ID': 1122137, 'Player 2 Name': 'Karel Kapras',
    'P1_ML': 225, # Actual Decimal: 3.25, Winner: P1
    'P2_ML': -300  # Actual Decimal: 1.33
},
{
    'Match ID': 10682245,
    'Player 1 ID': 607888, 'Player 1 Name': 'Lubor Sulava',
    'Player 2 ID': 874419, 'Player 2 Name': 'Tomas Regner',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P2
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682265,
    'Player 1 ID': 687549, 'Player 1 Name': 'Michal Vavrecka',
    'Player 2 ID': 339257, 'Player 2 Name': 'Petr Bradach',
    'P1_ML': 200, # Actual Decimal: 3.00, Winner: P1
    'P2_ML': -275  # Actual Decimal: 1.36
},
{
    'Match ID': 10683000,
    'Player 1 ID': 1182742, 'Player 1 Name': 'Vitezslav Bosak',
    'Player 2 ID': 1162474, 'Player 2 Name': 'David Heczko',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682223,
    'Player 1 ID': 689521, 'Player 1 Name': 'Jaroslav Strnad 1964',
    'Player 2 ID': 770057, 'Player 2 Name': 'Josef Pelikan',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P1
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10682255,
    'Player 1 ID': 699464, 'Player 1 Name': 'Martin Stefek',
    'Player 2 ID': 765084, 'Player 2 Name': 'Radomir Vidlicka',
    'P1_ML': 225, # Actual Decimal: 3.25, Winner: P2
    'P2_ML': -300  # Actual Decimal: 1.33
},
{
    'Match ID': 10682325,
    'Player 1 ID': 888755, 'Player 1 Name': 'Vratislav Petracek',
    'Player 2 ID': 1149975, 'Player 2 Name': 'Jan Mecl Jr',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P1
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10682995,
    'Player 1 ID': 765084, 'Player 1 Name': 'Radomir Vidlicka',
    'Player 2 ID': 1162474, 'Player 2 Name': 'David Heczko',
    'P1_ML': -250, # Actual Decimal: 1.40, Winner: P1
    'P2_ML': 175  # Actual Decimal: 2.75
},
{
    'Match ID': 10682283,
    'Player 1 ID': 1099292, 'Player 1 Name': 'Jiri Dedek',
    'Player 2 ID': 686099, 'Player 2 Name': 'Michal Vedmoch',
    'P1_ML': -400, # Actual Decimal: 1.25, Winner: P1
    'P2_ML': 275  # Actual Decimal: 3.75
},
{
    'Match ID': 10682266,
    'Player 1 ID': 770057, 'Player 1 Name': 'Josef Pelikan',
    'Player 2 ID': 874419, 'Player 2 Name': 'Tomas Regner',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682984,
    'Player 1 ID': 1107438, 'Player 1 Name': 'Pavel Vondra',
    'Player 2 ID': 1099292, 'Player 2 Name': 'Jiri Dedek',
    'P1_ML': 137, # Actual Decimal: 2.38, Winner: P2
    'P2_ML': -187  # Actual Decimal: 1.53
},
{
    'Match ID': 10682252,
    'Player 1 ID': 1149975, 'Player 1 Name': 'Jan Mecl Jr',
    'Player 2 ID': 339257, 'Player 2 Name': 'Petr Bradach',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10682233,
    'Player 1 ID': 797186, 'Player 1 Name': 'Simon Kadavy',
    'Player 2 ID': 685556, 'Player 2 Name': 'Lukas Tonar',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682203,
    'Player 1 ID': 1122137, 'Player 1 Name': 'Karel Kapras',
    'Player 2 ID': 685556, 'Player 2 Name': 'Lukas Tonar',
    'P1_ML': -300, # Actual Decimal: 1.33, Winner: P1
    'P2_ML': 225  # Actual Decimal: 3.25
},
{
    'Match ID': 10687643,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 881949, 'Player 2 Name': 'Tomas Barta',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682211,
    'Player 1 ID': 1182742, 'Player 1 Name': 'Vitezslav Bosak',
    'Player 2 ID': 699464, 'Player 2 Name': 'Martin Stefek',
    'P1_ML': -400, # Actual Decimal: 1.25, Winner: P1
    'P2_ML': 275  # Actual Decimal: 3.75
},
{
    'Match ID': 10682217,
    'Player 1 ID': 1162474, 'Player 1 Name': 'David Heczko',
    'Player 2 ID': 699464, 'Player 2 Name': 'Martin Stefek',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682333,
    'Player 1 ID': 1149975, 'Player 1 Name': 'Jan Mecl Jr',
    'Player 2 ID': 687549, 'Player 2 Name': 'Michal Vavrecka',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682329,
    'Player 1 ID': 1122137, 'Player 1 Name': 'Karel Kapras',
    'Player 2 ID': 797186, 'Player 2 Name': 'Simon Kadavy',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682310,
    'Player 1 ID': 770057, 'Player 1 Name': 'Josef Pelikan',
    'Player 2 ID': 607888, 'Player 2 Name': 'Lubor Sulava',
    'P1_ML': 137, # Actual Decimal: 2.38, Winner: P2
    'P2_ML': -187  # Actual Decimal: 1.53
},
{
    'Match ID': 10682232,
    'Player 1 ID': 765084, 'Player 1 Name': 'Radomir Vidlicka',
    'Player 2 ID': 1182742, 'Player 2 Name': 'Vitezslav Bosak',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682231,
    'Player 1 ID': 1099292, 'Player 1 Name': 'Jiri Dedek',
    'Player 2 ID': 781022, 'Player 2 Name': 'Milan Fisera',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10689217,
    'Player 1 ID': 561176, 'Player 1 Name': 'Jiri Ruzicka',
    'Player 2 ID': 1088851, 'Player 2 Name': 'Oleksandr Kolisnyk',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10689216,
    'Player 1 ID': 685221, 'Player 1 Name': 'Jiri Nesnera',
    'Player 2 ID': 338586, 'Player 2 Name': 'Marek Fabini',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P1
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10689214,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 881949, 'Player 2 Name': 'Tomas Barta',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10689213,
    'Player 1 ID': 380220, 'Player 1 Name': 'Patrik Pycha',
    'Player 2 ID': 358665, 'Player 2 Name': 'Marek Sedlak',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P2
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10682222,
    'Player 1 ID': 339257, 'Player 1 Name': 'Petr Bradach',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P1
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10682210,
    'Player 1 ID': 797186, 'Player 1 Name': 'Simon Kadavy',
    'Player 2 ID': 1015754, 'Player 2 Name': 'Marek Chlebecek',
    'P1_ML': 250, # Actual Decimal: 3.50, Winner: P1
    'P2_ML': -350  # Actual Decimal: 1.28
},
{
    'Match ID': 10682293,
    'Player 1 ID': 874419, 'Player 1 Name': 'Tomas Regner',
    'Player 2 ID': 689521, 'Player 2 Name': 'Jaroslav Strnad 1964',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P1
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10682249,
    'Player 1 ID': 888755, 'Player 1 Name': 'Vratislav Petracek',
    'Player 2 ID': 922416, 'Player 2 Name': 'Jakub Vales',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10682267,
    'Player 1 ID': 687549, 'Player 1 Name': 'Michal Vavrecka',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': 200, # Actual Decimal: 3.00, Winner: P2
    'P2_ML': -275  # Actual Decimal: 1.36
},
{
    'Match ID': 10682320,
    'Player 1 ID': 607888, 'Player 1 Name': 'Lubor Sulava',
    'Player 2 ID': 689521, 'Player 2 Name': 'Jaroslav Strnad 1964',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682989,
    'Player 1 ID': 781022, 'Player 1 Name': 'Milan Fisera',
    'Player 2 ID': 1107438, 'Player 2 Name': 'Pavel Vondra',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10689751,
    'Player 1 ID': 685556, 'Player 1 Name': 'Lukas Tonar',
    'Player 2 ID': 1122137, 'Player 2 Name': 'Karel Kapras',
    'P1_ML': 275, # Actual Decimal: 3.75, Winner: P1
    'P2_ML': -400  # Actual Decimal: 1.25
},
{
    'Match ID': 10689753,
    'Player 1 ID': 339257, 'Player 1 Name': 'Petr Bradach',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10689755,
    'Player 1 ID': 781022, 'Player 1 Name': 'Milan Fisera',
    'Player 2 ID': 1099292, 'Player 2 Name': 'Jiri Dedek',
    'P1_ML': -250, # Actual Decimal: 1.40, Winner: P1
    'P2_ML': 175  # Actual Decimal: 2.75
},
{
    'Match ID': 10682268,
    'Player 1 ID': 559309, 'Player 1 Name': 'Lukas Zeman',
    'Player 2 ID': 341888, 'Player 2 Name': 'Martin Huk',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10682307,
    'Player 1 ID': 1098619, 'Player 1 Name': 'Jaroslav Bares',
    'Player 2 ID': 553241, 'Player 2 Name': 'Martin Sobisek',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P2
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682987,
    'Player 1 ID': 685556, 'Player 1 Name': 'Lukas Tonar',
    'Player 2 ID': 1015754, 'Player 2 Name': 'Marek Chlebecek',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682234,
    'Player 1 ID': 355559, 'Player 1 Name': 'Michal Regner',
    'Player 2 ID': 342540, 'Player 2 Name': 'Matous Klimenta',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P1
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10687411,
    'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma',
    'Player 2 ID': 1166071, 'Player 2 Name': 'Jiri Havel',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P1
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10683002,
    'Player 1 ID': 380220, 'Player 1 Name': 'Patrik Pycha',
    'Player 2 ID': 1133447, 'Player 2 Name': 'Petr Sebera',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P1
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10682198,
    'Player 1 ID': 1013721, 'Player 1 Name': 'Jiri Jira',
    'Player 2 ID': 650467, 'Player 2 Name': 'Vladimir Kubat',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P2
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682342,
    'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma',
    'Player 2 ID': 1122093, 'Player 2 Name': 'Petr Franc',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682288,
    'Player 1 ID': 686799, 'Player 1 Name': 'Lukas Martinak',
    'Player 2 ID': 1157770, 'Player 2 Name': 'Evzen Rychlik',
    'P1_ML': 137, # Actual Decimal: 2.38, Winner: P2
    'P2_ML': -187  # Actual Decimal: 1.53
},
{
    'Match ID': 10682226,
    'Player 1 ID': 799638, 'Player 1 Name': 'Jindrich Vrba',
    'Player 2 ID': 635749, 'Player 2 Name': 'Cesta Havrda',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682215,
    'Player 1 ID': 341166, 'Player 1 Name': 'Milan Klement',
    'Player 2 ID': 339709, 'Player 2 Name': 'Jiri Louda',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682213,
    'Player 1 ID': 391465, 'Player 1 Name': 'Vaclav Dolezal',
    'Player 2 ID': 346257, 'Player 2 Name': 'Petr Sudek',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P1
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10682227,
    'Player 1 ID': 1166071, 'Player 1 Name': 'Jiri Havel',
    'Player 2 ID': 360808, 'Player 2 Name': 'Jiri Svec',
    'P1_ML': -275, # Actual Decimal: 1.36, Winner: P2
    'P2_ML': 200  # Actual Decimal: 3.00
},
{
    'Match ID': 10682341,
    'Player 1 ID': 1171338, 'Player 1 Name': 'Josef Cabak',
    'Player 2 ID': 686799, 'Player 2 Name': 'Lukas Martinak',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P2
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10682315,
    'Player 1 ID': 650467, 'Player 1 Name': 'Vladimir Kubat',
    'Player 2 ID': 799638, 'Player 2 Name': 'Jindrich Vrba',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10682250,
    'Player 1 ID': 689521, 'Player 1 Name': 'Jaroslav Strnad 1964',
    'Player 2 ID': 391465, 'Player 2 Name': 'Vaclav Dolezal',
    'P1_ML': 200, # Actual Decimal: 3.00, Winner: P1
    'P2_ML': -275  # Actual Decimal: 1.36
},
{
    'Match ID': 10682238,
    'Player 1 ID': 360808, 'Player 1 Name': 'Jiri Svec',
    'Player 2 ID': 1005699, 'Player 2 Name': 'Miroslav Tuma',
    'P1_ML': 137, # Actual Decimal: 2.38, Winner: P2
    'P2_ML': -187  # Actual Decimal: 1.53
},
{
    'Match ID': 10682335,
    'Player 1 ID': 1166071, 'Player 1 Name': 'Jiri Havel',
    'Player 2 ID': 1122093, 'Player 2 Name': 'Petr Franc',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682330,
    'Player 1 ID': 355559, 'Player 1 Name': 'Michal Regner',
    'Player 2 ID': 339709, 'Player 2 Name': 'Jiri Louda',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682314,
    'Player 1 ID': 606056, 'Player 1 Name': 'Vaclav Hruska Sr',
    'Player 2 ID': 346257, 'Player 2 Name': 'Petr Sudek',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10682340,
    'Player 1 ID': 1090473, 'Player 1 Name': 'Karel Schopf',
    'Player 2 ID': 341166, 'Player 2 Name': 'Milan Klement',
    'P1_ML': 120, # Actual Decimal: 2.20, Winner: P2
    'P2_ML': -162  # Actual Decimal: 1.61
},
{
    'Match ID': 10682299,
    'Player 1 ID': 925949, 'Player 1 Name': 'Ivan Jemelka',
    'Player 2 ID': 1157770, 'Player 2 Name': 'Evzen Rychlik',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P1
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10682228,
    'Player 1 ID': 355559, 'Player 1 Name': 'Michal Regner',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682337,
    'Player 1 ID': 925949, 'Player 1 Name': 'Ivan Jemelka',
    'Player 2 ID': 1171338, 'Player 2 Name': 'Josef Cabak',
    'P1_ML': 150, # Actual Decimal: 2.50, Winner: P2
    'P2_ML': -200  # Actual Decimal: 1.50
},
{
    'Match ID': 10682259,
    'Player 1 ID': 387783, 'Player 1 Name': 'Tomas Postelt',
    'Player 2 ID': 1149975, 'Player 2 Name': 'Jan Mecl Jr',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682258,
    'Player 1 ID': 341888, 'Player 1 Name': 'Martin Huk',
    'Player 2 ID': 976107, 'Player 2 Name': 'Dan Volhejn',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682207,
    'Player 1 ID': 697880, 'Player 1 Name': 'David Szotek',
    'Player 2 ID': 683470, 'Player 2 Name': 'Petr Kalias',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P2
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10687423,
    'Player 1 ID': 355559, 'Player 1 Name': 'Michal Regner',
    'Player 2 ID': 341166, 'Player 2 Name': 'Milan Klement',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P2
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10682985,
    'Player 1 ID': 1090473, 'Player 1 Name': 'Karel Schopf',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682331,
    'Player 1 ID': 606056, 'Player 1 Name': 'Vaclav Hruska Sr',
    'Player 2 ID': 689521, 'Player 2 Name': 'Jaroslav Strnad 1964',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10687418,
    'Player 1 ID': 925949, 'Player 1 Name': 'Ivan Jemelka',
    'Player 2 ID': 1171338, 'Player 2 Name': 'Josef Cabak',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P2
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10682202,
    'Player 1 ID': 781022, 'Player 1 Name': 'Milan Fisera',
    'Player 2 ID': 686099, 'Player 2 Name': 'Michal Vedmoch',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687367,
    'Player 1 ID': 1013721, 'Player 1 Name': 'Jiri Jira',
    'Player 2 ID': 635749, 'Player 2 Name': 'Cesta Havrda',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P1
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10687365,
    'Player 1 ID': 1090473, 'Player 1 Name': 'Karel Schopf',
    'Player 2 ID': 339709, 'Player 2 Name': 'Jiri Louda',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10687364,
    'Player 1 ID': 1122093, 'Player 1 Name': 'Petr Franc',
    'Player 2 ID': 360808, 'Player 2 Name': 'Jiri Svec',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10687417,
    'Player 1 ID': 689521, 'Player 1 Name': 'Jaroslav Strnad 1964',
    'Player 2 ID': 606056, 'Player 2 Name': 'Vaclav Hruska Sr',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P2
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682273,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 689390, 'Player 2 Name': 'Marek Kostal',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10682287,
    'Player 1 ID': 1013721, 'Player 1 Name': 'Jiri Jira',
    'Player 2 ID': 635749, 'Player 2 Name': 'Cesta Havrda',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682257,
    'Player 1 ID': 339709, 'Player 1 Name': 'Jiri Louda',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10676156,
    'Player 1 ID': 999484, 'Player 1 Name': 'Tomas Kucera',
    'Player 2 ID': 1007725, 'Player 2 Name': 'Jiri Grohsgott',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676134,
    'Player 1 ID': 636504, 'Player 1 Name': 'Vladimir Cermak',
    'Player 2 ID': 359407, 'Player 2 Name': 'Jaroslav Prokupek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676868,
    'Player 1 ID': 360789, 'Player 1 Name': 'Jakub Hradecky',
    'Player 2 ID': 1169654, 'Player 2 Name': 'Vojtech Svechota',
    'P1_ML': 2500, # Actual Decimal: 26.00, Winner: P2
    'P2_ML': -100000  # Actual Decimal: 1.00
},
{
    'Match ID': 10676867,
    'Player 1 ID': 600584, 'Player 1 Name': 'Denis Hofman',
    'Player 2 ID': 1078065, 'Player 2 Name': 'Martin Sopko',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676142,
    'Player 1 ID': 680338, 'Player 1 Name': 'Matej Pycha',
    'Player 2 ID': 654692, 'Player 2 Name': 'Pavel Sprynar',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676866,
    'Player 1 ID': 999484, 'Player 1 Name': 'Tomas Kucera',
    'Player 2 ID': 701362, 'Player 2 Name': 'Stanislav Mazanek',
    'P1_ML': -1999, # Actual Decimal: 1.05, Winner: P1
    'P2_ML': 750  # Actual Decimal: 8.50
},
{
    'Match ID': 10676157,
    'Player 1 ID': 954785, 'Player 1 Name': 'Martin Sychra',
    'Player 2 ID': 339048, 'Player 2 Name': 'Karel Brozhik',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676155,
    'Player 1 ID': 954785, 'Player 1 Name': 'Martin Sychra',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676202,
    'Player 1 ID': 359407, 'Player 1 Name': 'Jaroslav Prokupek',
    'Player 2 ID': 360789, 'Player 2 Name': 'Jakub Hradecky',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676192,
    'Player 1 ID': 339048, 'Player 1 Name': 'Karel Brozhik',
    'Player 2 ID': 600584, 'Player 2 Name': 'Denis Hofman',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676154,
    'Player 1 ID': 1007725, 'Player 1 Name': 'Jiri Grohsgott',
    'Player 2 ID': 680338, 'Player 2 Name': 'Matej Pycha',
    'P1_ML': -4000, # Actual Decimal: 1.02, Winner: P1
    'P2_ML': 950  # Actual Decimal: 10.50
},
{
    'Match ID': 10676174,
    'Player 1 ID': 654692, 'Player 1 Name': 'Pavel Sprynar',
    'Player 2 ID': 999484, 'Player 2 Name': 'Tomas Kucera',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676135,
    'Player 1 ID': 1078065, 'Player 1 Name': 'Martin Sopko',
    'Player 2 ID': 954785, 'Player 2 Name': 'Martin Sychra',
    'P1_ML': -6666, # Actual Decimal: 1.01, Winner: P1
    'P2_ML': 1100  # Actual Decimal: 12.00
},
{
    'Match ID': 10676098,
    'Player 1 ID': 1169654, 'Player 1 Name': 'Vojtech Svechota',
    'Player 2 ID': 636504, 'Player 2 Name': 'Vladimir Cermak',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676122,
    'Player 1 ID': 636504, 'Player 1 Name': 'Vladimir Cermak',
    'Player 2 ID': 553241, 'Player 2 Name': 'Martin Sobisek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682328,
    'Player 1 ID': 346257, 'Player 1 Name': 'Petr Sudek',
    'Player 2 ID': 689521, 'Player 2 Name': 'Jaroslav Strnad 1964',
    'P1_ML': -187, # Actual Decimal: 1.53, Winner: P2
    'P2_ML': 137  # Actual Decimal: 2.38
},
{
    'Match ID': 10676079,
    'Player 1 ID': 654692, 'Player 1 Name': 'Pavel Sprynar',
    'Player 2 ID': 701362, 'Player 2 Name': 'Stanislav Mazanek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10676869,
    'Player 1 ID': 1078065, 'Player 1 Name': 'Martin Sopko',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': 500, # Actual Decimal: 6.00, Winner: P2
    'P2_ML': -900  # Actual Decimal: 1.11
},
{
    'Match ID': 10682225,
    'Player 1 ID': 635749, 'Player 1 Name': 'Cesta Havrda',
    'Player 2 ID': 650467, 'Player 2 Name': 'Vladimir Kubat',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682220,
    'Player 1 ID': 1157770, 'Player 1 Name': 'Evzen Rychlik',
    'Player 2 ID': 1171338, 'Player 2 Name': 'Josef Cabak',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682219,
    'Player 1 ID': 1122093, 'Player 1 Name': 'Petr Franc',
    'Player 2 ID': 360808, 'Player 2 Name': 'Jiri Svec',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682339,
    'Player 1 ID': 686799, 'Player 1 Name': 'Lukas Martinak',
    'Player 2 ID': 925949, 'Player 2 Name': 'Ivan Jemelka',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682313,
    'Player 1 ID': 391465, 'Player 1 Name': 'Vaclav Dolezal',
    'Player 2 ID': 606056, 'Player 2 Name': 'Vaclav Hruska Sr',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682308,
    'Player 1 ID': 799638, 'Player 1 Name': 'Jindrich Vrba',
    'Player 2 ID': 1013721, 'Player 2 Name': 'Jiri Jira',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10676158,
    'Player 1 ID': 1169654, 'Player 1 Name': 'Vojtech Svechota',
    'Player 2 ID': 553241, 'Player 2 Name': 'Martin Sobisek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682281,
    'Player 1 ID': 341166, 'Player 1 Name': 'Milan Klement',
    'Player 2 ID': 355559, 'Player 2 Name': 'Michal Regner',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P2
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10686389,
    'Player 1 ID': 339048, 'Player 1 Name': 'Karel Brozhik',
    'Player 2 ID': 600584, 'Player 2 Name': 'Denis Hofman',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10686359,
    'Player 1 ID': 1007725, 'Player 1 Name': 'Jiri Grohsgott',
    'Player 2 ID': 999484, 'Player 2 Name': 'Tomas Kucera',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10686353,
    'Player 1 ID': 1169654, 'Player 1 Name': 'Vojtech Svechota',
    'Player 2 ID': 553241, 'Player 2 Name': 'Martin Sobisek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10686373,
    'Player 1 ID': 954785, 'Player 1 Name': 'Martin Sychra',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': 550, # Actual Decimal: 6.50, Winner: P2
    'P2_ML': -999  # Actual Decimal: 1.10
},
{
    'Match ID': 10686331,
    'Player 1 ID': 680338, 'Player 1 Name': 'Matej Pycha',
    'Player 2 ID': 701362, 'Player 2 Name': 'Stanislav Mazanek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10686323,
    'Player 1 ID': 636504, 'Player 1 Name': 'Vladimir Cermak',
    'Player 2 ID': 360789, 'Player 2 Name': 'Jakub Hradecky',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682248,
    'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma',
    'Player 2 ID': 1166071, 'Player 2 Name': 'Jiri Havel',
    'P1_ML': -300, # Actual Decimal: 1.33, Winner: P2
    'P2_ML': 225  # Actual Decimal: 3.25
},
{
    'Match ID': 10682278,
    'Player 1 ID': 686099, 'Player 1 Name': 'Michal Vedmoch',
    'Player 2 ID': 847624, 'Player 2 Name': 'Lukas Malek',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P1
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10687419,
    'Player 1 ID': 799638, 'Player 1 Name': 'Jindrich Vrba',
    'Player 2 ID': 650467, 'Player 2 Name': 'Vladimir Kubat',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P2
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10682260,
    'Player 1 ID': 678236, 'Player 1 Name': 'Tomas Turek',
    'Player 2 ID': 639253, 'Player 2 Name': 'Michal Zahradka',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682243,
    'Player 1 ID': 639253, 'Player 1 Name': 'Michal Zahradka',
    'Player 2 ID': 338586, 'Player 2 Name': 'Marek Fabini',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682857,
    'Player 1 ID': 358665, 'Player 1 Name': 'Marek Sedlak',
    'Player 2 ID': 1098619, 'Player 2 Name': 'Jaroslav Bares',
    'P1_ML': -400, # Actual Decimal: 1.25, Winner: P1
    'P2_ML': 275  # Actual Decimal: 3.75
},
{
    'Match ID': 10682306,
    'Player 1 ID': 741786, 'Player 1 Name': 'Ondrej Svacha',
    'Player 2 ID': 976107, 'Player 2 Name': 'Dan Volhejn',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682291,
    'Player 1 ID': 685221, 'Player 1 Name': 'Jiri Nesnera',
    'Player 2 ID': 387783, 'Player 2 Name': 'Tomas Postelt',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P1
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10682282,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 548705, 'Player 2 Name': 'Jiri Plachy',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682301,
    'Player 1 ID': 561176, 'Player 1 Name': 'Jiri Ruzicka',
    'Player 2 ID': 1088851, 'Player 2 Name': 'Oleksandr Kolisnyk',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682251,
    'Player 1 ID': 1163803, 'Player 1 Name': 'David Mikula',
    'Player 2 ID': 1159459, 'Player 2 Name': 'Roman Guliak',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P2
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10682253,
    'Player 1 ID': 1030283, 'Player 1 Name': 'Tomas Vinter',
    'Player 2 ID': 548705, 'Player 2 Name': 'Jiri Plachy',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682305,
    'Player 1 ID': 387783, 'Player 1 Name': 'Tomas Postelt',
    'Player 2 ID': 639253, 'Player 2 Name': 'Michal Zahradka',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682286,
    'Player 1 ID': 976107, 'Player 1 Name': 'Dan Volhejn',
    'Player 2 ID': 561176, 'Player 2 Name': 'Jiri Ruzicka',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682264,
    'Player 1 ID': 1098619, 'Player 1 Name': 'Jaroslav Bares',
    'Player 2 ID': 380220, 'Player 2 Name': 'Patrik Pycha',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682304,
    'Player 1 ID': 1088851, 'Player 1 Name': 'Oleksandr Kolisnyk',
    'Player 2 ID': 741786, 'Player 2 Name': 'Ondrej Svacha',
    'P1_ML': 150, # Actual Decimal: 2.50, Winner: P1
    'P2_ML': -200  # Actual Decimal: 1.50
},
{
    'Match ID': 10687619,
    'Player 1 ID': 548705, 'Player 1 Name': 'Jiri Plachy',
    'Player 2 ID': 881949, 'Player 2 Name': 'Tomas Barta',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682254,
    'Player 1 ID': 373963, 'Player 1 Name': 'Tomas Janata',
    'Player 2 ID': 387428, 'Player 2 Name': 'Daniel Tuma',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P2
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682319,
    'Player 1 ID': 380220, 'Player 1 Name': 'Patrik Pycha',
    'Player 2 ID': 798958, 'Player 2 Name': 'Jiri Zuzanek',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P1
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10682201,
    'Player 1 ID': 798958, 'Player 1 Name': 'Jiri Zuzanek',
    'Player 2 ID': 1098619, 'Player 2 Name': 'Jaroslav Bares',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682224,
    'Player 1 ID': 341888, 'Player 1 Name': 'Martin Huk',
    'Player 2 ID': 687549, 'Player 2 Name': 'Michal Vavrecka',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682294,
    'Player 1 ID': 342540, 'Player 1 Name': 'Matous Klimenta',
    'Player 2 ID': 546478, 'Player 2 Name': 'Jan Sucharda',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682321,
    'Player 1 ID': 553241, 'Player 1 Name': 'Martin Sobisek',
    'Player 2 ID': 1015754, 'Player 2 Name': 'Marek Chlebecek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682197,
    'Player 1 ID': 355559, 'Player 1 Name': 'Michal Regner',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682295,
    'Player 1 ID': 1098619, 'Player 1 Name': 'Jaroslav Bares',
    'Player 2 ID': 380220, 'Player 2 Name': 'Patrik Pycha',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682296,
    'Player 1 ID': 559309, 'Player 1 Name': 'Lukas Zeman',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10687618,
    'Player 1 ID': 881949, 'Player 1 Name': 'Tomas Barta',
    'Player 2 ID': 373963, 'Player 2 Name': 'Tomas Janata',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P1
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10682292,
    'Player 1 ID': 685221, 'Player 1 Name': 'Jiri Nesnera',
    'Player 2 ID': 639253, 'Player 2 Name': 'Michal Zahradka',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682277,
    'Player 1 ID': 358665, 'Player 1 Name': 'Marek Sedlak',
    'Player 2 ID': 380220, 'Player 2 Name': 'Patrik Pycha',
    'P1_ML': -300, # Actual Decimal: 1.33, Winner: P2
    'P2_ML': 225  # Actual Decimal: 3.25
},
{
    'Match ID': 10682221,
    'Player 1 ID': 741786, 'Player 1 Name': 'Ondrej Svacha',
    'Player 2 ID': 561176, 'Player 2 Name': 'Jiri Ruzicka',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P2
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682334,
    'Player 1 ID': 338586, 'Player 1 Name': 'Marek Fabini',
    'Player 2 ID': 387783, 'Player 2 Name': 'Tomas Postelt',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682241,
    'Player 1 ID': 1088851, 'Player 1 Name': 'Oleksandr Kolisnyk',
    'Player 2 ID': 976107, 'Player 2 Name': 'Dan Volhejn',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682230,
    'Player 1 ID': 373963, 'Player 1 Name': 'Tomas Janata',
    'Player 2 ID': 548705, 'Player 2 Name': 'Jiri Plachy',
    'P1_ML': -187, # Actual Decimal: 1.53, Winner: P1
    'P2_ML': 137  # Actual Decimal: 2.38
},
{
    'Match ID': 10682280,
    'Player 1 ID': 1163803, 'Player 1 Name': 'David Mikula',
    'Player 2 ID': 686100, 'Player 2 Name': 'Jakub Levicky',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P2
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10682209,
    'Player 1 ID': 338586, 'Player 1 Name': 'Marek Fabini',
    'Player 2 ID': 685221, 'Player 2 Name': 'Jiri Nesnera',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10682323,
    'Player 1 ID': 1159459, 'Player 1 Name': 'Roman Guliak',
    'Player 2 ID': 686100, 'Player 2 Name': 'Jakub Levicky',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P2
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10688387,
    'Player 1 ID': 1030283, 'Player 1 Name': 'Tomas Vinter',
    'Player 2 ID': 689390, 'Player 2 Name': 'Marek Kostal',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10682208,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 1030283, 'Player 2 Name': 'Tomas Vinter',
    'P1_ML': 120, # Actual Decimal: 2.20, Winner: P2
    'P2_ML': -162  # Actual Decimal: 1.61
},
{
    'Match ID': 10682371,
    'Player 1 ID': 1168056, 'Player 1 Name': 'Michal Bazalka',
    'Player 2 ID': 697880, 'Player 2 Name': 'David Szotek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682316,
    'Player 1 ID': 561176, 'Player 1 Name': 'Jiri Ruzicka',
    'Player 2 ID': 341888, 'Player 2 Name': 'Martin Huk',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682303,
    'Player 1 ID': 548705, 'Player 1 Name': 'Jiri Plachy',
    'Player 2 ID': 387428, 'Player 2 Name': 'Daniel Tuma',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682239,
    'Player 1 ID': 639253, 'Player 1 Name': 'Michal Zahradka',
    'Player 2 ID': 387783, 'Player 2 Name': 'Tomas Postelt',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10682199,
    'Player 1 ID': 798958, 'Player 1 Name': 'Jiri Zuzanek',
    'Player 2 ID': 358665, 'Player 2 Name': 'Marek Sedlak',
    'P1_ML': 200, # Actual Decimal: 3.00, Winner: P2
    'P2_ML': -275  # Actual Decimal: 1.36
},
{
    'Match ID': 10682214,
    'Player 1 ID': 387783, 'Player 1 Name': 'Tomas Postelt',
    'Player 2 ID': 678236, 'Player 2 Name': 'Tomas Turek',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682261,
    'Player 1 ID': 976107, 'Player 1 Name': 'Dan Volhejn',
    'Player 2 ID': 1088851, 'Player 2 Name': 'Oleksandr Kolisnyk',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682244,
    'Player 1 ID': 1149975, 'Player 1 Name': 'Jan Mecl Jr',
    'Player 2 ID': 678236, 'Player 2 Name': 'Tomas Turek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682240,
    'Player 1 ID': 689390, 'Player 1 Name': 'Marek Kostal',
    'Player 2 ID': 1030283, 'Player 2 Name': 'Tomas Vinter',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10682336,
    'Player 1 ID': 688067, 'Player 1 Name': 'Tadeas Zika',
    'Player 2 ID': 1168056, 'Player 2 Name': 'Michal Bazalka',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682309,
    'Player 1 ID': 1088851, 'Player 1 Name': 'Oleksandr Kolisnyk',
    'Player 2 ID': 561176, 'Player 2 Name': 'Jiri Ruzicka',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682332,
    'Player 1 ID': 683470, 'Player 1 Name': 'Petr Kalias',
    'Player 2 ID': 688067, 'Player 2 Name': 'Tadeas Zika',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10682229,
    'Player 1 ID': 341888, 'Player 1 Name': 'Martin Huk',
    'Player 2 ID': 1088851, 'Player 2 Name': 'Oleksandr Kolisnyk',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10682216,
    'Player 1 ID': 686972, 'Player 1 Name': 'Kyryl Darin',
    'Player 2 ID': 686099, 'Player 2 Name': 'Michal Vedmoch',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P2
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10682317,
    'Player 1 ID': 697880, 'Player 1 Name': 'David Szotek',
    'Player 2 ID': 688067, 'Player 2 Name': 'Tadeas Zika',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10688386,
    'Player 1 ID': 561176, 'Player 1 Name': 'Jiri Ruzicka',
    'Player 2 ID': 976107, 'Player 2 Name': 'Dan Volhejn',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10688384,
    'Player 1 ID': 688067, 'Player 1 Name': 'Tadeas Zika',
    'Player 2 ID': 683470, 'Player 2 Name': 'Petr Kalias',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10688383,
    'Player 1 ID': 639253, 'Player 1 Name': 'Michal Zahradka',
    'Player 2 ID': 1149975, 'Player 2 Name': 'Jan Mecl Jr',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688322,
    'Player 1 ID': 686099, 'Player 1 Name': 'Michal Vedmoch',
    'Player 2 ID': 686972, 'Player 2 Name': 'Kyryl Darin',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P1
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10682988,
    'Player 1 ID': 686099, 'Player 1 Name': 'Michal Vedmoch',
    'Player 2 ID': 1107438, 'Player 2 Name': 'Pavel Vondra',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P1
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10688319,
    'Player 1 ID': 1151479, 'Player 1 Name': 'BA Hoang Tai Nguyen',
    'Player 2 ID': 847624, 'Player 2 Name': 'Lukas Malek',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10682318,
    'Player 1 ID': 683470, 'Player 1 Name': 'Petr Kalias',
    'Player 2 ID': 1168056, 'Player 2 Name': 'Michal Bazalka',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P1
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10682289,
    'Player 1 ID': 689390, 'Player 1 Name': 'Marek Kostal',
    'Player 2 ID': 548705, 'Player 2 Name': 'Jiri Plachy',
    'P1_ML': -250, # Actual Decimal: 1.40, Winner: P1
    'P2_ML': 175  # Actual Decimal: 2.75
},
{
    'Match ID': 10682279,
    'Player 1 ID': 1149975, 'Player 1 Name': 'Jan Mecl Jr',
    'Player 2 ID': 639253, 'Player 2 Name': 'Michal Zahradka',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10682263,
    'Player 1 ID': 976107, 'Player 1 Name': 'Dan Volhejn',
    'Player 2 ID': 561176, 'Player 2 Name': 'Jiri Ruzicka',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P2
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10682262,
    'Player 1 ID': 847624, 'Player 1 Name': 'Lukas Malek',
    'Player 2 ID': 686972, 'Player 2 Name': 'Kyryl Darin',
    'P1_ML': 120, # Actual Decimal: 2.20, Winner: P2
    'P2_ML': -162  # Actual Decimal: 1.61
},

# --- Matches for 2025-09-18 ---
{
    'Match ID': 10694157,
    'Player 1 ID': 1088851, 'Player 1 Name': 'Oleksandr Kolisnyk',
    'Player 2 ID': 608764, 'Player 2 Name': 'Michal Jezek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10694154,
    'Player 1 ID': 959029, 'Player 1 Name': 'Pavel Berdych',
    'Player 2 ID': 1005699, 'Player 2 Name': 'Miroslav Tuma',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10694153,
    'Player 1 ID': 1149975, 'Player 1 Name': 'Jan Mecl Jr',
    'Player 2 ID': 341888, 'Player 2 Name': 'Martin Huk',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P2
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10694165,
    'Player 1 ID': 338649, 'Player 1 Name': 'Jan Zajicek',
    'Player 2 ID': 257586, 'Player 2 Name': 'Lukas Boruvka',
    'P1_ML': 300, # Actual Decimal: 4.00, Winner: P2
    'P2_ML': -450  # Actual Decimal: 1.22
},
{
    'Match ID': 10682235,
    'Player 1 ID': 546478, 'Player 1 Name': 'Jan Sucharda',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10688051,
    'Player 1 ID': 688067, 'Player 1 Name': 'Tadeas Zika',
    'Player 2 ID': 1157765, 'Player 2 Name': 'Michal Hrabec',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P1
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10688077,
    'Player 1 ID': 689390, 'Player 1 Name': 'Marek Kostal',
    'Player 2 ID': 1098619, 'Player 2 Name': 'Jaroslav Bares',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10691279,
    'Player 1 ID': 1080540, 'Player 1 Name': 'Martin Kir',
    'Player 2 ID': 388336, 'Player 2 Name': 'Vladimir Postelt',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10687730,
    'Player 1 ID': 687549, 'Player 1 Name': 'Michal Vavrecka',
    'Player 2 ID': 678236, 'Player 2 Name': 'Tomas Turek',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10688019,
    'Player 1 ID': 1090473, 'Player 1 Name': 'Karel Schopf',
    'Player 2 ID': 358665, 'Player 2 Name': 'Marek Sedlak',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688061,
    'Player 1 ID': 388336, 'Player 1 Name': 'Vladimir Postelt',
    'Player 2 ID': 608765, 'Player 2 Name': 'Tibor Kolenic',
    'P1_ML': -275, # Actual Decimal: 1.36, Winner: P2
    'P2_ML': 200  # Actual Decimal: 3.00
},
{
    'Match ID': 10688068,
    'Player 1 ID': 1098619, 'Player 1 Name': 'Jaroslav Bares',
    'Player 2 ID': 1012124, 'Player 2 Name': 'Michal Vondrak',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688193,
    'Player 1 ID': 1157765, 'Player 1 Name': 'Michal Hrabec',
    'Player 2 ID': 1179684, 'Player 2 Name': 'Tomas Dousek',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P2
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10688049,
    'Player 1 ID': 888755, 'Player 1 Name': 'Vratislav Petracek',
    'Player 2 ID': 687549, 'Player 2 Name': 'Michal Vavrecka',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10688020,
    'Player 1 ID': 1085799, 'Player 1 Name': 'Martin Vizek',
    'Player 2 ID': 689390, 'Player 2 Name': 'Marek Kostal',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10688052,
    'Player 1 ID': 955815, 'Player 1 Name': 'Vitek Dvorak',
    'Player 2 ID': 339709, 'Player 2 Name': 'Jiri Louda',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688053,
    'Player 1 ID': 696884, 'Player 1 Name': 'Michal Wollny',
    'Player 2 ID': 688067, 'Player 2 Name': 'Tadeas Zika',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10688178,
    'Player 1 ID': 338586, 'Player 1 Name': 'Marek Fabini',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10692039,
    'Player 1 ID': 807181, 'Player 1 Name': 'Frantisek Briza',
    'Player 2 ID': 1080540, 'Player 2 Name': 'Martin Kir',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P2
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10693122,
    'Player 1 ID': 608765, 'Player 1 Name': 'Tibor Kolenic',
    'Player 2 ID': 807181, 'Player 2 Name': 'Frantisek Briza',
    'P1_ML': -275, # Actual Decimal: 1.36, Winner: P2
    'P2_ML': 200  # Actual Decimal: 3.00
},
{
    'Match ID': 10688040,
    'Player 1 ID': 339709, 'Player 1 Name': 'Jiri Louda',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10688179,
    'Player 1 ID': 1085799, 'Player 1 Name': 'Martin Vizek',
    'Player 2 ID': 1012124, 'Player 2 Name': 'Michal Vondrak',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10688018,
    'Player 1 ID': 1157765, 'Player 1 Name': 'Michal Hrabec',
    'Player 2 ID': 696884, 'Player 2 Name': 'Michal Wollny',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10688062,
    'Player 1 ID': 687549, 'Player 1 Name': 'Michal Vavrecka',
    'Player 2 ID': 338586, 'Player 2 Name': 'Marek Fabini',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P2
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10688066,
    'Player 1 ID': 388336, 'Player 1 Name': 'Vladimir Postelt',
    'Player 2 ID': 807181, 'Player 2 Name': 'Frantisek Briza',
    'P1_ML': -275, # Actual Decimal: 1.36, Winner: P1
    'P2_ML': 200  # Actual Decimal: 3.00
},
{
    'Match ID': 10688195,
    'Player 1 ID': 696884, 'Player 1 Name': 'Michal Wollny',
    'Player 2 ID': 1179684, 'Player 2 Name': 'Tomas Dousek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10688186,
    'Player 1 ID': 1090473, 'Player 1 Name': 'Karel Schopf',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10687734,
    'Player 1 ID': 678236, 'Player 1 Name': 'Tomas Turek',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P2
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10688030,
    'Player 1 ID': 358665, 'Player 1 Name': 'Marek Sedlak',
    'Player 2 ID': 339709, 'Player 2 Name': 'Jiri Louda',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688037,
    'Player 1 ID': 1012124, 'Player 1 Name': 'Michal Vondrak',
    'Player 2 ID': 689390, 'Player 2 Name': 'Marek Kostal',
    'P1_ML': -250, # Actual Decimal: 1.40, Winner: P1
    'P2_ML': 175  # Actual Decimal: 2.75
},
{
    'Match ID': 10688038,
    'Player 1 ID': 1179684, 'Player 1 Name': 'Tomas Dousek',
    'Player 2 ID': 688067, 'Player 2 Name': 'Tadeas Zika',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10691280,
    'Player 1 ID': 608765, 'Player 1 Name': 'Tibor Kolenic',
    'Player 2 ID': 1080540, 'Player 2 Name': 'Martin Kir',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P2
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10687733,
    'Player 1 ID': 338586, 'Player 1 Name': 'Marek Fabini',
    'Player 2 ID': 678236, 'Player 2 Name': 'Tomas Turek',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688039,
    'Player 1 ID': 955815, 'Player 1 Name': 'Vitek Dvorak',
    'Player 2 ID': 358665, 'Player 2 Name': 'Marek Sedlak',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688063,
    'Player 1 ID': 807181, 'Player 1 Name': 'Frantisek Briza',
    'Player 2 ID': 608765, 'Player 2 Name': 'Tibor Kolenic',
    'P1_ML': 150, # Actual Decimal: 2.50, Winner: P2
    'P2_ML': -200  # Actual Decimal: 1.50
},
{
    'Match ID': 10693192,
    'Player 1 ID': 358665, 'Player 1 Name': 'Marek Sedlak',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10688075,
    'Player 1 ID': 1142181, 'Player 1 Name': 'Jan Srba',
    'Player 2 ID': 701363, 'Player 2 Name': 'Jaromir Cernik',
    'P1_ML': 150, # Actual Decimal: 2.50, Winner: P2
    'P2_ML': -200  # Actual Decimal: 1.50
},
{
    'Match ID': 10687871,
    'Player 1 ID': 1157770, 'Player 1 Name': 'Evzen Rychlik',
    'Player 2 ID': 1158338, 'Player 2 Name': 'Oldrich Vaclahovsky',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10687878,
    'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma',
    'Player 2 ID': 1089695, 'Player 2 Name': 'Milos Pospisil',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687890,
    'Player 1 ID': 611386, 'Player 1 Name': 'Ludek Madle',
    'Player 2 ID': 701363, 'Player 2 Name': 'Jaromir Cernik',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P2
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10688485,
    'Player 1 ID': 341166, 'Player 1 Name': 'Milan Klement',
    'Player 2 ID': 798958, 'Player 2 Name': 'Jiri Zuzanek',
    'P1_ML': -250, # Actual Decimal: 1.40, Winner: P1
    'P2_ML': 175  # Actual Decimal: 2.75
},
{
    'Match ID': 10687872,
    'Player 1 ID': 636504, 'Player 1 Name': 'Vladimir Cermak',
    'Player 2 ID': 347091, 'Player 2 Name': 'Tomas Varnuska',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10687879,
    'Player 1 ID': 339298, 'Player 1 Name': 'Tomas Holik',
    'Player 2 ID': 954785, 'Player 2 Name': 'Martin Sychra',
    'P1_ML': -187, # Actual Decimal: 1.53, Winner: P1
    'P2_ML': 137  # Actual Decimal: 2.38
},
{
    'Match ID': 10687880,
    'Player 1 ID': 1166071, 'Player 1 Name': 'Jiri Havel',
    'Player 2 ID': 750300, 'Player 2 Name': 'Michal Syroha',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687889,
    'Player 1 ID': 360808, 'Player 1 Name': 'Jiri Svec',
    'Player 2 ID': 654692, 'Player 2 Name': 'Pavel Sprynar',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10687891,
    'Player 1 ID': 1159074, 'Player 1 Name': 'Laurent Lasota',
    'Player 2 ID': 925949, 'Player 2 Name': 'Ivan Jemelka',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10687719,
    'Player 1 ID': 1030283, 'Player 1 Name': 'Tomas Vinter',
    'Player 2 ID': 387428, 'Player 2 Name': 'Daniel Tuma',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P2
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10687855,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 636504, 'Player 2 Name': 'Vladimir Cermak',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P1
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10687866,
    'Player 1 ID': 701363, 'Player 1 Name': 'Jaromir Cernik',
    'Player 2 ID': 360808, 'Player 2 Name': 'Jiri Svec',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687881,
    'Player 1 ID': 798958, 'Player 1 Name': 'Jiri Zuzanek',
    'Player 2 ID': 339298, 'Player 2 Name': 'Tomas Holik',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10687882,
    'Player 1 ID': 1158338, 'Player 1 Name': 'Oldrich Vaclahovsky',
    'Player 2 ID': 1159074, 'Player 2 Name': 'Laurent Lasota',
    'P1_ML': 200, # Actual Decimal: 3.00, Winner: P2
    'P2_ML': -275  # Actual Decimal: 1.36
},
{
    'Match ID': 10687718,
    'Player 1 ID': 347091, 'Player 1 Name': 'Tomas Varnuska',
    'Player 2 ID': 1030283, 'Player 2 Name': 'Tomas Vinter',
    'P1_ML': 137, # Actual Decimal: 2.38, Winner: P1
    'P2_ML': -187  # Actual Decimal: 1.53
},
{
    'Match ID': 10687861,
    'Player 1 ID': 750300, 'Player 1 Name': 'Michal Syroha',
    'Player 2 ID': 1005699, 'Player 2 Name': 'Miroslav Tuma',
    'P1_ML': 250, # Actual Decimal: 3.50, Winner: P2
    'P2_ML': -350  # Actual Decimal: 1.28
},
{
    'Match ID': 10687862,
    'Player 1 ID': 654692, 'Player 1 Name': 'Pavel Sprynar',
    'Player 2 ID': 611386, 'Player 2 Name': 'Ludek Madle',
    'P1_ML': 275, # Actual Decimal: 3.75, Winner: P2
    'P2_ML': -400  # Actual Decimal: 1.25
},
{
    'Match ID': 10687867,
    'Player 1 ID': 925949, 'Player 1 Name': 'Ivan Jemelka',
    'Player 2 ID': 1157770, 'Player 2 Name': 'Evzen Rychlik',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687883,
    'Player 1 ID': 954785, 'Player 1 Name': 'Martin Sychra',
    'Player 2 ID': 341166, 'Player 2 Name': 'Milan Klement',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P2
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10687720,
    'Player 1 ID': 1030283, 'Player 1 Name': 'Tomas Vinter',
    'Player 2 ID': 636504, 'Player 2 Name': 'Vladimir Cermak',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10687865,
    'Player 1 ID': 1089695, 'Player 1 Name': 'Milos Pospisil',
    'Player 2 ID': 1166071, 'Player 2 Name': 'Jiri Havel',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10687860,
    'Player 1 ID': 611386, 'Player 1 Name': 'Ludek Madle',
    'Player 2 ID': 360808, 'Player 2 Name': 'Jiri Svec',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10690393,
    'Player 1 ID': 1015754, 'Player 1 Name': 'Marek Chlebecek',
    'Player 2 ID': 380220, 'Player 2 Name': 'Patrik Pycha',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10690391,
    'Player 1 ID': 888755, 'Player 1 Name': 'Vratislav Petracek',
    'Player 2 ID': 559309, 'Player 2 Name': 'Lukas Zeman',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10688192,
    'Player 1 ID': 1163571, 'Player 1 Name': 'Jindrich Simecek',
    'Player 2 ID': 689440, 'Player 2 Name': 'Bohuslav Kaloc',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P2
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10682297,
    'Player 1 ID': 1090473, 'Player 1 Name': 'Karel Schopf',
    'Player 2 ID': 546478, 'Player 2 Name': 'Jan Sucharda',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10683008,
    'Player 1 ID': 922416, 'Player 1 Name': 'Jakub Vales',
    'Player 2 ID': 559309, 'Player 2 Name': 'Lukas Zeman',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682270,
    'Player 1 ID': 955815, 'Player 1 Name': 'Vitek Dvorak',
    'Player 2 ID': 355559, 'Player 2 Name': 'Michal Regner',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682269,
    'Player 1 ID': 1133447, 'Player 1 Name': 'Petr Sebera',
    'Player 2 ID': 1098619, 'Player 2 Name': 'Jaroslav Bares',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682986,
    'Player 1 ID': 342540, 'Player 1 Name': 'Matous Klimenta',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682311,
    'Player 1 ID': 553241, 'Player 1 Name': 'Martin Sobisek',
    'Player 2 ID': 380220, 'Player 2 Name': 'Patrik Pycha',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682236,
    'Player 1 ID': 341888, 'Player 1 Name': 'Martin Huk',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10683003,
    'Player 1 ID': 1015754, 'Player 1 Name': 'Marek Chlebecek',
    'Player 2 ID': 1133447, 'Player 2 Name': 'Petr Sebera',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10690392,
    'Player 1 ID': 355559, 'Player 1 Name': 'Michal Regner',
    'Player 2 ID': 546478, 'Player 2 Name': 'Jan Sucharda',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682284,
    'Player 1 ID': 687549, 'Player 1 Name': 'Michal Vavrecka',
    'Player 2 ID': 922416, 'Player 2 Name': 'Jakub Vales',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682272,
    'Player 1 ID': 1133447, 'Player 1 Name': 'Petr Sebera',
    'Player 2 ID': 553241, 'Player 2 Name': 'Martin Sobisek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682298,
    'Player 1 ID': 955815, 'Player 1 Name': 'Vitek Dvorak',
    'Player 2 ID': 342540, 'Player 2 Name': 'Matous Klimenta',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682302,
    'Player 1 ID': 922416, 'Player 1 Name': 'Jakub Vales',
    'Player 2 ID': 341888, 'Player 2 Name': 'Martin Huk',
    'P1_ML': 900, # Actual Decimal: 10.00, Winner: P2
    'P2_ML': -3333  # Actual Decimal: 1.03
},
{
    'Match ID': 10682237,
    'Player 1 ID': 687549, 'Player 1 Name': 'Michal Vavrecka',
    'Player 2 ID': 559309, 'Player 2 Name': 'Lukas Zeman',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682242,
    'Player 1 ID': 546478, 'Player 1 Name': 'Jan Sucharda',
    'Player 2 ID': 355559, 'Player 2 Name': 'Michal Regner',
    'P1_ML': 900, # Actual Decimal: 10.00, Winner: P2
    'P2_ML': -3333  # Actual Decimal: 1.03
},
{
    'Match ID': 10682322,
    'Player 1 ID': 1015754, 'Player 1 Name': 'Marek Chlebecek',
    'Player 2 ID': 1098619, 'Player 2 Name': 'Jaroslav Bares',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10690318,
    'Player 1 ID': 922416, 'Player 1 Name': 'Jakub Vales',
    'Player 2 ID': 341888, 'Player 2 Name': 'Martin Huk',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10690321,
    'Player 1 ID': 553241, 'Player 1 Name': 'Martin Sobisek',
    'Player 2 ID': 1133447, 'Player 2 Name': 'Petr Sebera',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10690322,
    'Player 1 ID': 342540, 'Player 1 Name': 'Matous Klimenta',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682326,
    'Player 1 ID': 888755, 'Player 1 Name': 'Vratislav Petracek',
    'Player 2 ID': 687549, 'Player 2 Name': 'Michal Vavrecka',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10687863,
    'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma',
    'Player 2 ID': 1166071, 'Player 2 Name': 'Jiri Havel',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P1
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10682271,
    'Player 1 ID': 380220, 'Player 1 Name': 'Patrik Pycha',
    'Player 2 ID': 1015754, 'Player 2 Name': 'Marek Chlebecek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10687888,
    'Player 1 ID': 341166, 'Player 1 Name': 'Milan Klement',
    'Player 2 ID': 339298, 'Player 2 Name': 'Tomas Holik',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10687886,
    'Player 1 ID': 654692, 'Player 1 Name': 'Pavel Sprynar',
    'Player 2 ID': 701363, 'Player 2 Name': 'Jaromir Cernik',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P1
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10687721,
    'Player 1 ID': 548705, 'Player 1 Name': 'Jiri Plachy',
    'Player 2 ID': 387428, 'Player 2 Name': 'Daniel Tuma',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10687887,
    'Player 1 ID': 954785, 'Player 1 Name': 'Martin Sychra',
    'Player 2 ID': 339298, 'Player 2 Name': 'Tomas Holik',
    'P1_ML': 150, # Actual Decimal: 2.50, Winner: P1
    'P2_ML': -200  # Actual Decimal: 1.50
},
{
    'Match ID': 10688029,
    'Player 1 ID': 1022513, 'Player 1 Name': 'Milan Cetner',
    'Player 2 ID': 1166071, 'Player 2 Name': 'Jiri Havel',
    'P1_ML': 200, # Actual Decimal: 3.00, Winner: P2
    'P2_ML': -275  # Actual Decimal: 1.36
},
{
    'Match ID': 10688071,
    'Player 1 ID': 1142181, 'Player 1 Name': 'Jan Srba',
    'Player 2 ID': 654692, 'Player 2 Name': 'Pavel Sprynar',
    'P1_ML': 150, # Actual Decimal: 2.50, Winner: P2
    'P2_ML': -200  # Actual Decimal: 1.50
},
{
    'Match ID': 10688207,
    'Player 1 ID': 1163571, 'Player 1 Name': 'Jindrich Simecek',
    'Player 2 ID': 686100, 'Player 2 Name': 'Jakub Levicky',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10688015,
    'Player 1 ID': 701363, 'Player 1 Name': 'Jaromir Cernik',
    'Player 2 ID': 770032, 'Player 2 Name': 'Miloslav Lubas',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P1
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10687873,
    'Player 1 ID': 1157770, 'Player 1 Name': 'Evzen Rychlik',
    'Player 2 ID': 1159074, 'Player 2 Name': 'Laurent Lasota',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10688035,
    'Player 1 ID': 750300, 'Player 1 Name': 'Michal Syroha',
    'Player 2 ID': 957132, 'Player 2 Name': 'Vaclav Kosar',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687858,
    'Player 1 ID': 1166071, 'Player 1 Name': 'Jiri Havel',
    'Player 2 ID': 750300, 'Player 2 Name': 'Michal Syroha',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P1
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10688067,
    'Player 1 ID': 798958, 'Player 1 Name': 'Jiri Zuzanek',
    'Player 2 ID': 1012124, 'Player 2 Name': 'Michal Vondrak',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688010,
    'Player 1 ID': 689390, 'Player 1 Name': 'Marek Kostal',
    'Player 2 ID': 387428, 'Player 2 Name': 'Daniel Tuma',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10688011,
    'Player 1 ID': 1012124, 'Player 1 Name': 'Michal Vondrak',
    'Player 2 ID': 339298, 'Player 2 Name': 'Tomas Holik',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P1
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10688012,
    'Player 1 ID': 770032, 'Player 1 Name': 'Miloslav Lubas',
    'Player 2 ID': 654692, 'Player 2 Name': 'Pavel Sprynar',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10688036,
    'Player 1 ID': 765084, 'Player 1 Name': 'Radomir Vidlicka',
    'Player 2 ID': 686100, 'Player 2 Name': 'Jakub Levicky',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10688060,
    'Player 1 ID': 957132, 'Player 1 Name': 'Vaclav Kosar',
    'Player 2 ID': 1166071, 'Player 2 Name': 'Jiri Havel',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687717,
    'Player 1 ID': 548705, 'Player 1 Name': 'Jiri Plachy',
    'Player 2 ID': 339299, 'Player 2 Name': 'Miroslav Svedik',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P1
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10688024,
    'Player 1 ID': 1022513, 'Player 1 Name': 'Milan Cetner',
    'Player 2 ID': 750300, 'Player 2 Name': 'Michal Syroha',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688072,
    'Player 1 ID': 954785, 'Player 1 Name': 'Martin Sychra',
    'Player 2 ID': 798958, 'Player 2 Name': 'Jiri Zuzanek',
    'P1_ML': 275, # Actual Decimal: 3.75, Winner: P2
    'P2_ML': -400  # Actual Decimal: 1.25
},
{
    'Match ID': 10688048,
    'Player 1 ID': 1098619, 'Player 1 Name': 'Jaroslav Bares',
    'Player 2 ID': 1085799, 'Player 2 Name': 'Martin Vizek',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P2
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10688069,
    'Player 1 ID': 689440, 'Player 1 Name': 'Bohuslav Kaloc',
    'Player 2 ID': 765084, 'Player 2 Name': 'Radomir Vidlicka',
    'P1_ML': 300, # Actual Decimal: 4.00, Winner: P2
    'P2_ML': -450  # Actual Decimal: 1.22
},
{
    'Match ID': 10687857,
    'Player 1 ID': 339298, 'Player 1 Name': 'Tomas Holik',
    'Player 2 ID': 798958, 'Player 2 Name': 'Jiri Zuzanek',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10688034,
    'Player 1 ID': 339299, 'Player 1 Name': 'Miroslav Svedik',
    'Player 2 ID': 689390, 'Player 2 Name': 'Marek Kostal',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P2
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10687716,
    'Player 1 ID': 689390, 'Player 1 Name': 'Marek Kostal',
    'Player 2 ID': 548705, 'Player 2 Name': 'Jiri Plachy',
    'P1_ML': -275, # Actual Decimal: 1.36, Winner: P1
    'P2_ML': 200  # Actual Decimal: 3.00
},
{
    'Match ID': 10687853,
    'Player 1 ID': 701363, 'Player 1 Name': 'Jaromir Cernik',
    'Player 2 ID': 654692, 'Player 2 Name': 'Pavel Sprynar',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P1
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10687859,
    'Player 1 ID': 1089695, 'Player 1 Name': 'Milos Pospisil',
    'Player 2 ID': 750300, 'Player 2 Name': 'Michal Syroha',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10687852,
    'Player 1 ID': 686100, 'Player 1 Name': 'Jakub Levicky',
    'Player 2 ID': 689440, 'Player 2 Name': 'Bohuslav Kaloc',
    'P1_ML': -500, # Actual Decimal: 1.20, Winner: P1
    'P2_ML': 333  # Actual Decimal: 4.33
},
{
    'Match ID': 10687864,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 347091, 'Player 2 Name': 'Tomas Varnuska',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P1
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10687868,
    'Player 1 ID': 1012124, 'Player 1 Name': 'Michal Vondrak',
    'Player 2 ID': 954785, 'Player 2 Name': 'Martin Sychra',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P1
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10687885,
    'Player 1 ID': 1158338, 'Player 1 Name': 'Oldrich Vaclahovsky',
    'Player 2 ID': 925949, 'Player 2 Name': 'Ivan Jemelka',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P2
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10691150,
    'Player 1 ID': 798958, 'Player 1 Name': 'Jiri Zuzanek',
    'Player 2 ID': 954785, 'Player 2 Name': 'Martin Sychra',
    'P1_ML': -333, # Actual Decimal: 1.30, Winner: P1
    'P2_ML': 240  # Actual Decimal: 3.40
},
{
    'Match ID': 10687884,
    'Player 1 ID': 798958, 'Player 1 Name': 'Jiri Zuzanek',
    'Player 2 ID': 954785, 'Player 2 Name': 'Martin Sychra',
    'P1_ML': -400, # Actual Decimal: 1.25, Winner: P1
    'P2_ML': 275  # Actual Decimal: 3.75
},
{
    'Match ID': 10687896,
    'Player 1 ID': 765084, 'Player 1 Name': 'Radomir Vidlicka',
    'Player 2 ID': 1163571, 'Player 2 Name': 'Jindrich Simecek',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10687869,
    'Player 1 ID': 957132, 'Player 1 Name': 'Vaclav Kosar',
    'Player 2 ID': 1022513, 'Player 2 Name': 'Milan Cetner',
    'P1_ML': -300, # Actual Decimal: 1.33, Winner: P1
    'P2_ML': 225  # Actual Decimal: 3.25
},
{
    'Match ID': 10687856,
    'Player 1 ID': 770032, 'Player 1 Name': 'Miloslav Lubas',
    'Player 2 ID': 1142181, 'Player 2 Name': 'Jan Srba',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10687870,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 339299, 'Player 2 Name': 'Miroslav Svedik',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10691218,
    'Player 1 ID': 341166, 'Player 1 Name': 'Milan Klement',
    'Player 2 ID': 339298, 'Player 2 Name': 'Tomas Holik',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P2
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10691212,
    'Player 1 ID': 701363, 'Player 1 Name': 'Jaromir Cernik',
    'Player 2 ID': 611386, 'Player 2 Name': 'Ludek Madle',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10691211,
    'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma',
    'Player 2 ID': 750300, 'Player 2 Name': 'Michal Syroha',
    'P1_ML': -400, # Actual Decimal: 1.25, Winner: P1
    'P2_ML': 275  # Actual Decimal: 3.75
},
{
    'Match ID': 10691223,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 636504, 'Player 2 Name': 'Vladimir Cermak',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10694165,
    'Player 1 ID': 338649, 'Player 1 Name': 'Jan Zajicek',
    'Player 2 ID': 257586, 'Player 2 Name': 'Lukas Boruvka',
    'P1_ML': 300, # Actual Decimal: 4.00, Winner: P2
    'P2_ML': -450  # Actual Decimal: 1.22
},
{
    'Match ID': 10694157,
    'Player 1 ID': 1088851, 'Player 1 Name': 'Oleksandr Kolisnyk',
    'Player 2 ID': 608764, 'Player 2 Name': 'Michal Jezek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10694154,
    'Player 1 ID': 959029, 'Player 1 Name': 'Pavel Berdych',
    'Player 2 ID': 1005699, 'Player 2 Name': 'Miroslav Tuma',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10694153,
    'Player 1 ID': 1149975, 'Player 1 Name': 'Jan Mecl Jr',
    'Player 2 ID': 341888, 'Player 2 Name': 'Martin Huk',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P2
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10693192,
    'Player 1 ID': 358665, 'Player 1 Name': 'Marek Sedlak',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10688018,
    'Player 1 ID': 1157765, 'Player 1 Name': 'Michal Hrabec',
    'Player 2 ID': 696884, 'Player 2 Name': 'Michal Wollny',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10688062,
    'Player 1 ID': 687549, 'Player 1 Name': 'Michal Vavrecka',
    'Player 2 ID': 338586, 'Player 2 Name': 'Marek Fabini',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P2
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10688066,
    'Player 1 ID': 388336, 'Player 1 Name': 'Vladimir Postelt',
    'Player 2 ID': 807181, 'Player 2 Name': 'Frantisek Briza',
    'P1_ML': -275, # Actual Decimal: 1.36, Winner: P1
    'P2_ML': 200  # Actual Decimal: 3.00
},
{
    'Match ID': 10688186,
    'Player 1 ID': 1090473, 'Player 1 Name': 'Karel Schopf',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10687734,
    'Player 1 ID': 678236, 'Player 1 Name': 'Tomas Turek',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P2
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10688030,
    'Player 1 ID': 358665, 'Player 1 Name': 'Marek Sedlak',
    'Player 2 ID': 339709, 'Player 2 Name': 'Jiri Louda',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688037,
    'Player 1 ID': 1012124, 'Player 1 Name': 'Michal Vondrak',
    'Player 2 ID': 689390, 'Player 2 Name': 'Marek Kostal',
    'P1_ML': -250, # Actual Decimal: 1.40, Winner: P1
    'P2_ML': 175  # Actual Decimal: 2.75
},
{
    'Match ID': 10688038,
    'Player 1 ID': 1179684, 'Player 1 Name': 'Tomas Dousek',
    'Player 2 ID': 688067, 'Player 2 Name': 'Tadeas Zika',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10691280,
    'Player 1 ID': 608765, 'Player 1 Name': 'Tibor Kolenic',
    'Player 2 ID': 1080540, 'Player 2 Name': 'Martin Kir',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P2
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10687733,
    'Player 1 ID': 338586, 'Player 1 Name': 'Marek Fabini',
    'Player 2 ID': 678236, 'Player 2 Name': 'Tomas Turek',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688039,
    'Player 1 ID': 955815, 'Player 1 Name': 'Vitek Dvorak',
    'Player 2 ID': 358665, 'Player 2 Name': 'Marek Sedlak',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688063,
    'Player 1 ID': 807181, 'Player 1 Name': 'Frantisek Briza',
    'Player 2 ID': 608765, 'Player 2 Name': 'Tibor Kolenic',
    'P1_ML': 150, # Actual Decimal: 2.50, Winner: P2
    'P2_ML': -200  # Actual Decimal: 1.50
},
{
    'Match ID': 10688179,
    'Player 1 ID': 1085799, 'Player 1 Name': 'Martin Vizek',
    'Player 2 ID': 1012124, 'Player 2 Name': 'Michal Vondrak',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10688195,
    'Player 1 ID': 696884, 'Player 1 Name': 'Michal Wollny',
    'Player 2 ID': 1179684, 'Player 2 Name': 'Tomas Dousek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10688040,
    'Player 1 ID': 339709, 'Player 1 Name': 'Jiri Louda',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10688071,
    'Player 1 ID': 1142181, 'Player 1 Name': 'Jan Srba',
    'Player 2 ID': 654692, 'Player 2 Name': 'Pavel Sprynar',
    'P1_ML': 150, # Actual Decimal: 2.50, Winner: P2
    'P2_ML': -200  # Actual Decimal: 1.50
},
{
    'Match ID': 10688207,
    'Player 1 ID': 1163571, 'Player 1 Name': 'Jindrich Simecek',
    'Player 2 ID': 686100, 'Player 2 Name': 'Jakub Levicky',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10688015,
    'Player 1 ID': 701363, 'Player 1 Name': 'Jaromir Cernik',
    'Player 2 ID': 770032, 'Player 2 Name': 'Miloslav Lubas',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P1
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10688034,
    'Player 1 ID': 339299, 'Player 1 Name': 'Miroslav Svedik',
    'Player 2 ID': 689390, 'Player 2 Name': 'Marek Kostal',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P2
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10688035,
    'Player 1 ID': 750300, 'Player 1 Name': 'Michal Syroha',
    'Player 2 ID': 957132, 'Player 2 Name': 'Vaclav Kosar',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10688067,
    'Player 1 ID': 798958, 'Player 1 Name': 'Jiri Zuzanek',
    'Player 2 ID': 1012124, 'Player 2 Name': 'Michal Vondrak',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688069,
    'Player 1 ID': 689440, 'Player 1 Name': 'Bohuslav Kaloc',
    'Player 2 ID': 765084, 'Player 2 Name': 'Radomir Vidlicka',
    'P1_ML': 300, # Actual Decimal: 4.00, Winner: P2
    'P2_ML': -450  # Actual Decimal: 1.22
},
{
    'Match ID': 10688010,
    'Player 1 ID': 689390, 'Player 1 Name': 'Marek Kostal',
    'Player 2 ID': 387428, 'Player 2 Name': 'Daniel Tuma',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10688011,
    'Player 1 ID': 1012124, 'Player 1 Name': 'Michal Vondrak',
    'Player 2 ID': 339298, 'Player 2 Name': 'Tomas Holik',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P1
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10688049,
    'Player 1 ID': 888755, 'Player 1 Name': 'Vratislav Petracek',
    'Player 2 ID': 687549, 'Player 2 Name': 'Michal Vavrecka',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10688012,
    'Player 1 ID': 770032, 'Player 1 Name': 'Miloslav Lubas',
    'Player 2 ID': 654692, 'Player 2 Name': 'Pavel Sprynar',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10688060,
    'Player 1 ID': 957132, 'Player 1 Name': 'Vaclav Kosar',
    'Player 2 ID': 1166071, 'Player 2 Name': 'Jiri Havel',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687717,
    'Player 1 ID': 548705, 'Player 1 Name': 'Jiri Plachy',
    'Player 2 ID': 339299, 'Player 2 Name': 'Miroslav Svedik',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P1
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10688024,
    'Player 1 ID': 1022513, 'Player 1 Name': 'Milan Cetner',
    'Player 2 ID': 750300, 'Player 2 Name': 'Michal Syroha',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688048,
    'Player 1 ID': 1098619, 'Player 1 Name': 'Jaroslav Bares',
    'Player 2 ID': 1085799, 'Player 2 Name': 'Martin Vizek',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P2
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10688072,
    'Player 1 ID': 954785, 'Player 1 Name': 'Martin Sychra',
    'Player 2 ID': 798958, 'Player 2 Name': 'Jiri Zuzanek',
    'P1_ML': 275, # Actual Decimal: 3.75, Winner: P2
    'P2_ML': -400  # Actual Decimal: 1.25
},
{
    'Match ID': 10688075,
    'Player 1 ID': 1142181, 'Player 1 Name': 'Jan Srba',
    'Player 2 ID': 701363, 'Player 2 Name': 'Jaromir Cernik',
    'P1_ML': 150, # Actual Decimal: 2.50, Winner: P2
    'P2_ML': -200  # Actual Decimal: 1.50
},
{
    'Match ID': 10688192,
    'Player 1 ID': 1163571, 'Player 1 Name': 'Jindrich Simecek',
    'Player 2 ID': 689440, 'Player 2 Name': 'Bohuslav Kaloc',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P2
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10688036,
    'Player 1 ID': 765084, 'Player 1 Name': 'Radomir Vidlicka',
    'Player 2 ID': 686100, 'Player 2 Name': 'Jakub Levicky',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10688029,
    'Player 1 ID': 1022513, 'Player 1 Name': 'Milan Cetner',
    'Player 2 ID': 1166071, 'Player 2 Name': 'Jiri Havel',
    'P1_ML': 200, # Actual Decimal: 3.00, Winner: P2
    'P2_ML': -275  # Actual Decimal: 1.36
},
{
    'Match ID': 10688077,
    'Player 1 ID': 689390, 'Player 1 Name': 'Marek Kostal',
    'Player 2 ID': 1098619, 'Player 2 Name': 'Jaroslav Bares',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10691279,
    'Player 1 ID': 1080540, 'Player 1 Name': 'Martin Kir',
    'Player 2 ID': 388336, 'Player 2 Name': 'Vladimir Postelt',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10687730,
    'Player 1 ID': 687549, 'Player 1 Name': 'Michal Vavrecka',
    'Player 2 ID': 678236, 'Player 2 Name': 'Tomas Turek',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10688019,
    'Player 1 ID': 1090473, 'Player 1 Name': 'Karel Schopf',
    'Player 2 ID': 358665, 'Player 2 Name': 'Marek Sedlak',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688068,
    'Player 1 ID': 1098619, 'Player 1 Name': 'Jaroslav Bares',
    'Player 2 ID': 1012124, 'Player 2 Name': 'Michal Vondrak',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688193,
    'Player 1 ID': 1157765, 'Player 1 Name': 'Michal Hrabec',
    'Player 2 ID': 1179684, 'Player 2 Name': 'Tomas Dousek',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P2
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10688020,
    'Player 1 ID': 1085799, 'Player 1 Name': 'Martin Vizek',
    'Player 2 ID': 689390, 'Player 2 Name': 'Marek Kostal',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10688052,
    'Player 1 ID': 955815, 'Player 1 Name': 'Vitek Dvorak',
    'Player 2 ID': 339709, 'Player 2 Name': 'Jiri Louda',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688053,
    'Player 1 ID': 696884, 'Player 1 Name': 'Michal Wollny',
    'Player 2 ID': 688067, 'Player 2 Name': 'Tadeas Zika',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10688051,
    'Player 1 ID': 688067, 'Player 1 Name': 'Tadeas Zika',
    'Player 2 ID': 1157765, 'Player 2 Name': 'Michal Hrabec',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P1
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10692039,
    'Player 1 ID': 807181, 'Player 1 Name': 'Frantisek Briza',
    'Player 2 ID': 1080540, 'Player 2 Name': 'Martin Kir',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P2
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10693122,
    'Player 1 ID': 608765, 'Player 1 Name': 'Tibor Kolenic',
    'Player 2 ID': 807181, 'Player 2 Name': 'Frantisek Briza',
    'P1_ML': -275, # Actual Decimal: 1.36, Winner: P2
    'P2_ML': 200  # Actual Decimal: 3.00
},
{
    'Match ID': 10688178,
    'Player 1 ID': 338586, 'Player 1 Name': 'Marek Fabini',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687887,
    'Player 1 ID': 954785, 'Player 1 Name': 'Martin Sychra',
    'Player 2 ID': 339298, 'Player 2 Name': 'Tomas Holik',
    'P1_ML': 150, # Actual Decimal: 2.50, Winner: P1
    'P2_ML': -200  # Actual Decimal: 1.50
},
{
    'Match ID': 10688061,
    'Player 1 ID': 388336, 'Player 1 Name': 'Vladimir Postelt',
    'Player 2 ID': 608765, 'Player 2 Name': 'Tibor Kolenic',
    'P1_ML': -275, # Actual Decimal: 1.36, Winner: P2
    'P2_ML': 200  # Actual Decimal: 3.00
},
{
    'Match ID': 10687886,
    'Player 1 ID': 654692, 'Player 1 Name': 'Pavel Sprynar',
    'Player 2 ID': 701363, 'Player 2 Name': 'Jaromir Cernik',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P1
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10682270,
    'Player 1 ID': 955815, 'Player 1 Name': 'Vitek Dvorak',
    'Player 2 ID': 355559, 'Player 2 Name': 'Michal Regner',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10683008,
    'Player 1 ID': 922416, 'Player 1 Name': 'Jakub Vales',
    'Player 2 ID': 559309, 'Player 2 Name': 'Lukas Zeman',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682271,
    'Player 1 ID': 380220, 'Player 1 Name': 'Patrik Pycha',
    'Player 2 ID': 1015754, 'Player 2 Name': 'Marek Chlebecek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682297,
    'Player 1 ID': 1090473, 'Player 1 Name': 'Karel Schopf',
    'Player 2 ID': 546478, 'Player 2 Name': 'Jan Sucharda',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682326,
    'Player 1 ID': 888755, 'Player 1 Name': 'Vratislav Petracek',
    'Player 2 ID': 687549, 'Player 2 Name': 'Michal Vavrecka',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682272,
    'Player 1 ID': 1133447, 'Player 1 Name': 'Petr Sebera',
    'Player 2 ID': 553241, 'Player 2 Name': 'Martin Sobisek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682298,
    'Player 1 ID': 955815, 'Player 1 Name': 'Vitek Dvorak',
    'Player 2 ID': 342540, 'Player 2 Name': 'Matous Klimenta',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682302,
    'Player 1 ID': 922416, 'Player 1 Name': 'Jakub Vales',
    'Player 2 ID': 341888, 'Player 2 Name': 'Martin Huk',
    'P1_ML': 900, # Actual Decimal: 10.00, Winner: P2
    'P2_ML': -3333  # Actual Decimal: 1.03
},
{
    'Match ID': 10682237,
    'Player 1 ID': 687549, 'Player 1 Name': 'Michal Vavrecka',
    'Player 2 ID': 559309, 'Player 2 Name': 'Lukas Zeman',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682242,
    'Player 1 ID': 546478, 'Player 1 Name': 'Jan Sucharda',
    'Player 2 ID': 355559, 'Player 2 Name': 'Michal Regner',
    'P1_ML': 900, # Actual Decimal: 10.00, Winner: P2
    'P2_ML': -3333  # Actual Decimal: 1.03
},
{
    'Match ID': 10682322,
    'Player 1 ID': 1015754, 'Player 1 Name': 'Marek Chlebecek',
    'Player 2 ID': 1098619, 'Player 2 Name': 'Jaroslav Bares',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10690318,
    'Player 1 ID': 922416, 'Player 1 Name': 'Jakub Vales',
    'Player 2 ID': 341888, 'Player 2 Name': 'Martin Huk',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10690321,
    'Player 1 ID': 553241, 'Player 1 Name': 'Martin Sobisek',
    'Player 2 ID': 1133447, 'Player 2 Name': 'Petr Sebera',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10690322,
    'Player 1 ID': 342540, 'Player 1 Name': 'Matous Klimenta',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10690391,
    'Player 1 ID': 888755, 'Player 1 Name': 'Vratislav Petracek',
    'Player 2 ID': 559309, 'Player 2 Name': 'Lukas Zeman',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10690392,
    'Player 1 ID': 355559, 'Player 1 Name': 'Michal Regner',
    'Player 2 ID': 546478, 'Player 2 Name': 'Jan Sucharda',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10690393,
    'Player 1 ID': 1015754, 'Player 1 Name': 'Marek Chlebecek',
    'Player 2 ID': 380220, 'Player 2 Name': 'Patrik Pycha',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10687719,
    'Player 1 ID': 1030283, 'Player 1 Name': 'Tomas Vinter',
    'Player 2 ID': 387428, 'Player 2 Name': 'Daniel Tuma',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P2
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10687871,
    'Player 1 ID': 1157770, 'Player 1 Name': 'Evzen Rychlik',
    'Player 2 ID': 1158338, 'Player 2 Name': 'Oldrich Vaclahovsky',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682269,
    'Player 1 ID': 1133447, 'Player 1 Name': 'Petr Sebera',
    'Player 2 ID': 1098619, 'Player 2 Name': 'Jaroslav Bares',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682986,
    'Player 1 ID': 342540, 'Player 1 Name': 'Matous Klimenta',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682311,
    'Player 1 ID': 553241, 'Player 1 Name': 'Martin Sobisek',
    'Player 2 ID': 380220, 'Player 2 Name': 'Patrik Pycha',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682236,
    'Player 1 ID': 341888, 'Player 1 Name': 'Martin Huk',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10687878,
    'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma',
    'Player 2 ID': 1089695, 'Player 2 Name': 'Milos Pospisil',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687721,
    'Player 1 ID': 548705, 'Player 1 Name': 'Jiri Plachy',
    'Player 2 ID': 387428, 'Player 2 Name': 'Daniel Tuma',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682235,
    'Player 1 ID': 546478, 'Player 1 Name': 'Jan Sucharda',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682284,
    'Player 1 ID': 687549, 'Player 1 Name': 'Michal Vavrecka',
    'Player 2 ID': 922416, 'Player 2 Name': 'Jakub Vales',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10683003,
    'Player 1 ID': 1015754, 'Player 1 Name': 'Marek Chlebecek',
    'Player 2 ID': 1133447, 'Player 2 Name': 'Petr Sebera',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10687890,
    'Player 1 ID': 611386, 'Player 1 Name': 'Ludek Madle',
    'Player 2 ID': 701363, 'Player 2 Name': 'Jaromir Cernik',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P2
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10687872,
    'Player 1 ID': 636504, 'Player 1 Name': 'Vladimir Cermak',
    'Player 2 ID': 347091, 'Player 2 Name': 'Tomas Varnuska',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10691150,
    'Player 1 ID': 798958, 'Player 1 Name': 'Jiri Zuzanek',
    'Player 2 ID': 954785, 'Player 2 Name': 'Martin Sychra',
    'P1_ML': -333, # Actual Decimal: 1.30, Winner: P1
    'P2_ML': 240  # Actual Decimal: 3.40
},
{
    'Match ID': 10687884,
    'Player 1 ID': 798958, 'Player 1 Name': 'Jiri Zuzanek',
    'Player 2 ID': 954785, 'Player 2 Name': 'Martin Sychra',
    'P1_ML': -400, # Actual Decimal: 1.25, Winner: P1
    'P2_ML': 275  # Actual Decimal: 3.75
},
{
    'Match ID': 10687857,
    'Player 1 ID': 339298, 'Player 1 Name': 'Tomas Holik',
    'Player 2 ID': 798958, 'Player 2 Name': 'Jiri Zuzanek',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10687852,
    'Player 1 ID': 686100, 'Player 1 Name': 'Jakub Levicky',
    'Player 2 ID': 689440, 'Player 2 Name': 'Bohuslav Kaloc',
    'P1_ML': -500, # Actual Decimal: 1.20, Winner: P1
    'P2_ML': 333  # Actual Decimal: 4.33
},
{
    'Match ID': 10687716,
    'Player 1 ID': 689390, 'Player 1 Name': 'Marek Kostal',
    'Player 2 ID': 548705, 'Player 2 Name': 'Jiri Plachy',
    'P1_ML': -275, # Actual Decimal: 1.36, Winner: P1
    'P2_ML': 200  # Actual Decimal: 3.00
},
{
    'Match ID': 10687885,
    'Player 1 ID': 1158338, 'Player 1 Name': 'Oldrich Vaclahovsky',
    'Player 2 ID': 925949, 'Player 2 Name': 'Ivan Jemelka',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P2
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10687858,
    'Player 1 ID': 1166071, 'Player 1 Name': 'Jiri Havel',
    'Player 2 ID': 750300, 'Player 2 Name': 'Michal Syroha',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P1
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10687896,
    'Player 1 ID': 765084, 'Player 1 Name': 'Radomir Vidlicka',
    'Player 2 ID': 1163571, 'Player 2 Name': 'Jindrich Simecek',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10687870,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 339299, 'Player 2 Name': 'Miroslav Svedik',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10687869,
    'Player 1 ID': 957132, 'Player 1 Name': 'Vaclav Kosar',
    'Player 2 ID': 1022513, 'Player 2 Name': 'Milan Cetner',
    'P1_ML': -300, # Actual Decimal: 1.33, Winner: P1
    'P2_ML': 225  # Actual Decimal: 3.25
},
{
    'Match ID': 10687856,
    'Player 1 ID': 770032, 'Player 1 Name': 'Miloslav Lubas',
    'Player 2 ID': 1142181, 'Player 2 Name': 'Jan Srba',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10691223,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 636504, 'Player 2 Name': 'Vladimir Cermak',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10691218,
    'Player 1 ID': 341166, 'Player 1 Name': 'Milan Klement',
    'Player 2 ID': 339298, 'Player 2 Name': 'Tomas Holik',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P2
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10691212,
    'Player 1 ID': 701363, 'Player 1 Name': 'Jaromir Cernik',
    'Player 2 ID': 611386, 'Player 2 Name': 'Ludek Madle',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10691211,
    'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma',
    'Player 2 ID': 750300, 'Player 2 Name': 'Michal Syroha',
    'P1_ML': -400, # Actual Decimal: 1.25, Winner: P1
    'P2_ML': 275  # Actual Decimal: 3.75
},
{
    'Match ID': 10687864,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 347091, 'Player 2 Name': 'Tomas Varnuska',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P1
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10687868,
    'Player 1 ID': 1012124, 'Player 1 Name': 'Michal Vondrak',
    'Player 2 ID': 954785, 'Player 2 Name': 'Martin Sychra',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P1
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10687882,
    'Player 1 ID': 1158338, 'Player 1 Name': 'Oldrich Vaclahovsky',
    'Player 2 ID': 1159074, 'Player 2 Name': 'Laurent Lasota',
    'P1_ML': 200, # Actual Decimal: 3.00, Winner: P2
    'P2_ML': -275  # Actual Decimal: 1.36
},
{
    'Match ID': 10687879,
    'Player 1 ID': 339298, 'Player 1 Name': 'Tomas Holik',
    'Player 2 ID': 954785, 'Player 2 Name': 'Martin Sychra',
    'P1_ML': -187, # Actual Decimal: 1.53, Winner: P1
    'P2_ML': 137  # Actual Decimal: 2.38
},
{
    'Match ID': 10687880,
    'Player 1 ID': 1166071, 'Player 1 Name': 'Jiri Havel',
    'Player 2 ID': 750300, 'Player 2 Name': 'Michal Syroha',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687889,
    'Player 1 ID': 360808, 'Player 1 Name': 'Jiri Svec',
    'Player 2 ID': 654692, 'Player 2 Name': 'Pavel Sprynar',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10687891,
    'Player 1 ID': 1159074, 'Player 1 Name': 'Laurent Lasota',
    'Player 2 ID': 925949, 'Player 2 Name': 'Ivan Jemelka',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10687855,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 636504, 'Player 2 Name': 'Vladimir Cermak',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P1
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10687865,
    'Player 1 ID': 1089695, 'Player 1 Name': 'Milos Pospisil',
    'Player 2 ID': 1166071, 'Player 2 Name': 'Jiri Havel',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10687866,
    'Player 1 ID': 701363, 'Player 1 Name': 'Jaromir Cernik',
    'Player 2 ID': 360808, 'Player 2 Name': 'Jiri Svec',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687881,
    'Player 1 ID': 798958, 'Player 1 Name': 'Jiri Zuzanek',
    'Player 2 ID': 339298, 'Player 2 Name': 'Tomas Holik',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10687859,
    'Player 1 ID': 1089695, 'Player 1 Name': 'Milos Pospisil',
    'Player 2 ID': 750300, 'Player 2 Name': 'Michal Syroha',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10687718,
    'Player 1 ID': 347091, 'Player 1 Name': 'Tomas Varnuska',
    'Player 2 ID': 1030283, 'Player 2 Name': 'Tomas Vinter',
    'P1_ML': 137, # Actual Decimal: 2.38, Winner: P1
    'P2_ML': -187  # Actual Decimal: 1.53
},
{
    'Match ID': 10688485,
    'Player 1 ID': 341166, 'Player 1 Name': 'Milan Klement',
    'Player 2 ID': 798958, 'Player 2 Name': 'Jiri Zuzanek',
    'P1_ML': -250, # Actual Decimal: 1.40, Winner: P1
    'P2_ML': 175  # Actual Decimal: 2.75
},
{
    'Match ID': 10687862,
    'Player 1 ID': 654692, 'Player 1 Name': 'Pavel Sprynar',
    'Player 2 ID': 611386, 'Player 2 Name': 'Ludek Madle',
    'P1_ML': 275, # Actual Decimal: 3.75, Winner: P2
    'P2_ML': -400  # Actual Decimal: 1.25
},
{
    'Match ID': 10687867,
    'Player 1 ID': 925949, 'Player 1 Name': 'Ivan Jemelka',
    'Player 2 ID': 1157770, 'Player 2 Name': 'Evzen Rychlik',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687883,
    'Player 1 ID': 954785, 'Player 1 Name': 'Martin Sychra',
    'Player 2 ID': 341166, 'Player 2 Name': 'Milan Klement',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P2
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10687720,
    'Player 1 ID': 1030283, 'Player 1 Name': 'Tomas Vinter',
    'Player 2 ID': 636504, 'Player 2 Name': 'Vladimir Cermak',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10687860,
    'Player 1 ID': 611386, 'Player 1 Name': 'Ludek Madle',
    'Player 2 ID': 360808, 'Player 2 Name': 'Jiri Svec',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10687863,
    'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma',
    'Player 2 ID': 1166071, 'Player 2 Name': 'Jiri Havel',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P1
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10687873,
    'Player 1 ID': 1157770, 'Player 1 Name': 'Evzen Rychlik',
    'Player 2 ID': 1159074, 'Player 2 Name': 'Laurent Lasota',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10687888,
    'Player 1 ID': 341166, 'Player 1 Name': 'Milan Klement',
    'Player 2 ID': 339298, 'Player 2 Name': 'Tomas Holik',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10687853,
    'Player 1 ID': 701363, 'Player 1 Name': 'Jaromir Cernik',
    'Player 2 ID': 654692, 'Player 2 Name': 'Pavel Sprynar',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P1
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10687861,
    'Player 1 ID': 750300, 'Player 1 Name': 'Michal Syroha',
    'Player 2 ID': 1005699, 'Player 2 Name': 'Miroslav Tuma',
    'P1_ML': 250, # Actual Decimal: 3.50, Winner: P2
    'P2_ML': -350  # Actual Decimal: 1.28
},
{
    'Match ID': 10694165,
    'Player 1 ID': 338649, 'Player 1 Name': 'Jan Zajicek',
    'Player 2 ID': 257586, 'Player 2 Name': 'Lukas Boruvka',
    'P1_ML': 300, # Actual Decimal: 4.00, Winner: P2
    'P2_ML': -450  # Actual Decimal: 1.22
},
{
    'Match ID': 10694157,
    'Player 1 ID': 1088851, 'Player 1 Name': 'Oleksandr Kolisnyk',
    'Player 2 ID': 608764, 'Player 2 Name': 'Michal Jezek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10694154,
    'Player 1 ID': 959029, 'Player 1 Name': 'Pavel Berdych',
    'Player 2 ID': 1005699, 'Player 2 Name': 'Miroslav Tuma',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10694153,
    'Player 1 ID': 1149975, 'Player 1 Name': 'Jan Mecl Jr',
    'Player 2 ID': 341888, 'Player 2 Name': 'Martin Huk',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P2
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10693192,
    'Player 1 ID': 358665, 'Player 1 Name': 'Marek Sedlak',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10688018,
    'Player 1 ID': 1157765, 'Player 1 Name': 'Michal Hrabec',
    'Player 2 ID': 696884, 'Player 2 Name': 'Michal Wollny',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P1
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10688062,
    'Player 1 ID': 687549, 'Player 1 Name': 'Michal Vavrecka',
    'Player 2 ID': 338586, 'Player 2 Name': 'Marek Fabini',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P2
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10688066,
    'Player 1 ID': 388336, 'Player 1 Name': 'Vladimir Postelt',
    'Player 2 ID': 807181, 'Player 2 Name': 'Frantisek Briza',
    'P1_ML': -275, # Actual Decimal: 1.36, Winner: P1
    'P2_ML': 200  # Actual Decimal: 3.00
},
{
    'Match ID': 10688186,
    'Player 1 ID': 1090473, 'Player 1 Name': 'Karel Schopf',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10687734,
    'Player 1 ID': 678236, 'Player 1 Name': 'Tomas Turek',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P2
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10688030,
    'Player 1 ID': 358665, 'Player 1 Name': 'Marek Sedlak',
    'Player 2 ID': 339709, 'Player 2 Name': 'Jiri Louda',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688037,
    'Player 1 ID': 1012124, 'Player 1 Name': 'Michal Vondrak',
    'Player 2 ID': 689390, 'Player 2 Name': 'Marek Kostal',
    'P1_ML': -250, # Actual Decimal: 1.40, Winner: P1
    'P2_ML': 175  # Actual Decimal: 2.75
},
{
    'Match ID': 10688038,
    'Player 1 ID': 1179684, 'Player 1 Name': 'Tomas Dousek',
    'Player 2 ID': 688067, 'Player 2 Name': 'Tadeas Zika',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10691280,
    'Player 1 ID': 608765, 'Player 1 Name': 'Tibor Kolenic',
    'Player 2 ID': 1080540, 'Player 2 Name': 'Martin Kir',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P2
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10687733,
    'Player 1 ID': 338586, 'Player 1 Name': 'Marek Fabini',
    'Player 2 ID': 678236, 'Player 2 Name': 'Tomas Turek',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688039,
    'Player 1 ID': 955815, 'Player 1 Name': 'Vitek Dvorak',
    'Player 2 ID': 358665, 'Player 2 Name': 'Marek Sedlak',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688063,
    'Player 1 ID': 807181, 'Player 1 Name': 'Frantisek Briza',
    'Player 2 ID': 608765, 'Player 2 Name': 'Tibor Kolenic',
    'P1_ML': 150, # Actual Decimal: 2.50, Winner: P2
    'P2_ML': -200  # Actual Decimal: 1.50
},
{
    'Match ID': 10688179,
    'Player 1 ID': 1085799, 'Player 1 Name': 'Martin Vizek',
    'Player 2 ID': 1012124, 'Player 2 Name': 'Michal Vondrak',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10688195,
    'Player 1 ID': 696884, 'Player 1 Name': 'Michal Wollny',
    'Player 2 ID': 1179684, 'Player 2 Name': 'Tomas Dousek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10688040,
    'Player 1 ID': 339709, 'Player 1 Name': 'Jiri Louda',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10688071,
    'Player 1 ID': 1142181, 'Player 1 Name': 'Jan Srba',
    'Player 2 ID': 654692, 'Player 2 Name': 'Pavel Sprynar',
    'P1_ML': 150, # Actual Decimal: 2.50, Winner: P2
    'P2_ML': -200  # Actual Decimal: 1.50
},
{
    'Match ID': 10688207,
    'Player 1 ID': 1163571, 'Player 1 Name': 'Jindrich Simecek',
    'Player 2 ID': 686100, 'Player 2 Name': 'Jakub Levicky',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10688015,
    'Player 1 ID': 701363, 'Player 1 Name': 'Jaromir Cernik',
    'Player 2 ID': 770032, 'Player 2 Name': 'Miloslav Lubas',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P1
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10688034,
    'Player 1 ID': 339299, 'Player 1 Name': 'Miroslav Svedik',
    'Player 2 ID': 689390, 'Player 2 Name': 'Marek Kostal',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P2
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10688035,
    'Player 1 ID': 750300, 'Player 1 Name': 'Michal Syroha',
    'Player 2 ID': 957132, 'Player 2 Name': 'Vaclav Kosar',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10688067,
    'Player 1 ID': 798958, 'Player 1 Name': 'Jiri Zuzanek',
    'Player 2 ID': 1012124, 'Player 2 Name': 'Michal Vondrak',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688069,
    'Player 1 ID': 689440, 'Player 1 Name': 'Bohuslav Kaloc',
    'Player 2 ID': 765084, 'Player 2 Name': 'Radomir Vidlicka',
    'P1_ML': 300, # Actual Decimal: 4.00, Winner: P2
    'P2_ML': -450  # Actual Decimal: 1.22
},
{
    'Match ID': 10688010,
    'Player 1 ID': 689390, 'Player 1 Name': 'Marek Kostal',
    'Player 2 ID': 387428, 'Player 2 Name': 'Daniel Tuma',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10688011,
    'Player 1 ID': 1012124, 'Player 1 Name': 'Michal Vondrak',
    'Player 2 ID': 339298, 'Player 2 Name': 'Tomas Holik',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P1
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10688049,
    'Player 1 ID': 888755, 'Player 1 Name': 'Vratislav Petracek',
    'Player 2 ID': 687549, 'Player 2 Name': 'Michal Vavrecka',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10688012,
    'Player 1 ID': 770032, 'Player 1 Name': 'Miloslav Lubas',
    'Player 2 ID': 654692, 'Player 2 Name': 'Pavel Sprynar',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10688060,
    'Player 1 ID': 957132, 'Player 1 Name': 'Vaclav Kosar',
    'Player 2 ID': 1166071, 'Player 2 Name': 'Jiri Havel',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687717,
    'Player 1 ID': 548705, 'Player 1 Name': 'Jiri Plachy',
    'Player 2 ID': 339299, 'Player 2 Name': 'Miroslav Svedik',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P1
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10688024,
    'Player 1 ID': 1022513, 'Player 1 Name': 'Milan Cetner',
    'Player 2 ID': 750300, 'Player 2 Name': 'Michal Syroha',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688048,
    'Player 1 ID': 1098619, 'Player 1 Name': 'Jaroslav Bares',
    'Player 2 ID': 1085799, 'Player 2 Name': 'Martin Vizek',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P2
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10688072,
    'Player 1 ID': 954785, 'Player 1 Name': 'Martin Sychra',
    'Player 2 ID': 798958, 'Player 2 Name': 'Jiri Zuzanek',
    'P1_ML': 275, # Actual Decimal: 3.75, Winner: P2
    'P2_ML': -400  # Actual Decimal: 1.25
},
{
    'Match ID': 10688075,
    'Player 1 ID': 1142181, 'Player 1 Name': 'Jan Srba',
    'Player 2 ID': 701363, 'Player 2 Name': 'Jaromir Cernik',
    'P1_ML': 150, # Actual Decimal: 2.50, Winner: P2
    'P2_ML': -200  # Actual Decimal: 1.50
},
{
    'Match ID': 10688192,
    'Player 1 ID': 1163571, 'Player 1 Name': 'Jindrich Simecek',
    'Player 2 ID': 689440, 'Player 2 Name': 'Bohuslav Kaloc',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P2
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10688036,
    'Player 1 ID': 765084, 'Player 1 Name': 'Radomir Vidlicka',
    'Player 2 ID': 686100, 'Player 2 Name': 'Jakub Levicky',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10688029,
    'Player 1 ID': 1022513, 'Player 1 Name': 'Milan Cetner',
    'Player 2 ID': 1166071, 'Player 2 Name': 'Jiri Havel',
    'P1_ML': 200, # Actual Decimal: 3.00, Winner: P2
    'P2_ML': -275  # Actual Decimal: 1.36
},
{
    'Match ID': 10688077,
    'Player 1 ID': 689390, 'Player 1 Name': 'Marek Kostal',
    'Player 2 ID': 1098619, 'Player 2 Name': 'Jaroslav Bares',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10691279,
    'Player 1 ID': 1080540, 'Player 1 Name': 'Martin Kir',
    'Player 2 ID': 388336, 'Player 2 Name': 'Vladimir Postelt',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10687730,
    'Player 1 ID': 687549, 'Player 1 Name': 'Michal Vavrecka',
    'Player 2 ID': 678236, 'Player 2 Name': 'Tomas Turek',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10688019,
    'Player 1 ID': 1090473, 'Player 1 Name': 'Karel Schopf',
    'Player 2 ID': 358665, 'Player 2 Name': 'Marek Sedlak',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688068,
    'Player 1 ID': 1098619, 'Player 1 Name': 'Jaroslav Bares',
    'Player 2 ID': 1012124, 'Player 2 Name': 'Michal Vondrak',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688193,
    'Player 1 ID': 1157765, 'Player 1 Name': 'Michal Hrabec',
    'Player 2 ID': 1179684, 'Player 2 Name': 'Tomas Dousek',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P2
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10688020,
    'Player 1 ID': 1085799, 'Player 1 Name': 'Martin Vizek',
    'Player 2 ID': 689390, 'Player 2 Name': 'Marek Kostal',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10688052,
    'Player 1 ID': 955815, 'Player 1 Name': 'Vitek Dvorak',
    'Player 2 ID': 339709, 'Player 2 Name': 'Jiri Louda',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10688053,
    'Player 1 ID': 696884, 'Player 1 Name': 'Michal Wollny',
    'Player 2 ID': 688067, 'Player 2 Name': 'Tadeas Zika',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10688051,
    'Player 1 ID': 688067, 'Player 1 Name': 'Tadeas Zika',
    'Player 2 ID': 1157765, 'Player 2 Name': 'Michal Hrabec',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P1
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10692039,
    'Player 1 ID': 807181, 'Player 1 Name': 'Frantisek Briza',
    'Player 2 ID': 1080540, 'Player 2 Name': 'Martin Kir',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P2
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10693122,
    'Player 1 ID': 608765, 'Player 1 Name': 'Tibor Kolenic',
    'Player 2 ID': 807181, 'Player 2 Name': 'Frantisek Briza',
    'P1_ML': -275, # Actual Decimal: 1.36, Winner: P2
    'P2_ML': 200  # Actual Decimal: 3.00
},
{
    'Match ID': 10688178,
    'Player 1 ID': 338586, 'Player 1 Name': 'Marek Fabini',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687887,
    'Player 1 ID': 954785, 'Player 1 Name': 'Martin Sychra',
    'Player 2 ID': 339298, 'Player 2 Name': 'Tomas Holik',
    'P1_ML': 150, # Actual Decimal: 2.50, Winner: P1
    'P2_ML': -200  # Actual Decimal: 1.50
},
{
    'Match ID': 10688061,
    'Player 1 ID': 388336, 'Player 1 Name': 'Vladimir Postelt',
    'Player 2 ID': 608765, 'Player 2 Name': 'Tibor Kolenic',
    'P1_ML': -275, # Actual Decimal: 1.36, Winner: P2
    'P2_ML': 200  # Actual Decimal: 3.00
},
{
    'Match ID': 10687886,
    'Player 1 ID': 654692, 'Player 1 Name': 'Pavel Sprynar',
    'Player 2 ID': 701363, 'Player 2 Name': 'Jaromir Cernik',
    'P1_ML': 162, # Actual Decimal: 2.62, Winner: P1
    'P2_ML': -225  # Actual Decimal: 1.44
},
{
    'Match ID': 10682270,
    'Player 1 ID': 955815, 'Player 1 Name': 'Vitek Dvorak',
    'Player 2 ID': 355559, 'Player 2 Name': 'Michal Regner',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10683008,
    'Player 1 ID': 922416, 'Player 1 Name': 'Jakub Vales',
    'Player 2 ID': 559309, 'Player 2 Name': 'Lukas Zeman',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682271,
    'Player 1 ID': 380220, 'Player 1 Name': 'Patrik Pycha',
    'Player 2 ID': 1015754, 'Player 2 Name': 'Marek Chlebecek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682297,
    'Player 1 ID': 1090473, 'Player 1 Name': 'Karel Schopf',
    'Player 2 ID': 546478, 'Player 2 Name': 'Jan Sucharda',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682326,
    'Player 1 ID': 888755, 'Player 1 Name': 'Vratislav Petracek',
    'Player 2 ID': 687549, 'Player 2 Name': 'Michal Vavrecka',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682272,
    'Player 1 ID': 1133447, 'Player 1 Name': 'Petr Sebera',
    'Player 2 ID': 553241, 'Player 2 Name': 'Martin Sobisek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682298,
    'Player 1 ID': 955815, 'Player 1 Name': 'Vitek Dvorak',
    'Player 2 ID': 342540, 'Player 2 Name': 'Matous Klimenta',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682302,
    'Player 1 ID': 922416, 'Player 1 Name': 'Jakub Vales',
    'Player 2 ID': 341888, 'Player 2 Name': 'Martin Huk',
    'P1_ML': 900, # Actual Decimal: 10.00, Winner: P2
    'P2_ML': -3333  # Actual Decimal: 1.03
},
{
    'Match ID': 10682237,
    'Player 1 ID': 687549, 'Player 1 Name': 'Michal Vavrecka',
    'Player 2 ID': 559309, 'Player 2 Name': 'Lukas Zeman',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682242,
    'Player 1 ID': 546478, 'Player 1 Name': 'Jan Sucharda',
    'Player 2 ID': 355559, 'Player 2 Name': 'Michal Regner',
    'P1_ML': 900, # Actual Decimal: 10.00, Winner: P2
    'P2_ML': -3333  # Actual Decimal: 1.03
},
{
    'Match ID': 10682322,
    'Player 1 ID': 1015754, 'Player 1 Name': 'Marek Chlebecek',
    'Player 2 ID': 1098619, 'Player 2 Name': 'Jaroslav Bares',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10690318,
    'Player 1 ID': 922416, 'Player 1 Name': 'Jakub Vales',
    'Player 2 ID': 341888, 'Player 2 Name': 'Martin Huk',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10690321,
    'Player 1 ID': 553241, 'Player 1 Name': 'Martin Sobisek',
    'Player 2 ID': 1133447, 'Player 2 Name': 'Petr Sebera',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10690322,
    'Player 1 ID': 342540, 'Player 1 Name': 'Matous Klimenta',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10690391,
    'Player 1 ID': 888755, 'Player 1 Name': 'Vratislav Petracek',
    'Player 2 ID': 559309, 'Player 2 Name': 'Lukas Zeman',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10690392,
    'Player 1 ID': 355559, 'Player 1 Name': 'Michal Regner',
    'Player 2 ID': 546478, 'Player 2 Name': 'Jan Sucharda',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10690393,
    'Player 1 ID': 1015754, 'Player 1 Name': 'Marek Chlebecek',
    'Player 2 ID': 380220, 'Player 2 Name': 'Patrik Pycha',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10687719,
    'Player 1 ID': 1030283, 'Player 1 Name': 'Tomas Vinter',
    'Player 2 ID': 387428, 'Player 2 Name': 'Daniel Tuma',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P2
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10687871,
    'Player 1 ID': 1157770, 'Player 1 Name': 'Evzen Rychlik',
    'Player 2 ID': 1158338, 'Player 2 Name': 'Oldrich Vaclahovsky',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P1
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10682269,
    'Player 1 ID': 1133447, 'Player 1 Name': 'Petr Sebera',
    'Player 2 ID': 1098619, 'Player 2 Name': 'Jaroslav Bares',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682986,
    'Player 1 ID': 342540, 'Player 1 Name': 'Matous Klimenta',
    'Player 2 ID': 1090473, 'Player 2 Name': 'Karel Schopf',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682311,
    'Player 1 ID': 553241, 'Player 1 Name': 'Martin Sobisek',
    'Player 2 ID': 380220, 'Player 2 Name': 'Patrik Pycha',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682236,
    'Player 1 ID': 341888, 'Player 1 Name': 'Martin Huk',
    'Player 2 ID': 888755, 'Player 2 Name': 'Vratislav Petracek',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10687878,
    'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma',
    'Player 2 ID': 1089695, 'Player 2 Name': 'Milos Pospisil',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687721,
    'Player 1 ID': 548705, 'Player 1 Name': 'Jiri Plachy',
    'Player 2 ID': 387428, 'Player 2 Name': 'Daniel Tuma',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10682235,
    'Player 1 ID': 546478, 'Player 1 Name': 'Jan Sucharda',
    'Player 2 ID': 955815, 'Player 2 Name': 'Vitek Dvorak',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10682284,
    'Player 1 ID': 687549, 'Player 1 Name': 'Michal Vavrecka',
    'Player 2 ID': 922416, 'Player 2 Name': 'Jakub Vales',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P2
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10683003,
    'Player 1 ID': 1015754, 'Player 1 Name': 'Marek Chlebecek',
    'Player 2 ID': 1133447, 'Player 2 Name': 'Petr Sebera',
    'P1_ML': None, # Actual Decimal: N/A, Winner: P1
    'P2_ML': None  # Actual Decimal: N/A
},
{
    'Match ID': 10687890,
    'Player 1 ID': 611386, 'Player 1 Name': 'Ludek Madle',
    'Player 2 ID': 701363, 'Player 2 Name': 'Jaromir Cernik',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P2
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10687872,
    'Player 1 ID': 636504, 'Player 1 Name': 'Vladimir Cermak',
    'Player 2 ID': 347091, 'Player 2 Name': 'Tomas Varnuska',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P1
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10691150,
    'Player 1 ID': 798958, 'Player 1 Name': 'Jiri Zuzanek',
    'Player 2 ID': 954785, 'Player 2 Name': 'Martin Sychra',
    'P1_ML': -333, # Actual Decimal: 1.30, Winner: P1
    'P2_ML': 240  # Actual Decimal: 3.40
},
{
    'Match ID': 10687884,
    'Player 1 ID': 798958, 'Player 1 Name': 'Jiri Zuzanek',
    'Player 2 ID': 954785, 'Player 2 Name': 'Martin Sychra',
    'P1_ML': -400, # Actual Decimal: 1.25, Winner: P1
    'P2_ML': 275  # Actual Decimal: 3.75
},
{
    'Match ID': 10687857,
    'Player 1 ID': 339298, 'Player 1 Name': 'Tomas Holik',
    'Player 2 ID': 798958, 'Player 2 Name': 'Jiri Zuzanek',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10687852,
    'Player 1 ID': 686100, 'Player 1 Name': 'Jakub Levicky',
    'Player 2 ID': 689440, 'Player 2 Name': 'Bohuslav Kaloc',
    'P1_ML': -500, # Actual Decimal: 1.20, Winner: P1
    'P2_ML': 333  # Actual Decimal: 4.33
},
{
    'Match ID': 10687716,
    'Player 1 ID': 689390, 'Player 1 Name': 'Marek Kostal',
    'Player 2 ID': 548705, 'Player 2 Name': 'Jiri Plachy',
    'P1_ML': -275, # Actual Decimal: 1.36, Winner: P1
    'P2_ML': 200  # Actual Decimal: 3.00
},
{
    'Match ID': 10687885,
    'Player 1 ID': 1158338, 'Player 1 Name': 'Oldrich Vaclahovsky',
    'Player 2 ID': 925949, 'Player 2 Name': 'Ivan Jemelka',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P2
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10687858,
    'Player 1 ID': 1166071, 'Player 1 Name': 'Jiri Havel',
    'Player 2 ID': 750300, 'Player 2 Name': 'Michal Syroha',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P1
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10687896,
    'Player 1 ID': 765084, 'Player 1 Name': 'Radomir Vidlicka',
    'Player 2 ID': 1163571, 'Player 2 Name': 'Jindrich Simecek',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10687870,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 339299, 'Player 2 Name': 'Miroslav Svedik',
    'P1_ML': -175, # Actual Decimal: 1.57, Winner: P1
    'P2_ML': 125  # Actual Decimal: 2.25
},
{
    'Match ID': 10687869,
    'Player 1 ID': 957132, 'Player 1 Name': 'Vaclav Kosar',
    'Player 2 ID': 1022513, 'Player 2 Name': 'Milan Cetner',
    'P1_ML': -300, # Actual Decimal: 1.33, Winner: P1
    'P2_ML': 225  # Actual Decimal: 3.25
},
{
    'Match ID': 10687856,
    'Player 1 ID': 770032, 'Player 1 Name': 'Miloslav Lubas',
    'Player 2 ID': 1142181, 'Player 2 Name': 'Jan Srba',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10691223,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 636504, 'Player 2 Name': 'Vladimir Cermak',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10691218,
    'Player 1 ID': 341166, 'Player 1 Name': 'Milan Klement',
    'Player 2 ID': 339298, 'Player 2 Name': 'Tomas Holik',
    'P1_ML': -125, # Actual Decimal: 1.80, Winner: P2
    'P2_ML': -110  # Actual Decimal: 1.91
},
{
    'Match ID': 10691212,
    'Player 1 ID': 701363, 'Player 1 Name': 'Jaromir Cernik',
    'Player 2 ID': 611386, 'Player 2 Name': 'Ludek Madle',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10691211,
    'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma',
    'Player 2 ID': 750300, 'Player 2 Name': 'Michal Syroha',
    'P1_ML': -400, # Actual Decimal: 1.25, Winner: P1
    'P2_ML': 275  # Actual Decimal: 3.75
},
{
    'Match ID': 10687864,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 347091, 'Player 2 Name': 'Tomas Varnuska',
    'P1_ML': -110, # Actual Decimal: 1.91, Winner: P1
    'P2_ML': -125  # Actual Decimal: 1.80
},
{
    'Match ID': 10687868,
    'Player 1 ID': 1012124, 'Player 1 Name': 'Michal Vondrak',
    'Player 2 ID': 954785, 'Player 2 Name': 'Martin Sychra',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P1
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10687882,
    'Player 1 ID': 1158338, 'Player 1 Name': 'Oldrich Vaclahovsky',
    'Player 2 ID': 1159074, 'Player 2 Name': 'Laurent Lasota',
    'P1_ML': 200, # Actual Decimal: 3.00, Winner: P2
    'P2_ML': -275  # Actual Decimal: 1.36
},
{
    'Match ID': 10687879,
    'Player 1 ID': 339298, 'Player 1 Name': 'Tomas Holik',
    'Player 2 ID': 954785, 'Player 2 Name': 'Martin Sychra',
    'P1_ML': -187, # Actual Decimal: 1.53, Winner: P1
    'P2_ML': 137  # Actual Decimal: 2.38
},
{
    'Match ID': 10687880,
    'Player 1 ID': 1166071, 'Player 1 Name': 'Jiri Havel',
    'Player 2 ID': 750300, 'Player 2 Name': 'Michal Syroha',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687889,
    'Player 1 ID': 360808, 'Player 1 Name': 'Jiri Svec',
    'Player 2 ID': 654692, 'Player 2 Name': 'Pavel Sprynar',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P2
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10687891,
    'Player 1 ID': 1159074, 'Player 1 Name': 'Laurent Lasota',
    'Player 2 ID': 925949, 'Player 2 Name': 'Ivan Jemelka',
    'P1_ML': -150, # Actual Decimal: 1.67, Winner: P1
    'P2_ML': 110  # Actual Decimal: 2.10
},
{
    'Match ID': 10687855,
    'Player 1 ID': 387428, 'Player 1 Name': 'Daniel Tuma',
    'Player 2 ID': 636504, 'Player 2 Name': 'Vladimir Cermak',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P1
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10687865,
    'Player 1 ID': 1089695, 'Player 1 Name': 'Milos Pospisil',
    'Player 2 ID': 1166071, 'Player 2 Name': 'Jiri Havel',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10687866,
    'Player 1 ID': 701363, 'Player 1 Name': 'Jaromir Cernik',
    'Player 2 ID': 360808, 'Player 2 Name': 'Jiri Svec',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P1
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687881,
    'Player 1 ID': 798958, 'Player 1 Name': 'Jiri Zuzanek',
    'Player 2 ID': 339298, 'Player 2 Name': 'Tomas Holik',
    'P1_ML': 125, # Actual Decimal: 2.25, Winner: P2
    'P2_ML': -175  # Actual Decimal: 1.57
},
{
    'Match ID': 10687859,
    'Player 1 ID': 1089695, 'Player 1 Name': 'Milos Pospisil',
    'Player 2 ID': 750300, 'Player 2 Name': 'Michal Syroha',
    'P1_ML': -162, # Actual Decimal: 1.61, Winner: P2
    'P2_ML': 120  # Actual Decimal: 2.20
},
{
    'Match ID': 10687718,
    'Player 1 ID': 347091, 'Player 1 Name': 'Tomas Varnuska',
    'Player 2 ID': 1030283, 'Player 2 Name': 'Tomas Vinter',
    'P1_ML': 137, # Actual Decimal: 2.38, Winner: P1
    'P2_ML': -187  # Actual Decimal: 1.53
},
{
    'Match ID': 10688485,
    'Player 1 ID': 341166, 'Player 1 Name': 'Milan Klement',
    'Player 2 ID': 798958, 'Player 2 Name': 'Jiri Zuzanek',
    'P1_ML': -250, # Actual Decimal: 1.40, Winner: P1
    'P2_ML': 175  # Actual Decimal: 2.75
},
{
    'Match ID': 10687862,
    'Player 1 ID': 654692, 'Player 1 Name': 'Pavel Sprynar',
    'Player 2 ID': 611386, 'Player 2 Name': 'Ludek Madle',
    'P1_ML': 275, # Actual Decimal: 3.75, Winner: P2
    'P2_ML': -400  # Actual Decimal: 1.25
},
{
    'Match ID': 10687867,
    'Player 1 ID': 925949, 'Player 1 Name': 'Ivan Jemelka',
    'Player 2 ID': 1157770, 'Player 2 Name': 'Evzen Rychlik',
    'P1_ML': 110, # Actual Decimal: 2.10, Winner: P2
    'P2_ML': -150  # Actual Decimal: 1.67
},
{
    'Match ID': 10687883,
    'Player 1 ID': 954785, 'Player 1 Name': 'Martin Sychra',
    'Player 2 ID': 341166, 'Player 2 Name': 'Milan Klement',
    'P1_ML': 175, # Actual Decimal: 2.75, Winner: P2
    'P2_ML': -250  # Actual Decimal: 1.40
},
{
    'Match ID': 10687720,
    'Player 1 ID': 1030283, 'Player 1 Name': 'Tomas Vinter',
    'Player 2 ID': 636504, 'Player 2 Name': 'Vladimir Cermak',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P1
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10687860,
    'Player 1 ID': 611386, 'Player 1 Name': 'Ludek Madle',
    'Player 2 ID': 360808, 'Player 2 Name': 'Jiri Svec',
    'P1_ML': 100, # Actual Decimal: 2.00, Winner: P2
    'P2_ML': -137  # Actual Decimal: 1.73
},
{
    'Match ID': 10687863,
    'Player 1 ID': 1005699, 'Player 1 Name': 'Miroslav Tuma',
    'Player 2 ID': 1166071, 'Player 2 Name': 'Jiri Havel',
    'P1_ML': -225, # Actual Decimal: 1.44, Winner: P1
    'P2_ML': 162  # Actual Decimal: 2.62
},
{
    'Match ID': 10687873,
    'Player 1 ID': 1157770, 'Player 1 Name': 'Evzen Rychlik',
    'Player 2 ID': 1159074, 'Player 2 Name': 'Laurent Lasota',
    'P1_ML': -137, # Actual Decimal: 1.73, Winner: P2
    'P2_ML': 100  # Actual Decimal: 2.00
},
{
    'Match ID': 10687888,
    'Player 1 ID': 341166, 'Player 1 Name': 'Milan Klement',
    'Player 2 ID': 339298, 'Player 2 Name': 'Tomas Holik',
    'P1_ML': -120, # Actual Decimal: 1.83, Winner: P1
    'P2_ML': -120  # Actual Decimal: 1.83
},
{
    'Match ID': 10687853,
    'Player 1 ID': 701363, 'Player 1 Name': 'Jaromir Cernik',
    'Player 2 ID': 654692, 'Player 2 Name': 'Pavel Sprynar',
    'P1_ML': -200, # Actual Decimal: 1.50, Winner: P1
    'P2_ML': 150  # Actual Decimal: 2.50
},
{
    'Match ID': 10687861,
    'Player 1 ID': 750300, 'Player 1 Name': 'Michal Syroha',
    'Player 2 ID': 1005699, 'Player 2 Name': 'Miroslav Tuma',
    'P1_ML': 250, # Actual Decimal: 3.50, Winner: P2
    'P2_ML': -350  # Actual Decimal: 1.28
},

]
# ---!!!--- ---!!!--- ---!!!--- ---!!!--- ---!!!--- ---!!!---

# --- File Paths (Using the stable v7.4 models) ---
# HISTORICAL_DATA_FILE = "final_dataset_v7.4.csv"
HISTORICAL_DATA_FILE = "paper_trade_history.csv"        # CHANGE BACK WHEN RUNNING REAL-TIME PREDICTIONS

GBM_MODEL_FILE = "cpr_v7.4_gbm_specialist.joblib"
GBM_PREPROCESSOR_FILE = "gbm_preprocessor_v7.4.joblib"
LSTM_MODEL_FILE = "cpr_v7.1_lstm_specialist.h5"
LSTM_SCALER_FILE = "lstm_scaler_v7.1.joblib"
META_MODEL_FILE = "cpr_v7.4_meta_model.pkl"

# --- FINAL STRATEGIC FILTERS v7.4 (Synchronized with Back-tester) ---
EDGE_THRESHOLD_MIN = 0.10
EDGE_THRESHOLD_MAX = 0.99
ODDS_THRESHOLD_MAX = 3.0
H2H_DISADVANTAGE_THRESHOLD = 0.40
FORM_EQUALITY_BAND = 0.10
MIN_PLAYER_HISTORY = 10 # Basic check to ensure player exists and has some data

# --- Model Parameters ---
SEQUENCE_LENGTH = 5
ROLLING_WINDOW = 10

# --- Helper function to convert odds ---
def moneyline_to_decimal(moneyline_odds):
    try:
        moneyline_odds = float(moneyline_odds)
        if moneyline_odds >= 100: return (moneyline_odds / 100) + 1
        elif moneyline_odds < 0: return (100 / abs(moneyline_odds)) + 1
        else: return np.nan
    except (ValueError, TypeError): return np.nan

# --- 2. Load All Models and Historical Data ---
try:
    print("--- Loading All Models and Historical Data for Prediction ---")
    df_history = pd.read_csv(HISTORICAL_DATA_FILE)
    df_history['Date'] = pd.to_datetime(df_history['Date'])

    gbm_model = joblib.load(GBM_MODEL_FILE)
    gbm_preprocessor = joblib.load(GBM_PREPROCESSOR_FILE)
    lstm_model = load_model(LSTM_MODEL_FILE)
    lstm_scaler = joblib.load(LSTM_SCALER_FILE)
    meta_model = joblib.load(META_MODEL_FILE)
    print("All v7.4 models and filters loaded successfully.")

    # --- 3. Feature Engineering and Prediction for Each Match ---
    print("\n--- Analyzing Upcoming Matches with Final v7.4 Strategy ---")
    
    setup_log_file() # <--- ADD THIS LINE TO START WITH A FRESH LOG
    logged_match_ids = set() # <--- ADD THIS LINE
    
    col_group1 = ['P1_Rolling_Win_Rate_L10', 'P1_Rolling_Pressure_Points_L10', 'P1_Rest_Days']
    col_group2 = ['P2_Rolling_Win_Rate_L10', 'P2_Rolling_Pressure_Points_L10', 'P2_Rest_Days']


    for match in upcoming_matches:
        p1_id, p2_id = match['Player 1 ID'], match['Player 2 ID']
        p1_name, p2_name = match['Player 1 Name'], match['Player 2 Name']
        
        # --- Point-in-Time Feature Engineering (Synchronized with Back-tester) ---
        # --- MORE EFFICIENT VERSION ---
        p1_games = df_history[(df_history['Player 1 ID'] == p1_id) | (df_history['Player 2 ID'] == p1_id)]
        p2_games = df_history[(df_history['Player 1 ID'] == p2_id) | (df_history['Player 2 ID'] == p2_id)]

        print("\n---------------------------------")
        print(f"Matchup: {p1_name} vs. {p2_name}")

        if len(p1_games) < MIN_PLAYER_HISTORY or len(p2_games) < MIN_PLAYER_HISTORY:
            print("RECOMMENDATION: NO BET (Insufficient player history)")
            print("---------------------------------")
            continue

        p1_rolling_games = p1_games.tail(ROLLING_WINDOW)
        p2_rolling_games = p2_games.tail(ROLLING_WINDOW)
        
        # --- Calculate Rest Advantage ---
        today = datetime.now()
        p1_last_date = p1_games['Date'].max()
        p2_last_date = p2_games['Date'].max()
        p1_rest = (today - p1_last_date).days if pd.notna(p1_last_date) else 30
        p2_rest = (today - p2_last_date).days if pd.notna(p2_last_date) else 30
        rest_advantage = p1_rest - p2_rest
        
        # --- Calculate other features ---
        p1_win_rate = p1_rolling_games.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p1_rolling_games.empty else 0.5
        p2_win_rate = p2_rolling_games.apply(lambda r: 1 if (r['Player 1 ID'] == p2_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p2_id and r['P1_Win'] == 0) else 0, axis=1).mean() if not p2_rolling_games.empty else 0.5
        win_rate_advantage = p1_win_rate - p2_win_rate

        p1_pressure_points = p1_rolling_games.apply(lambda r: r['P1 Pressure Points'] if r['Player 1 ID'] == p1_id else r['P2 Pressure Points'], axis=1).mean() if not p1_rolling_games.empty else 0.0
        p2_pressure_points = p2_rolling_games.apply(lambda r: r['P1 Pressure Points'] if r['Player 1 ID'] == p2_id else r['P2 Pressure Points'], axis=1).mean() if not p2_rolling_games.empty else 0.0
        pressure_points_advantage = p1_pressure_points - p2_pressure_points
        
        h2h_df = df_history[((df_history['Player 1 ID'] == p1_id) & (df_history['Player 2 ID'] == p2_id)) | ((df_history['Player 1 ID'] == p2_id) & (df_history['Player 2 ID'] == p1_id))]
        p1_h2h_wins = h2h_df.apply(lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0, axis=1).sum()
        h2h_p1_win_rate = p1_h2h_wins / len(h2h_df) if len(h2h_df) > 0 else 0.5
        
        p1_seq_df, p2_seq_df = p1_games.tail(SEQUENCE_LENGTH), p2_games.tail(SEQUENCE_LENGTH)
        if len(p1_seq_df) < SEQUENCE_LENGTH or len(p2_seq_df) < SEQUENCE_LENGTH:
            print("RECOMMENDATION: NO BET (Insufficient data for LSTM sequences)")
            print("---------------------------------")
            continue

        p1_seq, p2_seq = [], []
        for _, row in p1_seq_df.iterrows():
            if row['Player 1 ID'] == p1_id: p1_seq.append(np.concatenate([row[col_group1].values, row[col_group2].values, [row['H2H_P1_Win_Rate']]]))
            else: p1_seq.append(np.concatenate([row[col_group2].values, row[col_group1].values, [1 - row['H2H_P1_Win_Rate']]]))
        for _, row in p2_seq_df.iterrows():
            if row['Player 1 ID'] == p2_id: p2_seq.append(np.concatenate([row[col_group1].values, row[col_group2].values, [row['H2H_P1_Win_Rate']]]))
            else: p2_seq.append(np.concatenate([row[col_group2].values, row[col_group1].values, [1 - row['H2H_P1_Win_Rate']]]))

        # --- Full Ensemble Prediction Logic ---
        gbm_features = pd.DataFrame([{
            'Win_Rate_Advantage': win_rate_advantage,
            'Pressure_Points_Advantage': pressure_points_advantage,
            'Player 1 ID': p1_id,
            'Player 2 ID': p2_id
        }])
        X_gbm_processed = gbm_preprocessor.transform(gbm_features)
        gbm_pred = gbm_model.predict_proba(X_gbm_processed)[0, 1]
        
        X_p1, X_p2 = np.array([p1_seq]), np.array([p2_seq])
        nsamples, nsteps, nfeatures = X_p1.shape
        X_p1_scaled = lstm_scaler.transform(X_p1.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
        X_p2_scaled = lstm_scaler.transform(X_p2.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
        lstm_pred = lstm_model.predict([X_p1_scaled, X_p2_scaled], verbose=0)[0][0]
        
        X_meta = np.array([[gbm_pred, lstm_pred]])
        model_prob_p1 = meta_model.predict_proba(X_meta)[0, 1]
        model_prob_p2 = 1 - model_prob_p1

        # --- Apply Filters and Display Results ---
        p1_market_odds = moneyline_to_decimal(match['P1_ML'])
        p2_market_odds = moneyline_to_decimal(match['P2_ML'])
        
        edge_p1 = model_prob_p1 * p1_market_odds - 1 if pd.notna(p1_market_odds) else -1
        edge_p2 = model_prob_p2 * p2_market_odds - 1 if pd.notna(p2_market_odds) else -1
        
        p1_form_advantage = win_rate_advantage
        p2_form_advantage = -win_rate_advantage
        h2h_p2_win_rate = 1 - h2h_p1_win_rate

        # --- REFACTORED: Filter conditions mirroring the back-tester ---
        p1_conditions = {
            "Edge Min": edge_p1 > EDGE_THRESHOLD_MIN,
            "Edge Max": edge_p1 < EDGE_THRESHOLD_MAX,
            "Max Odds": p1_market_odds <= ODDS_THRESHOLD_MAX if pd.notna(p1_market_odds) else False,
            "H2H Adv": h2h_p1_win_rate >= H2H_DISADVANTAGE_THRESHOLD,
            "Form Adv": abs(p1_form_advantage) > FORM_EQUALITY_BAND
        }

        p2_conditions = {
            "Edge Min": edge_p2 > EDGE_THRESHOLD_MIN,
            "Edge Max": edge_p2 < EDGE_THRESHOLD_MAX,
            "Max Odds": p2_market_odds <= ODDS_THRESHOLD_MAX if pd.notna(p2_market_odds) else False,
            "H2H Adv": h2h_p2_win_rate >= H2H_DISADVANTAGE_THRESHOLD,
            "Form Adv": abs(p2_form_advantage) > FORM_EQUALITY_BAND
        }
        
        p1_pass_all = all(p1_conditions.values())
        p2_pass_all = all(p2_conditions.values())
        
        print(f"Model Prediction: {p1_name} ({model_prob_p1:.2%}) vs. {p2_name} ({model_prob_p2:.2%})")
        
        print(f"\nAnalysis for {p1_name}:")
        print(f"  - Market Odds: {match['P1_ML']} ({p1_market_odds:.2f} dec) -> {'PASS' if p1_conditions['Max Odds'] else 'FAIL'}")
        print(f"  - Model Edge: {edge_p1:.2%} -> {'PASS' if p1_conditions['Edge Min'] and p1_conditions['Edge Max'] else 'FAIL'}")
        print(f"  - H2H Win Rate: {h2h_p1_win_rate:.2%} -> {'PASS' if p1_conditions['H2H Adv'] else 'FAIL'}")
        print(f"  - Form Adv: {p1_form_advantage:+.2f} -> {'PASS' if p1_conditions['Form Adv'] else 'FAIL'}")
        
        # ... Inside the loop, in the Analysis for Player 1 section ...
        if p1_pass_all:
            match_id = match['Match ID']
            if match_id not in logged_match_ids:
                print(f"  RECOMMENDATION: BET on {p1_name}")
                log_bet(match_id, p1_id, p1_name, p2_id, p2_name, p1_market_odds, edge_p1)
                logged_match_ids.add(match_id) # Add the ID to our memory
            else:
                print(f"  RECOMMENDATION: NO BET (Already logged a bet for this match ID: {match_id})")
        else:
            print(f"  RECOMMENDATION: NO BET")
            
        print(f"\nAnalysis for {p2_name}:")
        print(f"  - Market Odds: {match['P2_ML']} ({p2_market_odds:.2f} dec) -> {'PASS' if p2_conditions['Max Odds'] else 'FAIL'}")
        print(f"  - Model Edge: {edge_p2:.2%} -> {'PASS' if p2_conditions['Edge Min'] and p2_conditions['Edge Max'] else 'FAIL'}")
        print(f"  - H2H Win Rate: {h2h_p2_win_rate:.2%} -> {'PASS' if p2_conditions['H2H Adv'] else 'FAIL'}")
        print(f"  - Form Adv: {p2_form_advantage:+.2f} -> {'PASS' if p2_conditions['Form Adv'] else 'FAIL'}")

        # ... Inside the loop, in the Analysis for Player 2 section ...
        if p2_pass_all:
            match_id = match['Match ID']
            if match_id not in logged_match_ids:
                print(f"  RECOMMENDATION: BET on {p2_name}")
                log_bet(match_id, p2_id, p2_name, p1_id, p1_name, p2_market_odds, edge_p2)
                logged_match_ids.add(match_id) # Add the ID to our memory
            else:
                print(f"  RECOMMENDATION: NO BET (Already logged a bet for this match ID: {match_id})")
        else:
            print(f"  RECOMMENDATION: NO BET")

        print("---------------------------------")

except FileNotFoundError as e:
    print(f"Error: A required model or data file was not found. Ensure all trainers have been run and files are in the correct folder.")
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()