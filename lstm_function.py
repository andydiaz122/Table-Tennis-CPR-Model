import pandas as pd
import numpy as np

SEQUENCE_LENGTH = 5 # Must match the value used in training

def create_lstm_sequences_for_prediction(p1_id, p2_id, historical_df):
    """
    Creates the sequences for Player 1 and Player 2 needed for an LSTM prediction.

    Args:
        p1_id: The ID of player 1 for the match to be predicted.
        p2_id: The ID of player 2 for the match to be predicted.
        historical_df: A DataFrame containing all historical matches, sorted by date.

    Returns:
        A tuple (p1_sequence, p2_sequence) ready for scaling and prediction,
        or (None, None) if either player doesn't have enough history.
    """
    # Filter historical matches for both players
    p1_matches = historical_df[(historical_df['Player 1 ID'] == p1_id) | (historical_df['Player 2 ID'] == p1_id)].copy()
    p2_matches = historical_df[(historical_df['Player 1 ID'] == p2_id) | (historical_df['Player 2 ID'] == p2_id)].copy()

    # Check if we have enough history
    if len(p1_matches) < SEQUENCE_LENGTH or len(p2_matches) < SEQUENCE_LENGTH:
        return None, None

    # Get the last N matches for each player
    p1_last_n = p1_matches.tail(SEQUENCE_LENGTH)
    p2_last_n = p2_matches.tail(SEQUENCE_LENGTH)

    p1_hist_vectors, p2_hist_vectors = [], []

    # Build the sequence for Player 1
    for _, row in p1_last_n.iterrows():
        if row['Player 1 ID'] == p1_id: # P1 was Player 1 in this historical match
            vec = [
                row['P1_Win'], row['P1_Pressure_Points'], row['P1_Rest_Days'],
                1.0 - row['P1_Win'], row['P2_Pressure_Points'], row['P2_Rest_Days'],
                row['H2H_P1_Win_Rate']
            ]
        else: # P1 was Player 2 in this historical match
            vec = [
                row['P2_Win'], row['P2_Pressure_Points'], row['P2_Rest_Days'],
                1.0 - row['P2_Win'], row['P1_Pressure_Points'], row['P1_Rest_Days'],
                1.0 - row['H2H_P1_Win_Rate']
            ]
        p1_hist_vectors.append(vec)

    # Build the sequence for Player 2
    for _, row in p2_last_n.iterrows():
        if row['Player 1 ID'] == p2_id: # P2 was Player 1 in this historical match
            vec = [
                row['P1_Win'], row['P1_Pressure_Points'], row['P1_Rest_Days'],
                1.0 - row['P1_Win'], row['P2_Pressure_Points'], row['P2_Rest_Days'],
                row['H2H_P1_Win_Rate']
            ]
        else: # P2 was Player 2 in this historical match
            vec = [
                row['P2_Win'], row['P2_Pressure_Points'], row['P2_Rest_Days'],
                1.0 - row['P2_Win'], row['P1_Pressure_Points'], row['P1_Rest_Days'],
                1.0 - row['H2H_P1_Win_Rate']
            ]
        p2_hist_vectors.append(vec)

    return np.array([p1_hist_vectors]), np.array([p2_hist_vectors])