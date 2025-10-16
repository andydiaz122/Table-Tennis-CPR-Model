import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1. Configuration ---
FEATURES_FILE = "training_dataset.csv"
MODEL_OUTPUT_FILE = "cpr_v7.1_lstm_specialist.h5"
SCALER_OUTPUT_FILE = "lstm_scaler_v7.1.joblib"
SEQUENCE_LENGTH = 5 # How many past matches to look at for momentum

# --- 2. Main Script Logic ---
try:
    print(f"Loading feature data from '{FEATURES_FILE}'...")
    df = pd.read_csv(FEATURES_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    # Note: Column names with spaces like 'P1 Pressure Points' are changed to 'P1_Pressure_Points' for consistency
    df.rename(columns={'P1 Pressure Points': 'P1_Pressure_Points', 'P2 Pressure Points': 'P2_Pressure_Points'}, inplace=True)
    
    # --- 3. Sequence Creation with LEAK-PROOF Features ---
    print("Creating match sequences with PRE-MATCH features to prevent data leakage...")
    player_history = {}
    X_p1_sequences, X_p2_sequences, y_train = [], [], []

    # Define the features that are known BEFORE a match starts.
    # We REMOVED post-match stats like Pressure Points from the sequence.
    features_to_use = [
        'P1_Win', # The historical outcome is okay to use
        'P1_Rest_Days',
        'P2_Rest_Days',
        'H2H_P1_Win_Rate'
    ]
    
    # We need to ensure the columns used for creating vectors exist before dropping NaNs
    df_sequence_features = df[['Player 1 ID', 'Player 2 ID'] + features_to_use].copy()
    df_sequence_features.dropna(inplace=True) 


    for index, row in df_sequence_features.iterrows():
        p1_id, p2_id = row['Player 1 ID'], row['Player 2 ID']
        p1_hist = player_history.get(p1_id, [])
        p2_hist = player_history.get(p2_id, [])

        # Only create a training sample if we have enough historical data for BOTH players
        if len(p1_hist) >= SEQUENCE_LENGTH and len(p2_hist) >= SEQUENCE_LENGTH:
            X_p1_sequences.append(p1_hist[-SEQUENCE_LENGTH:])
            X_p2_sequences.append(p2_hist[-SEQUENCE_LENGTH:])
            y_train.append(row['P1_Win'])

        # Create the feature vector from P1's perspective
        p1_perspective_vector = [
            row['P1_Win'],              # My result in this match (1.0 or 0.0)
            row['P1_Rest_Days'],        # My rest days
            1.0 - row['P1_Win'],        # Opponent's result (P2_Win)
            row['P2_Rest_Days'],        # Opponent's rest days
            row['H2H_P1_Win_Rate']      # H2H from my perspective
        ]
        if p1_id not in player_history: player_history[p1_id] = []
        player_history[p1_id].append(p1_perspective_vector)

        # Create the feature vector from P2's perspective
        p2_perspective_vector = [
            1.0 - row['P1_Win'],        # My result in this match (P2_Win)
            row['P2_Rest_Days'],        # My rest days
            row['P1_Win'],              # Opponent's result
            row['P1_Rest_Days'],        # Opponent's rest days
            1.0 - row['H2H_P1_Win_Rate']# H2H must be inverted for P2's perspective
        ]
        if p2_id not in player_history: player_history[p2_id] = []
        player_history[p2_id].append(p2_perspective_vector)

    X_p1, X_p2, y = np.array(X_p1_sequences), np.array(X_p2_sequences), np.array(y_train)
    
    if len(y) == 0:
        raise ValueError(f"Created 0 training sequences. Check if any player has at least {SEQUENCE_LENGTH} matches in the dataset.")
        
    print(f"Created {len(y)} training sequences.")

    # --- 4. Data Scaling ---
    nsamples, nsteps, nfeatures = X_p1.shape
    X_p1_reshaped = X_p1.reshape((nsamples * nsteps, nfeatures))
    X_p2_reshaped = X_p2.reshape((nsamples * nsteps, nfeatures))

    scaler = StandardScaler()
    scaler.fit(np.vstack([X_p1_reshaped, X_p2_reshaped]))

    X_p1_scaled = scaler.transform(X_p1_reshaped).reshape(nsamples, nsteps, nfeatures)
    X_p2_scaled = scaler.transform(X_p2_reshaped).reshape(nsamples, nsteps, nfeatures)

    # --- 5. LSTM Model Training ---
    print("\nBuilding and training the LSTM model with Dropout regularization...")
    input_p1, input_p2 = Input(shape=(SEQUENCE_LENGTH, nfeatures)), Input(shape=(SEQUENCE_LENGTH, nfeatures))

    lstm_layer1 = LSTM(16)
    lstm_layer2 = LSTM(16)
    lstm_p1 = lstm_layer1(input_p1)
    lstm_p2 = lstm_layer2(input_p2)
    dropout_lstm_p1 = Dropout(0.3)(lstm_p1)
    dropout_lstm_p2 = Dropout(0.3)(lstm_p2)
    concatenated = Concatenate()([dropout_lstm_p1, dropout_lstm_p2])
    dense1 = Dense(16, activation='relu')(concatenated)
    dropout_dense1 = Dropout(0.3)(dense1)
    output = Dense(1, activation='sigmoid')(dropout_dense1)

    model = Model(inputs=[input_p1, input_p2], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit([X_p1_scaled, X_p2_scaled], y, epochs=15, batch_size=32, validation_split=0.2, verbose=1)

    # --- 6. Save the Final Model ---
    model.save(MODEL_OUTPUT_FILE)
    joblib.dump(scaler, SCALER_OUTPUT_FILE)

    print(f"\n✅ Successfully re-trained and saved the LSTM model to '{MODEL_OUTPUT_FILE}'")

except FileNotFoundError:
    print(f"Error: The input file '{FEATURES_FILE}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")