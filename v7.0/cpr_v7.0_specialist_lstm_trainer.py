import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1. Configuration ---
FEATURES_FILE = "final_engineered_features.csv"
MODEL_OUTPUT_FILE = "cpr_v7.0_lstm_specialist.h5"
SCALER_OUTPUT_FILE = "lstm_scaler_v7.0.joblib"
SEQUENCE_LENGTH = 5 # How many past matches to look at for momentum

# --- 2. Main Script Logic ---
try:
    print(f"Loading feature data from '{FEATURES_FILE}'...")
    df = pd.read_csv(FEATURES_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    df.dropna(inplace=True)

    # --- 3. Sequence Creation with New Features ---
    print("Creating match sequences with advanced features...")
    player_history = {}
    X_p1_sequences, X_p2_sequences, y_train = [], [], []
    
    # Define the features that represent a player's form and the match context at a point in time
    # Note: We are creating a feature vector for each player's perspective in each match
    features_to_sequence = [
        'P1_Rolling_Win_Rate_L10', 'P1_Rolling_Pressure_Points_L10', 'P1_Rest_Days',
        'P2_Rolling_Win_Rate_L10', 'P2_Rolling_Pressure_Points_L10', 'P2_Rest_Days',
        'H2H_P1_Win_Rate'
    ]
    df_sequence_features = df[['Player 1 ID', 'Player 2 ID', 'P1_Win'] + features_to_sequence]

    for index, row in df_sequence_features.iterrows():
        p1_id, p2_id = row['Player 1 ID'], row['Player 2 ID']
        p1_hist = player_history.get(p1_id, [])
        p2_hist = player_history.get(p2_id, [])
        
        # Only create a training sample if we have enough historical data for BOTH players
        if len(p1_hist) >= SEQUENCE_LENGTH and len(p2_hist) >= SEQUENCE_LENGTH:
            X_p1_sequences.append(p1_hist[-SEQUENCE_LENGTH:])
            X_p2_sequences.append(p2_hist[-SEQUENCE_LENGTH:])
            y_train.append(row['P1_Win'])
            
        # Create the feature vector for P1's perspective and add to their history
        p1_perspective_vector = [
            row['P1_Rolling_Win_Rate_L10'], row['P1_Rolling_Pressure_Points_L10'], row['P1_Rest_Days'], # My stats
            row['P2_Rolling_Win_Rate_L10'], row['P2_Rolling_Pressure_Points_L10'], row['P2_Rest_Days'], # Opponent stats
            row['H2H_P1_Win_Rate'] # H2H stat
        ]
        if p1_id not in player_history: player_history[p1_id] = []
        player_history[p1_id].append(p1_perspective_vector)

        # Create the feature vector for P2's perspective and add to their history
        p2_perspective_vector = [
            row['P2_Rolling_Win_Rate_L10'], row['P2_Rolling_Pressure_Points_L10'], row['P2_Rest_Days'], # My stats
            row['P1_Rolling_Win_Rate_L10'], row['P1_Rolling_Pressure_Points_L10'], row['P1_Rest_Days'], # Opponent stats
            1 - row['H2H_P1_Win_Rate'] # H2H must be inverted for P2's perspective
        ]
        if p2_id not in player_history: player_history[p2_id] = []
        player_history[p2_id].append(p2_perspective_vector)

    X_p1, X_p2, y = np.array(X_p1_sequences), np.array(X_p2_sequences), np.array(y_train)
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
    print("\nBuilding and training the LSTM model on new features...")
    input_p1, input_p2 = Input(shape=(SEQUENCE_LENGTH, nfeatures)), Input(shape=(SEQUENCE_LENGTH, nfeatures))
    lstm_p1, lstm_p2 = LSTM(16)(input_p1), LSTM(16)(input_p2)
    concatenated = Concatenate()([lstm_p1, lstm_p2])
    dense1 = Dense(16, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(dense1)
    
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