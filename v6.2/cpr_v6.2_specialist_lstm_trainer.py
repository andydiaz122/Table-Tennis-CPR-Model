import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1. Configuration ---
FEATURES_FILE = "training_features.csv"
MODEL_OUTPUT_FILE = "cpr_v6.2_lstm_specialist.h5"
SCALER_OUTPUT_FILE = "lstm_scaler.joblib"
SEQUENCE_LENGTH = 5 # How many past matches to look at for momentum

# --- 2. Main Script Logic ---
try:
    print(f"Loading feature data from '{FEATURES_FILE}'...")
    df = pd.read_csv(FEATURES_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    print(f"Loaded {len(df)} matches for LSTM training.")

    # --- 3. Sequence Creation ---
    print("Creating match sequences for each player...")
    
    # Create a dictionary to store each player's match history
    player_history = {}
    
    X_p1_sequences, X_p2_sequences, y_train = [], [], []
    
    features_to_sequence = ['Total_Points_Advantage', 'Pressure_Points_Advantage', 'Comeback_Advantage']
    
    for index, row in df.iterrows():
        p1_id, p2_id = row['Player 1 ID'], row['Player 2 ID']
        
        # Get last N matches for each player
        p1_hist = player_history.get(p1_id, [])
        p2_hist = player_history.get(p2_id, [])
        
        # If we have enough history for both players, create a sequence
        if len(p1_hist) >= SEQUENCE_LENGTH and len(p2_hist) >= SEQUENCE_LENGTH:
            X_p1_sequences.append(p1_hist[-SEQUENCE_LENGTH:])
            X_p2_sequences.append(p2_hist[-SEQUENCE_LENGTH:])
            y_train.append(row['P1_Win'])
            
        # Update history for both players with the current match's stats
        current_p1_stats = row[features_to_sequence].values
        if p1_id not in player_history: player_history[p1_id] = []
        player_history[p1_id].append(current_p1_stats)

        current_p2_stats = -row[features_to_sequence].values # Invert stats for P2's perspective
        if p2_id not in player_history: player_history[p2_id] = []
        player_history[p2_id].append(current_p2_stats)

    # Convert to numpy arrays
    X_p1 = np.array(X_p1_sequences)
    X_p2 = np.array(X_p2_sequences)
    y = np.array(y_train)

    print(f"Created {len(y)} training sequences.")

    # --- 4. Data Scaling ---
    # Reshape data to 2D for scaler, then back to 3D for LSTM
    nsamples, nsteps, nfeatures = X_p1.shape
    X_p1_reshaped = X_p1.reshape((nsamples * nsteps, nfeatures))
    X_p2_reshaped = X_p2.reshape((nsamples * nsteps, nfeatures))
    
    scaler = StandardScaler()
    scaler.fit(np.vstack([X_p1_reshaped, X_p2_reshaped])) # Fit on all data
    
    X_p1_scaled = scaler.transform(X_p1_reshaped).reshape(nsamples, nsteps, nfeatures)
    X_p2_scaled = scaler.transform(X_p2_reshaped).reshape(nsamples, nsteps, nfeatures)

    # --- 5. LSTM Model Training ---
    print("\nBuilding and training the LSTM model...")
    
    # Input layers for each player's sequence
    input_p1 = Input(shape=(SEQUENCE_LENGTH, len(features_to_sequence)))
    input_p2 = Input(shape=(SEQUENCE_LENGTH, len(features_to_sequence)))
    
    # LSTM layers
    lstm_p1 = LSTM(16)(input_p1)
    lstm_p2 = LSTM(16)(input_p2)
    
    # Combine the outputs
    concatenated = Concatenate()([lstm_p1, lstm_p2])
    
    # Dense layers for final prediction
    dense1 = Dense(16, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(dense1)
    
    model = Model(inputs=[input_p1, input_p2], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit([X_p1_scaled, X_p2_scaled], y, epochs=10, batch_size=32, validation_split=0.2)

    # --- 6. Save the Final Model ---
    model.save(MODEL_OUTPUT_FILE)
    joblib.dump(scaler, SCALER_OUTPUT_FILE)
    
    print(f"\n✅ Successfully trained and saved the LSTM model to '{MODEL_OUTPUT_FILE}'")
    print(f"✅ Saved the LSTM data scaler to '{SCALER_OUTPUT_FILE}'")

except FileNotFoundError:
    print(f"Error: The input file '{FEATURES_FILE}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")