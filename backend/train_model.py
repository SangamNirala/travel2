import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

def build_model(input_dim):
    """
    Build neural network for travel time and distance prediction
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')  # Predict travel duration
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model():
    """
    Train the neural network model
    """
    print("Loading training data...")
    X = np.load('X_train.npy')
    y = np.load('y_train.npy')
    
    print(f"Training data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build and train model
    print("\nBuilding model...")
    model = build_model(X_train.shape[1])
    print(model.summary())
    
    print("\nTraining model...")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating model...")
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
    print(f"Test MAE: {test_mae:.2f} minutes")
    
    # Save model and scaler
    print("\nSaving model...")
    model.save('railway_model.keras')
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\nTraining complete!")
    print(f"Model saved as 'railway_model.keras'")
    print(f"Scaler saved as 'scaler.pkl'")
    
    return history

if __name__ == '__main__':
    train_model()