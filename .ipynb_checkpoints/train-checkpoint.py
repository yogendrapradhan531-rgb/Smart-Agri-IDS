import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from preprocess import clean_and_normalize

def train_ids_model():
    print("\n" + "="*55)
    print("   SMART AGRI-IDS: MODEL TRAINING PIPELINE")
    print("="*55 + "\n")

    data_path = 'data/raw/DNN-EdgeIIoT-dataset.csv'
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'trained_ids.h5')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    X_scaled, y_final, _ = clean_and_normalize(data_path)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    X_train, X_val, y_train, y_val = train_test_split(
        X_reshaped, y_final, test_size=0.2, random_state=42
    )

    print("[INFO] Calculating Class Weights...")
    y_train_1d = np.argmax(y_train, axis=1)
    
    raw_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_1d),
        y=y_train_1d
    )
    
    class_weight_dict = {i: min(w, 5.0) for i, w in enumerate(raw_weights)}
    print("[SUCCESS] Smoothed Class Weights applied.")

    print("[INFO] Compiling BiGRU-LSTM Hybrid Architecture...")
    num_classes = y_final.shape[1]
    
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Bidirectional(GRU(64, return_sequences=True)),
        LSTM(64),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    custom_optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print("[INFO] Starting Training Phase...")
    model.fit(
        X_train, y_train, 
        epochs=10,  
        batch_size=64, 
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict, 
        verbose=1
    )

    model.save(model_path)
    print(f"\n[SUCCESS] Smart Agri-IDS model saved to: {model_path}")

if __name__ == "__main__":
    train_ids_model()