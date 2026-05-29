import sys
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, 'models', 'trained_ids.h5')
DATA_PATH = os.path.join(current_dir, 'data', 'raw', 'DNN-EdgeIIoT-dataset.csv')

def run_edge_simulation():
    print("\n" + "="*60)
    print("   SMART AGRI-IDS: LIGHTWEIGHT EDGE GATEWAY SIMULATION")
    print("="*60 + "\n", flush=True)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Trained model not found at: {MODEL_PATH}")
        return

    print(f"[INFO] Initializing AI Brain (BiGRU-LSTM Hybrid)...", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Dynamically read how many features the AI expects (e.g., 53)
    expected_features = model.input_shape[2]
    print(f"[INFO] AI Engine loaded. Expecting {expected_features} features per packet.")
    
    print(f"[INFO] Accessing sensor traffic logs...", flush=True)
    try:
        sample_df = pd.read_csv(DATA_PATH, nrows=100)
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'Attack_type' in numeric_cols:
            numeric_cols.remove('Attack_type')
        
        df_sample = pd.read_csv(
            DATA_PATH, 
            nrows=500, 
            usecols=numeric_cols, 
            low_memory=False
        )
        
        # Coerce dirty data (strings/IPs) into zeros
        df_sample = df_sample.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_raw = df_sample.values
        
        # ==========================================
        # THE FIX: Dynamic Shape Matching
        # ==========================================
        if X_raw.shape[1] < expected_features:
            # If the mock data is missing columns, pad them with zeros
            padding = np.zeros((X_raw.shape[0], expected_features - X_raw.shape[1]))
            X_test = np.hstack((X_raw, padding))
        else:
            # Slice to exact expected features
            X_test = X_raw[:, :expected_features]
            
        X_sim = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    except Exception as e:
        print(f"[ERROR] High-speed data pull failed: {e}")
        return

    print(f"[READY] Monitoring Smart Agriculture sensors. (Ctrl+C to stop)\n", flush=True)

    try:
        for i in range(len(X_sim)):
            packet = X_sim[i:i+1]
            start_time = time.time()
            
            prediction = model.predict(packet, verbose=0)
            latency = (time.time() - start_time) * 1000 
            
            predicted_class = np.argmax(prediction)
            is_threat = (predicted_class != 0) 
            
            status = "!!! THREAT DETECTED !!!" if is_threat else "NORMAL"
            action = "BLOCKED" if is_threat else "PASSED"

            print(f"[SCAN] Packet {i+1:04d} | Status: {status:<22} | Action: {action:<7} | Latency: {latency:.2f}ms", flush=True)
            time.sleep(0.4)

    except KeyboardInterrupt:
        print("\n" + "-"*60)
        print("[SHUTDOWN] Gateway security system deactivated.")
        print("-"*60)

if __name__ == "__main__":
    run_edge_simulation()