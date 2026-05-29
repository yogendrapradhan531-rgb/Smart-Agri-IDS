import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
 
def generate_ids_visuals():
    # --- CONFIGURATION (relative paths, works on any machine) ---
    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'trained_ids.h5')
    DATA_PATH   = os.path.join(BASE_DIR, 'data', 'raw', 'DNN-EdgeIIoT-dataset.csv')
    OUTPUT_PATH = os.path.join(BASE_DIR, 'visual_results.png')
 
    # Human-readable names for all 15 attack classes
    # Index 0 = Normal; adjust the list if your encoder maps differently
    CLASS_NAMES = [
        'Normal', 'DDoS_UDP', 'DDoS_ICMP', 'SQL_Injection',
        'Vulnerability_Scanner', 'Password', 'DDoS_TCP',
        'DDoS_HTTP', 'Backdoor', 'Port_Scanning',
        'XSS', 'Ransomware', 'Uploading', 'MITM', 'Fingerprinting'
    ]
 
    # ----------------------------------------------------------------
    # 1. Load model
    # ----------------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at: {MODEL_PATH}")
        print("  → Run train.py first to generate 'models/trained_ids.h5'")
        return
 
    print("[INFO] Loading trained BiGRU-LSTM model...")
    model = load_model(MODEL_PATH)
    num_classes = model.output_shape[-1]
 
    # FIX: Read true feature count from GRU weight matrix, not Input layer
    # model.input_shape[2] can be wrong if the model was saved inconsistently
    gru_layer = next(l for l in model.layers if 'gru' in l.name.lower()
                     or 'bidirectional' in l.name.lower())
    EXPECTED_FEATURES = gru_layer.get_weights()[0].shape[0]
    print(f"[INFO] True feature count from GRU weights: {EXPECTED_FEATURES}")
    print(f"[INFO] Model predicts {num_classes} classes.")
 
    # ----------------------------------------------------------------
    # 2. Load dataset
    # ----------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at: {DATA_PATH}")
        return
 
    print("[INFO] Reading dataset sample (first 500 rows)...")
    sample_df = pd.read_csv(DATA_PATH, nrows=100)
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
 
    label_col    = 'Attack_type'
    feature_cols = [c for c in numeric_cols if c != label_col]
 
    df = pd.read_csv(
        DATA_PATH,
        nrows=500,
        usecols=feature_cols + ([label_col] if label_col in sample_df.columns else []),
        low_memory=False
    )
 
    # Coerce any dirty values (IPs, strings) → 0
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
 
    X_raw = df[feature_cols].values
 
    # ----------------------------------------------------------------
    # 3. Feature alignment
    # FIX: Slice to exact feature count the GRU weights expect — no padding
    # ----------------------------------------------------------------
    X_aligned = X_raw[:, :EXPECTED_FEATURES]
    print(f"[INFO] Features aligned to: {X_aligned.shape[1]}")
 
    # ----------------------------------------------------------------
    # 4. Scale to match training normalization
    # ----------------------------------------------------------------
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_aligned)
 
    # Reshape to (samples, timesteps=1, features) for BiGRU-LSTM
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
 
    # ----------------------------------------------------------------
    # 5. Predict
    # ----------------------------------------------------------------
    print("[INFO] Generating predictions...")
    y_prob         = model.predict(X_reshaped, verbose=0)   # shape: (N, num_classes)
    y_pred_classes = np.argmax(y_prob, axis=1)              # shape: (N,)
 
    # ----------------------------------------------------------------
    # 6. Build readable label list
    # ----------------------------------------------------------------
    labels_available = CLASS_NAMES[:num_classes]
    pred_labels = [labels_available[c] if c < len(labels_available)
                   else f"Class_{c}" for c in y_pred_classes]
 
    # ----------------------------------------------------------------
    # 7. Visualisation — two panels
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Smart Agri-IDS — Detection Results", fontsize=15, fontweight='bold')
 
    # Panel A: count of every predicted class
    ax1 = axes[0]
    pred_series = pd.Series(pred_labels)
    order = pred_series.value_counts().index.tolist()
    sns.countplot(y=pred_series, order=order, palette='viridis', ax=ax1)
    ax1.set_title("Predicted Class Distribution")
    ax1.set_xlabel("Count")
    ax1.set_ylabel("Class")
 
    # Panel B: Normal vs Threat summary
    ax2 = axes[1]
    binary_labels = ['Normal' if c == 0 else 'Threat' for c in y_pred_classes]
    sns.countplot(x=binary_labels, order=['Normal', 'Threat'],
                  palette={'Normal': '#2ecc71', 'Threat': '#e74c3c'}, ax=ax2)
    ax2.set_title("Normal vs Threat Summary")
    ax2.set_xlabel("Prediction")
    ax2.set_ylabel("Count")
    for bar in ax2.patches:
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 int(bar.get_height()),
                 ha='center', va='bottom', fontweight='bold')
 
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"[SUCCESS] Visualization saved → {OUTPUT_PATH}")
    plt.show()
 
if __name__ == "__main__":
    generate_ids_visuals()