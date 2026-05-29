import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical

def clean_and_normalize(file_path):
    print(f"[INFO] Accessing dataset securely (RAM-Safe Chunking)...")
    
    # 1. Read first 100 rows to accurately find numeric columns
    sample_df = pd.read_csv(file_path, nrows=100)
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'Attack_type' in sample_df.columns and 'Attack_type' not in numeric_cols:
        numeric_cols.append('Attack_type')

    print("[INFO] Processing file in chunks to clean dirty data...")
    chunk_size = 200000
    collected_chunks = []
    
    for chunk in pd.read_csv(file_path, usecols=numeric_cols, chunksize=chunk_size):
        feature_cols = [c for c in chunk.columns if c != 'Attack_type']
        # Aggressive string coercion: turns IP addresses into NaN, then 0.0
        chunk[feature_cols] = chunk[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        collected_chunks.append(chunk)

    print("[INFO] Reassembling cleaned dataset...")
    df = pd.concat(collected_chunks, ignore_index=True)
    
    print("[INFO] Balancing classes to cap at 3,000 samples each...")
    balanced_chunks = []
    
    for attack_class in df['Attack_type'].unique():
        class_subset = df[df['Attack_type'] == attack_class]
        sampled_subset = class_subset.sample(n=min(len(class_subset), 3000), random_state=42)
        balanced_chunks.append(sampled_subset)
        
    df_balanced = pd.concat(balanced_chunks, axis=0).reset_index(drop=True)
    print(f"[SUCCESS] Dataset balanced! Total rows: {len(df_balanced)}")

    X = df_balanced.drop('Attack_type', axis=1, errors='ignore').values
    y = df_balanced['Attack_type'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_final = to_categorical(y_encoded)
    
    print(f"[INFO] Features extracted: {X_scaled.shape[1]}") 
    print(f"[INFO] Classes detected: {y_final.shape[1]} (Should be 15)")
    
    return X_scaled, y_final, encoder