import os
import pandas as pd
import numpy as np

print("Generating Mock Smart Agri-IDS Dataset...")

# 1. Create the data/raw directory if it doesn't exist
os.makedirs(os.path.join('data', 'raw'), exist_ok=True)
file_path = os.path.join('data', 'raw', 'DNN-EdgeIIoT-dataset.csv')

# 2. Generate 43 columns of random numeric network data
data = np.random.rand(500, 43)
columns = [f"network_feature_{i}" for i in range(1, 44)]
df = pd.DataFrame(data, columns=columns)

# 3. Add a "dirty" string column to test your new string coercion fix
df['arp.src.proto_ipv4'] = '192.168.0.1' 

# 4. Add the Attack_type labels
attacks = ['Normal', 'DDoS_HTTP', 'SQL_injection', 'Backdoor', 'XSS', 'Ransomware']
df['Attack_type'] = np.random.choice(attacks, 500)

# 5. Save to the exact location your scripts expect
df.to_csv(file_path, index=False)

print(f"[SUCCESS] Mock dataset created at: {file_path}")
print("You can now run 'python edge_gateway_ids.py' to test your simulation!")