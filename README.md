# Smart Agri-IDS: AI-Powered Cybersecurity for Agricultural IoT 🛡️🌱

Smart farms rely on connected IoT sensors (soil moisture, irrigation controllers, weather stations), which are increasingly targeted by cyberattacks. These edge devices lack the processing power for traditional security software, and relying on cloud defense introduces dangerous latency.

**Smart Agri-IDS** is a lightweight, edge-deployed machine learning firewall designed to protect agricultural networks. By combining Bidirectional GRU (BiGRU) and LSTM layers, this system captures complex, multi-step sequential network threats in real-time.

### ✨ Key Features
* **Hybrid Deep Learning:** Utilizes a BiGRU-LSTM architecture to classify network traffic into 15 categories (14 specific attack types + Normal traffic).
* **Edge-Native Inference:** Optimized to scan packets and block threats with less than 10ms of latency, ensuring instantaneous protection.
* **Real-Time Dashboard:** Includes a live Flask-based web dashboard with Server-Sent Events (SSE) to monitor passing and blocked traffic dynamically.
* **RAM-Safe Processing:** Built-in chunking and aggressive coercion to handle massive, imbalanced IoT datasets (DNN-EdgeIIoT-dataset) efficiently.

### 🛠️ Tech Stack
* **Machine Learning:** Python, TensorFlow 2.x, Keras, Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Web Dashboard:** Flask, HTML/CSS, JavaScript (SSE Streaming)

### 🚀 How to Run
1. Clone the repository:
   `git clone https://github.com/yogendrapradhan531-rgb/Smart-Agri-IDS.git`
2. Install the required dependencies:
   `pip install -r requirements.txt`
3. Start the Edge Gateway Dashboard:
   `python app.py`
4. Open your browser and navigate to `http://localhost:5000` to view the live threat monitor.
