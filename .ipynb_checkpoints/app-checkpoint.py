"""
Smart Agri-IDS — Flask System Application
==========================================
Run:  python app.py
Open: http://localhost:5000
"""

import os
import time
import json
import threading
import queue
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, Response, stream_with_context
from sklearn.preprocessing import StandardScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "trained_ids.h5")
DATA_PATH   = os.path.join(BASE_DIR, "data", "raw", "ML-EdgeIIoT-dataset.csv")
MAX_ROWS    = 500
SCAN_DELAY  = 0.35  # seconds between packets

CLASS_NAMES = [
    "Normal", "DDoS_UDP", "DDoS_ICMP", "SQL_Injection",
    "Vulnerability_Scanner", "Password", "DDoS_TCP",
    "DDoS_HTTP", "Backdoor", "Port_Scanning",
    "XSS", "Ransomware", "Uploading", "MITM", "Fingerprinting",
]

# ─────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────
model          = None
X_sim          = None
scaler         = None
expected_feats = 0

scan_running  = False
scan_thread   = None
event_queue   = queue.Queue(maxsize=500)

stats = {
    "total": 0, "normal": 0, "threats": 0,
    "latencies": [], "class_counts": {n: 0 for n in CLASS_NAMES},
}


# ─────────────────────────────────────────────
# Model + data loading
# ─────────────────────────────────────────────
def load_model_once():
    global model, X_sim, scaler, expected_feats

    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] Model not found at {MODEL_PATH}. Running in demo mode.")
        return False
    if not os.path.exists(DATA_PATH):
        print(f"[WARN] Dataset not found at {DATA_PATH}. Running in demo mode.")
        return False

    print("[INFO] Loading BiGRU-LSTM model ...")
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)

    gru_layer = next(
        (l for l in model.layers
         if "gru" in l.name.lower() or "bidirectional" in l.name.lower()),
        None,
    )
    expected_feats = (
        gru_layer.get_weights()[0].shape[0] if gru_layer else model.input_shape[2]
    )
    print(f"[INFO] Features expected by model: {expected_feats}")

    sample_df = pd.read_csv(DATA_PATH, nrows=100)
    num_cols  = sample_df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c != "Attack_type"]

    df = pd.read_csv(DATA_PATH, nrows=MAX_ROWS, usecols=feat_cols, low_memory=False)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    X_raw  = df.values[:, :expected_feats]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_sim  = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    print(f"[INFO] Loaded {len(X_sim)} packets for simulation.")
    return True


# ─────────────────────────────────────────────
# Scan loop (background thread)
# ─────────────────────────────────────────────
def scan_loop():
    global scan_running
    idx = 0
    total_rows = len(X_sim) if X_sim is not None else 0
    # Demo weights mirror real class imbalance
    demo_weights = [0.45,0.08,0.07,0.06,0.06,0.05,0.05,0.04,0.03,0.03,0.03,0.03,0.02,0.025,0.015]

    while scan_running:
        t0 = time.time()

        if model is not None and total_rows > 0:
            row   = X_sim[idx % total_rows : idx % total_rows + 1]
            probs = model.predict(row, verbose=0)
            cls   = int(np.argmax(probs))
        else:
            rnd, cumsum, cls = np.random.random(), 0, 0
            for i, w in enumerate(demo_weights):
                cumsum += w
                if rnd < cumsum:
                    cls = i
                    break

        latency_ms = (time.time() - t0) * 1000 + np.random.uniform(0.5, 3.0)

        stats["total"] += 1
        if cls == 0:
            stats["normal"] += 1
        else:
            stats["threats"] += 1
        stats["latencies"].append(round(latency_ms, 2))
        if len(stats["latencies"]) > 200:
            stats["latencies"].pop(0)
        cname = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class_{cls}"
        stats["class_counts"][cname] = stats["class_counts"].get(cname, 0) + 1

        event = {
            "idx":     stats["total"],
            "cls":     cls,
            "name":    cname,
            "threat":  cls != 0,
            "lat":     round(latency_ms, 2),
            "total":   stats["total"],
            "normal":  stats["normal"],
            "threats": stats["threats"],
            "avg_lat": round(sum(stats["latencies"]) / len(stats["latencies"]), 2),
            "class_counts": dict(stats["class_counts"]),
        }

        try:
            event_queue.put_nowait(event)
        except queue.Full:
            event_queue.get_nowait()
            event_queue.put_nowait(event)

        idx += 1
        time.sleep(SCAN_DELAY)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", class_names=CLASS_NAMES)


@app.route("/api/status")
def api_status():
    return jsonify({
        "model_loaded":   model is not None,
        "data_loaded":    X_sim is not None,
        "scanning":       scan_running,
        "expected_feats": int(expected_feats),
        "demo_mode":      model is None,
    })


@app.route("/api/stats")
def api_stats():
    avg = (
        round(sum(stats["latencies"]) / len(stats["latencies"]), 2)
        if stats["latencies"] else 0
    )
    return jsonify({**stats, "avg_lat": avg, "latencies": []})


@app.route("/api/start", methods=["POST"])
def api_start():
    global scan_running, scan_thread
    if scan_running:
        return jsonify({"status": "already_running"})
    scan_running = True
    scan_thread  = threading.Thread(target=scan_loop, daemon=True)
    scan_thread.start()
    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global scan_running
    scan_running = False
    return jsonify({"status": "stopped"})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global scan_running
    scan_running = False
    stats.update({"total": 0, "normal": 0, "threats": 0,
                  "latencies": [], "class_counts": {n: 0 for n in CLASS_NAMES}})
    while not event_queue.empty():
        event_queue.get_nowait()
    return jsonify({"status": "reset"})


@app.route("/api/stream")
def api_stream():
    """Server-Sent Events — pushes each scanned packet to the browser in real time."""
    def generate():
        yield "data: {\"ping\":true}\n\n"
        while True:
            try:
                event = event_queue.get(timeout=5)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                yield "data: {\"ping\":true}\n\n"   # keepalive

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    load_model_once()
    print("\n" + "=" * 52)
    print("  Smart Agri-IDS  →  http://localhost:5000")
    print("=" * 52 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
