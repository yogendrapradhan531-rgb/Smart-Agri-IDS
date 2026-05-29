from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def run_evaluation(model, X_test, y_test):
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Standard metrics [cite: 283, 285, 300]
    metrics = {
        "ACC": accuracy_score(y_true, y_pred),
        "PRE": precision_score(y_true, y_pred, average='macro'),
        "REC": recall_score(y_true, y_pred, average='macro'),
        "F1": f1_score(y_true, y_pred, average='macro'),
        "CM": confusion_matrix(y_true, y_pred) # Table II, III, IV [cite: 308]
    }
    
    for key, value in metrics.items():
        print(f"{key}: {value}")