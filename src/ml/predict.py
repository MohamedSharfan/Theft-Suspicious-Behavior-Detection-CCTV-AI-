import joblib 
import numpy as np

class AnomalyDetector:
    def __init__(self):
        self.model = joblib.load("models/anomaly_model.pkl")

    def predict(self, features):
        X = np.array([features])
        pred = self.model.predict(X)

        return 1 if pred[0] == -1 else 0