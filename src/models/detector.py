import joblib
import numpy as np


class AnomalyDetector:
    def __init__(self):
        self.model = joblib.load("./models/anomaly_model.pkl")
        self.scaler = joblib.load("./models/scaler.pkl")

    
    def predict(self, features):
        X = np.array([[
            features["speed"],
            features["angle"],
            features["acceleration"],
            features["time_in_zone"],
            features["hand_distance"],
            features["stop_count"],
            features["hand_speed"],
            features["body_expansion"]
        ]])

        X_scaled = self.scaler.transform(X)

        pred = self.model.predict(X_scaled)

        return 1 if pred[0] == -1 else 0