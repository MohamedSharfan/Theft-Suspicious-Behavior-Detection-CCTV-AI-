import joblib 
import numpy as np

class AnomalyDetector:
    def __init__(self):
        self.model = joblib.load("./models/anomaly_model.pkl")
        self.scaler = joblib.load("./models/scaler.pkl")

    def predict(self, features):
        if len(features) != 4:
            raise ValueError("Invalid feature lenght")
        
        X = np.array([features])
        X = self.scaler.transform(X)

        pred = self.model.predict(X)
        return 1 if pred[0] == -1 else 0