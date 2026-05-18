import joblib 
import numpy as np

class AnomalyDetector:
    def __init__(self):
        self.model = joblib.load("./models/anomaly_model.pkl")
        self.scaler = joblib.load("./models/scaler.pkl")

    def predict(self, features):

        X = np.array([[
            features["speed_mean"],
            features["speed_std"],

            features["angle_mean"],
            features["angle_std"],

            features["acc_mean"],
            features["acc_std"],

            features["time_mean"],

            features["hand_mean"],
            features["hand_std"],

            features["stop_mean"],
            features["stop_std"],

            features["crouch_ratio"],

            features["hand_speed"],
            features["body_expansion"],

            features["crowd_count"],
            features["crowd_density_ratio"],
            features["avg_person_distance"],
            features["crowd_mean_30"],
            features["crowd_std_30"]
        ]])

        X_scaled = self.scaler.transform(X)

        pred = self.model.predict(X_scaled)

        return 1 if pred[0] == -1 else 0