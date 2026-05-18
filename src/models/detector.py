import joblib
import numpy as np
import pandas as pd


class AnomalyDetector:
    def __init__(self):
        self.model = joblib.load("./models/anomaly_model.pkl")
        self.scaler = joblib.load("./models/scaler.pkl")

    
    def predict(self, features):
        X = pd.DataFrame([{
            "speed_mean": features["speed_mean"],
            "speed_std": features["speed_std"],

            "angle_mean": features["angle_mean"],
            "angle_std": features["angle_std"],

            "acc_mean": features["acc_mean"],
            "acc_std": features["acc_std"],

            "time_mean": features["time_mean"],

            "hand_mean": features["hand_mean"],
            "hand_std": features["hand_std"],

            "stop_mean": features["stop_mean"],
            "stop_std": features["stop_std"],

            "crouch_ratio": features["crouch_ratio"],

            "hand_speed": features["hand_speed"],
            "body_expansion": features["body_expansion"],

            "crowd_count": features["crowd_count"],
            "crowd_density_ratio": features["crowd_density_ratio"],
            "avg_person_distance": features["avg_person_distance"],
            "crowd_mean_30": features["crowd_mean_30"],
            "crowd_std_30": features["crowd_std_30"]
        }])

        # X = np.array([[
        #     features["speed_mean"],
        #     features["speed_std"],
        #     features["angle_mean"],
        #     features["angle_std"],
        #     features["acc_mean"],
        #     features["acc_std"],
        #     features["time_mean"],
        #     features["hand_mean"],
        #     features["hand_std"],
        #     features["stop_mean"],
        #     features["stop_std"],
        #     features["crouch_ratio"],
        #     features["hand_speed"],
        #     features["body_expansion"],

        #     features["crowd_count"],
        #     features["crowd_density_ratio"],
        #     features["avg_person_distance"],
        #     features["crowd_mean_30"],
        #     features["crowd_std_30"]
        # ]])

        X_scaled = self.scaler.transform(X)

        pred = self.model.predict(X_scaled)

        return 1 if pred[0] == -1 else 0