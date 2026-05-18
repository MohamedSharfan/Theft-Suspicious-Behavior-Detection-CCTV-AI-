import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib


df = pd.read_csv("./data/raw/features.csv")

df = df[df["speed_mean"] > 0.01]

X = df[[
    "speed_mean",
    "speed_std",

    "angle_mean", 
    "angle_std",

    "acc_mean", 
    "acc_std",

    "time_mean",

    "hand_mean",
    "hand_std",

    "stop_mean",
    "stop_std",

    "crouch_ratio",

    "hand_speed", 
    "body_expansion",

    "crowd_count",
    "crowd_density_ratio",
    "avg_person_distance",
    "crowd_mean_30",
    "crowd_std_30"
    ]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(contamination=0.05, 
                        n_estimators=200, 
                        random_state=42)
model.fit(X_scaled)

joblib.dump(model, "./models/anomaly_model.pkl")
joblib.dump(scaler, "./models/scaler.pkl")

print("model trained and saved")
