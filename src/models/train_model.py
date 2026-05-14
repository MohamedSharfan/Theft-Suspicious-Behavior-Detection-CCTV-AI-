import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib


df = pd.read_csv("./data/raw/features.csv")

df = df[df["speed"] > 0.01]

x = df[["speed", "angle", "acceleration", "time_in_zone", "hand_distance", "stop_count"]]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(x_scaled)

joblib.dump(model, "./models/anomaly_model.pkl")
joblib.dump(scaler, "./models/scaler.pkl")

print("model trained and saved")
