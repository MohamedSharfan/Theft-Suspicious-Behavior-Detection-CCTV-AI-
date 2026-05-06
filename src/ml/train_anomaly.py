import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("./data/raw/features.csv")
x = df[["speed", "direction", "acceleration", "time"]]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

model = IsolationForest(
    contamination=0.03,
    n_estimators=100
)

model.fit(x_scaled)

joblib.dump(model, "./models/anomaly_model.pkl")
joblib.dump(scaler, "./models/scaler.pkl")

print("model trained")