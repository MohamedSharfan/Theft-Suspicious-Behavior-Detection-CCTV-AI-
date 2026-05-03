import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

df = pd.read_csv("data/raw/features.csv")
x = df[["speed", "direction", "acceleration", "time"]]

model = IsolationForest(
    contamination=0.03,
    n_estimators=100
)

model.fit(x)

joblib.dump(model, "models/anomaly_model.pkl")

print("model trained")