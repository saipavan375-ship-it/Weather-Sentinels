"""
train_model.py
Trains an Isolation Forest on engineered features from weather_data.csv,
evaluates it against the known synthetic labels, and saves the model.

Run after generate_data.py.
Output: model.joblib, scaler.joblib
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temperature"] = df["temperature"].interpolate(limit=2)  # fill only tiny gaps
    df["missing_flag"] = df["temperature"].isna().astype(int)
    df["temperature"] = df["temperature"].ffill().bfill()

    df["rate_of_change"] = df["temperature"].diff().fillna(0)
    df["rolling_mean_6h"] = df["temperature"].rolling(6, min_periods=1).mean()
    df["rolling_std_6h"] = df["temperature"].rolling(6, min_periods=1).std().fillna(0)
    df["z_score"] = (df["temperature"] - df["rolling_mean_6h"]) / (df["rolling_std_6h"] + 1e-6)

    # persistence: how many of the last 5 readings are (near) identical -> stuck-sensor signal
    def persistence(window):
        return int(np.sum(np.abs(window - window.iloc[-1]) < 0.05))
    df["persistence_5h"] = df["temperature"].rolling(5, min_periods=1).apply(persistence, raw=False)

    df["hour"] = df["timestamp"].dt.hour
    return df


FEATURES = [
    "temperature", "rate_of_change", "rolling_mean_6h",
    "rolling_std_6h", "z_score", "persistence_5h", "hour",
]


def main():
    df = pd.read_csv("weather_data.csv", parse_dates=["timestamp"])
    df = engineer_features(df)

    X = df[FEATURES].values
    y_true = df["is_anomaly"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # contamination ~ expected anomaly rate; matches the ~6% fault_rate used in generate_data.py
    model = IsolationForest(
        n_estimators=200,
        contamination=0.06,
        random_state=42,
    )
    model.fit(X_scaled)

    # IsolationForest: -1 = anomaly, 1 = normal -> convert to 1/0 to match our labels
    raw_pred = model.predict(X_scaled)
    y_pred = np.where(raw_pred == -1, 1, 0)

    print("=== Evaluation against synthetic labels ===")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
    print("Confusion matrix [rows=true, cols=predicted]:")
    print(confusion_matrix(y_true, y_pred))

    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    print("\nSaved model.joblib and scaler.joblib")


if __name__ == "__main__":
    main()
