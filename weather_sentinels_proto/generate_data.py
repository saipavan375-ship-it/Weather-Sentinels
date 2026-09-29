"""
generate_data.py
Creates a realistic AWS (Automatic Weather Station) time series with
seasonal patterns, then injects labeled synthetic faults on top of it.

Output: weather_data.csv
Columns:
  timestamp, temperature, humidity, rainfall, wind_speed, pressure,
  is_anomaly (0/1), anomaly_type (Normal/Spike/Drop/Stuck/Drift/Noise/Missing)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_DAYS = 180          # ~6 months of hourly data
FREQ_HOURS = 1
N_POINTS = N_DAYS * 24 // FREQ_HOURS
STATION_ID = "AWS_073"


def base_signal(n_points: int):
    """Builds smooth, seasonal + daily-cycle base signals for each sensor."""
    t = np.arange(n_points)
    day_frac = (t % 24) / 24.0
    season_frac = (t / (24 * 365)) * 2 * np.pi

    # Temperature: daily sine (warm afternoons) + seasonal sine + noise
    temperature = (
        26
        + 6 * np.sin(2 * np.pi * day_frac - np.pi / 2)   # daily cycle
        + 5 * np.sin(season_frac)                         # seasonal cycle
        + RNG.normal(0, 0.6, n_points)
    )

    humidity = np.clip(
        65 - 15 * np.sin(2 * np.pi * day_frac - np.pi / 2) + RNG.normal(0, 2, n_points),
        10, 100
    )

    rainfall = np.clip(
        RNG.exponential(0.3, n_points) * (RNG.random(n_points) < 0.08),
        0, None
    )

    wind_speed = np.clip(8 + 4 * np.sin(season_frac + 1) + RNG.normal(0, 1.5, n_points), 0, None)

    pressure = 1012 + 3 * np.sin(season_frac + 2) + RNG.normal(0, 0.5, n_points)

    return pd.DataFrame({
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "wind_speed": wind_speed,
        "pressure": pressure,
    })


def inject_faults(df: pd.DataFrame, fault_rate: float = 0.06):
    """Randomly injects labeled faults into the temperature column."""
    n = len(df)
    df["is_anomaly"] = 0
    df["anomaly_type"] = "Normal"

    n_faults = int(n * fault_rate)
    fault_types = ["Spike", "Drop", "Stuck", "Drift", "Noise", "Missing"]

    used_indices = set()
    faults_added = 0
    attempts = 0

    while faults_added < n_faults and attempts < n_faults * 20:
        attempts += 1
        ftype = RNG.choice(fault_types)
        start = RNG.integers(20, n - 20)

        if ftype in ("Spike", "Drop"):
            span = 1
        elif ftype == "Stuck":
            span = RNG.integers(4, 10)
        elif ftype == "Drift":
            span = RNG.integers(10, 30)
        elif ftype == "Noise":
            span = RNG.integers(5, 15)
        else:  # Missing
            span = RNG.integers(3, 8)

        idx_range = range(start, min(start + span, n))
        if any(i in used_indices for i in idx_range):
            continue

        if ftype == "Spike":
            df.loc[start, "temperature"] += RNG.uniform(12, 25)
        elif ftype == "Drop":
            df.loc[start, "temperature"] -= RNG.uniform(12, 25)
        elif ftype == "Stuck":
            stuck_val = df.loc[start, "temperature"]
            df.loc[list(idx_range), "temperature"] = stuck_val
        elif ftype == "Drift":
            offsets = np.linspace(0, RNG.uniform(6, 12), len(list(idx_range)))
            df.loc[list(idx_range), "temperature"] += offsets
        elif ftype == "Noise":
            df.loc[list(idx_range), "temperature"] += RNG.normal(0, 4, len(list(idx_range)))
        elif ftype == "Missing":
            df.loc[list(idx_range), "temperature"] = np.nan

        df.loc[list(idx_range), "is_anomaly"] = 1
        df.loc[list(idx_range), "anomaly_type"] = ftype
        used_indices.update(idx_range)
        faults_added += len(list(idx_range))

    return df


def main():
    df = base_signal(N_POINTS)
    df = inject_faults(df, fault_rate=0.06)

    timestamps = pd.date_range("2025-01-01", periods=N_POINTS, freq=f"{FREQ_HOURS}h")
    df.insert(0, "timestamp", timestamps)
    df.insert(1, "station_id", STATION_ID)

    df.to_csv("weather_data.csv", index=False)

    print(f"Generated {len(df)} rows -> weather_data.csv")
    print(df["anomaly_type"].value_counts())


if __name__ == "__main__":
    main()
