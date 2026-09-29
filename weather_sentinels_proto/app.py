"""
app.py
Weather Sentinels — Streamlit prototype dashboard.

Loads weather_data.csv + the trained Isolation Forest model, replays the
series as a simulated live feed, and shows detected anomalies with
explanations, severity, and a rolling sensor health score.

Run:
    streamlit run app.py
"""

import time
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

from train_model import engineer_features, FEATURES

st.set_page_config(page_title="Weather Sentinels", page_icon="🛡️", layout="wide")

# ---------------- load data + model ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("weather_data.csv", parse_dates=["timestamp"])
    df = engineer_features(df)
    return df

@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

df = load_data()
model, scaler = load_model()

X_scaled = scaler.transform(df[FEATURES].values)
scores = model.decision_function(X_scaled)          # higher = more normal
raw_pred = model.predict(X_scaled)                   # -1 anomaly, 1 normal
df["ml_flag"] = np.where(raw_pred == -1, 1, 0)
df["anomaly_score"] = (1 - (scores - scores.min()) / (scores.max() - scores.min())) * 100


def classify_row(row) -> dict:
    reasons = []
    score = row["anomaly_score"]

    if row["missing_flag"] == 1:
        reasons.append("Value missing from feed (possible comms/power failure)")
    if abs(row["z_score"]) > 3:
        reasons.append(f"Reading far outside recent distribution (z ≈ {row['z_score']:.1f})")
    elif abs(row["z_score"]) > 2:
        reasons.append(f"Reading moderately outside recent pattern (z ≈ {row['z_score']:.1f})")
    if abs(row["rate_of_change"]) > 10:
        reasons.append(f"Sudden change of {row['rate_of_change']:+.1f}° between readings")
    if row["persistence_5h"] >= 4:
        reasons.append(f"Identical value repeated {int(row['persistence_5h'])}x — possible stuck sensor")

    if row["ml_flag"] == 1 and score >= 70:
        status, cause = "Anomaly", ("Possible stuck sensor" if row["persistence_5h"] >= 4
                                     else "Possible sensor spike/fault")
    elif row["ml_flag"] == 1 or score >= 45:
        status, cause = "Suspicious", "Needs monitoring"
    else:
        status, cause = "Normal", None

    if not reasons and status != "Normal":
        reasons.append("Flagged by model as statistically unusual pattern")

    return {"status": status, "cause": cause, "reasons": reasons}


# ---------------- sidebar controls ----------------
st.sidebar.title("🛡️ Weather Sentinels")
st.sidebar.caption("AI/ML anomaly detection prototype")
st.sidebar.divider()

mode = st.sidebar.radio("View", ["Live replay (simulated stream)", "Full dataset explorer"])
window_size = st.sidebar.slider("Chart window (hours)", 24, 200, 72)

if mode == "Live replay (simulated stream)":
    speed = st.sidebar.slider("Replay speed (rows/tick)", 1, 20, 4)
    auto_play = st.sidebar.checkbox("Auto-play", value=True)

st.sidebar.divider()
total_anom = int(df["ml_flag"].sum())
st.sidebar.metric("Total flagged (full dataset)", total_anom)
st.sidebar.metric("Flag rate", f"{100*total_anom/len(df):.1f}%")

st.title("Weather Sentinels — Live Prototype")
st.caption("Station AWS_073 · Isolation Forest trained on engineered features from the synthetic dataset")


def render_chart(view_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=view_df["timestamp"], y=view_df["temperature"],
        mode="lines", name="Temperature", line=dict(color="#5fc4d6", width=2)
    ))
    for status, color in [("Suspicious", "#f2b544"), ("Anomaly", "#ef5350")]:
        pts = view_df[view_df["_status"] == status]
        if len(pts):
            fig.add_trace(go.Scatter(
                x=pts["timestamp"], y=pts["temperature"], mode="markers",
                name=status, marker=dict(color=color, size=9, symbol="circle")
            ))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#0f2035", paper_bgcolor="#0f2035",
        font=dict(color="#e7eef5"),
        xaxis=dict(gridcolor="#1f3a56"), yaxis=dict(gridcolor="#1f3a56", title="°C"),
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def render_feed(view_df):
    flagged = view_df[view_df["_status"] != "Normal"].sort_values("timestamp", ascending=False)
    if len(flagged) == 0:
        st.info("No anomalies in current window.")
        return
    for _, row in flagged.head(12).iterrows():
        color = "🔴" if row["_status"] == "Anomaly" else "🟡"
        with st.container(border=True):
            st.markdown(f"**{color} {row['_status']}** · {row['timestamp']} · "
                        f"score {row['anomaly_score']:.0f}%  ·  value {row['temperature']:.1f}°C")
            for r in row["_reasons"]:
                st.markdown(f"- {r}")
            if row["_cause"]:
                st.caption(f"→ Suggested cause: {row['_cause']}")


if mode == "Full dataset explorer":
    view = df.tail(window_size).copy()
    classified = view.apply(classify_row, axis=1, result_type="expand")
    view["_status"] = classified["status"]
    view["_cause"] = classified["cause"]
    view["_reasons"] = classified["reasons"]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(render_chart(view), use_container_width=True)
    with col2:
        st.subheader("Anomaly feed")
        render_feed(view)

    st.subheader("Raw data (last rows in window)")
    st.dataframe(
        view[["timestamp", "temperature", "anomaly_score", "_status", "anomaly_type"]]
        .rename(columns={"anomaly_type": "true_label (synthetic)"}),
        use_container_width=True, height=250,
    )

else:
    if "cursor" not in st.session_state:
        st.session_state.cursor = window_size

    placeholder = st.empty()

    def draw(cursor):
        start = max(0, cursor - window_size)
        view = df.iloc[start:cursor].copy()
        classified = view.apply(classify_row, axis=1, result_type="expand")
        view["_status"] = classified["status"]
        view["_cause"] = classified["cause"]
        view["_reasons"] = classified["reasons"]

        with placeholder.container():
            latest = view.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Latest reading", f"{latest['temperature']:.1f} °C")
            c2.metric("Status", classify_row(latest)["status"])
            c3.metric("Anomaly score", f"{latest['anomaly_score']:.0f}%")
            n_anom_window = int((view["_status"] != "Normal").sum())
            c4.metric("Flags in window", n_anom_window)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(render_chart(view), use_container_width=True, key=f"chart_{cursor}")
            with col2:
                st.subheader("Anomaly feed")
                render_feed(view)

    draw(st.session_state.cursor)

    if auto_play and st.session_state.cursor < len(df):
        st.session_state.cursor = min(len(df), st.session_state.cursor + speed)
        time.sleep(0.6)
        st.rerun()
    elif st.session_state.cursor >= len(df):
        st.success("Reached end of dataset — restart the app or switch to Full dataset explorer.")
