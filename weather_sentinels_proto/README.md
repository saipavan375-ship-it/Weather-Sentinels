# Weather Sentinels — Prototype

A real (not simulated-in-browser) anomaly detection prototype: synthetic
AWS weather data with labeled faults, an Isolation Forest trained and
evaluated against those labels, and a Streamlit dashboard that replays
the data as a live feed with anomaly explanations.

## Setup in VS Code

1. Open this folder in VS Code (`File > Open Folder`).
2. Open a terminal in VS Code (`` Ctrl+` ``) and create a virtual environment:

   ```bash
   python -m venv venv
   ```

   Activate it:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run it — 3 steps

```bash
python generate_data.py     # creates weather_data.csv with labeled faults
python train_model.py       # trains Isolation Forest, prints precision/recall
streamlit run app.py        # opens the dashboard in your browser
```

`train_model.py` will print something like:

```
=== Evaluation against synthetic labels ===
              precision    recall  f1-score   support
      Normal       0.98      0.97      0.97      ...
     Anomaly       0.71      0.80      0.75      ...
```

Use these real numbers in your presentation — don't estimate them.

## What to show your mentor / judges

- `generate_data.py` — proves you understand what real sensor faults look
  like and can generate labeled ground truth for evaluation.
- `train_model.py` output — your actual measured precision/recall/confusion
  matrix, not a claimed number.
- `app.py` (Streamlit) — the live dashboard: run "Live replay" mode and let
  an anomaly appear naturally while you talk.

## Next steps (mention as roadmap, don't fake it live)

- Swap `generate_data.py`'s synthetic base signal for real NOAA / NASA POWER
  / Indian AWS data, keeping the same fault-injection step on top of it.
- Add an LSTM Autoencoder (`tensorflow`/`keras`) trained on the same
  features for pattern-level (not just point) anomaly detection.
- Add a context engine comparing multiple nearby stations before finalizing
  a "real weather event" vs "sensor fault" classification.



