# AgriPredict Ghana — Crop Yield Prediction System

AI-powered crop yield forecasting for Ghanaian maize and tomato farms,
built on a Random Forest + XGBoost ensemble trained on regional agro-ecological data.

---

## Quick Start (3 Steps)

### Step 1 — Install Dependencies
```bash
python -m pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn flask
```

### Step 2 — Generate Dataset & Train Model
```bash
# Generate the synthetic Ghanaian dataset (3000 records)
python dataset/generate_ghana_dataset.py

# Train the ML model (Random Forest + XGBoost)
python model/train_model.py
```

### Step 3 — Run the Web App
```bash
python app.py
```
Then open **http://localhost:5000** in your browser.

---

## Model Performance (Trained Results)

| Crop   | Best Model | R²     | RMSE   | CV R² (5-Fold)       |
|--------|-----------|--------|--------|----------------------|
| Maize  | XGBoost   | 0.8756 | 1.221  | 0.8923 ± 0.0078 ✅   |
| Tomato | XGBoost   | 0.9062 | 266.3  | 0.8979 ± 0.0080 ✅   |

> Target benchmark (Asamoah et al. 2024): R² ≥ 0.75 — **both models exceed this** ✅

---

## Project Structure

```
yield prediction/
├── app.py                          ← Flask web app (run this)
├── requirements.txt
├── dataset/
│   ├── generate_ghana_dataset.py   ← Synthetic dataset generator
│   └── ghana_yield_data.csv        ← Generated dataset (3000 rows)
├── model/
│   ├── train_model.py              ← ML training script
│   ├── predict.py                  ← Inference engine
│   ├── yield_model_maize.pkl       ← Trained XGBoost (Maize)
│   ├── yield_model_tomato.pkl      ← Trained XGBoost (Tomato)
│   ├── label_encoders.pkl          ← Categorical encoders
│   ├── feature_names.json          ← Feature registry
│   ├── regional_defaults.json      ← Cold-start baselines
│   └── plots/
│       ├── feature_importance_maize.png
│       ├── feature_importance_tomato.png
│       ├── actual_vs_predicted_maize.png
│       └── actual_vs_predicted_tomato.png
└── templates/
    └── index.html                  ← Web app frontend
```

---

## Features

- **6 Ghana Regions**: Ashanti, Western, Brong-Ahafo, Northern, Volta, Eastern
- **2 Crops**: Maize (bags/acre) and Tomato (kg/acre)
- **Cold-Start Algorithm**: First-time farmers get regional defaults; live ESP32 data replaces these progressively
- **REST API**: POST `/api/predict` — integrate with any mobile or IoT app
- **Feature Importance**: NPK, pH, and rainfall shown as top predictors

---

## REST API Usage

**Endpoint**: `POST /api/predict`

```json
{
  "crop": "Maize",
  "region": "Ashanti",
  "farm_size_acres": 2.5,
  "crop_variety": "Hybrid OPV",
  "season": "Major",
  "soil_ph": 6.2,
  "soil_moisture_pct": 55,
  "nitrogen_N_mg_kg": 110,
  "phosphorus_P_mg_kg": 42,
  "potassium_K_mg_kg": 155
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "crop": "Maize",
    "region": "Ashanti",
    "predicted_yield_per_acre": 21.4,
    "predicted_total_yield": 53.5,
    "yield_unit": "bags/acre",
    "yield_category": "High",
    "confidence_interval": { "low": 18.8, "high": 24.0 },
    "advice": "Excellent conditions! ...",
    "cold_start_applied": false
  }
}
```

---

## Deploying for AgriVault Integration

The AgriVault Ghana web app (React + Supabase) calls this API through a
`predict-yield` Supabase Edge Function, which runs in Supabase's cloud —
so `localhost:5000` is **not** reachable from it. This API needs a public
URL before that integration works end-to-end.

**Recommended: Render (free tier)**
1. Push this repo to GitHub (already done: `Qharny/yield-prediction`).
2. On [render.com](https://render.com), create a **Web Service** from the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app` (a `Procfile` with this is already included).
5. Optionally set an env var `YIELD_API_KEY` to a random secret string — this
   locks down `/api/predict` so only requests with a matching `X-API-Key`
   header succeed.
6. Once deployed, copy the public URL (e.g. `https://yield-prediction.onrender.com`).

Any other Python host (Railway, Fly.io, PythonAnywhere) works the same way —
they just need `gunicorn app:app` as the start command.

**Wire it into AgriVault:**
In the Supabase project's Edge Function secrets, set:
- `YIELD_API_URL` = the public URL from above (no trailing slash)
- `YIELD_API_KEY` = the same value used above, if you set one

The `predict-yield` edge function forwards requests to `${YIELD_API_URL}/api/predict`.

> Free tiers on Render/Railway sleep after inactivity — the first prediction
> after idle time may take 20-30s while the instance wakes up.

---

## Methodology

Based on system design from `system.txt`:

1. **Offline Training** on synthetic regional data (Asamoah et al. 2024 parameters)
2. **Live Inference** using ESP32 sensor readings as feature inputs
3. **Cold-Start** algorithm uses regional defaults for first-time farmers
4. **Transfer Learning** approach: historical regional profiles → personalized farm forecasts
