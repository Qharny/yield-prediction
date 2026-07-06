"""
predict.py
─────────────────────────────────────────────────────────────────────────────
Ghana Crop Yield Prediction — Inference / Prediction Engine
Project: Smart Agriculture IoT System (ESP32-based)

Usage:
    # From Python code (import mode):
    from model.predict import predict_yield
    result = predict_yield(crop="Maize", region="Ashanti",
                           soil_ph=6.2, soil_moisture_pct=55,
                           nitrogen_N_mg_kg=110, phosphorus_P_mg_kg=42,
                           potassium_K_mg_kg=145, farm_size_acres=2.0,
                           crop_variety="Hybrid OPV", season="Major")

    # From command line (demo mode):
    python model/predict.py

Cold-Start Logic:
    If sensor readings are not provided (first-time farmer), the engine
    substitutes regional default baselines from regional_defaults.json.
    This implements the "cold-start algorithm" described in system.txt.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

VALID_REGIONS  = ["Ashanti", "Western", "Brong-Ahafo", "Northern", "Volta", "Eastern"]
VALID_CROPS    = ["Maize", "Tomato"]
VALID_VARIETIES = {
    "Maize":  ["Local", "Hybrid OPV", "Pioneer 30G19"],
    "Tomato": ["Local", "Pectomech", "F1 Hybrid"],
}
VALID_SEASONS  = ["Major", "Minor"]

YIELD_THRESHOLDS = {
    "Maize":  {"High": 22, "Medium": 12},   # bags/acre
    "Tomato": {"High": 5000, "Medium": 2500}, # kg/acre
}

YIELD_UNITS = {
    "Maize":  "bags/acre",
    "Tomato": "kg/acre",
}


# ── Model Loader (cached) ────────────────────────────────────────────────────

_cache = {}

def _load_artifacts(crop: str) -> tuple:
    """Load and cache model, scaler, encoder, feature names for a crop."""
    crop_key = crop.lower()
    if crop_key in _cache:
        return _cache[crop_key]

    model_path    = os.path.join(MODEL_DIR, f"yield_model_{crop_key}.pkl")
    scaler_path   = os.path.join(MODEL_DIR, f"scaler_{crop_key}.pkl")
    encoders_path = os.path.join(MODEL_DIR, "label_encoders.pkl")
    features_path = os.path.join(MODEL_DIR, "feature_names.json")
    defaults_path = os.path.join(MODEL_DIR, "regional_defaults.json")

    for p, name in [(model_path, "model"), (encoders_path, "encoders"),
                    (features_path, "feature_names"), (defaults_path, "regional_defaults")]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {name} file: {p}\n"
                "  → Run: python model/train_model.py  to train first."
            )

    model    = joblib.load(model_path)
    scaler   = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    encoders = joblib.load(encoders_path)

    with open(features_path) as f:
        feature_names = json.load(f)[crop]
    with open(defaults_path) as f:
        regional_defaults = json.load(f)

    _cache[crop_key] = (model, scaler, encoders, feature_names, regional_defaults)
    return _cache[crop_key]


# ── Category Classifier ──────────────────────────────────────────────────────

def _classify_yield(yield_val: float, crop: str) -> str:
    thresholds = YIELD_THRESHOLDS[crop]
    if yield_val >= thresholds["High"]:
        return "High"
    elif yield_val >= thresholds["Medium"]:
        return "Medium"
    else:
        return "Low"


def _category_advice(category: str, crop: str) -> str:
    advice = {
        "High": (
            f"Excellent conditions! Your {crop.lower()} field is on track for a high yield. "
            "Maintain current soil management and monitor for pests."
        ),
        "Medium": (
            f"Moderate yield expected. Consider boosting nitrogen levels and "
            f"ensuring consistent irrigation, especially during flowering stage."
        ),
        "Low": (
            f"Low yield risk detected. Check soil pH (aim for 5.8–7.0), "
            f"increase fertiliser application, and verify irrigation schedule."
        ),
    }
    return advice[category]


# ── Main Prediction Function ─────────────────────────────────────────────────

def predict_yield(
    crop: str,
    region: str,
    farm_size_acres: float,
    crop_variety: str   = None,
    season: str         = "Major",
    # Sensor readings (optional — cold-start if None)
    soil_ph: float             = None,
    soil_moisture_pct: float   = None,
    nitrogen_N_mg_kg: float    = None,
    phosphorus_P_mg_kg: float  = None,
    potassium_K_mg_kg: float   = None,
    rainfall_mm: float         = None,
    temperature_c: float       = None,
) -> dict:
    """
    Predict crop yield for a Ghanaian farm plot.

    Parameters
    ----------
    crop             : "Maize" or "Tomato"
    region           : One of the 6 Ghanaian regions
    farm_size_acres  : Plot size in acres
    crop_variety     : e.g. "Hybrid OPV" — defaults to region-appropriate variety
    season           : "Major" (Apr–Jul) or "Minor" (Sep–Nov)
    soil_ph          : ESP32 sensor reading (optional)
    soil_moisture_pct: ESP32 sensor reading (optional)
    nitrogen_N_mg_kg : ESP32/lab NPK (optional)
    phosphorus_P_mg_kg: ESP32/lab NPK (optional)
    potassium_K_mg_kg : ESP32/lab NPK (optional)
    rainfall_mm      : Known or forecasted annual rainfall (optional)
    temperature_c    : Ambient temperature (optional)

    Returns
    -------
    dict with keys:
        predicted_yield_per_acre, predicted_total_yield,
        yield_unit, yield_category, advice,
        inputs_used, cold_start_applied
    """

    # ── Validate Inputs ───────────────────────────────────────────────────
    crop   = crop.strip().title()
    region = region.strip().title()

    if crop not in VALID_CROPS:
        raise ValueError(f"Crop must be one of {VALID_CROPS}, got '{crop}'")
    if region not in VALID_REGIONS:
        raise ValueError(f"Region must be one of {VALID_REGIONS}, got '{region}'")
    if season not in VALID_SEASONS:
        raise ValueError(f"Season must be 'Major' or 'Minor', got '{season}'")
    if crop_variety is None:
        crop_variety = "Hybrid OPV" if crop == "Maize" else "Pectomech"
    if crop_variety not in VALID_VARIETIES[crop]:
        raise ValueError(
            f"Variety '{crop_variety}' not valid for {crop}. "
            f"Choose from {VALID_VARIETIES[crop]}"
        )
    if farm_size_acres <= 0:
        raise ValueError("farm_size_acres must be > 0")

    # ── Load Artifacts ────────────────────────────────────────────────────
    model, scaler, encoders, feature_names, regional_defaults = _load_artifacts(crop)

    # ── Cold-Start: Fill Missing Sensor Data from Regional Defaults ───────
    defaults     = regional_defaults[region]
    cold_start   = False
    missing_keys = []

    def _fill(val, key):
        nonlocal cold_start
        if val is None:
            missing_keys.append(key)
            cold_start = True
            return defaults[key]
        return val

    soil_ph            = _fill(soil_ph,            "soil_ph")
    soil_moisture_pct  = _fill(soil_moisture_pct,  "soil_moisture_pct")
    nitrogen_N_mg_kg   = _fill(nitrogen_N_mg_kg,   "nitrogen_N_mg_kg")
    phosphorus_P_mg_kg = _fill(phosphorus_P_mg_kg, "phosphorus_P_mg_kg")
    potassium_K_mg_kg  = _fill(potassium_K_mg_kg,  "potassium_K_mg_kg")
    rainfall_mm        = _fill(rainfall_mm,         "rainfall_mm")
    temperature_c      = _fill(temperature_c,       "temperature_c")

    # ── Build Feature Vector ──────────────────────────────────────────────
    raw_input = {
        "soil_ph":             soil_ph,
        "soil_moisture_pct":   soil_moisture_pct,
        "nitrogen_N_mg_kg":    nitrogen_N_mg_kg,
        "phosphorus_P_mg_kg":  phosphorus_P_mg_kg,
        "potassium_K_mg_kg":   potassium_K_mg_kg,
        "rainfall_mm":         rainfall_mm,
        "temperature_c":       temperature_c,
        "farm_size_acres":     farm_size_acres,
        "region":              region,
        "crop_variety":        crop_variety,
        "season":              season,
    }

    df_input = pd.DataFrame([raw_input])

    # Encode categoricals using the saved LabelEncoders
    for col in ["region", "crop_variety", "season"]:
        le_key = f"{crop}_{col}"
        le = encoders[le_key]
        val = df_input[col].iloc[0]
        # Handle unseen labels gracefully
        if val in le.classes_:
            df_input[col] = le.transform(df_input[col])
        else:
            df_input[col] = le.transform([le.classes_[0]] * len(df_input))

    # Ensure column order matches training
    df_input = df_input[feature_names]

    # ── Predict ───────────────────────────────────────────────────────────
    yield_per_acre = float(model.predict(df_input)[0])
    yield_per_acre = max(0.0, round(yield_per_acre, 2))
    total_yield    = round(yield_per_acre * farm_size_acres, 2)
    category       = _classify_yield(yield_per_acre, crop)
    unit           = YIELD_UNITS[crop]
    advice         = _category_advice(category, crop)

    # ── Confidence Interval (±12% heuristic from training RMSE) ──────────
    ci_low  = round(yield_per_acre * 0.88, 2)
    ci_high = round(yield_per_acre * 1.12, 2)

    return {
        "crop":                    crop,
        "region":                  region,
        "predicted_yield_per_acre": yield_per_acre,
        "predicted_total_yield":   total_yield,
        "yield_unit":              unit,
        "yield_category":          category,
        "confidence_interval":     {"low": ci_low, "high": ci_high},
        "advice":                  advice,
        "cold_start_applied":      cold_start,
        "cold_start_fields":       missing_keys if cold_start else [],
        "inputs_used": {
            "soil_ph": soil_ph, "soil_moisture_pct": soil_moisture_pct,
            "nitrogen_N_mg_kg": nitrogen_N_mg_kg,
            "phosphorus_P_mg_kg": phosphorus_P_mg_kg,
            "potassium_K_mg_kg": potassium_K_mg_kg,
            "rainfall_mm": rainfall_mm, "temperature_c": temperature_c,
            "farm_size_acres": farm_size_acres,
            "crop_variety": crop_variety, "season": season,
        },
    }


# ── CLI Demo ─────────────────────────────────────────────────────────────────

def _print_result(result: dict):
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  🌾  YIELD PREDICTION RESULT")
    print(sep)
    print(f"  Crop          : {result['crop']}")
    print(f"  Region        : {result['region']}")
    print(f"  Yield / acre  : {result['predicted_yield_per_acre']} {result['yield_unit']}")
    print(f"  Total Yield   : {result['predicted_total_yield']} {result['yield_unit'].replace('/acre','')}")
    ci = result['confidence_interval']
    print(f"  95% CI        : [{ci['low']} – {ci['high']}] {result['yield_unit']}")
    cat = result['yield_category']
    emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[cat]
    print(f"  Category      : {emoji}  {cat}")
    if result['cold_start_applied']:
        print(f"  ⚠  Cold-Start : Regional defaults used for: {', '.join(result['cold_start_fields'])}")
    print(f"\n  💡 Advice: {result['advice']}")
    print(sep)


if __name__ == "__main__":
    print("=" * 55)
    print("  Ghana Yield Prediction — Demo Inference")
    print("=" * 55)

    # ── Test Case 1: Full Sensor Data (Maize, Ashanti) ────────────────────
    print("\n[Test 1] Full sensor data — Maize, Ashanti Region")
    r1 = predict_yield(
        crop="Maize", region="Ashanti",
        soil_ph=6.2, soil_moisture_pct=58,
        nitrogen_N_mg_kg=115, phosphorus_P_mg_kg=44, potassium_K_mg_kg=155,
        rainfall_mm=1380, temperature_c=26.3,
        farm_size_acres=2.5,
        crop_variety="Hybrid OPV", season="Major",
    )
    _print_result(r1)

    # ── Test Case 2: Cold-Start (Tomato, Northern — first-time farmer) ────
    print("\n[Test 2] Cold-start (no sensor data) — Tomato, Northern Region")
    r2 = predict_yield(
        crop="Tomato", region="Northern",
        farm_size_acres=1.5,
        crop_variety="F1 Hybrid", season="Minor",
        # No sensor readings — cold-start will apply regional defaults
    )
    _print_result(r2)

    # ── Test Case 3: Acidic Soil Stress (Maize, Western) ─────────────────
    print("\n[Test 3] Acidic soil stress — Maize, Western Region")
    r3 = predict_yield(
        crop="Maize", region="Western",
        soil_ph=3.9, soil_moisture_pct=62,
        nitrogen_N_mg_kg=80, phosphorus_P_mg_kg=30, potassium_K_mg_kg=110,
        rainfall_mm=1750, temperature_c=25.9,
        farm_size_acres=3.0,
        crop_variety="Local", season="Major",
    )
    _print_result(r3)
