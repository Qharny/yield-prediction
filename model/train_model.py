"""
train_model.py
─────────────────────────────────────────────────────────────────────────────
Ghana Crop Yield Prediction — Offline ML Training Script
Project: Smart Agriculture IoT System (ESP32-based)

Models Trained:
    1. Random Forest Regressor  (per-crop)
    2. XGBoost Regressor        (per-crop)
    → Best model saved as yield_model_{crop}.pkl

Features Used:
    Sensor   : soil_ph, soil_moisture_pct, nitrogen_N_mg_kg,
               phosphorus_P_mg_kg, potassium_K_mg_kg
    Macro    : rainfall_mm, temperature_c
    User     : farm_size_acres, crop_variety (encoded), season (encoded)
    Regional : region (encoded)

Target:
    yield_value  (bags/acre for Maize | kg/acre for Tomato)

Outputs:
    model/yield_model_maize.pkl
    model/yield_model_tomato.pkl
    model/scaler_maize.pkl
    model/scaler_tomato.pkl
    model/feature_names.json
    model/plots/feature_importance_maize.png
    model/plots/feature_importance_tomato.png
    model/plots/actual_vs_predicted.png
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠  XGBoost not installed — will use GradientBoosting as fallback.")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")
DATA_PATH   = os.path.join(DATASET_DIR, "ghana_yield_data.csv")
MODEL_DIR   = BASE_DIR
PLOTS_DIR   = os.path.join(MODEL_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Categorical Columns ──────────────────────────────────────────────────────
CAT_COLS = ["region", "crop_variety", "season"]

FEATURE_COLS = [
    "soil_ph", "soil_moisture_pct",
    "nitrogen_N_mg_kg", "phosphorus_P_mg_kg", "potassium_K_mg_kg",
    "rainfall_mm", "temperature_c",
    "farm_size_acres",
    "region", "crop_variety", "season",
]

TARGET_COL = "yield_value"


# ── Helpers ──────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def evaluate_model(model, X_test, y_test, model_name: str, crop: str) -> dict:
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"\n  [{crop}] {model_name}")
    print(f"    RMSE : {rmse:.3f}")
    print(f"    MAE  : {mae:.3f}")
    print(f"    R²   : {r2:.4f}  {'✅' if r2 >= 0.75 else '⚠ below target 0.75'}")
    return {"model": model_name, "crop": crop, "RMSE": rmse, "MAE": mae, "R2": r2}


def plot_feature_importance(model, feature_names: list, crop: str, model_name: str):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return

    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71" if i < 3 else "#3498db" for i in range(len(feature_names))]
    bars = ax.barh(
        [feature_names[i] for i in indices],
        [importances[i] for i in indices],
        color=[colors[i] for i in range(len(feature_names))],
        edgecolor="white", linewidth=0.5
    )
    ax.set_xlabel("Feature Importance Score", fontsize=11)
    ax.set_title(f"Feature Importance — {crop} Yield ({model_name})", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    for bar, val in zip(bars, [importances[i] for i in indices]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"feature_importance_{crop.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    📊 Feature importance plot → {path}")


def plot_actual_vs_predicted(y_test, y_pred, crop: str, r2: float):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.45, s=18, color="#3498db", edgecolors="none")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
    ax.set_xlabel(f"Actual Yield", fontsize=11)
    ax.set_ylabel(f"Predicted Yield", fontsize=11)
    unit = "bags/acre" if crop == "Maize" else "kg/acre"
    ax.set_title(f"{crop} — Actual vs Predicted Yield ({unit})\nR² = {r2:.4f}", fontsize=12)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"actual_vs_predicted_{crop.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    📊 Actual vs Predicted plot → {path}")


# ── Per-Crop Training Pipeline ───────────────────────────────────────────────

def train_crop(df_crop: pd.DataFrame, crop: str, encoders: dict) -> dict:
    print_section(f"Training — {crop.upper()}")

    X = df_crop[FEATURE_COLS].copy()
    y = df_crop[TARGET_COL].copy()

    # ── Encode Categoricals ───────────────────────────────────────────────
    for col in CAT_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[f"{crop}_{col}"] = le

    feature_names = list(X.columns)

    # ── Train / Test Split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Scale Features ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    results = []
    best_r2 = -np.inf
    best_model = None
    best_name = ""

    # ── Model 1: Random Forest ────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)   # RF doesn't need scaled features
    res_rf = evaluate_model(rf, X_test, y_test, "Random Forest", crop)
    results.append(res_rf)
    plot_feature_importance(rf, feature_names, crop, "Random Forest")

    if res_rf["R2"] > best_r2:
        best_r2 = res_rf["R2"]
        best_model = rf
        best_name = "RandomForest"

    # ── Model 2: XGBoost / GradientBoosting ──────────────────────────────
    if XGBOOST_AVAILABLE:
        xgb = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
        xgb.fit(X_train, y_train)
        res_xgb = evaluate_model(xgb, X_test, y_test, "XGBoost", crop)
        results.append(res_xgb)
        plot_feature_importance(xgb, feature_names, crop, "XGBoost")

        if res_xgb["R2"] > best_r2:
            best_r2 = res_xgb["R2"]
            best_model = xgb
            best_name = "XGBoost"
    else:
        gb = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.07,
            subsample=0.8, random_state=42
        )
        gb.fit(X_train, y_train)
        res_gb = evaluate_model(gb, X_test, y_test, "GradientBoosting", crop)
        results.append(res_gb)
        plot_feature_importance(gb, feature_names, crop, "GradientBoosting")

        if res_gb["R2"] > best_r2:
            best_r2 = res_gb["R2"]
            best_model = gb
            best_name = "GradientBoosting"

    # ── Cross-Validation (5-Fold) ─────────────────────────────────────────
    print(f"\n  5-Fold Cross-Validation ({crop}, best model = {best_name}):")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X, y, cv=kf, scoring="r2", n_jobs=-1)
    print(f"    CV R² scores : {np.round(cv_scores, 4)}")
    print(f"    CV Mean R²   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Actual vs Predicted Plot ──────────────────────────────────────────
    y_pred_best = best_model.predict(X_test)
    plot_actual_vs_predicted(y_test.values, y_pred_best, crop, best_r2)

    # ── Save Best Model + Scaler ──────────────────────────────────────────
    model_path  = os.path.join(MODEL_DIR, f"yield_model_{crop.lower()}.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{crop.lower()}.pkl")
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n  💾 Best model saved  → {model_path}")
    print(f"  💾 Scaler saved      → {scaler_path}")

    return {
        "crop": crop,
        "best_model": best_name,
        "best_r2": round(best_r2, 4),
        "feature_names": feature_names,
        "results": results,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print_section("Ghana Crop Yield Prediction — Model Training")

    # ── Load Dataset ──────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found: {DATA_PATH}")
        print("   Run: python dataset/generate_ghana_dataset.py  first.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"\n  Dataset loaded : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Crops          : {df['crop'].unique().tolist()}")
    print(f"  Regions        : {df['region'].unique().tolist()}")

    encoders = {}
    summary  = []
    feature_registry = {}

    # ── Train Per-Crop ─────────────────────────────────────────────────────
    for crop in ["Maize", "Tomato"]:
        df_crop = df[df["crop"] == crop].copy().reset_index(drop=True)
        result  = train_crop(df_crop, crop, encoders)
        summary.append(result)
        feature_registry[crop] = result["feature_names"]

    # ── Save Encoders & Feature Registry ──────────────────────────────────
    enc_path = os.path.join(MODEL_DIR, "label_encoders.pkl")
    joblib.dump(encoders, enc_path)

    reg_path = os.path.join(MODEL_DIR, "feature_names.json")
    with open(reg_path, "w") as f:
        json.dump(feature_registry, f, indent=2)

    # ── Regional Baselines (Cold-Start) ────────────────────────────────────
    regional_defaults = {
        "Ashanti":     {"soil_ph": 6.0, "soil_moisture_pct": 52, "nitrogen_N_mg_kg": 95,
                        "phosphorus_P_mg_kg": 38, "potassium_K_mg_kg": 130,
                        "rainfall_mm": 1400, "temperature_c": 26.5},
        "Western":     {"soil_ph": 5.8, "soil_moisture_pct": 55, "nitrogen_N_mg_kg": 100,
                        "phosphorus_P_mg_kg": 40, "potassium_K_mg_kg": 140,
                        "rainfall_mm": 1800, "temperature_c": 25.8},
        "Brong-Ahafo": {"soil_ph": 6.2, "soil_moisture_pct": 50, "nitrogen_N_mg_kg": 88,
                        "phosphorus_P_mg_kg": 34, "potassium_K_mg_kg": 120,
                        "rainfall_mm": 1200, "temperature_c": 27.2},
        "Northern":    {"soil_ph": 6.5, "soil_moisture_pct": 42, "nitrogen_N_mg_kg": 70,
                        "phosphorus_P_mg_kg": 28, "potassium_K_mg_kg": 100,
                        "rainfall_mm": 900,  "temperature_c": 29.5},
        "Volta":       {"soil_ph": 6.1, "soil_moisture_pct": 48, "nitrogen_N_mg_kg": 82,
                        "phosphorus_P_mg_kg": 32, "potassium_K_mg_kg": 115,
                        "rainfall_mm": 1100, "temperature_c": 27.8},
        "Eastern":     {"soil_ph": 5.9, "soil_moisture_pct": 51, "nitrogen_N_mg_kg": 90,
                        "phosphorus_P_mg_kg": 36, "potassium_K_mg_kg": 125,
                        "rainfall_mm": 1300, "temperature_c": 26.9},
    }
    defaults_path = os.path.join(MODEL_DIR, "regional_defaults.json")
    with open(defaults_path, "w") as f:
        json.dump(regional_defaults, f, indent=2)

    # ── Training Summary ───────────────────────────────────────────────────
    print_section("Training Summary")
    for s in summary:
        status = "✅ PASS" if s["best_r2"] >= 0.75 else "⚠  BELOW TARGET"
        print(f"  {s['crop']:8s} | Best: {s['best_model']:20s} | R² = {s['best_r2']:.4f}  {status}")

    print(f"\n  Encoders saved   → {enc_path}")
    print(f"  Features saved   → {reg_path}")
    print(f"  Defaults saved   → {defaults_path}")
    print("\n  ✅ Training complete. Run predict.py to test inference.\n")


if __name__ == "__main__":
    main()
