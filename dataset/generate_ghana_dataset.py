"""
generate_ghana_dataset.py
─────────────────────────────────────────────────────────────────────────────
Synthetic Dataset Generator for Ghanaian Crop Yield Prediction
Project: Smart Agriculture IoT System (ESP32-based)

Methodology:
    Generates a statistically grounded synthetic dataset of ~3,000 records
    calibrated against published Ghanaian agro-ecological statistics and the
    Asamoah et al. (2024) finding that soil + fertiliser variables explain
    up to 81% of variance in Ghanaian crop yields (R² = 0.81).

    Regional parameters (rainfall, temperature baselines) are drawn from:
    - Ghana Meteorological Agency (GMet) regional normals
    - FAO GAEZ v4 West Africa layer statistics
    - MoFA (Ministry of Food & Agriculture Ghana) district yield tables

Output:
    dataset/ghana_yield_data.csv  — 3000 rows × 15 columns
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ─────────────────────────────────────────────────────────
np.random.seed(42)

# ── Regional Agro-Ecological Baselines (from GMet / FAO-GAEZ) ───────────────
REGIONAL_PROFILES = {
    "Ashanti": {
        "rainfall_mean": 1400, "rainfall_std": 180,
        "temp_mean": 26.5,
        "soil_ph_mean": 6.0, "soil_ph_std": 0.5,
        "yield_base_maize": 17.0,   # bags/acre
        "yield_base_tomato": 3800,  # kg/acre
    },
    "Western": {
        "rainfall_mean": 1800, "rainfall_std": 220,
        "temp_mean": 25.8,
        "soil_ph_mean": 5.8, "soil_ph_std": 0.6,
        "yield_base_maize": 18.5,
        "yield_base_tomato": 4200,
    },
    "Brong-Ahafo": {
        "rainfall_mean": 1200, "rainfall_std": 160,
        "temp_mean": 27.2,
        "soil_ph_mean": 6.2, "soil_ph_std": 0.55,
        "yield_base_maize": 16.0,
        "yield_base_tomato": 3500,
    },
    "Northern": {
        "rainfall_mean": 900, "rainfall_std": 140,
        "temp_mean": 29.5,
        "soil_ph_mean": 6.5, "soil_ph_std": 0.6,
        "yield_base_maize": 11.0,
        "yield_base_tomato": 2800,
    },
    "Volta": {
        "rainfall_mean": 1100, "rainfall_std": 155,
        "temp_mean": 27.8,
        "soil_ph_mean": 6.1, "soil_ph_std": 0.5,
        "yield_base_maize": 14.5,
        "yield_base_tomato": 3200,
    },
    "Eastern": {
        "rainfall_mean": 1300, "rainfall_std": 170,
        "temp_mean": 26.9,
        "soil_ph_mean": 5.9, "soil_ph_std": 0.5,
        "yield_base_maize": 15.5,
        "yield_base_tomato": 3600,
    },
}

CROPS = ["Maize", "Tomato"]
CROP_VARIETIES = {
    "Maize":  {"Local": 0.75, "Hybrid OPV": 1.0, "Pioneer 30G19": 1.30},
    "Tomato": {"Local":  0.70, "Pectomech": 1.0,  "F1 Hybrid":     1.35},
}

# Optimal soil pH ranges per crop
PH_OPTIMAL = {
    "Maize":  (5.8, 7.0),
    "Tomato": (6.0, 7.0),
}

# ── Yield Simulation Functions ───────────────────────────────────────────────

def ph_penalty(ph: float, crop: str) -> float:
    """
    Returns a multiplier [0.35 – 1.0] based on how far pH deviates
    from the crop-optimal range. Severe acidity (<4.5) or alkalinity (>8.0)
    collapses yield — consistent with CSIR-SARI Ghanaian field studies.
    """
    lo, hi = PH_OPTIMAL[crop]
    if lo <= ph <= hi:
        return 1.0
    elif ph < lo:
        deficit = lo - ph
        return max(0.35, 1.0 - (deficit ** 1.6) * 0.35)
    else:
        deficit = ph - hi
        return max(0.55, 1.0 - deficit * 0.18)


def npk_score(N: float, P: float, K: float, crop: str) -> float:
    """
    Combines NPK into a normalised fertility index [0 – 1.05].
    Uses Mitscherlich-style diminishing returns so that excess application
    does not linearly boost yield. Weights reflect Asamoah et al. importance.
    """
    if crop == "Maize":
        n_opt, p_opt, k_opt = 120.0, 45.0, 150.0
        w = (0.45, 0.30, 0.25)
    else:  # Tomato
        n_opt, p_opt, k_opt = 100.0, 55.0, 180.0
        w = (0.35, 0.35, 0.30)

    def mitscherlich(val, opt):
        ratio = val / opt
        return min(1.05, 1.0 - np.exp(-2.5 * ratio))

    return (w[0] * mitscherlich(N, n_opt)
            + w[1] * mitscherlich(P, p_opt)
            + w[2] * mitscherlich(K, k_opt))


def moisture_factor(moisture_pct: float, crop: str) -> float:
    """Returns [0.40 – 1.05] based on soil moisture %. Optimal 45–70%."""
    if crop == "Maize":
        lo, hi = 40, 70
    else:
        lo, hi = 45, 75
    if lo <= moisture_pct <= hi:
        return 1.0 + 0.05 * ((moisture_pct - lo) / (hi - lo) - 0.5)
    elif moisture_pct < lo:
        return max(0.40, 0.40 + 0.60 * (moisture_pct / lo) ** 1.4)
    else:
        return max(0.55, 1.0 - (moisture_pct - hi) / hi * 0.8)


def rainfall_factor(rainfall_mm: float, crop: str) -> float:
    """Rainfall contribution [0.5 – 1.1]."""
    if crop == "Maize":
        opt_lo, opt_hi = 900, 1500
    else:
        opt_lo, opt_hi = 800, 1800
    if opt_lo <= rainfall_mm <= opt_hi:
        return 1.0
    elif rainfall_mm < opt_lo:
        return max(0.50, rainfall_mm / opt_lo)
    else:
        return max(0.70, 1.0 - (rainfall_mm - opt_hi) / opt_hi * 0.35)


# ── Main Dataset Generator ───────────────────────────────────────────────────

def generate_dataset(n_records: int = 3000) -> pd.DataFrame:
    records = []

    n_per_region_crop = n_records // (len(REGIONAL_PROFILES) * len(CROPS))

    for region, profile in REGIONAL_PROFILES.items():
        for crop in CROPS:
            for _ in range(n_per_region_crop):
                # ── Soil & Environmental Sensor Features ─────────────────
                soil_ph = np.clip(
                    np.random.normal(profile["soil_ph_mean"], profile["soil_ph_std"]),
                    3.5, 8.5
                )
                soil_moisture = np.clip(np.random.normal(52, 15), 10, 95)
                N = np.clip(np.random.normal(90, 40), 5, 250)
                P = np.clip(np.random.normal(35, 20), 2, 100)
                K = np.clip(np.random.normal(130, 55), 20, 350)
                rainfall_mm = np.clip(
                    np.random.normal(profile["rainfall_mean"], profile["rainfall_std"]),
                    400, 2500
                )
                temperature_c = np.clip(
                    np.random.normal(profile["temp_mean"], 1.5), 18, 38
                )

                # ── Farm Management Features ──────────────────────────────
                farm_size_acres = np.clip(np.random.exponential(2.5), 0.5, 15)
                variety = np.random.choice(
                    list(CROP_VARIETIES[crop].keys()),
                    p=[1/len(CROP_VARIETIES[crop])] * len(CROP_VARIETIES[crop])
                )
                variety_multiplier = CROP_VARIETIES[crop][variety]

                # Season: Major (Apr–Jul) or Minor (Sep–Nov) — affects yield ~10%
                season = np.random.choice(["Major", "Minor"], p=[0.60, 0.40])
                season_factor = 1.0 if season == "Major" else 0.88

                # ── Yield Calculation (Mechanistic + Noise) ───────────────
                base = (profile["yield_base_maize"]
                        if crop == "Maize"
                        else profile["yield_base_tomato"])

                composite = (
                    ph_penalty(soil_ph, crop)
                    * npk_score(N, P, K, crop)
                    * moisture_factor(soil_moisture, crop)
                    * rainfall_factor(rainfall_mm, crop)
                    * variety_multiplier
                    * season_factor
                )

                # Add realistic noise (σ = 8% of base yield)
                noise = np.random.normal(1.0, 0.08)
                raw_yield = base * composite * noise

                if crop == "Maize":
                    yield_val = np.clip(round(raw_yield, 2), 1.0, 50.0)
                    yield_unit = "bags_per_acre"
                else:
                    yield_val = np.clip(round(raw_yield, 1), 100.0, 12000.0)
                    yield_unit = "kg_per_acre"

                # ── Yield Category Label ──────────────────────────────────
                if crop == "Maize":
                    cat = "High" if yield_val >= 22 else ("Medium" if yield_val >= 12 else "Low")
                else:
                    cat = "High" if yield_val >= 5000 else ("Medium" if yield_val >= 2500 else "Low")

                records.append({
                    # Identifiers
                    "region":            region,
                    "crop":              crop,
                    "crop_variety":      variety,
                    "season":            season,
                    # Sensor Features (ESP32)
                    "soil_ph":           round(soil_ph, 2),
                    "soil_moisture_pct": round(soil_moisture, 1),
                    "nitrogen_N_mg_kg":  round(N, 1),
                    "phosphorus_P_mg_kg": round(P, 1),
                    "potassium_K_mg_kg": round(K, 1),
                    # Macro / Regional Features
                    "rainfall_mm":       round(rainfall_mm, 1),
                    "temperature_c":     round(temperature_c, 1),
                    # Farm Management (User Input)
                    "farm_size_acres":   round(farm_size_acres, 2),
                    # Targets
                    "yield_value":       yield_val,
                    "yield_unit":        yield_unit,
                    "yield_category":    cat,
                })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ghana_yield_data.csv")

    print("=" * 60)
    print("  Ghana Crop Yield Dataset Generator")
    print("=" * 60)
    df = generate_dataset(n_records=3000)
    df.to_csv(out_path, index=False)

    print(f"\n✅ Dataset saved → {out_path}")
    print(f"   Shape        : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n── Column Summary ──────────────────────────────────────────")
    print(df.dtypes.to_string())
    print(f"\n── Yield Statistics by Crop ────────────────────────────────")
    print(df.groupby("crop")["yield_value"].describe().round(2).to_string())
    print(f"\n── Yield Category Distribution ─────────────────────────────")
    print(df.groupby(["crop", "yield_category"]).size().unstack(fill_value=0).to_string())
    print(f"\n── Region × Crop Mean Yield ─────────────────────────────────")
    print(df.groupby(["region", "crop"])["yield_value"].mean().round(2).unstack().to_string())
