"""
app.py — Ghana Crop Yield Prediction Web App
Flask backend that serves the ML prediction engine via REST API + web UI.

Run:
    python app.py
Then open: http://localhost:5000
"""

import os
import sys

# Ensure model directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
from model.predict import predict_yield

app = Flask(__name__)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for yield prediction.

    Accepts JSON body:
    {
        "crop": "Maize",
        "region": "Ashanti",
        "farm_size_acres": 2.5,
        "crop_variety": "Hybrid OPV",
        "season": "Major",
        "soil_ph": 6.2,            // optional — cold-start if omitted
        "soil_moisture_pct": 55,   // optional
        "nitrogen_N_mg_kg": 110,   // optional
        "phosphorus_P_mg_kg": 42,  // optional
        "potassium_K_mg_kg": 155,  // optional
        "rainfall_mm": 1380,       // optional
        "temperature_c": 26.3      // optional
    }

    Returns JSON prediction result.
    """
    try:
        data = request.get_json(force=True)

        # Required fields
        crop           = data.get("crop")
        region         = data.get("region")
        farm_size      = float(data.get("farm_size_acres", 1.0))
        crop_variety   = data.get("crop_variety")
        season         = data.get("season", "Major")

        # Optional sensor fields (cold-start if missing)
        def _opt(key):
            val = data.get(key)
            return float(val) if val not in (None, "", "null") else None

        result = predict_yield(
            crop=crop,
            region=region,
            farm_size_acres=farm_size,
            crop_variety=crop_variety,
            season=season,
            soil_ph=_opt("soil_ph"),
            soil_moisture_pct=_opt("soil_moisture_pct"),
            nitrogen_N_mg_kg=_opt("nitrogen_N_mg_kg"),
            phosphorus_P_mg_kg=_opt("phosphorus_P_mg_kg"),
            potassium_K_mg_kg=_opt("potassium_K_mg_kg"),
            rainfall_mm=_opt("rainfall_mm"),
            temperature_c=_opt("temperature_c"),
        )

        return jsonify({"success": True, "data": result})

    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": "Ghana Yield Predictor v1.0"})


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Ghana Crop Yield Prediction App")
    print("  Open: http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
