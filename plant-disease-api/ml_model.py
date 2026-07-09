import os
import json
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "final_plant_disease_model.keras"
)
OOD_CENTROIDS_PATH = os.path.join(BASE_DIR, "model", "ood_centroids.npy")
OOD_CONFIG_PATH = os.path.join(BASE_DIR, "model", "ood_config.json")

model = tf.keras.models.load_model(MODEL_PATH)

# layers[-1] is the softmax head; layers[-2] is Dropout, which is an identity
# pass-through at inference time, so its output is the 256-d embedding used
# for out-of-distribution (OOD) detection.
embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

ood_enabled = os.path.exists(OOD_CENTROIDS_PATH) and os.path.exists(OOD_CONFIG_PATH)
ood_centroids = None
ood_threshold = None

if ood_enabled:
    ood_centroids = np.load(OOD_CENTROIDS_PATH)
    with open(OOD_CONFIG_PATH, "r", encoding="utf-8") as f:
        ood_config = json.load(f)
    ood_threshold = ood_config["threshold"]
else:
    print(
        "Warning: OOD calibration files not found "
        f"('{OOD_CENTROIDS_PATH}', '{OOD_CONFIG_PATH}'). "
        "Out-of-distribution rejection is disabled; run training/calibrate_ood.py to enable it."
    )