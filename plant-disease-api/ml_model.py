import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "final_plant_disease_model.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)