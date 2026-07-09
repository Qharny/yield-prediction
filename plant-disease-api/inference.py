import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.vgg19 import preprocess_input

from ml_model import model, embedding_model, ood_enabled, ood_centroids, ood_threshold
from class_names import class_names


def predict_image(image):

    image = tf.image.resize(image, (224, 224))

    image = np.array(image, dtype=np.float32).copy()

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    if ood_enabled:
        embedding = embedding_model.predict(image, verbose=0)[0]
        embedding = embedding / max(np.linalg.norm(embedding), 1e-8)
        similarities = ood_centroids @ embedding
        max_similarity = float(np.max(similarities))

        if max_similarity < ood_threshold:
            return {
                "prediction": "NOT_A_PLANT_LEAF",
                "status": "rejected",
                "confidence": max_similarity,
                "message": "This image doesn't look like a plant leaf. Please upload a clear photo of a single leaf.",
                "top_predictions": []
            }

    preds = model.predict(image, verbose=0)[0]

    top1 = float(np.max(preds))

    top2 = float(np.partition(preds, -2)[-2])

    confidence_gap = top1 - top2

    class_id = int(np.argmax(preds))

    predicted_label = class_names[class_id]

    # Top 3 predictions (useful for debugging)
    top3_idx = np.argsort(preds)[-3:][::-1]

    top3_predictions = [
        {
            "class": class_names[i],
            "confidence": float(preds[i])
        }
        for i in top3_idx
    ]

    # Rejection rule
    if top1 < 0.50:
        return {
            "prediction": "UNKNOWN",
            "status": "rejected",
            "confidence": top1,
            "message": "Low confidence prediction.",
            "top_predictions": top3_predictions
        }

    if top1 < 0.55 and confidence_gap < 0.10:
        return {
            "prediction": "UNKNOWN",
            "status": "uncertain",
            "confidence": top1,
            "message": "Model is uncertain between classes.",
            "top_predictions": top3_predictions
        }

    return {
        "prediction": predicted_label,
        "status": "accepted",
        "confidence": top1,
        "confidence_gap": confidence_gap,
        "top_predictions": top3_predictions
    }