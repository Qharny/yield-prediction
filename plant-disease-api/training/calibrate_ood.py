import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf

from dataset import load_datasets, prepare_dataset

# Add parent directory to sys.path to import class_names
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from class_names import class_names


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate out-of-distribution (OOD) detection for the plant disease model"
    )
    parser.add_argument("--data_dir", type=str, default="./data_capped")
    parser.add_argument("--model_path", type=str, default="../model/final_plant_disease_model.keras")
    parser.add_argument("--centroids_output", type=str, default="../model/ood_centroids.npy")
    parser.add_argument("--config_output", type=str, default="../model/ood_config.json")
    parser.add_argument(
        "--percentile",
        type=float,
        default=1.0,
        help="Percentile of in-distribution (validation) similarity scores used as the rejection threshold"
    )
    return parser.parse_args()


def l2_normalize(x, axis=-1, eps=1e-8):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        sys.exit(1)

    print("Loading datasets...")
    raw_train_ds, raw_val_ds = load_datasets(args.data_dir, batch_size=32)
    train_ds = prepare_dataset(raw_train_ds, augment=False)
    val_ds = prepare_dataset(raw_val_ds, augment=False)

    print(f"Loading model from '{args.model_path}'...")
    model = tf.keras.models.load_model(args.model_path)

    # layers[-1] is the softmax Dense head; layers[-2] is Dropout, which is an
    # identity pass-through at inference time, so its output equals the
    # BatchNormalization-ed 256-d embedding right before classification.
    embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    embed_dim = embedding_model.output_shape[-1]
    num_classes = len(class_names)
    print(f"Embedding dimension: {embed_dim}")

    print("Computing per-class centroids from training data...")
    sums = np.zeros((num_classes, embed_dim), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)

    for images, labels in train_ds:
        embs = embedding_model.predict(images, verbose=0)
        embs = l2_normalize(embs)
        label_idx = np.argmax(labels.numpy(), axis=1)
        for i, li in enumerate(label_idx):
            sums[li] += embs[i]
            counts[li] += 1

    if np.any(counts == 0):
        missing = [class_names[i] for i in range(num_classes) if counts[i] == 0]
        print(f"Error: no training samples found for classes: {missing}")
        sys.exit(1)

    centroids = sums / counts[:, None]
    centroids = l2_normalize(centroids)

    print("Calibrating rejection threshold on validation data...")
    max_sims = []
    for images, labels in val_ds:
        embs = embedding_model.predict(images, verbose=0)
        embs = l2_normalize(embs)
        sims = embs @ centroids.T
        max_sims.extend(np.max(sims, axis=1).tolist())

    max_sims = np.array(max_sims)
    threshold = float(np.percentile(max_sims, args.percentile))

    print("\n=== Similarity stats on validation set (in-distribution) ===")
    print(f"  min:    {max_sims.min():.4f}")
    print(f"  p1:     {np.percentile(max_sims, 1):.4f}")
    print(f"  p5:     {np.percentile(max_sims, 5):.4f}")
    print(f"  median: {np.median(max_sims):.4f}")
    print(f"  mean:   {max_sims.mean():.4f}")
    print(f"  max:    {max_sims.max():.4f}")
    print(f"\nChosen threshold (percentile={args.percentile}): {threshold:.4f}")
    print("Images with max cosine similarity below this threshold will be rejected as 'not a plant leaf'.")

    np.save(args.centroids_output, centroids.astype(np.float32))
    with open(args.config_output, "w", encoding="utf-8") as f:
        json.dump({
            "threshold": threshold,
            "embedding_dim": int(embed_dim),
            "class_names": class_names,
            "percentile_used": args.percentile
        }, f, indent=2)

    print(f"\nSaved centroids to '{args.centroids_output}'.")
    print(f"Saved config to '{args.config_output}'.")


if __name__ == "__main__":
    main()
