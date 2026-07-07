import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from dataset import load_datasets, prepare_dataset

# Add parent directory to sys.path to import class_names
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from class_names import class_names

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Plant Disease Detection Model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to the dataset directory structured by class folders"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../model/final_plant_disease_model.keras",
        help="Path to the trained Keras model file"
    )
    parser.add_argument(
        "--report_output",
        type=str,
        default="evaluation_report.txt",
        help="Path to save the text classification report"
    )
    parser.add_argument(
        "--matrix_output",
        type=str,
        default="confusion_matrix.png",
        help="Path to save the confusion matrix plot"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        sys.exit(1)

    print("Loading datasets...")
    try:
        _, raw_val_ds = load_datasets(
            args.data_dir,
            batch_size=32
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Prepare dataset (preprocess using VGG19 pipeline, no augmentation)
    val_ds = prepare_dataset(raw_val_ds, augment=False)

    print(f"Loading model from '{args.model_path}'...")
    model = tf.keras.models.load_model(args.model_path)

    print("Generating predictions...")
    y_true = []
    y_pred = []

    # Iterate over dataset to compile true labels and predictions
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=list(range(len(class_names))),
        zero_division=0
    )

    print("\n=== Classification Report ===")
    print(report)

    # Save text report
    with open(args.report_output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved classification report to '{args.report_output}'.")

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    
    # Plot Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("True Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    plt.savefig(args.matrix_output, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to '{args.matrix_output}'.")

if __name__ == "__main__":
    main()
