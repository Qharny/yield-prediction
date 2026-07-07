import os
import sys
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf

from dataset import load_datasets, prepare_dataset
from model import build_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train Plant Disease Detection Model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to the dataset directory structured by class folders"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of epochs to train the classification head"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=5,
        help="Number of epochs for fine-tuning. Set to 0 to skip fine-tuning."
    )
    parser.add_argument(
        "--fine_tune_layers",
        type=int,
        default=4,
        help="Number of top layers of VGG19 to unfreeze during fine-tuning"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="../model/final_plant_disease_model.keras",
        help="Path where the final model will be saved"
    )
    return parser.parse_args()

def plot_history(history, fine_tune_history=None, output_path="training_history.png"):
    """
    Plots and saves loss and accuracy history for train and validation sets.
    Supports concatenating fine-tuning history if present.
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    if fine_tune_history:
        acc += fine_tune_history.history["accuracy"]
        val_acc += fine_tune_history.history["val_accuracy"]
        loss += fine_tune_history.history["loss"]
        val_loss += fine_tune_history.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#3b82f6", linewidth=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#10b981", linewidth=2)
    if fine_tune_history:
        # Draw a vertical line representing start of fine-tuning
        ft_start = len(history.history["accuracy"]) - 1
        plt.axvline(x=ft_start, color="#ef4444", linestyle="--", label="Fine-tuning Start")
    plt.title("Training and Validation Accuracy", fontsize=14, fontweight="bold", pad=10)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#3b82f6", linewidth=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#10b981", linewidth=2)
    if fine_tune_history:
        ft_start = len(history.history["loss"]) - 1
        plt.axvline(x=ft_start, color="#ef4444", linestyle="--", label="Fine-tuning Start")
    plt.title("Training and Validation Loss", fontsize=14, fontweight="bold", pad=10)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Training history plot saved to '{output_path}'.")

def main():
    args = parse_args()

    # Ensure output model directory exists
    model_dir = os.path.dirname(os.path.abspath(args.model_save_path))
    os.makedirs(model_dir, exist_ok=True)
    print(f"Ensured target model directory exists: {model_dir}")

    # Load datasets
    print("Loading datasets...")
    try:
        raw_train_ds, raw_val_ds = load_datasets(
            args.data_dir,
            batch_size=args.batch_size
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the dataset path or run 'prepare_dataset.py --mock' to generate sample structures.")
        sys.exit(1)

    # Prepare datasets (apply augmentation/VGG19 preprocessing)
    train_ds = prepare_dataset(raw_train_ds, augment=True)
    val_ds = prepare_dataset(raw_val_ds, augment=False)

    num_classes = len(raw_train_ds.class_names)
    print(f"Dataset successfully loaded with {num_classes} classes.")

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.model_save_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    # Phase 1: Train Classification Head (VGG19 Frozen)
    print("\n--- Phase 1: Training Classification Head ---")
    model = build_model(num_classes=num_classes, fine_tune_layers=0)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )

    # Phase 2: Fine-Tuning (Optional)
    fine_tune_history = None
    if args.fine_tune_epochs > 0:
        print("\n--- Phase 2: Fine-Tuning Base Model ---")
        
        # Save head weights temporarily
        temp_weights_path = "temp_weights.weights.h5"
        model.save_weights(temp_weights_path)

        # Build fine-tune model with unfrozen layers
        ft_model = build_model(
            num_classes=num_classes,
            fine_tune_layers=args.fine_tune_layers
        )
        ft_model.load_weights(temp_weights_path)
        
        # Clean up temp file
        if os.path.exists(temp_weights_path):
            os.remove(temp_weights_path)

        # Recompile with a significantly lower learning rate for fine-tuning
        ft_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr * 0.1),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Train again
        fine_tune_history = ft_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=[early_stopping, reduce_lr, checkpoint]
        )
        
        # Update model variable to point to the best fine-tuned model (restored by checkpoint)
        if os.path.exists(args.model_save_path):
            print(f"Loading best saved model from {args.model_save_path} for final steps...")
            model = tf.keras.models.load_model(args.model_save_path)
        else:
            model = ft_model

    # Save history plot
    plot_history(history, fine_tune_history)
    print(f"\nTraining completed. Final model is saved at '{args.model_save_path}'")

if __name__ == "__main__":
    main()
