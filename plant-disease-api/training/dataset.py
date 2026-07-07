import os
import sys
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input

# Add parent directory to sys.path to import class_names
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from class_names import class_names

# Define data augmentation layers
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

def load_datasets(data_dir, batch_size=32, val_split=0.2, seed=42):
    """
    Loads training and validation datasets from a directory structured by classes.
    Enforces the class ordering from class_names.py.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found.")

    # Check if all classes exist or create placeholders/warn user
    missing_classes = []
    for cls in class_names:
        class_path = os.path.join(data_dir, cls)
        if not os.path.exists(class_path):
            missing_classes.append(cls)
    
    if missing_classes:
        print(f"Warning: The following directories are missing from '{data_dir}':")
        for mc in missing_classes:
            print(f"  - {mc}")
        print("Creating placeholder folders for missing classes...")
        for mc in missing_classes:
            os.makedirs(os.path.join(data_dir, mc), exist_ok=True)

    # Load raw training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode="categorical",
        class_names=class_names
    )

    # Load raw validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode="categorical",
        class_names=class_names
    )

    return train_ds, val_ds

def prepare_dataset(ds, augment=False):
    """
    Applies VGG19 preprocessing and optional data augmentation to the dataset.
    Optimizes loading using prefetching and caching.
    """
    # Use buffered prefetching to load images from disk without I/O blocking
    AUTOTUNE = tf.data.AUTOTUNE

    if augment:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )

    # Apply VGG19 preprocess_input
    ds = ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=AUTOTUNE
    )

    return ds.prefetch(buffer_size=AUTOTUNE)
