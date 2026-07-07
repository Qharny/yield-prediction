import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization

def build_model(num_classes=15, fine_tune_layers=0):
    """
    Builds a transfer learning model based on VGG19.
    
    Args:
        num_classes (int): Number of target classification classes.
        fine_tune_layers (int): Number of top layers of VGG19 to unfreeze. 
                               If 0, the entire base VGG19 is frozen.
    
    Returns:
        tf.keras.Model: The constructed VGG19-based model.
    """
    # Load VGG19 pre-trained on ImageNet, excluding the classification head
    base_model = VGG19(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Configure base model layers trainability
    if fine_tune_layers > 0:
        base_model.trainable = True
        # Freeze all layers except the last 'fine_tune_layers' layers
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        print(f"Base model set to trainable. Unfrozen the top {fine_tune_layers} layers.")
    else:
        base_model.trainable = False
        print("Base VGG19 model fully frozen for feature extraction.")

    # Create classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    # Instantiate model
    model = Model(inputs=base_model.input, outputs=predictions)

    return model
