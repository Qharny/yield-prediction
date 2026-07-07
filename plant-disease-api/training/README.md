# Plant Disease Detection Model Training

This directory contains the training pipeline for the TensorFlow/Keras-based image classification model used by the API.

## Requirements

Install the dependencies needed for training:
```bash
pip install -r requirements.txt
```

## Dataset Structure

The training scripts expect the dataset to be organized in a directory containing subdirectories for each of the 15 classes. The class names must match the categories defined in `class_names.py`:

```
dataset/
├── Bacterial spot (Pepper, bell)/
│   ├── image1.jpg
│   └── image2.jpg
├── Bacterial spot (Tomato)/
│   ├── image1.jpg
│   └── image2.jpg
...
└── healthy (Tomato)/
    ├── image1.jpg
    └── image2.jpg
```

### Supported Classes (15):
1. `Bacterial spot (Pepper, bell)`
2. `Bacterial spot (Tomato)`
3. `Cercospora leaf spot Gray leaf spot (Corn (maize))`
4. `Common rust (Corn (maize))`
5. `Early blight (Tomato)`
6. `Late blight (Tomato)`
7. `Leaf Mold (Tomato)`
8. `Northern Leaf Blight (Corn (maize))`
9. `Septoria leaf spot (Tomato)`
10. `Target Spot (Tomato)`
11. `Tomato Yellow Leaf Curl Virus (Tomato)`
12. `Tomato mosaic virus (Tomato)`
13. `healthy (Corn (maize))`
14. `healthy (Pepper, bell)`
15. `healthy (Tomato)`

---

## Getting Started

### 1. Set Up Mock Dataset (Optional/Testing)
For testing the training script without the full dataset, you can generate a mock directory structure with small dummy images:
```bash
python prepare_dataset.py --mock
```

### 2. Run Training
Start model training using:
```bash
python train.py --data_dir ./data --epochs 15 --batch_size 32 --lr 0.0001
```

**Parameters:**
- `--data_dir`: Path to the root directory of the dataset (defaults to `./data`).
- `--epochs`: Number of epochs to train (defaults to `15`).
- `--batch_size`: Batch size (defaults to `32`).
- `--lr`: Initial learning rate (defaults to `0.0001`).
- `--fine_tune_epochs`: Number of fine-tuning epochs after unfreezing VGG19 top layers (defaults to `5`, set to `0` to skip fine-tuning).

When training finishes, the best model will be saved automatically to `../model/final_plant_disease_model.keras`.

### 3. Evaluate the Model
Evaluate the model against validation/test images to generate a confusion matrix and classification report:
```bash
python evaluate.py --data_dir ./data --model_path ../model/final_plant_disease_model.keras
```
This generates:
- `evaluation_report.txt` (Accuracy, Precision, Recall, F1-score per class)
- `confusion_matrix.png` (Visual matrix of correct/incorrect classifications)
