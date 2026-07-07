import os
import sys
import argparse
import numpy as np
from PIL import Image

# Add parent directory to sys.path to import class_names
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from class_names import class_names

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Plant Disease Dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to create or check the dataset directory"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="If set, generates a dummy dataset containing small mock images for validation"
    )
    return parser.parse_args()

def generate_mock_data(data_dir, num_images_per_class=10):
    """
    Generates dummy JPEG images for each class to test training/inference pipelines.
    """
    print(f"Generating mock dataset in '{data_dir}'...")
    os.makedirs(data_dir, exist_ok=True)

    for cls in class_names:
        class_path = os.path.join(data_dir, cls)
        os.makedirs(class_path, exist_ok=True)
        print(f"Creating mock images for class: '{cls}'")

        for i in range(num_images_per_class):
            img_name = f"mock_{i}.jpg"
            img_path = os.path.join(class_path, img_name)
            
            # Generate random color image of size 224x224
            color = tuple(np.random.randint(0, 256, size=3))
            img = Image.new("RGB", (224, 224), color=color)
            img.save(img_path, "JPEG")
            
    print(f"\nMock dataset successfully generated at '{data_dir}' with {num_images_per_class} images per class ({len(class_names) * num_images_per_class} total images).")

def main():
    args = parse_args()
    
    if args.mock:
        generate_mock_data(args.data_dir)
    else:
        print("=== Dataset Structure Verification ===")
        print(f"Checking dataset root directory: {args.data_dir}")
        
        # Ensure directory exists
        os.makedirs(args.data_dir, exist_ok=True)
        
        missing = []
        for cls in class_names:
            class_path = os.path.join(args.data_dir, cls)
            if not os.path.exists(class_path):
                missing.append(cls)
                os.makedirs(class_path, exist_ok=True)
                
        if missing:
            print(f"Created {len(missing)} missing class directories:")
            for m in missing:
                print(f"  - {args.data_dir}/{m}")
            print("\nPlease download the PlantVillage dataset subset for Pepper, Tomato, and Corn classes and organize them into the directories listed above.")
        else:
            print("All class directories exist and are structured correctly.")
            
        print("\nNote: You can run `python prepare_dataset.py --mock` to generate a lightweight dummy dataset for testing the pipeline instantly.")

if __name__ == "__main__":
    main()
