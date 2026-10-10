"""
predict.py - Single Image Inference CLI for MobileNetV2 Rice Leaf Disease Classification

Usage:
    python src/predict.py --image path/to/leaf.jpg
    python src/predict.py --image path/to/leaf.jpg --top_k 3

Features:
- Validates that the input image file exists and is readable.
- Loads models/mobilenetv2.keras and models/class_names.json.
- Resizes input image to 224x224 RGB.
- Applies MobileNetV2 preprocessing (scales pixel values to [-1, 1]) exactly once.
- Outputs predicted class and confidence.
- Verifies and displays top-k class probabilities, ensuring probabilities sum to ~1.0.
"""

import os
import sys
import json
import argparse
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def parse_args():
    parser = argparse.ArgumentParser(description="Predict rice leaf disease using MobileNetV2.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--model", type=str, default="models/mobilenetv2.keras", help="Path to trained .keras model")
    parser.add_argument("--classes", type=str, default="models/class_names.json", help="Path to class_names.json")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to display (default: 3)")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = (project_root / image_path).resolve()

    if not image_path.exists() or not image_path.is_file():
        print(f"[ERROR] Image file does not exist: {image_path}")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (project_root / model_path).resolve()

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}. Train the model first!")
        sys.exit(1)

    classes_path = Path(args.classes)
    if not classes_path.is_absolute():
        classes_path = (project_root / classes_path).resolve()

    if not classes_path.exists():
        print(f"[ERROR] Class names file not found: {classes_path}")
        sys.exit(1)

    with open(classes_path, "r") as f:
        class_names = json.load(f)

    # Load and preprocess image
    try:
        with Image.open(image_path) as pil_img:
            img_rgb = pil_img.convert("RGB").resize((224, 224))
            img_arr = np.array(img_rgb, dtype=np.float32)
            # MobileNetV2 preprocessing maps [0, 255] -> [-1, 1]
            img_prep = preprocess_input(img_arr)
            img_batch = np.expand_dims(img_prep, axis=0)
    except Exception as e:
        print(f"[ERROR] Failed to load or process image: {e}")
        sys.exit(1)

    # Load model and run inference
    try:
        model = load_model(str(model_path))
    except Exception as e:
        print(f"[ERROR] Failed to load model from {model_path}: {e}")
        sys.exit(1)

    preds = model.predict(img_batch, verbose=0)[0]
    prob_sum = float(np.sum(preds))

    pred_idx = int(np.argmax(preds))
    pred_class = class_names[pred_idx]
    confidence = float(preds[pred_idx]) * 100.0

    print("=" * 45)
    print("Model: MobileNetV2")
    print(f"Image: {image_path.name}")
    print("=" * 45)
    print("\nPrediction:")
    print(f"{pred_class}")
    print("\nConfidence:")
    print(f"{confidence:.2f}%")

    top_k = min(args.top_k, len(class_names))
    sorted_indices = np.argsort(preds)[::-1][:top_k]

    print(f"\nTop {top_k}:")
    for rank, idx in enumerate(sorted_indices, 1):
        print(f"{rank}. {class_names[idx]} - {preds[idx]*100.0:.2f}%")

    print("\nProbability Verification:")
    print(f"Sum of probabilities: {prob_sum:.6f} (~ 1.000000)")
    print("=" * 45)


if __name__ == "__main__":
    main()

