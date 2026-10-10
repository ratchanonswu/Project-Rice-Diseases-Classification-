"""
evaluate.py - Test Set Evaluation, Metrics, Confusion Matrix, and Efficiency Profiling

This script:
1. Loads models/resnet50.keras and models/class_names.json.
2. Evaluates the model exclusively on the unseen test set (outputs/test.csv).
3. Computes Accuracy, Precision, Recall, F1-Score, and ROC-AUC (OvR).
4. Generates outputs/classification_report.csv and outputs/metrics.json.
5. Computes and plots outputs/resnet50_confusion_matrix.png and outputs/resnet50_confusion_matrix.csv.
6. Generates outputs/test_predictions.csv and saves visual examples into
   outputs/correct_predictions/ and outputs/incorrect_predictions/.
7. Accurately profiles model efficiency (parameters, model size, inference latency)
   and saves outputs/efficiency.json.
8. Gathers complete experiment metadata and logs outputs/experiment_metadata.json.
"""

import os
import sys
import time
import json
import argparse
import platform
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


def load_and_preprocess_test_data(test_csv: Path, img_size: tuple = (224, 224)):
    """
    Loads all images in test.csv and applies deterministic ResNet50 preprocessing.
    Never applies data augmentation to test data.
    """
    df = pd.read_csv(test_csv)
    image_paths = df["image_path"].tolist()
    labels = df["numeric_class_index"].values
    class_names = df["class_name"].tolist()

    images = []
    valid_indices = []

    for i, path_str in enumerate(image_paths):
        p = Path(path_str)
        try:
            with Image.open(p) as pil_img:
                img_rgb = pil_img.convert("RGB").resize(img_size)
                arr = np.array(img_rgb, dtype=np.float32)
                # ResNet50 preprocessing expects [0, 255] float array
                arr_prep = preprocess_input(arr)
                images.append(arr_prep)
                valid_indices.append(i)
        except Exception as e:
            print(f"[WARNING] Skipping unreadable test image: {path_str} ({e})")

    images_arr = np.array(images, dtype=np.float32)
    labels_arr = labels[valid_indices]
    paths_arr = [image_paths[i] for i in valid_indices]
    class_names_arr = [class_names[i] for i in valid_indices]

    return images_arr, labels_arr, paths_arr, class_names_arr


def measure_inference_efficiency(model: tf.keras.Model, test_images: np.ndarray, num_profile_samples: int = 50, warmup: int = 10):
    """
    Accurately measures inference latency per image and overall throughput.
    Uses num_profile_samples to ensure reliable latency estimation without CPU timeouts.
    """
    n_samples = min(num_profile_samples, len(test_images))
    print(f"\n[INFO] Profiling inference latency on {n_samples} test images (warmup={warmup})...")

    # Warmup
    for i in range(min(warmup, len(test_images))):
        _ = model(test_images[i:i+1], training=False)

    latencies = []
    # Measure per-sample single image inference
    for i in range(n_samples):
        t0 = time.perf_counter()
        _ = model(test_images[i:i+1], training=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # milliseconds

    avg_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    median_ms = float(np.median(latencies))
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    return {
        "profiled_samples": n_samples,
        "average_inference_time_ms": round(avg_ms, 3),
        "median_inference_time_ms": round(median_ms, 3),
        "std_inference_time_ms": round(std_ms, 3),
        "throughput_fps": round(fps, 2)
    }


def save_prediction_visuals(
    test_paths: list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    class_names: list,
    output_dir: Path,
    num_examples: int = 8
):
    """
    Saves visual example cards of correct and incorrect predictions.
    """
    correct_dir = output_dir / "correct_predictions"
    incorrect_dir = output_dir / "incorrect_predictions"
    correct_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir.mkdir(parents=True, exist_ok=True)

    correct_indices = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]
    incorrect_indices = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]

    print(f"[INFO] Saving up to {num_examples} visual prediction samples each...")

    def export_samples(indices, target_folder, category_name):
        selected = indices[:num_examples]
        for rank, idx in enumerate(selected):
            img_path = test_paths[idx]
            true_label = class_names[y_true[idx]]
            pred_label = class_names[y_pred[idx]]
            conf = confidences[idx] * 100.0

            try:
                with Image.open(img_path) as pil_img:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(pil_img)
                    ax.axis("off")
                    color = "#2ca02c" if category_name == "correct" else "#d62728"
                    title = f"True: {true_label}\nPred: {pred_label} ({conf:.2f}%)"
                    ax.set_title(title, fontsize=12, fontweight="bold", color=color, pad=10)
                    out_name = f"{category_name}_{rank+1:02d}_{Path(img_path).stem}.png"
                    fig.savefig(target_folder / out_name, bbox_inches="tight", dpi=150)
                    plt.close(fig)
            except Exception as e:
                print(f"[WARNING] Could not save visualization for {img_path}: {e}")

    export_samples(correct_indices, correct_dir, "correct")
    export_samples(incorrect_indices, incorrect_dir, "incorrect")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ResNet50 model on test set")
    parser.add_argument("--model_path", type=str, default="models/resnet50.keras", help="Path to trained model")
    parser.add_argument("--class_names", type=str, default="models/class_names.json", help="Path to class names JSON")
    parser.add_argument("--test_csv", type=str, default="outputs/test.csv", help="Path to test set CSV")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for output reports")
    parser.add_argument("--num_examples", type=int, default=8, help="Number of visual examples to save")
    parser.add_argument("--profile_samples", type=int, default=50, help="Number of samples for latency profiling (default: 50)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    model_path = (project_root / args.model_path).resolve()
    class_names_path = (project_root / args.class_names).resolve()
    test_csv = (project_root / args.test_csv).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Train the model first!")
    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names file not found: {class_names_path}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test split CSV not found: {test_csv}")

    # Load class names
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    print("=" * 60)
    print("EVALUATING RESNET50 MODEL ON UNBIASED TEST SET")
    print(f"Model: {model_path}")
    print(f"Classes ({num_classes}): {class_names}")
    print("=" * 60)

    # Load trained model
    model = load_model(str(model_path))
    print("[INFO] Model loaded successfully.")

    # Load test data
    test_images, y_true, test_paths, test_class_names = load_and_preprocess_test_data(test_csv)
    print(f"[INFO] Loaded {len(test_images)} test samples.")

    # Predict probabilities on test set
    y_probs = model.predict(test_images, batch_size=32, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)

    # 1. Metrics Computation
    accuracy = float(accuracy_score(y_true, y_pred))
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    precision_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    recall_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # ROC AUC (One-vs-Rest)
    try:
        # Convert y_true to one-hot for multi-class ROC AUC
        y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
        roc_auc_macro = float(roc_auc_score(y_true_one_hot, y_probs, multi_class="ovr", average="macro"))
        roc_auc_weighted = float(roc_auc_score(y_true_one_hot, y_probs, multi_class="ovr", average="weighted"))
    except Exception as e:
        print(f"[WARNING] Could not compute ROC AUC: {e}")
        roc_auc_macro = None
        roc_auc_weighted = None

    metrics_data = {
        "model": "ResNet50",
        "test_sample_count": len(y_true),
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision_macro, 4),
        "precision_weighted": round(precision_weighted, 4),
        "recall_macro": round(recall_macro, 4),
        "recall_weighted": round(recall_weighted, 4),
        "f1_score_macro": round(f1_macro, 4),
        "f1_score_weighted": round(f1_weighted, 4),
        "roc_auc_macro": round(roc_auc_macro, 4) if roc_auc_macro is not None else "N/A",
        "roc_auc_weighted": round(roc_auc_weighted, 4) if roc_auc_weighted is not None else "N/A"
    }

    metrics_json_path = output_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\n[INFO] Overall metrics saved to: {metrics_json_path}")
    print(json.dumps(metrics_data, indent=2))

    # 2. Classification Report
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    df_report = pd.DataFrame(report_dict).transpose()
    report_csv_path = output_dir / "classification_report.csv"
    df_report.to_csv(report_csv_path, index=True)
    print(f"[INFO] Classification report saved to: {report_csv_path}")

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = output_dir / "resnet50_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)
    print(f"[INFO] Raw confusion matrix saved to: {cm_csv_path}")

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title("ResNet50 Confusion Matrix (Test Set)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    cm_png_path = output_dir / "resnet50_confusion_matrix.png"
    plt.savefig(cm_png_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix plot saved to: {cm_png_path}")

    # 4. Save Test Predictions CSV
    pred_class_names = [class_names[idx] for idx in y_pred]
    is_correct = [bool(t == p) for t, p in zip(y_true, y_pred)]
    preds_dict = {
        "image_path": test_paths,
        "true_class": test_class_names,
        "predicted_class": pred_class_names,
        "confidence": [round(float(c), 4) for c in confidences],
        "correct": is_correct
    }
    for c_idx, c_name in enumerate(class_names):
        preds_dict[f"prob_{c_name.replace(' ', '_')}"] = [round(float(p), 4) for p in y_probs[:, c_idx]]
    df_preds = pd.DataFrame(preds_dict)
    test_preds_path = output_dir / "test_predictions.csv"
    df_preds.to_csv(test_preds_path, index=False)
    print(f"[INFO] Test predictions saved to: {test_preds_path}")

    # 5. Save Visual Examples (Correct & Incorrect)
    save_prediction_visuals(
        test_paths=test_paths,
        y_true=y_true,
        y_pred=y_pred,
        confidences=confidences,
        class_names=class_names,
        output_dir=output_dir,
        num_examples=args.num_examples
    )

    # 6. Model Efficiency Profiling
    total_params = int(model.count_params())
    trainable_params = int(sum([tf.size(w).numpy() for w in model.trainable_weights]))
    file_size_bytes = model_path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    # Read training time if logged
    train_time_file = output_dir / ".training_time.txt"
    training_time_sec = None
    if train_time_file.exists():
        with open(train_time_file, "r", encoding="utf-8") as f:
            training_time_sec = float(f.read().strip())

    latency_stats = measure_inference_efficiency(model, test_images, num_profile_samples=args.profile_samples)

    efficiency_data = {
        "model_architecture": "ResNet50",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_file_path": str(model_path.relative_to(project_root)).replace("\\", "/"),
        "model_file_size_bytes": file_size_bytes,
        "model_file_size_mb": file_size_mb,
        "training_time_seconds": round(training_time_sec, 2) if training_time_sec is not None else "N/A",
        "training_time_minutes": round(training_time_sec / 60.0, 2) if training_time_sec is not None else "N/A",
        "inference_latency": latency_stats
    }
    efficiency_path = output_dir / "efficiency.json"
    with open(efficiency_path, "w", encoding="utf-8") as f:
        json.dump(efficiency_data, f, indent=2)
    print(f"[INFO] Efficiency metrics saved to: {efficiency_path}")

    # 7. Experiment Metadata
    # Load training config if exists
    training_config = {}
    config_file = output_dir / "training_config.json"
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            training_config = json.load(f)

    # Load dataset summary for distribution
    dataset_summary_file = output_dir / "dataset_summary.csv"
    dataset_size = len(test_images)
    class_distribution = {}
    if dataset_summary_file.exists():
        df_ds = pd.read_csv(dataset_summary_file)
        class_distribution = dict(zip(df_ds["class_name"], df_ds["image_count"]))
        dataset_size = int(df_ds.loc[df_ds["class_name"] == "TOTAL", "image_count"].values[0])

    experiment_metadata = {
        "experiment_title": "Rice Leaf Disease Classification using ResNet50 Transfer Learning",
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "operating_system": platform.platform(),
        "random_seed": training_config.get("random_seed", 42),
        "dataset_size": dataset_size,
        "dataset_split": {
            "train": int(len(pd.read_csv(output_dir / "train.csv"))),
            "validation": int(len(pd.read_csv(output_dir / "validation.csv"))),
            "test": int(len(pd.read_csv(output_dir / "test.csv")))
        },
        "class_distribution": class_distribution,
        "classes": class_names,
        "image_size": training_config.get("input_shape", [224, 224, 3]),
        "batch_size": training_config.get("batch_size", 32),
        "learning_rate": training_config.get("learning_rate", 0.0001),
        "optimizer": training_config.get("optimizer", "Adam"),
        "max_epochs": training_config.get("max_epochs", 25),
        "augmentation_settings": training_config.get("augmentation_settings", {}),
        "model_architecture": {
            "backbone": "ResNet50 (ImageNet Pretrained, Frozen)",
            "pooling": "GlobalAveragePooling2D",
            "dense_head": training_config.get("dense_units", 256),
            "dropout": training_config.get("dropout_rate", 0.3),
            "output_units": num_classes,
            "output_activation": "softmax"
        },
        "test_results": metrics_data
    }
    metadata_path = output_dir / "experiment_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(experiment_metadata, f, indent=2)
    print(f"[INFO] Experiment metadata saved to: {metadata_path}")

    print("\n[SUCCESS] Evaluation and profiling completed.")


if __name__ == "__main__":
    main()

