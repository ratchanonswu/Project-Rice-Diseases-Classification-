"""
evaluate.py - Test Set Evaluation, Metrics, Confusion Matrix, and Efficiency Profiling for MobileNetV2

This script:
1. Loads models/mobilenetv2.keras and models/class_names.json.
2. Evaluates the model exclusively on the unseen test set (outputs/test.csv).
3. Computes Accuracy, Macro Precision, Macro Recall, Macro F1-Score, and Macro ROC-AUC (OvR).
4. Generates outputs/classification_report.csv and outputs/metrics.json.
5. Computes and plots outputs/mobilenetv2_confusion_matrix.png and outputs/mobilenetv2_confusion_matrix.csv.
6. Generates outputs/test_predictions.csv and saves visual examples into
   outputs/correct_predictions/ and outputs/incorrect_predictions/.
7. Profiles model efficiency (parameters, model size, inference latency)
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
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def load_and_preprocess_test_data(test_csv: Path, img_size: tuple = (224, 224)):
    """
    Loads all images in test.csv and applies deterministic MobileNetV2 preprocessing.
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
                # MobileNetV2 preprocessing expects [0, 255] float array and scales to [-1, 1]
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
    for i in range(n_samples):
        t0 = time.perf_counter()
        _ = model(test_images[i:i+1], training=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

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
    """Saves visual example cards of correct and incorrect predictions."""
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
                    title = f"True: {true_label}\nPred: {pred_label} ({conf:.1f}%)"
                    ax.set_title(title, fontsize=11, fontweight="bold", color=color, pad=10)
                    fname = f"{category_name}_{rank+1:02d}_{Path(img_path).stem}.png"
                    plt.tight_layout()
                    plt.savefig(target_folder / fname, dpi=150, bbox_inches="tight")
                    plt.close()
            except Exception as e:
                print(f"[WARNING] Could not save card for {img_path}: {e}")

    export_samples(correct_indices, correct_dir, "correct")
    export_samples(incorrect_indices, incorrect_dir, "incorrect")
    print(f"[INFO] Saved {min(num_examples, len(correct_indices))} correct and {min(num_examples, len(incorrect_indices))} incorrect example cards.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2 on rice leaf test set.")
    parser.add_argument("--num_examples", type=int, default=8, help="Number of visual examples to export (default: 8)")
    parser.add_argument("--profile_samples", type=int, default=50, help="Number of samples for latency profiling (default: 50)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "mobilenetv2.keras"
    class_names_path = models_dir / "class_names.json"
    test_csv = output_dir / "test.csv"

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"[INFO] Canonical classes ({num_classes}): {class_names}")

    print("\n" + "=" * 65)
    print(f"LOADING MOBILENETV2 MODEL: {model_path}")
    print("=" * 65)
    model = load_model(str(model_path))
    print("[INFO] Model loaded successfully.")

    # Load test data
    print(f"\n[INFO] Loading unseen test data from: {test_csv}")
    X_test, y_true, test_paths, test_class_names = load_and_preprocess_test_data(test_csv)
    n_test = len(y_true)
    print(f"[INFO] Evaluated test samples count: {n_test}")

    # Run predictions
    print("[INFO] Computing predictions on test set...")
    y_probs = model.predict(X_test, batch_size=32, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)

    # 1. Metrics calculation
    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    p_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    r_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    try:
        y_true_oh = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
        roc_macro = float(roc_auc_score(y_true_oh, y_probs, multi_class="ovr", average="macro"))
        roc_weighted = float(roc_auc_score(y_true_oh, y_probs, multi_class="ovr", average="weighted"))
    except Exception as e:
        print(f"[WARNING] ROC-AUC calculation issue: {e}")
        roc_macro = None
        roc_weighted = None

    metrics_dict = {
        "model": "MobileNetV2",
        "test_sample_count": n_test,
        "accuracy": round(acc, 4),
        "precision_macro": round(p_macro, 4),
        "precision_weighted": round(p_weighted, 4),
        "recall_macro": round(r_macro, 4),
        "recall_weighted": round(r_weighted, 4),
        "f1_score_macro": round(f1_m, 4),
        "f1_score_weighted": round(f1_w, 4),
        "roc_auc_macro": round(roc_macro, 4) if roc_macro is not None else None,
        "roc_auc_weighted": round(roc_weighted, 4) if roc_weighted is not None else None
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\n[INFO] Saved test metrics to: {metrics_path}")
    print(json.dumps(metrics_dict, indent=2))

    # 2. Classification report
    clf_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    clf_df = pd.DataFrame(clf_dict).transpose()
    clf_csv_path = output_dir / "classification_report.csv"
    clf_df.to_csv(clf_csv_path)
    print(f"[INFO] Saved classification report to: {clf_csv_path}")

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = output_dir / "mobilenetv2_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)
    print(f"[INFO] Saved confusion matrix CSV to: {cm_csv_path}")

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
    plt.title("MobileNetV2 Confusion Matrix (Test Set)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    cm_plot_path = output_dir / "mobilenetv2_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrix plot to: {cm_plot_path}")

    # 4. Detailed Predictions CSV
    pred_records = []
    for i in range(n_test):
        rec = {
            "image_path": test_paths[i],
            "true_label": class_names[y_true[i]],
            "predicted_label": class_names[y_pred[i]],
            "is_correct": bool(y_true[i] == y_pred[i]),
            "confidence": float(confidences[i])
        }
        for c_idx, c_name in enumerate(class_names):
            rec[f"prob_{c_name.replace(' ', '_')}"] = float(y_probs[i, c_idx])
        pred_records.append(rec)

    pred_df = pd.DataFrame(pred_records)
    pred_csv_path = output_dir / "test_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"[INFO] Saved test predictions CSV to: {pred_csv_path}")

    # 5. Visual cards
    save_prediction_visuals(
        test_paths,
        y_true,
        y_pred,
        confidences,
        class_names,
        output_dir,
        num_examples=args.num_examples
    )

    # 6. Efficiency profiling
    print("\n" + "=" * 65)
    print("PROFILING COMPUTATIONAL EFFICIENCY")
    print("=" * 65)
    latency_stats = measure_inference_efficiency(model, X_test, num_profile_samples=args.profile_samples)

    total_params = int(model.count_params())
    trainable_params = int(sum([tf.size(w).numpy() for w in model.trainable_weights]))
    non_trainable_params = total_params - trainable_params
    file_size_bytes = model_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    training_time = 0.0
    tt_file = output_dir / ".training_time.txt"
    if tt_file.exists():
        with open(tt_file, "r") as f:
            try:
                training_time = float(f.read().strip())
            except:
                pass

    efficiency_data = {
        "model_architecture": "MobileNetV2",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "model_file_size_bytes": file_size_bytes,
        "model_file_size_mb": round(file_size_mb, 2),
        "training_time_seconds": round(training_time, 2),
        "training_time_minutes": round(training_time / 60.0, 2),
        "hardware_profile": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "device": "CPU"
        },
        "inference_latency": latency_stats
    }
    eff_path = output_dir / "efficiency.json"
    with open(eff_path, "w") as f:
        json.dump(efficiency_data, f, indent=2)
    print(f"[INFO] Saved efficiency profile to: {eff_path}")
    print(json.dumps(efficiency_data, indent=2))

    # 7. Experiment metadata
    training_config = {}
    tc_path = output_dir / "training_config.json"
    if tc_path.exists():
        with open(tc_path, "r") as f:
            training_config = json.load(f)

    experiment_metadata = {
        "experiment_title": "Rice Leaf Disease Classification using MobileNetV2 Transfer Learning",
        "model": "MobileNetV2",
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "operating_system": platform.platform(),
        "random_seed": training_config.get("random_seed", 42),
        "dataset_split": {
            "train": int(len(pd.read_csv(output_dir / "train.csv"))),
            "validation": int(len(pd.read_csv(output_dir / "validation.csv"))),
            "test": int(len(pd.read_csv(output_dir / "test.csv")))
        },
        "classes": class_names,
        "image_size": [224, 224, 3],
        "batch_size": training_config.get("batch_size", 32),
        "learning_rate": training_config.get("learning_rate", 0.0001),
        "optimizer": training_config.get("optimizer", "Adam"),
        "max_epochs": training_config.get("max_epochs", 25),
        "actual_epochs_trained": training_config.get("actual_epochs_trained", None),
        "best_epoch": training_config.get("best_epoch", None),
        "augmentation_settings": training_config.get("augmentation_settings", {}),
        "model_architecture": {
            "backbone": "MobileNetV2 (ImageNet Pretrained, Frozen)",
            "pooling": "GlobalAveragePooling2D",
            "dense_head": training_config.get("dense_units", 256),
            "dropout": training_config.get("dropout_rate", 0.3),
            "output_units": num_classes,
            "output_activation": "softmax"
        },
        "test_results": metrics_dict,
        "efficiency": efficiency_data
    }
    meta_path = output_dir / "experiment_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(experiment_metadata, f, indent=2)
    print(f"[INFO] Saved experiment metadata to: {meta_path}")

    print("\n[SUCCESS] MobileNetV2 Evaluation and Profiling Complete!")


if __name__ == "__main__":
    main()

