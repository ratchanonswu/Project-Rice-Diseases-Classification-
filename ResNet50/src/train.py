"""
train.py - Transfer Learning Pipeline for ResNet50 on Rice Leaf Disease Classification

This script:
1. Configures all hyperparameters and logs them to outputs/training_config.json.
2. Builds the ResNet50 transfer learning architecture with frozen ImageNet backbone.
3. Records model summary and parameter counts to outputs/model_summary.txt.
4. Loads training and validation datasets using tf.keras.applications.resnet50.preprocess_input.
5. Applies data augmentation exclusively to training data.
6. Trains the model with Adam optimizer, EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.
7. Saves the complete trained model to models/resnet50.keras.
8. Generates training vs. validation accuracy and loss curves to outputs/.
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path

# Suppress noisy TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


def set_seed(seed: int = 42):
    """Enforces deterministic behavior where possible for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dataset_from_csv(
    csv_path: Path,
    img_size: tuple = (224, 224),
    batch_size: int = 32,
    is_training: bool = False,
    augmenter=None
):
    """
    Constructs a high-performance tf.data.Dataset from a split CSV.
    Applies ResNet50 preprocessing (RGB to BGR zero-centered ImageNet means).
    Applies augmentation exclusively if is_training=True.
    """
    df = pd.read_csv(csv_path)
    image_paths = df["image_path"].tolist()
    labels = df["numeric_class_index"].tolist()

    def process_path(img_path, label):
        img_raw = tf.io.read_file(img_path)
        img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        # ResNet50 expects inputs in range [0, 255] for preprocess_input
        img = preprocess_input(img)
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training and augmenter is not None:
        dataset = dataset.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset, len(image_paths)


def build_augmentation_layer():
    """
    Defines the training data augmentation pipeline.
    Augmentation is applied ONLY to the training dataset.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.1),
        layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        layers.RandomTranslation(height_factor=0.08, width_factor=0.08)
    ], name="data_augmentation")


def build_resnet50_model(
    num_classes: int = 6,
    input_shape: tuple = (224, 224, 3),
    dense_units: int = 256,
    dropout_rate: float = 0.3
) -> tf.keras.Model:
    """
    Builds the ResNet50 transfer learning model.
    Structure:
    Input (224, 224, 3)
      -> ResNet50 Backbone (ImageNet weights, Frozen)
      -> GlobalAveragePooling2D
      -> Dense(dense_units, activation='relu')
      -> Dropout(dropout_rate)
      -> Dense(num_classes, activation='softmax')
    """
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    # Freeze the ResNet50 feature extractor
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    if dense_units and dense_units > 0:
        x = layers.Dense(dense_units, activation="relu", name="fc_dense")(x)
    if dropout_rate and dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="RiceLeaf_ResNet50")
    return model, base_model


def plot_and_save_curves(history, output_dir: Path):
    """Generates and saves training and validation accuracy/loss curve plots."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    # 1. Training vs Validation Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#1f77b4", linewidth=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    plt.title("ResNet50 - Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    acc_plot_path = output_dir / "resnet50_training_accuracy.png"
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Accuracy curve saved to: {acc_plot_path}")

    # 2. Training vs Validation Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, loss, label="Training Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#d62728", linewidth=2)
    plt.title("ResNet50 - Training vs Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    loss_plot_path = output_dir / "resnet50_training_loss.png"
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Loss curve saved to: {loss_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ResNet50 model for Rice Leaf Disease Classification")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Directory containing dataset")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for output files")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory for saved model and classes")
    parser.add_argument("--img_height", type=int, default=224, help="Input image height")
    parser.add_argument("--img_width", type=int, default=224, help="Input image width")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum epochs to train")
    parser.add_argument("--lr", type=float, default=0.0001, help="Adam optimizer learning rate")
    parser.add_argument("--dense_units", type=int, default=256, help="Units in classification head dense layer")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate before output layer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--patience_es", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--patience_lr", type=int, default=2, help="ReduceLROnPlateau patience")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_dir = (project_root / args.output_dir).resolve()
    model_dir = (project_root / args.model_dir).resolve()
    dataset_dir = (project_root / args.dataset_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    # Check for split files, run prepare_dataset if missing
    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "validation.csv"
    test_csv = output_dir / "test.csv"
    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        print("[INFO] Split CSVs not found. Running prepare_dataset.py...")
        from prepare_dataset import scan_and_summarize, create_stratified_split
        valid_records = scan_and_summarize(dataset_dir, output_dir)
        create_stratified_split(valid_records, output_dir, seed=args.seed)

    # Load canonical classes
    class_names_path = model_dir / "class_names.json"
    if not class_names_path.exists():
        from prepare_dataset import EXPECTED_CLASSES
        with open(class_names_path, "w", encoding="utf-8") as f:
            json.dump(EXPECTED_CLASSES, f, indent=2)
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    num_classes = len(class_names)
    print(f"\n[INFO] Loaded {num_classes} classes from {class_names_path}: {class_names}")

    # Build Augmenter for training pipeline
    augmenter = build_augmentation_layer()

    # Create tf.data datasets
    img_size = (args.img_height, args.img_width)
    train_ds, n_train = load_dataset_from_csv(
        train_csv,
        img_size=img_size,
        batch_size=args.batch_size,
        is_training=True,
        augmenter=augmenter
    )
    val_ds, n_val = load_dataset_from_csv(
        val_csv,
        img_size=img_size,
        batch_size=args.batch_size,
        is_training=False,
        augmenter=None
    )

    print(f"[INFO] Train samples: {n_train} | Validation samples: {n_val}")

    # Build Model
    model, base_model = build_resnet50_model(
        num_classes=num_classes,
        input_shape=(args.img_height, args.img_width, 3),
        dense_units=args.dense_units,
        dropout_rate=args.dropout
    )

    optimizer = optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Compute and save parameter counts and summary
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    model_summary_str = "\n".join(summary_lines)

    total_params = int(model.count_params())
    trainable_params = int(sum([tf.size(w).numpy() for w in model.trainable_weights]))
    non_trainable_params = int(sum([tf.size(w).numpy() for w in model.non_trainable_weights]))

    summary_file_path = output_dir / "model_summary.txt"
    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RESNET50 RICE LEAF DISEASE CLASSIFIER - MODEL ARCHITECTURE SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(model_summary_str + "\n\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Parameters:         {total_params:,}\n")
        f.write(f"Trainable Parameters:     {trainable_params:,}\n")
        f.write(f"Non-Trainable Parameters: {non_trainable_params:,}\n")
        f.write("-" * 70 + "\n")

    print(f"[INFO] Model summary saved to: {summary_file_path}")
    print(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,} | Non-trainable: {non_trainable_params:,}")

    # Save training configuration
    training_config = {
        "model_name": "ResNet50",
        "pretrained_weights": "imagenet",
        "input_shape": [args.img_height, args.img_width, 3],
        "classes": class_names,
        "num_classes": num_classes,
        "random_seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": "Adam",
        "loss_function": "sparse_categorical_crossentropy",
        "max_epochs": args.epochs,
        "dense_units": args.dense_units,
        "dropout_rate": args.dropout,
        "backbone_trainable": False,
        "augmentation_settings": {
            "random_flip": "horizontal",
            "random_rotation": 0.1,
            "random_zoom": 0.1,
            "random_translation": 0.08
        },
        "early_stopping_patience": args.patience_es,
        "reduce_lr_patience": args.patience_lr,
        "reduce_lr_factor": 0.5,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params
    }
    config_path = output_dir / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)
    print(f"[INFO] Training configuration saved to: {config_path}")

    # Set up callbacks
    model_save_path = model_dir / "resnet50.keras"
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience_es,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=args.patience_lr,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    print("\n" + "=" * 60)
    print("STARTING TRAINING PIPELINE")
    print(f"Epochs: up to {args.epochs} | Batch size: {args.batch_size} | Learning rate: {args.lr}")
    print("=" * 60)

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=cb_list,
        verbose=1
    )
    training_time_sec = time.time() - start_time
    print(f"\n[INFO] Training finished in {training_time_sec:.2f} seconds ({training_time_sec / 60:.2f} minutes).")

    # Ensure best model is saved to models/resnet50.keras
    model.save(str(model_save_path))
    print(f"[SUCCESS] Trained model successfully saved to: {model_save_path}")

    # Generate and save curves
    plot_and_save_curves(history, output_dir)

    # Save training time for efficiency report
    train_time_file = output_dir / ".training_time.txt"
    with open(train_time_file, "w", encoding="utf-8") as f:
        f.write(f"{training_time_sec:.4f}")

    print("\n[SUCCESS] Training pipeline completed successfully.")


if __name__ == "__main__":
    main()

