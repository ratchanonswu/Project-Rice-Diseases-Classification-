"""
train.py - Transfer Learning Pipeline for MobileNetV2 on Rice Leaf Disease Classification

This script:
1. Configures all hyperparameters and logs them to outputs/training_config.json.
2. Builds the MobileNetV2 transfer learning architecture with frozen ImageNet backbone.
3. Records model summary and parameter counts to outputs/model_summary.txt.
4. Loads training and validation datasets using tf.keras.applications.mobilenet_v2.preprocess_input.
5. Applies data augmentation exclusively to training data.
6. Trains the model with Adam optimizer, EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.
7. Saves the complete trained model to models/mobilenetv2.keras.
8. Generates training vs. validation accuracy and loss curves to outputs/.
9. Saves training history to outputs/training_history.csv.
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
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


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
    Constructs a tf.data.Dataset from a split CSV.
    Applies MobileNetV2 preprocessing (scales pixel values to [-1, 1]).
    Applies augmentation exclusively if is_training=True.
    """
    df = pd.read_csv(csv_path)
    image_paths = df["image_path"].tolist()
    labels = df["numeric_class_index"].tolist()

    def process_path(img_path, label):
        img_raw = tf.io.read_file(img_path)
        img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        # MobileNetV2 expects float inputs in range [0, 255] for preprocess_input, mapping to [-1, 1]
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
    Defines the training data augmentation pipeline matching ResNet50:
    - Random horizontal flip
    - Random rotation (±10%)
    - Random zoom (±10%)
    - Random translation (±8%)
    Augmentation is applied ONLY to the training dataset.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.1),
        layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        layers.RandomTranslation(height_factor=0.08, width_factor=0.08)
    ], name="data_augmentation")


def build_mobilenetv2_model(
    num_classes: int = 6,
    input_shape: tuple = (224, 224, 3),
    dense_units: int = 256,
    dropout_rate: float = 0.3
) -> tuple:
    """
    Builds the MobileNetV2 transfer learning model.
    Structure:
    Input (224, 224, 3)
      -> MobileNetV2 Backbone (ImageNet weights, Frozen)
      -> GlobalAveragePooling2D
      -> Dense(dense_units, activation='relu')
      -> Dropout(dropout_rate)
      -> Dense(num_classes, activation='softmax')
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    # Freeze the pretrained backbone
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    if dense_units and dense_units > 0:
        x = layers.Dense(dense_units, activation="relu", name="fc_dense")(x)
    if dropout_rate and dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="RiceLeaf_MobileNetV2")
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
    plt.title("MobileNetV2 - Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    acc_plot_path = output_dir / "mobilenetv2_training_accuracy.png"
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved accuracy curve: {acc_plot_path}")

    # 2. Training vs Validation Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, loss, label="Training Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#ff7f0e", linewidth=2)
    plt.title("MobileNetV2 - Training vs Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Sparse Categorical Crossentropy)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    loss_plot_path = output_dir / "mobilenetv2_training_loss.png"
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved loss curve: {loss_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on Rice Leaf Disease Dataset.")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum epochs (default: 25)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate (default: 0.0001)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for output files (default: outputs)")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory for saved model and classes (default: models)")
    parser.add_argument("--dense_units", type=int, default=256, help="Units in top dense layer (default: 256)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate before output (default: 0.3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_dir = (project_root / args.output_dir).resolve()
    models_dir = (project_root / args.models_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "validation.csv"
    class_names_path = models_dir / "class_names.json"

    if not train_csv.exists() or not val_csv.exists():
        print("[ERROR] Split files not found in outputs/! Run prepare_dataset.py first.")
        sys.exit(1)

    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"[INFO] Loaded {num_classes} classes: {class_names}")

    # Build model
    input_shape = (224, 224, 3)
    model, base_model = build_mobilenetv2_model(
        num_classes=num_classes,
        input_shape=input_shape,
        dense_units=args.dense_units,
        dropout_rate=args.dropout
    )

    optimizer = optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Record model summary
    summary_path = output_dir / "model_summary.txt"
    with open(summary_path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print(f"[INFO] Saved model architecture summary to: {summary_path}")

    total_params = int(model.count_params())
    trainable_params = int(sum([tf.size(w).numpy() for w in model.trainable_weights]))
    non_trainable_params = total_params - trainable_params
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,} | Non-trainable: {non_trainable_params:,}")

    # Log training config
    training_config = {
        "model_architecture": "MobileNetV2",
        "pretrained_weights": "imagenet",
        "frozen_backbone": True,
        "input_shape": list(input_shape),
        "num_classes": num_classes,
        "classes": class_names,
        "random_seed": args.seed,
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": "Adam",
        "loss": "sparse_categorical_crossentropy",
        "dense_units": args.dense_units,
        "dropout_rate": args.dropout,
        "augmentation_settings": {
            "random_flip": "horizontal",
            "random_rotation": 0.1,
            "random_zoom": 0.1,
            "random_translation": 0.08
        },
        "callbacks": {
            "early_stopping": {"monitor": "val_loss", "patience": 5, "restore_best_weights": True},
            "reduce_lr_on_plateau": {"monitor": "val_loss", "factor": 0.5, "patience": 2, "min_lr": 1e-6},
            "model_checkpoint": {"monitor": "val_loss", "save_best_only": True}
        }
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    # Prepare datasets
    augmenter = build_augmentation_layer()
    train_ds, train_count = load_dataset_from_csv(
        train_csv,
        img_size=(224, 224),
        batch_size=args.batch_size,
        is_training=True,
        augmenter=augmenter
    )
    val_ds, val_count = load_dataset_from_csv(
        val_csv,
        img_size=(224, 224),
        batch_size=args.batch_size,
        is_training=False,
        augmenter=None
    )
    print(f"[INFO] Train samples: {train_count} | Validation samples: {val_count}")

    # Callbacks
    model_save_path = models_dir / "mobilenetv2.keras"
    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=str(model_save_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    cb_earlystop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    print("\n" + "=" * 65)
    print(f"STARTING MOBILENETV2 TRAINING (Max Epochs: {args.epochs}, Batch Size: {args.batch_size})")
    print("=" * 65)

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[cb_checkpoint, cb_earlystop, cb_reduce_lr],
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"\n[INFO] Training finished in {training_time:.2f} seconds ({training_time/60:.2f} minutes).")

    # Record training duration for efficiency profiling
    with open(output_dir / ".training_time.txt", "w") as f:
        f.write(str(training_time))

    # Save training history to CSV
    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", range(1, len(hist_df) + 1))
    hist_csv = output_dir / "training_history.csv"
    hist_df.to_csv(hist_csv, index=False)
    print(f"[INFO] Saved training history to: {hist_csv}")

    # Record best epoch details into config
    val_losses = history.history.get("val_loss", [])
    if val_losses:
        best_epoch_idx = int(np.argmin(val_losses))
        best_epoch = best_epoch_idx + 1
        best_val_loss = float(val_losses[best_epoch_idx])
        best_val_acc = float(history.history.get("val_accuracy", [0])[best_epoch_idx])
        training_config["best_epoch"] = best_epoch
        training_config["best_val_loss"] = best_val_loss
        training_config["best_val_accuracy"] = best_val_acc
        training_config["selection_criterion"] = "Minimum validation loss (val_loss)"
        training_config["actual_epochs_trained"] = len(val_losses)
        training_config["training_time_seconds"] = round(training_time, 2)
        with open(output_dir / "training_config.json", "w") as f:
            json.dump(training_config, f, indent=2)
        print(f"[INFO] Best Epoch: {best_epoch} (val_loss: {best_val_loss:.4f}, val_accuracy: {best_val_acc*100:.2f}%)")

    # Ensure best model is saved
    model.save(str(model_save_path))
    print(f"[SUCCESS] Final/best model saved to: {model_save_path}")

    # Generate curves
    plot_and_save_curves(history, output_dir)
    print("\n[SUCCESS] MobileNetV2 Training pipeline complete!")


if __name__ == "__main__":
    main()

