# ResNet50 Rice Leaf Disease Classification

A university research project implementing **ResNet50 transfer learning** for 6-class rice leaf disease classification using TensorFlow/Keras.

---

## Dataset Structure

Place (or symlink) your dataset in the `dataset/` directory with the following structure:

```
dataset/
├── Bacterial Leaf Blight/
├── Brown Spot/
├── Healthy Rice Leaf/
├── Leaf Blast/
├── Leaf Scald/
└── Sheath Blight/
```

**Dataset statistics (3,829 images):**

| Class | Images | Split (Train/Val/Test) |
|---|---|---|
| Bacterial Leaf Blight | 636 | 476 / 64 / 96 |
| Brown Spot | 646 | 484 / 65 / 97 |
| Healthy Rice Leaf | 653 | 490 / 65 / 98 |
| Leaf Blast | 634 | 476 / 63 / 95 |
| Leaf Scald | 628 | 471 / 63 / 94 |
| Sheath Blight | 632 | 474 / 63 / 95 |
| **Total** | **3,829** | **2,871 / 383 / 575** |

---

## Installation

### Prerequisites
- Python 3.11 (TensorFlow-compatible)
- ~4 GB disk space for the virtual environment + model

### Setup

```bash
# 1. Create virtual environment with Python 3.11
py -3.11 -m venv .venv

# 2. Activate (Windows)
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
ResNet50/
├── dataset/                          ← Put dataset here
│   ├── Bacterial Leaf Blight/
│   ├── Brown Spot/
│   ├── Healthy Rice Leaf/
│   ├── Leaf Blast/
│   ├── Leaf Scald/
│   └── Sheath Blight/
│
├── models/
│   ├── resnet50.keras                ← Trained model (generated after training)
│   └── class_names.json             ← Fixed class order
│
├── outputs/
│   ├── dataset_summary.csv
│   ├── train.csv / validation.csv / test.csv
│   ├── training_config.json
│   ├── model_summary.txt
│   ├── metrics.json
│   ├── classification_report.csv
│   ├── resnet50_confusion_matrix.png / .csv
│   ├── resnet50_training_accuracy.png
│   ├── resnet50_training_loss.png
│   ├── test_predictions.csv
│   ├── efficiency.json
│   ├── experiment_metadata.json
│   ├── correct_predictions/
│   └── incorrect_predictions/
│
├── src/
│   ├── prepare_dataset.py            ← Step 1: Dataset validation & splitting
│   ├── train.py                      ← Step 2: Model training
│   ├── evaluate.py                   ← Step 3: Evaluation & reporting
│   └── predict.py                    ← Inference CLI
│
├── requirements.txt
└── README.md
```

---

## Step 1: Dataset Preparation

Validates the dataset, scans for corrupt images, generates a summary, and creates a reproducible stratified split.

```bash
python src/prepare_dataset.py --dataset_dir dataset --output_dir outputs
```

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--dataset_dir` | `dataset` | Path to image dataset |
| `--output_dir` | `outputs` | Path for output files |
| `--seed` | `42` | Random seed for reproducibility |
| `--train_ratio` | `0.75` | Training set proportion |
| `--val_ratio` | `0.10` | Validation set proportion |
| `--test_ratio` | `0.15` | Test set proportion |

**Outputs:**
- `outputs/dataset_summary.csv` — Per-class image counts, dimensions, corrupted count
- `outputs/train.csv` — 2,871 training images
- `outputs/validation.csv` — 383 validation images
- `outputs/test.csv` — 575 test images

---

## Step 2: Training

Trains the ResNet50 transfer learning model with frozen ImageNet backbone.

```bash
python src/train.py
```

**All options:**

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `25` | Maximum training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--lr` | `0.0001` | Adam learning rate |
| `--dense_units` | `256` | Dense head units |
| `--dropout` | `0.3` | Dropout rate |
| `--seed` | `42` | Random seed |

**Model architecture:**
```
Input (224×224×3)
  → ResNet50 Backbone (ImageNet, Frozen — 23.6M non-trainable params)
  → GlobalAveragePooling2D
  → Dense(256, ReLU)
  → Dropout(0.3)
  → Dense(6, Softmax)
```

**Training augmentation (training pipeline only):**
- RandomFlip (horizontal)
- RandomRotation (±10%)
- RandomZoom (±10%)
- RandomTranslation (±8%)

**Callbacks:**
- `EarlyStopping(patience=5, restore_best_weights=True)`
- `ReduceLROnPlateau(patience=2, factor=0.5)`
- `ModelCheckpoint(save_best_only=True)`

**Outputs:**
- `models/resnet50.keras` — Complete trained model
- `outputs/training_config.json` — All hyperparameters
- `outputs/model_summary.txt` — Architecture and parameter counts
- `outputs/resnet50_training_accuracy.png`
- `outputs/resnet50_training_loss.png`

---

## Step 3: Evaluation

Evaluates the trained model exclusively on the unseen test set.

```bash
python src/evaluate.py
```

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--num_examples` | `8` | Visual prediction samples to save |

**Outputs:**
- `outputs/metrics.json` — Accuracy, Precision, Recall, F1, ROC-AUC
- `outputs/classification_report.csv` — Per-class metrics
- `outputs/resnet50_confusion_matrix.png/csv` — 6×6 confusion matrix
- `outputs/test_predictions.csv` — Per-image predictions + confidence
- `outputs/efficiency.json` — Parameters, model size, inference latency
- `outputs/experiment_metadata.json` — Full reproducibility metadata
- `outputs/correct_predictions/` — Sample correct prediction images
- `outputs/incorrect_predictions/` — Sample incorrect prediction images

---

## Loading the Trained Model

```python
from tensorflow.keras.models import load_model
import json

model = load_model("models/resnet50.keras")

with open("models/class_names.json") as f:
    class_names = json.load(f)
```

---

## Single Image Prediction

```bash
python src/predict.py --image path/to/rice_leaf.jpg
```

**Example output:**
```
Model: ResNet50

Prediction:
Leaf Blast

Confidence:
94.32%

Top 3:
1. Leaf Blast - 94.32%
2. Brown Spot - 3.41%
3. Leaf Scald - 1.27%
```

---

## Experiment Results

> Results are from actual training — not fabricated.

| Metric | Value |
|---|---|
| Test Accuracy | **90.26%** |
| Precision (macro) | 90.24% |
| Recall (macro) | 90.20% |
| F1-Score (macro) | 90.21% |
| ROC-AUC (macro) | 98.83% |
| Training Time | ~37.7 minutes (25 epochs, CPU) |
| Best Epoch | Epoch 24 (val_acc: 88.51%) |
| Model File Size | See `outputs/efficiency.json` |

**Per-class F1-Scores:**

| Class | F1-Score |
|---|---|
| Bacterial Leaf Blight | 86.6% |
| Brown Spot | 94.2% |
| Healthy Rice Leaf | 98.5% |
| Leaf Blast | 86.3% |
| Leaf Scald | 83.4% |
| Sheath Blight | 92.2% |

---

## Reproducibility

All experiments are reproducible using:
- Fixed random seed: `42`
- Stratified train/val/test split (75/10/15)
- Augmentation applied **only** to training pipeline
- Test set never seen during training or validation

Configuration logged to `outputs/experiment_metadata.json` and `outputs/training_config.json`.

---

## Experiment Configuration

To reproduce exactly:

```bash
python src/prepare_dataset.py --seed 42
python src/train.py --epochs 25 --batch_size 32 --lr 0.0001 --dense_units 256 --dropout 0.3 --seed 42
python src/evaluate.py
```

---

## Requirements

See [`requirements.txt`](requirements.txt). Key dependencies:
- `tensorflow >= 2.15.0`
- `numpy`, `pandas`, `matplotlib`
- `scikit-learn`, `seaborn`, `pillow`
