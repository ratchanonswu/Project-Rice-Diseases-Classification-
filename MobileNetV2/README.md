# MobileNetV2 Rice Leaf Disease Classification

A self-contained, reproducible transfer learning pipeline using **MobileNetV2** for 6-class rice leaf disease classification in TensorFlow/Keras.

This project is designed as part of a comparative university research initiative to benchmark lightweight convolutional architectures against larger deep networks (specifically comparing **MobileNetV2** against **ResNet50**).

---

## 1. Project Objective

The goal of this project is to build an efficient, highly compact deep learning model capable of accurately classifying rice leaf diseases from RGB leaf imagery. 

Crucially, this pipeline preserves the exact dataset split assignments and class indexing from the baseline ResNet50 experiment (`C:\Gitwork\Rice diseases\ResNet50`), ensuring a fair, rigorous, and head-to-head comparison on identical test samples.

---

## 2. Six Dataset Classes

The dataset consists of 3,829 images organized across the following 6 canonical classes in fixed index order:

| Index | Canonical Class Name | Description | Total Images |
|---|---|---|---|
| `0` | **Bacterial Leaf Blight** | Caused by *Xanthomonas oryzae*, produces water-soaked lesions along leaf margins. | 636 |
| `1` | **Brown Spot** | Caused by *Bipolaris oryzae*, produces circular to oval dark brown spots. | 646 |
| `2` | **Healthy Rice Leaf** | Normal rice leaves with no disease symptoms. | 653 |
| `3` | **Leaf Blast** | Caused by *Magnaporthe oryzae*, displays diamond/spindle-shaped lesions. | 634 |
| `4` | **Leaf Scald** | Caused by *Microdochium oryzae*, displays zoned oblong leaf lesions. | 628 |
| `5` | **Sheath Blight** | Caused by *Rhizoctonia solani*, exhibits greenish-gray water-soaked spots. | 632 |
| **Total** | | | **3,829** |

---

## 3. Project Structure

```text
MobileNetV2/
├── dataset/                                ← Image dataset organized by class (junction/copy)
│   ├── Bacterial Leaf Blight/
│   ├── Brown Spot/
│   ├── Healthy Rice Leaf/
│   ├── Leaf Blast/
│   ├── Leaf Scald/
│   └── Sheath Blight/
│
├── models/
│   ├── mobilenetv2.keras                  ← Fully trained Keras model (12.96 MB)
│   └── class_names.json                   ← Canonical 6-class mapping
│
├── outputs/
│   ├── dataset_summary.csv                ← Image counts, dimensions, corrupt checks
│   ├── train.csv                          ← Training set split manifest (2,871 samples)
│   ├── validation.csv                     ← Validation set split manifest (383 samples)
│   ├── test.csv                           ← Unseen test set split manifest (575 samples)
│   ├── training_config.json               ← Hyperparameters & training run record
│   ├── training_history.csv               ← Full per-epoch loss and accuracy metrics
│   ├── model_summary.txt                  ← Architecture layer layout & parameter counts
│   ├── metrics.json                       ← Final test set metrics (accuracy, F1, AUC, etc.)
│   ├── classification_report.csv          ← Per-class precision, recall, F1, and support
│   ├── mobilenetv2_confusion_matrix.png   ← High-resolution confusion matrix heatmap
│   ├── mobilenetv2_confusion_matrix.csv   ← Raw numerical confusion matrix
│   ├── mobilenetv2_training_accuracy.png  ← Training vs validation accuracy curve
│   ├── mobilenetv2_training_loss.png      ← Training vs validation loss curve
│   ├── test_predictions.csv              ← Per-sample predictions with all 6 probabilities
│   ├── efficiency.json                    ← Model size, latency, throughput, and hardware
│   ├── experiment_metadata.json           ← Environment, packages, and full reproducibility log
│   ├── correct_predictions/               ← Visual cards of correct test predictions
│   └── incorrect_predictions/             ← Visual cards of misclassified test samples
│
├── src/
│   ├── prepare_dataset.py                 ← Dataset scanner & ResNet50 split alignment
│   ├── train.py                           ← Transfer learning training pipeline
│   ├── evaluate.py                        ← Test set evaluation & latency profiling
│   └── predict.py                         ← CLI inference for single images
│
├── requirements.txt
└── README.md
```

---

## 4. Installation Instructions (Windows)

### Prerequisites
- Python 3.11.x (recommended for TensorFlow compatibility on Windows)
- Windows 10/11 64-bit

### Setup Virtual Environment

```powershell
# 1. From the MobileNetV2 project root, create a virtual environment:
py -3.11 -m venv .venv

# 2. Activate the virtual environment:
.venv\Scripts\activate

# 3. Install required packages:
pip install -r requirements.txt
```

---

## 5. Dataset and Split Setup

To maintain parity with the baseline experiment, `src/prepare_dataset.py` inspects the dataset and imports the exact split manifests from `ResNet50/outputs/`:

```powershell
python src/prepare_dataset.py --dataset_dir dataset --resnet_dir ../ResNet50 --output_dir outputs
```

**Split Distribution (Identical to ResNet50 baseline):**

| Class | Train (75%) | Validation (10%) | Test (15%) | Total |
|---|---|---|---|---|
| Bacterial Leaf Blight | 476 | 64 | 96 | 636 |
| Brown Spot | 484 | 65 | 97 | 646 |
| Healthy Rice Leaf | 490 | 65 | 98 | 653 |
| Leaf Blast | 476 | 63 | 95 | 634 |
| Leaf Scald | 471 | 63 | 94 | 628 |
| Sheath Blight | 474 | 63 | 95 | 632 |
| **Total** | **2,871** | **383** | **575** | **3,829** |

---

## 6. Training Command

Train MobileNetV2 using transfer learning with a frozen ImageNet feature extractor:

```powershell
python src/train.py --epochs 25 --batch_size 32 --lr 0.0001 --dense_units 256 --dropout 0.3 --seed 42
```

### Architecture Details
- **Backbone**: `MobileNetV2` (`include_top=False`, `weights="imagenet"`, frozen backbone: 2,257,984 non-trainable parameters).
- **Head**: `GlobalAveragePooling2D` -> `Dense(256, ReLU)` -> `Dropout(0.3)` -> `Dense(6, Softmax)`.
- **Total Parameters**: 2,587,462 (329,478 trainable).
- **Preprocessing**: `tf.keras.applications.mobilenet_v2.preprocess_input` (scales pixel values to `[-1, 1]`).
- **Data Augmentation** (applied strictly to training data only): Random horizontal flip, random rotation (±10%), random zoom (±10%), random translation (±8%).
- **Callbacks**: Early stopping (`patience=5`), ReduceLROnPlateau (`factor=0.5, patience=2`), ModelCheckpoint saving best weights on `val_loss`.

---

## 7. Evaluation Command

Evaluate the trained checkpoint exclusively on the unseen test set (575 samples):

```powershell
python src/evaluate.py --num_examples 8 --profile_samples 50
```

---

## 8. Single Image Prediction Command

Run inference on any individual leaf image:

```powershell
python src/predict.py --image dataset/Healthy Rice Leaf/Healthy_rice_leaf (108).jpg
```

**Example Output:**
```text
=============================================
Model: MobileNetV2
Image: Healthy_rice_leaf (108).jpg
=============================================

Prediction:
Healthy Rice Leaf

Confidence:
99.60%

Top 3:
1. Healthy Rice Leaf - 99.60%
2. Brown Spot - 0.35%
3. Sheath Blight - 0.02%

Probability Verification:
Sum of probabilities: 1.000000 (~ 1.000000)
=============================================
```

---

## 9. Generated Outputs & Results

### Test Set Performance (575 Unseen Images)

| Metric | MobileNetV2 | ResNet50 (Baseline) |
|---|---|---|
| **Test Accuracy** | **83.65%** | 90.26% |
| **Precision (macro)** | **84.00%** | 90.24% |
| **Recall (macro)** | **83.56%** | 90.20% |
| **F1-Score (macro)** | **83.47%** | 90.21% |
| **ROC-AUC (macro OvR)** | **97.58%** | 98.83% |

### Per-Class Test Performance (MobileNetV2)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Bacterial Leaf Blight | 79.59% | 81.25% | **80.41%** | 96 |
| Brown Spot | 81.55% | 86.60% | **84.00%** | 97 |
| Healthy Rice Leaf | 97.87% | 93.88% | **95.83%** | 98 |
| Leaf Blast | 72.55% | 77.89% | **75.13%** | 95 |
| Leaf Scald | 87.50% | 67.02% | **75.90%** | 94 |
| Sheath Blight | 84.91% | 94.74% | **89.55%** | 95 |

### Computational Efficiency Comparison

| Metric | MobileNetV2 | ResNet50 | Advantage |
|---|---|---|---|
| **Total Parameters** | **2,587,462** | 24,113,798 | **9.3× fewer parameters** |
| **Trainable Parameters** | **329,478** | 526,086 | **1.6× fewer trainable params** |
| **Model Disk Size** | **12.96 MB** | 96.64 MB | **7.5× smaller file size** |
| **Training Duration** | **11.98 min** (25 ep) | 37.65 min (25 ep) | **3.1× faster training** |
| **Inference Latency (CPU)**| **129.5 ms/image** | 206.1 ms/image | **1.6× faster inference** |
| **Throughput (CPU)** | **7.72 FPS** | 4.85 FPS | **+59% throughput** |

---

## 10. Important Limitations

1. **Native Windows TensorFlow Limitations**: TensorFlow >= 2.11 on native Windows does not utilize native GPU acceleration via CUDA; CPU training was utilized with oneDNN optimizations enabled.
2. **Backbone Depth vs. Edge Efficiency**: While ResNet50 achieves higher raw test accuracy (90.26% vs. 83.65%), MobileNetV2 requires only ~13% of the disk footprint and trains in less than one-third the time, making it suitable for edge and mobile deployment.
3. **Fine-Tuning Potential**: In both baseline experiments, feature backbones were frozen. Unfreezing top inverted residual blocks (fine-tuning) is a natural next step to narrow the accuracy gap.

---

## 11. Comparative Research Website Integration

This project outputs standard structured artifacts (`metrics.json`, `efficiency.json`, `experiment_metadata.json`, `classification_report.csv`, and `test_predictions.csv`) that match the exact schema used in `ResNet50`. 

The downstream comparison website can consume these files directly to render:
- Interactive radar charts comparing accuracy, parameter efficiency, and latency.
- Side-by-side confusion matrix comparisons.
- Per-class precision, recall, and F1 trade-off bars.
- Visual inspection of agreement and disagreement across individual test images.

