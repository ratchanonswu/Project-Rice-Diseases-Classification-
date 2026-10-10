# Comparative Research Report: ResNet50 vs MobileNetV2 on Leakage-Safe Rice Leaf Disease Dataset

**Date:** October 10, 2026  
**Auditor / ML Data Engineer:** Senior Machine Learning Data Engineer  
**Experiment Identifier:** `clean_split_v1`  
**Dataset Source:** `C:\Gitwork\Rice diseases\DatasetAudit\manifests\`  
**Target Architectures:** ResNet50 vs MobileNetV2 (Transfer Learning, ImageNet frozen backbones)

---

## 1. Executive Summary

This report delivers the first controlled, head-to-head empirical comparison of **ResNet50** and **MobileNetV2** on the validated, deduplicated, leakage-safe Rice Leaf Disease dataset (`clean_split_v1`).

Both models were trained and evaluated using strictly identical dataset partitions (Clean Train: 2,847 images, Clean Validation: 381 images, Clean Test: 570 images), identical image resolution (224×224×3), identical class index mapping `[0..5]`, identical batch size (32), identical learning rate schedule (Adam, $10^{-4}$ with ReduceLROnPlateau), identical augmentation policies, and identical checkpoint selection criteria (best `val_loss`).

### Key Findings
1. **Classification Accuracy Lead:** **ResNet50** outperforms **MobileNetV2** by **+7.02%** in test accuracy (**88.60%** vs **81.58%**) and by **+7.07%** in Macro F1-score (**88.39%** vs **81.32%**).
2. **Computational & Edge Efficiency Lead:** **MobileNetV2** demonstrates massive computational advantages:
   - **9.3× fewer total parameters** (2.59M vs 24.11M).
   - **7.5× smaller model storage footprint** (12.96 MB vs 96.64 MB).
   - **2.9× faster training convergence** (11.65 min vs 33.69 min).
   - **1.87× faster single-image inference** on CPU (105.07 ms vs 196.54 ms; throughput of 9.52 FPS vs 5.09 FPS).
3. **Data Leakage Fully Eliminated:** Zero exact SHA-256 byte duplicates and zero confirmed near-duplicate burst sequences cross the Train, Validation, and Test boundaries.
4. **Historical vs Clean Trajectory:** On the original contaminated split, ResNet50 achieved 90.26% and MobileNetV2 achieved 83.65%. On the clean split, ResNet50 scored 88.60% and MobileNetV2 scored 81.58%. While this reflects an apparent drop of ~1.7–2.1%, it must be understood that test membership has changed (exact duplicates removed, burst sequences isolated to single splits, and test set reduced from 575 to 570 images).

---

## 2. Experimental Setup & Protocol Alignment

| Parameter / Protocol Setting | ResNet50 (`clean_split_v1`) | MobileNetV2 (`clean_split_v1`) | Alignment Status | Notes |
|---|---|---|:---:|---|
| **Authoritative Split Source** | `DatasetAudit/manifests/` | `DatasetAudit/manifests/` | **IDENTICAL** | Byte-exact manifests verified |
| **Train Samples** | 2,847 (74.96%) | 2,847 (74.96%) | **IDENTICAL** | Exact same image files |
| **Validation Samples** | 381 (10.03%) | 381 (10.03%) | **IDENTICAL** | Exact same image files |
| **Test Samples** | 570 (15.01%) | 570 (15.01%) | **IDENTICAL** | Exact same image files |
| **Input Dimensions** | (224, 224, 3) | (224, 224, 3) | **MATCH** | Standard transfer learning resolution |
| **Input Preprocessing** | Zero-centered BGR ImageNet | Normalized RGB $[-1, 1]$ | **MATCH (Architecture-Native)** | ResNet50 vs MobileNetV2 native Keras |
| **Data Augmentation (Train only)** | Flip, Rot(0.1), Zoom(0.1), Trans(0.08) | Flip, Rot(0.1), Zoom(0.1), Trans(0.08) | **MATCH** | Identical augmentation pipeline |
| **Pretrained Weights** | ImageNet (`include_top=False`) | ImageNet (`include_top=False`) | **MATCH** | Identical initial feature extractor |
| **Backbone Trainability** | Frozen (`trainable=False`) | Frozen (`trainable=False`) | **MATCH** | Evaluates fixed representations |
| **Classifier Head** | GAP -> Dense(256, relu) -> Dropout(0.3) -> Dense(6, softmax) | GAP -> Dense(256, relu) -> Dropout(0.3) -> Dense(6, softmax) | **MATCH** | Identical head capacity |
| **Optimizer & Initial LR** | Adam ($1\times 10^{-4}$) | Adam ($1\times 10^{-4}$) | **MATCH** | Identical optimization dynamics |
| **Loss Function** | Sparse Categorical Cross-Entropy | Sparse Categorical Cross-Entropy | **MATCH** | Identical training objective |
| **Batch Size** | 32 | 32 | **MATCH** | 89 steps per epoch |
| **Max Epochs** | 25 | 25 | **MATCH** | Equal training budget |
| **Model Selection Metric** | Best `val_loss` | Best `val_loss` | **MATCH** | Epoch 25 for ResNet50, Epoch 23 for MobileNetV2 |
| **Early Stopping Policy** | `patience=5`, restore best | `patience=5`, restore best | **MATCH** | Neither stopped early |
| **LR Scheduler** | ReduceLROnPlateau (`pat=2, factor=0.5`) | ReduceLROnPlateau (`pat=2, factor=0.5`) | **MATCH** | Dynamic rate scaling |
| **Random Seed** | 42 | 42 | **MATCH** | Python, NumPy, TensorFlow seeds set |
| **Hardware** | AMD64 CPU, Windows native | AMD64 CPU, Windows native | **MATCH** | Identical CPU execution environment |

---

## 3. Side-by-Side Performance Comparison

### A. Test Set Classification Metrics (570 Clean Test Images)

| Evaluation Metric | Historical ResNet50 (Original Split) | **New ResNet50 (Clean Split)** | Historical MobileNetV2 (Original Split) | **New MobileNetV2 (Clean Split)** | **Clean Gap ($\Delta = \text{ResNet} - \text{MobileNet}$)** |
|---|:---:|:---:|:---:|:---:|:---:|
| **Test Accuracy** | 90.26% | **88.60%** | 83.65% | **81.58%** | **+7.02%** |
| **Precision (Macro)** | 90.24% | **88.89%** | 84.00% | **81.67%** | **+7.22%** |
| **Precision (Weighted)**| 90.31% | **88.91%** | 84.05% | **81.78%** | **+7.13%** |
| **Recall (Macro)** | 90.20% | **88.50%** | 83.56% | **81.46%** | **+7.04%** |
| **Recall (Weighted)** | 90.26% | **88.60%** | 83.65% | **81.58%** | **+7.02%** |
| **F1-Score (Macro)** | 90.21% | **88.39%** | 83.47% | **81.32%** | **+7.07%** |
| **F1-Score (Weighted)**| 90.27% | **88.45%** | 83.55% | **81.44%** | **+7.01%** |
| **Macro ROC-AUC (OvR)** | 98.83% | **98.93%** | 97.58% | **97.09%** | **+0.0184** |

### B. Computational Efficiency & Deployment Footprint

| Efficiency Metric | ResNet50 (`clean_split_v1`) | MobileNetV2 (`clean_split_v1`) | Advantage / Efficiency Multiplier |
|---|:---:|:---:|---|
| **Total Parameters** | 24,113,798 | **2,587,462** | **MobileNetV2 has 9.3× fewer parameters** |
| **Trainable Head Parameters** | 526,086 | **329,478** | **MobileNetV2 has 37.4% fewer trainable weights** |
| **Non-Trainable Weights** | 23,587,712 | **2,257,984** | MobileNetV2 backbone is 10.4× smaller |
| **Keras Model File Size** | 96.64 MB | **12.96 MB** | **MobileNetV2 is 7.5× more compact** |
| **Training Duration (25 Epochs)**| 2,021.67 s (33.69 min) | **698.71 s (11.65 min)**| **MobileNetV2 trained 2.9× faster** |
| **Avg Inference Latency (N=50 CPU)**| 196.54 ms/image | **105.07 ms/image** | **MobileNetV2 is 1.87× faster per image** |
| **Median Inference Latency** | 195.81 ms/image | **104.89 ms/image** | MobileNetV2 is 1.87× faster |
| **Throughput (Images/sec)** | 5.09 FPS | **9.52 FPS** | **MobileNetV2 throughput is +87.0% higher** |

---

## 4. Per-Class Performance and Error Analysis

### Class-by-Class Metric Breakdown on Clean Test Partition

| Canonical Class Name | Support | ResNet50 Precision | ResNet50 Recall | ResNet50 F1 | MobileNetV2 Precision | MobileNetV2 Recall | MobileNetV2 F1 | F1 Delta ($\Delta$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Bacterial Leaf Blight** | 95 | 83.16% | 83.16% | 83.16% | 78.22% | 83.16% | 80.61% | **+2.55%** |
| **Brown Spot** | 96 | 82.73% | 94.79% | 88.35% | 77.78% | 87.50% | 82.35% | **+6.00%** |
| **Healthy Rice Leaf** | 98 | 97.94% | 96.94% | 97.44% | 97.80% | 90.82% | 94.18% | **+3.26%** |
| **Leaf Blast** | 93 | 93.06% | 72.04% | 81.21% | 72.62% | 65.59% | 68.93% | **+12.28%** |
| **Leaf Scald** | 94 | 84.54% | 87.23% | 85.86% | 82.50% | 70.21% | 75.86% | **+10.00%** |
| **Sheath Blight** | 94 | 91.92% | 96.81% | 94.30% | 81.13% | 91.49% | 86.00% | **+8.30%** |

### Confusion Matrix Analysis

#### ResNet50 Confusion Matrix (Rows = True, Columns = Predicted):
```text
                     Bacterial  BrownSpot  Healthy  LeafBlast  LeafScald  SheathBlight
Bacterial Leaf Blight:      79          5        0          4          6             1
Brown Spot:                  1         91        2          0          1             1
Healthy Rice Leaf:           0          3       95          0          0             0
Leaf Blast:                  6         10        0         67          7             3
Leaf Scald:                  7          1        0          1         82             3
Sheath Blight:               2          0        0          0          1            91
```

#### MobileNetV2 Confusion Matrix (Rows = True, Columns = Predicted):
```text
                     Bacterial  BrownSpot  Healthy  LeafBlast  LeafScald  SheathBlight
Bacterial Leaf Blight:      79          0        0          9          3             4
Brown Spot:                  1         84        1          5          3             2
Healthy Rice Leaf:           0          7       89          1          0             1
Leaf Blast:                  7         13        0         61          7             5
Leaf Scald:                 10          4        1          5         66             8
Sheath Blight:               4          0        0          3          1            86
```

### Detailed Diagnostic Insights:
1. **The Leaf Blast Challenge:** Both models experienced their lowest recall on **Leaf Blast** (ResNet50: 72.04%, MobileNetV2: 65.59%). In particular, Leaf Blast lesions were most frequently confused with **Brown Spot** (10 samples misclassified by ResNet50, 13 by MobileNetV2) due to similarities in necrotic spot morphology and leaf-edge discoloration. However, ResNet50's deeper receptive field retained significantly higher precision (93.06% vs 72.62%), yielding a +12.28% F1 advantage.
2. **Leaf Scald Differentiation:** MobileNetV2 confused 10 Leaf Scald samples with Bacterial Leaf Blight and 8 with Sheath Blight (recall dropped to 70.21%). ResNet50 maintained 87.23% recall (+10.00% F1 advantage) by resolving chevron-like scald banding.
3. **Healthy Leaf Reliability:** Both models proved reliable at detecting healthy foliage, achieving >97.8% precision, with only rare confusions against early-stage brown spots.

---

## 5. Limitations & Research Disclosures

1. **Non-Direct Comparability with Historical Split:** Do not interpret the difference between the original split (90.26% vs 83.65%) and clean split (88.60% vs 81.58%) as caused solely by duplicate exclusion. The test set membership was substantially restructured: 31 redundant duplicate files were excluded, burst sequence groups were assigned atomically, and the test set size changed from 575 to 570 images.
2. **Source-Level Foliage Correlation:** While all 31 exact duplicate groups and 125 confirmed burst-sequence relationships were strictly partitioned to prevent data leakage, agricultural field datasets can contain subtle natural correlations (e.g., leaves collected from the same rice paddy or lighting on the same day). The 79 candidate pairs flagged with $d \in [4, 6]$ in `DatasetAudit/reports/manual_review_required.csv` were preserved to avoid artificial data pruning.
3. **Execution Platform:** Training and evaluation were conducted on an AMD x86_64 CPU using native Windows TensorFlow with oneDNN acceleration. Numerical determinism was ensured across data pipelines via seed 42.

---

## 6. Artifact & File Directory Registry

All artifacts, checkpoints, configurations, logs, and evaluation reports have been verified on disk:

### ResNet50 Clean Experiment Artifacts
- **Model Checkpoint:** [`ResNet50/models/clean_split_v1/resnet50.keras`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/ResNet50/models/clean_split_v1/resnet50.keras) (96.64 MB)
- **Class Index JSON:** [`ResNet50/models/clean_split_v1/class_names.json`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/ResNet50/models/clean_split_v1/class_names.json)
- **Training Config:** [`ResNet50/outputs/clean_split_v1/training_config.json`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/ResNet50/outputs/clean_split_v1/training_config.json)
- **Model Summary:** [`ResNet50/outputs/clean_split_v1/model_summary.txt`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/ResNet50/outputs/clean_split_v1/model_summary.txt)
- **Training Curves:**
  - Accuracy: [`ResNet50/outputs/clean_split_v1/resnet50_training_accuracy.png`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/ResNet50/outputs/clean_split_v1/resnet50_training_accuracy.png)
  - Loss: [`ResNet50/outputs/clean_split_v1/resnet50_training_loss.png`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/ResNet50/outputs/clean_split_v1/resnet50_training_loss.png)
- **Evaluation Metrics:** [`ResNet50/outputs/clean_split_v1/metrics.json`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/ResNet50/outputs/clean_split_v1/metrics.json)
- **Classification Report:** [`ResNet50/outputs/clean_split_v1/classification_report.csv`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/ResNet50/outputs/clean_split_v1/classification_report.csv)
- **Confusion Matrix:**
  - CSV: [`ResNet50/outputs/clean_split_v1/resnet50_confusion_matrix.csv`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/ResNet50/outputs/clean_split_v1/resnet50_confusion_matrix.csv)
  - Plot: [`ResNet50/outputs/clean_split_v1/resnet50_confusion_matrix.png`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/ResNet50/outputs/clean_split_v1/resnet50_confusion_matrix.png)
- **Prediction Manifest:** [`ResNet50/outputs/clean_split_v1/test_predictions.csv`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/ResNet50/outputs/clean_split_v1/test_predictions.csv)
- **Efficiency Profile:** [`ResNet50/outputs/clean_split_v1/efficiency.json`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/ResNet50/outputs/clean_split_v1/efficiency.json)
- **Experiment Metadata:** [`ResNet50/outputs/clean_split_v1/experiment_metadata.json`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/ResNet50/outputs/clean_split_v1/experiment_metadata.json)

### MobileNetV2 Clean Experiment Artifacts
- **Model Checkpoint:** [`MobileNetV2/models/clean_split_v1/mobilenetv2.keras`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/MobileNetV2/models/clean_split_v1/mobilenetv2.keras) (12.96 MB)
- **Class Index JSON:** [`MobileNetV2/models/clean_split_v1/class_names.json`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/MobileNetV2/models/clean_split_v1/class_names.json)
- **Training Config:** [`MobileNetV2/outputs/clean_split_v1/training_config.json`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/MobileNetV2/outputs/clean_split_v1/training_config.json)
- **Training History:** [`MobileNetV2/outputs/clean_split_v1/training_history.csv`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/MobileNetV2/outputs/clean_split_v1/training_history.csv)
- **Model Summary:** [`MobileNetV2/outputs/clean_split_v1/model_summary.txt`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/MobileNetV2/outputs/clean_split_v1/model_summary.txt)
- **Training Curves:**
  - Accuracy: [`MobileNetV2/outputs/clean_split_v1/mobilenetv2_training_accuracy.png`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/MobileNetV2/outputs/clean_split_v1/mobilenetv2_training_accuracy.png)
  - Loss: [`MobileNetV2/outputs/clean_split_v1/mobilenetv2_training_loss.png`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/MobileNetV2/outputs/clean_split_v1/mobilenetv2_training_loss.png)
- **Evaluation Metrics:** [`MobileNetV2/outputs/clean_split_v1/metrics.json`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/MobileNetV2/outputs/clean_split_v1/metrics.json)
- **Classification Report:** [`MobileNetV2/outputs/clean_split_v1/classification_report.csv`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/MobileNetV2/outputs/clean_split_v1/classification_report.csv)
- **Confusion Matrix:**
  - CSV: [`MobileNetV2/outputs/clean_split_v1/mobilenetv2_confusion_matrix.csv`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/MobileNetV2/outputs/clean_split_v1/mobilenetv2_confusion_matrix.csv)
  - Plot: [`MobileNetV2/outputs/clean_split_v1/mobilenetv2_confusion_matrix.png`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/MobileNetV2/outputs/clean_split_v1/mobilenetv2_confusion_matrix.png)
- **Prediction Manifest:** [`MobileNetV2/outputs/clean_split_v1/test_predictions.csv`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/MobileNetV2/outputs/clean_split_v1/test_predictions.csv)
- **Efficiency Profile:** [`MobileNetV2/outputs/clean_split_v1/efficiency.json`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/MobileNetV2/outputs/clean_split_v1/efficiency.json)
- **Experiment Metadata:** [`MobileNetV2/outputs/clean_split_v1/experiment_metadata.json`](file:///C:/Gitwork/Rice%20diseases/Rice%20Diseases%20Classification/MobileNetV2/outputs/clean_split_v1/experiment_metadata.json)

---

## 7. Strategic Deployment Recommendations

1. **Server / Cloud Diagnostic Systems:** Deploy **ResNet50** where classification accuracy and subtle lesion discrimination (particularly differentiating Leaf Blast from Brown Spot) are paramount.
2. **Mobile / Edge / IoT Field Scanning:** Deploy **MobileNetV2** where model weight size (12.96 MB vs 96.64 MB), low memory footprint, and high-framerate processing (9.52 FPS vs 5.09 FPS on CPU) are required for on-device farmer applications.

