# Research Audit Report: ResNet50 vs MobileNetV2 Rice Leaf Disease Classification

**Date of Audit:** October 10, 2026  
**Auditor:** Machine Learning Research Auditor  
**Scope:** Experimental setup, data integrity, training configuration, evaluation validity, and computational efficiency.  
**Audited Projects:**
- **ResNet50:** `C:\Gitwork\Rice diseases\Rice Diseases Classification\ResNet50`
- **MobileNetV2:** `C:\Gitwork\Rice diseases\Rice Diseases Classification\MobileNetV2`

---

## 1. Executive Verdict

### **Verdict: PASS WITH LIMITATIONS**

### Justification:
1. **Flawless Experimental Isolation & Alignment:** The MobileNetV2 project did **not** regenerate a split or reorder classes. Both projects use the **exact same 2,871 training, 383 validation, and 575 test images** in identical order with identical canonical class mappings (`[0..5]`). There is **zero path overlap** across splits.
2. **Correct Preprocessing & Architecture:** Both architectures strictly use their corresponding Keras preprocessing (`preprocess_input` for ResNet50: zero-centered ImageNet means; `preprocess_input` for MobileNetV2: `[-1, 1]` scaling). Data augmentation is isolated strictly to the training pipeline after splitting.
3. **Transparent & Reproducible Metrics:** Metrics were recomputed directly from the saved prediction manifests (`test_predictions.csv`) without retraining and matched the reported test set metrics.
4. **Identified Limitations (Non-fatal):**
   - **Pre-existing Dataset Duplication:** 9 out of 575 test images (1.57%) share identical SHA-256 byte hashes with images in the training set due to duplicate files present in the raw dataset source (e.g., `IMG_... - Copy.jpg`). This affects both models symmetrically (+0.15% on ResNet50, +0.08% on MobileNetV2) and does not invalidate the relative comparison, but must be disclosed.
   - **Evaluation Metric Logging Format:** ResNet50's `outputs/test_predictions.csv` logged hard class predictions and confidence without per-class probability columns, preventing post-hoc recomputation of multiclass ROC-AUC from the CSV alone (though ROC-AUC was computed by `evaluate.py` during execution and stored in `metrics.json`).

---

## 2. Side-by-Side Configuration Table

| Parameter / Setting | ResNet50 | MobileNetV2 | Status | Auditor Interpretation |
|---|---|---|---|---|
| **Input Image Dimensions** | `(224, 224, 3)` | `(224, 224, 3)` | **MATCH** | Standard transfer learning resolution. |
| **Color Order / Scale** | BGR, `[-123.68, 151.06]` | RGB, `[-1.0, 1.0]` | **MATCH (Architecture-specific)** | Both models use their native Keras `preprocess_input`. |
| **Batch Size** | `32` | `32` | **MATCH** | Identical mini-batch dynamics. |
| **Optimizer** | Adam | Adam | **MATCH** | Fair comparison baseline. |
| **Initial Learning Rate** | `0.0001` (`1e-4`) | `0.0001` (`1e-4`) | **MATCH** | Identical initial optimization scale. |
| **Loss Function** | Sparse Categorical Cross-Entropy | Sparse Categorical Cross-Entropy | **MATCH** | Identical objective function. |
| **Maximum Epochs** | `25` | `25` | **MATCH** | Equal budget. |
| **Actual Epochs Trained** | `25` | `25` | **MATCH** | Neither triggered early stopping termination. |
| **Early Stopping Monitor / Patience** | `val_loss`, patience=5, restore best | `val_loss`, patience=5, restore best | **MATCH** | Identical criteria. |
| **LR Scheduler** | ReduceLROnPlateau (`factor=0.5, pat=2, min=1e-6`) | ReduceLROnPlateau (`factor=0.5, pat=2, min=1e-6`) | **MATCH** | ResNet50 stepped at Ep 21; MobileNetV2 stepped at Ep 21. |
| **Model Checkpoint** | Save best on `val_loss` | Save best on `val_loss` | **MATCH** | Best validation weights retained. |
| **Selected Checkpoint Epoch** | Epoch 24 (`val_loss`: 0.3297) | Epoch 25 (`val_loss`: 0.5475) | **DIFFERENT (Expected)** | Architecture convergence rates differ. |
| **Pretrained Weights** | ImageNet | ImageNet | **MATCH** | Identical pretraining source. |
| **Backbone Frozen?** | Yes (`trainable = False`) | Yes (`trainable = False`) | **MATCH** | Both evaluate fixed feature extraction heads. |
| **Pooling Layer** | `GlobalAveragePooling2D` | `GlobalAveragePooling2D` | **MATCH** | Standard spatial pooling. |
| **Dense Head** | `Dense(256, relu)` | `Dense(256, relu)` | **MATCH** | Identical capacity in top classifier. |
| **Dropout Rate** | `0.3` | `0.3` | **MATCH** | Identical regularization strength. |
| **Output Activation / Units** | `Dense(6, softmax)` | `Dense(6, softmax)` | **MATCH** | Identical 6-class output layer. |
| **Data Augmentation (Train Only)** | Flip, Rot(0.1), Zoom(0.1), Trans(0.08) | Flip, Rot(0.1), Zoom(0.1), Trans(0.08) | **MATCH** | Exact same pipeline parameters. |
| **Class Weighting / Sampling** | None (uniform) | None (uniform) | **MATCH** | Naturally balanced classes (~630–650/class). |

---

## 3. Dataset and Split Verification

### A. Split Manifests and Isolation
Inspection of `train.csv`, `validation.csv`, and `test.csv` in both project directories confirms:
- **Exact Path Match:** 100% of image paths across all three CSV files are identical between ResNet50 and MobileNetV2.
- **Identical Order:** Rows appear in identical sequence.
- **Physical Image Existence:** All **3,829** referenced images exist on disk with zero missing files.
- **Zero Path Overlap:**
  - $\text{Train} \cap \text{Val} = 0$
  - $\text{Train} \cap \text{Test} = 0$
  - $\text{Val} \cap \text{Test} = 0$

### B. Sample Distribution by Class

| Canonical Class Name | Canonical Index | Train (75%) | Validation (10%) | Test (15%) | Total Valid Images |
|---|:---:|:---:|:---:|:---:|:---:|
| **Bacterial Leaf Blight** | 0 | 476 | 64 | 96 | 636 |
| **Brown Spot** | 1 | 484 | 65 | 97 | 646 |
| **Healthy Rice Leaf** | 2 | 490 | 65 | 98 | 653 |
| **Leaf Blast** | 3 | 476 | 63 | 95 | 634 |
| **Leaf Scald** | 4 | 471 | 63 | 94 | 628 |
| **Sheath Blight** | 5 | 474 | 63 | 95 | 632 |
| **Total** | — | **2,871** | **383** | **575** | **3,829** |

### C. Duplicate & Data Leakage Analysis (SHA-256 Hash Audit)
- **Within Train:** 15 redundant duplicate files found (2,856 unique hashes out of 2,871 images).
- **Within Validation:** 2 redundant duplicate files found (381 unique hashes out of 383 images).
- **Within Test:** 0 redundant duplicate files found (575 unique hashes out of 575 images).
- **Cross-Split Hash Overlap:**
  - **Train & Validation:** 5 exact duplicate file pairs.
  - **Train & Test:** **9 exact duplicate file pairs** (1.57% of test set).
  - **Validation & Test:** 0 duplicate file pairs.

**Specific Test Leakage Pairs Found in Raw Dataset:**
1. `IMG_20231018_143841 - Copy.jpg` (Train) $\leftrightarrow$ `IMG_20231018_143841.jpg` (Test)
2. `Brown_spot (132).jpg` (Train) $\leftrightarrow$ `Brown_spot (131).jpg` (Test)
3. `IMG_20231018_143750.jpg` (Train) $\leftrightarrow$ `IMG_20231018_143750 - Copy.jpg` (Test)
4. `IMG_20231018_143844 - Copy.jpg` (Train) $\leftrightarrow$ `IMG_20231018_143844.jpg` (Test)
5. `IMG_20231018_143839.jpg` (Train) $\leftrightarrow$ `IMG_20231018_143839 - Copy.jpg` (Test)
6. `IMG_20231018_143855.jpg` (Train) $\leftrightarrow$ `IMG_20231018_143855 - Copy.jpg` (Test)
7. `IMG_20231006_163221.jpg` (Train) $\leftrightarrow$ `IMG_20231006_163221 - Copy.jpg` (Test)
8. `Healthy_rice_leaf (7).jpg` (Train) $\leftrightarrow$ `Healthy_rice_leaf (6).jpg` (Test)
9. `Brown_spot (134).jpg` (Train) $\leftrightarrow$ `Brown_spot (133).jpg` (Test)

*Impact Assessment:* Both models classified these leaked duplicates with near-perfect accuracy (ResNet50: 9/9, MobileNetV2: 8/9). When re-evaluating exclusively on the **566 strictly unique test images**:
- **ResNet50 Clean Accuracy:** **90.11%** (vs. 90.26% reported, $\Delta = -0.15\%$)
- **MobileNetV2 Clean Accuracy:** **83.57%** (vs. 83.65% reported, $\Delta = -0.08\%$)
The performance gap remains essentially unchanged ($6.54\%$ vs. $6.61\%$), meaning the relative research comparison remains sound.

---

## 4. Seed and Reproducibility Findings

### Code Locations:
- **ResNet50:** `src/train.py`, `src/prepare_dataset.py`
- **MobileNetV2:** `src/train.py`, `src/prepare_dataset.py`

### Actual Seed Implementations:
```python
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
```

### Reproducibility Scope & Limitations:
- **Seed value:** Fixed at `42` in both projects.
- **Stratified Split:** Split was executed once with `seed=42` in ResNet50, then imported without re-shuffling into MobileNetV2.
- **Nondeterminism Caveat:** Multi-threaded CPU operations (such as multi-threaded BLAS/oneDNN kernels and `tf.data.AUTOTUNE` parallel mapping) introduce minor floating-point nondeterminism across runs. Setting the seed guarantees identical weight initialization and initial dataset shuffling order, but bit-exact numerical reproducibility across hardware is not guaranteed without `tf.config.experimental.enable_op_determinism()`.

---

## 5. Evaluation and Efficiency Verification

### A. Independent Test Metrics Verification (575 Test Samples)

| Metric | ResNet50 Reported | ResNet50 Recomputed | MobileNetV2 Reported | MobileNetV2 Recomputed | Match Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **Accuracy** | **0.9026** | **0.9026** | **0.8365** | **0.8365** | **EXACT MATCH** |
| **Precision (macro)** | **0.9024** | **0.9024** | **0.8400** | **0.8400** | **EXACT MATCH** |
| **Recall (macro)** | **0.9020** | **0.9020** | **0.8356** | **0.8356** | **EXACT MATCH** |
| **F1-Score (macro)** | **0.9021** | **0.9021** | **0.8347** | **0.8347** | **EXACT MATCH** |
| **ROC-AUC (macro OvR)**| **0.9883** | *Unverified from CSV\** | **0.9758** | **0.9758** | **VERIFIED** |

*\*Note on ResNet50 ROC-AUC:* ResNet50's `test_predictions.csv` stores only `image_path`, `true_class`, `predicted_class`, `confidence`, and `correct`. The full 6-column probability array was passed directly in-memory to `roc_auc_score` inside `src/evaluate.py` and logged to `metrics.json`, but omitted from the CSV export. In MobileNetV2, all 6 class probability columns were logged to the CSV.

### B. Efficiency Benchmarking Verification

| Metric | ResNet50 | MobileNetV2 | Comparison / Efficiency Gain |
|---|---|---|---|
| **Total Parameters** | 24,113,798 | 2,587,462 | **MobileNetV2 has 9.3× fewer parameters** |
| **Trainable Parameters** | 526,086 | 329,478 | **MobileNetV2 has 37.4% fewer trainable weights** |
| **Non-Trainable Parameters** | 23,587,712 | 2,257,984 | **MobileNetV2 backbone is 10.4× smaller** |
| **Model File Size** | 96.64 MB | 12.96 MB | **MobileNetV2 is 7.5× more compact** |
| **Training Time (25 Epochs)** | 2,259.16 s (37.65 min) | 718.73 s (11.98 min) | **MobileNetV2 trained 3.1× faster** |
| **Hardware Used** | AMD CPU, Windows native | AMD CPU, Windows native | Identical test environment |
| **Inference Latency (Avg)** | 206.09 ms/image | 129.52 ms/image | **MobileNetV2 is 1.59× faster per image** |
| **Inference Latency (Median)**| 199.99 ms/image | 126.99 ms/image | **MobileNetV2 is 1.57× faster** |
| **Throughput (FPS)** | 4.85 FPS | 7.72 FPS | **MobileNetV2 throughput +59.2%** |
| **Timing Methodology** | In-memory warmup + single-image forward pass | In-memory warmup + single-image forward pass | Comparable methodologies |

---

## 6. Findings by Severity

### Major
- **Finding 1: Cross-Split Exact File Duplicates in Raw Dataset (Data Leakage Risk)**
  - *Location:* Raw image files under `dataset/` (affecting both projects).
  - *Detail:* 9 test images are exact bit-for-bit SHA-256 duplicates of training images (originating from renamed/duplicate files in the source dataset, e.g., `IMG_... - Copy.jpg` or duplicate numbering like `Brown_spot (131).jpg` vs `(132).jpg`).
  - *Impact:* Confirmed minor metric inflation (+0.15% on ResNet50, +0.08% on MobileNetV2). The comparative delta is unaffected (ResNet50 leads by 6.54% on strictly unique test data vs 6.61% reported).
  - *Status:* Confirmed dataset artifact.

### Minor
- **Finding 2: ResNet50 Test Predictions CSV Lacks Full Class Probabilities**
  - *Location:* `ResNet50/outputs/test_predictions.csv`
  - *Detail:* While MobileNetV2 logs `prob_Bacterial_Leaf_Blight`, `prob_Brown_Spot`, etc., ResNet50 only logged `confidence` (the maximum probability).
  - *Impact:* Multiclass ROC-AUC for ResNet50 cannot be recomputed post-hoc from the CSV alone without running model inference.
  - *Status:* Confirmed formatting discrepancy.

- **Finding 3: Latency Profiling Repetition Count Discrepancy**
  - *Location:* ResNet50 evaluated efficiency over 30 test samples, whereas MobileNetV2 evaluated over 50 test samples.
  - *Impact:* Both sample counts are sufficient for central limit convergence on CPU, but standard practice requires identical sample counts (e.g., $N=50$) for reporting.
  - *Status:* Minor procedural difference.

### Informational
- **Finding 4: TensorFlow Native Windows CPU Execution**
  - *Location:* Native Windows TensorFlow $\ge 2.11$ environment.
  - *Detail:* TensorFlow on native Windows lacks direct CUDA support. Both models trained on CPU. Because both models experienced the exact same hardware constraints, the training time and CPU inference comparisons are completely fair.

---

## 7. Prioritized Action Plan

### 1. Must Fix Before Reporting Results (Documentation Only — No Retraining Required)
- **Disclose the Raw Dataset Duplication:** In research papers or project presentations, explicitly disclose that hash auditing revealed 9 duplicate image pairs across train and test splits (1.57% of the test set), and report both the overall test accuracy (90.26% vs 83.65%) and the clean subset accuracy (90.11% vs 83.57%).

### 2. Should Fix to Strengthen Reliability (Optional Re-evaluation)
- **Update ResNet50 `test_predictions.csv` with Probability Columns:** If external evaluators require verifying ROC-AUC directly from static CSV files, re-run `src/evaluate.py` in ResNet50 with probability export enabled (requires loading `resnet50.keras` without retraining).

### 3. Optional Long-term Improvements
- **Perceptual Deduplication Pipeline:** For future dataset expansions, apply perceptual hash (`pHash`) deduplication prior to generating stratified train/validation/test splits.

---

## 8. Final Research-Readiness Statement

1. **Can the existing ResNet50 and MobileNetV2 results be fairly compared?**  
   **Yes.** Both experiments share the exact same dataset splits, image paths, class mappings, image resolution, augmentation rules, optimization framework, and evaluation metrics.
2. **Are both models evaluated on the exact same test images?**  
   **Yes.** Both models were evaluated on the exact same 575 test images in identical order.
3. **Is there evidence of data leakage?**  
   **No synthetic or procedural data leakage exists** (no test augmentation, no split leakage from code). However, **9 pre-existing duplicate files (1.57%) exist in the source dataset across train and test splits**. Its symmetrical impact has been quantified and does not alter the scientific conclusion.
4. **Are the reported metrics independently verifiable?**  
   **Yes.** Test accuracy, macro precision, macro recall, and macro F1 were recomputed from raw prediction records and verified to be a 100% exact match. MobileNetV2 ROC-AUC was also verified to match.
5. **Is the seed configuration adequate for the stated experimental claims?**  
   **Yes.** Seed 42 was consistently applied across both pipelines.
6. **Is another training run necessary?**  
   **No.** Retraining is not necessary. The existing model checkpoints, training histories, and test predictions are consistent, scientifically defensible, and ready for publication or comparative web application integration.
7. **What limitations must be disclosed?**  
   Disclose that training occurred on CPU via native Windows TensorFlow, and document the 9-image raw dataset duplicate pair finding alongside the clean test accuracy figures.

