# Rice Leaf Disease Classification: ResNet50 vs. MobileNetV2

A controlled empirical comparison of **ResNet50** and **MobileNetV2** for classifying rice leaf diseases, evaluated on a deduplicated, leakage-resistant dataset split with reproducible training pipelines and interactive web dashboards.

---

## 📊 Benchmark Results (Clean Split v1)

Both models were evaluated on the exact same **570 held-out test images** under identical experimental protocols.

| Metric | ResNet50 | MobileNetV2 | Difference (Δ) | Winner |
|---|:---:|:---:|:---:|:---:|
| **Test Accuracy** | **88.60%** | 81.58% | +7.02% | **ResNet50** |
| **Macro F1-Score** | **88.39%** | 81.32% | +7.07% | **ResNet50** |
| **Macro Precision** | **88.89%** | 81.67% | +7.22% | **ResNet50** |
| **Macro Recall** | **88.50%** | 81.46% | +7.04% | **ResNet50** |
| **ROC-AUC (Macro)** | **98.93%** | 97.09% | +1.84% | **ResNet50** |
| **Model Checkpoint Size** | 96.64 MB | **12.96 MB** | −86.6% | **MobileNetV2** |
| **Total Parameters** | 24,113,798 | **2,587,462** | −89.3% | **MobileNetV2** |
| **Training Time (30 ep)** | 33.69 min | **11.65 min** | −65.4% | **MobileNetV2** |
| **Avg Inference Latency (CPU)** | 196.5 ms | **105.1 ms** | −46.5% | **MobileNetV2** |
| **Throughput (FPS on CPU)** | 5.09 fps | **9.52 fps** | +87.0% | **MobileNetV2** |

> **Key Finding:** ResNet50 delivers +7.0% higher classification accuracy and F1 score, while MobileNetV2 is 7.5× smaller in storage and 1.87× faster on edge/CPU hardware.

---

## 🌾 The 6 Disease Classes

The dataset covers six critical rice foliage conditions:

| Index | Class Name | Description | ResNet50 F1 | MobileNetV2 F1 |
|:---:|---|---|:---:|:---:|
| `0` | **Bacterial Leaf Blight** | Caused by *Xanthomonas oryzae*; marginal yellowing and wilting. | 83.16% | 80.61% |
| `1` | **Brown Spot** | Caused by *Cochliobolus miyabeanus*; oval brown spots with yellow halos. | 88.35% | 82.35% |
| `2` | **Healthy Rice Leaf** | Clean, symptom-free foliage; baseline control. | **97.44%** | **94.18%** |
| `3` | **Leaf Blast** | Caused by *Magnaporthe oryzae*; diamond-shaped necrotic lesions. | 81.21% | 68.93% |
| `4` | **Leaf Scald** | Caused by *Microdochium oryzae*; zonate tan scalds from leaf tips. | 85.86% | 75.86% |
| `5` | **Sheath Blight** | Caused by *Rhizoctonia solani*; elliptical gray-white sheath lesions. | 94.30% | 86.00% |

---

## 🛡️ Dataset Integrity & Leakage Prevention Pipeline

Data leakage from duplicate images across train/val/test splits artificially inflates test metrics. A rigorous dataset audit was conducted before retraining:

1. **Exact Duplicate Detection**: MD5 hash analysis identified 31 exact duplicate groups (62 redundant files).
2. **Deduplication Strategy**: Exactly 31 redundant copies were excluded from manifests without altering raw source images on disk.
3. **Near-Duplicate Audit**: Perceptual hashing (pHash, Hamming distance ≤ 10) identified visually similar pairs with rotations/crops for documented inspection.
4. **Stratified Leakage-Safe Splits (Clean Split v1)**:
   - **Total Clean Images:** 3,798 images
   - **Train:** 2,847 images (74.96%)
   - **Validation:** 381 images (10.03%)
   - **Test:** 570 images (15.01%)

---

## 📁 Repository Structure

```
Rice Diseases Classification/
├── ResNet50/                     # ResNet50 training, evaluation, checkpoints
│   ├── dataset/                  # Primary dataset directory (6 classes)
│   ├── models/clean_split_v1/    # Checkpoint (resnet50.keras) & class_names.json
│   ├── outputs/clean_split_v1/   # Metrics, confusion matrix, classification report
│   ├── src/                      # train.py, evaluate.py, prepare_dataset.py
│   └── .venv/                    # Shared Python 3.11 virtual environment
│
├── MobileNetV2/                  # MobileNetV2 training, evaluation, checkpoints
│   ├── dataset/                  # NTFS Junction linked to ResNet50/dataset
│   ├── models/clean_split_v1/    # Checkpoint (mobilenetv2.keras) & class_names.json
│   ├── outputs/clean_split_v1/   # Metrics, confusion matrix, classification report
│   └── src/                      # train.py, evaluate.py, prepare_dataset.py
│
├── DatasetAudit/                 # Full audit logs & reproducible manifests
│   ├── manifests/                # clean_train.csv, clean_validation.csv, clean_test.csv
│   ├── reports/                  # audit_report.md, model_comparison_report.md
│   └── review/                   # Near-duplicate visual contact sheets
│
├── RiceVisionAI/                 # Full-Stack Web Application (FastAPI + React)
│   ├── backend/                  # FastAPI inference engine with lazy model loading
│   ├── frontend/                 # React 19 + TypeScript + Tailwind CSS dashboard
│   ├── start_backend.bat         # Start backend on http://localhost:8000
│   └── start_frontend.bat        # Start frontend on http://localhost:5173
│
├── RiceVisionAI-Static/          # Zero-Server Static Showcase
│   ├── src/staticData.ts         # Pre-compiled benchmark metrics & thumbnails
│   ├── start_static.bat          # Preview locally on http://localhost:4173
│   └── README.md                 # 2-minute Vercel / Netlify deployment guide
│
└── README.md                     # Root project documentation (this file)
```

---

## 🚀 Interactive Dashboards

### Option A: RiceVisionAI (Full-Stack with Live Inference)
Provides live real-time image prediction with your trained `.keras` checkpoints.

```powershell
# Terminal 1: Backend
cd "RiceVisionAI"
.\start_backend.bat

# Terminal 2: Frontend
cd "RiceVisionAI"
.\start_frontend.bat
```
- Frontend UI: `http://localhost:5173`
- FastAPI Docs: `http://localhost:8000/docs`

### Option B: RiceVisionAI-Static (Zero-Server Cloud Showcase)
Designed for free, instant cloud deployment on **Vercel** or **Netlify** without requiring a Python server or GPU:
- All benchmark metrics, charts, and confusion matrices are bundled into static assets.
- Includes an interactive benchmark sample inspector.
- Preview locally:
  ```powershell
  cd "RiceVisionAI-Static"
  .\start_static.bat
  ```

---

## ⚙️ Reproducing Training & Evaluation

All commands use the project virtual environment located in `ResNet50\.venv`.

### 1. Training from Scratch

```powershell
# Train ResNet50
cd "ResNet50"
& ".\.venv\Scripts\python.exe" src/train.py `
  --train-csv "..\DatasetAudit\manifests\clean_train.csv" `
  --val-csv "..\DatasetAudit\manifests\clean_validation.csv" `
  --class-names "..\DatasetAudit\manifests\class_names.json" `
  --dataset-root "dataset" `
  --output-dir "outputs/clean_split_v1" `
  --model-dir "models/clean_split_v1" `
  --epochs 30 --batch-size 32

# Train MobileNetV2
cd "..\MobileNetV2"
& "..\ResNet50\.venv\Scripts\python.exe" src/train.py `
  --train-csv "..\DatasetAudit\manifests\clean_train.csv" `
  --val-csv "..\DatasetAudit\manifests\clean_validation.csv" `
  --class-names "..\DatasetAudit\manifests\class_names.json" `
  --dataset-root "dataset" `
  --output-dir "outputs/clean_split_v1" `
  --model-dir "models/clean_split_v1" `
  --epochs 30 --batch-size 32
```

### 2. Evaluating on Held-Out Test Set

```powershell
# Evaluate ResNet50
cd "ResNet50"
& ".\.venv\Scripts\python.exe" src/evaluate.py `
  --model "models/clean_split_v1/resnet50.keras" `
  --test-csv "..\DatasetAudit\manifests\clean_test.csv" `
  --dataset-root "dataset" `
  --class-names "..\DatasetAudit\manifests\class_names.json" `
  --output-dir "outputs/clean_split_v1"

# Evaluate MobileNetV2
cd "..\MobileNetV2"
& "..\ResNet50\.venv\Scripts\python.exe" src/evaluate.py `
  --model "models/clean_split_v1/mobilenetv2.keras" `
  --test-csv "..\DatasetAudit\manifests\clean_test.csv" `
  --dataset-root "dataset" `
  --class-names "..\DatasetAudit\manifests\class_names.json" `
  --output-dir "outputs/clean_split_v1"
```

---

## ⚠️ Academic & Practical Disclaimer

This system is an **educational research prototype** developed for comparative deep learning research. Model prediction scores are raw softmax probabilities and may vary under unstructured field conditions (lighting variance, overlapping disease symptoms, non-standard camera angles). This project is not certified as a commercial or agricultural diagnostic device.

