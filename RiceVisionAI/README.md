# RiceVision AI

A research dashboard and live prediction demo for **Rice Leaf Disease Classification** comparing ResNet50 and MobileNetV2 on a leakage-safe deduplicated dataset.

## Overview

This application was built as part of a university research project. It provides:

- **Overview dashboard** – dataset statistics, model summary, and feature highlights  
- **Model Comparison** – side-by-side metrics, per-class F1 chart, confusion matrices, and efficiency table  
- **Disease Classes** – visual browser of all six disease classes with descriptions and representative images  
- **Image Prediction** – upload a rice leaf image and receive a real-time prediction from either model  
- **Methodology** – full pipeline documentation from data collection to deployment  

## Project Structure

```
RiceVisionAI/
├── backend/
│   └── main.py              ← FastAPI application
├── frontend/
│   ├── src/
│   │   ├── pages/           ← Overview, Comparison, Classes, Predict, Methodology
│   │   ├── components/      ← Navbar, UI primitives
│   │   ├── api.ts           ← API client
│   │   ├── types.ts         ← TypeScript types
│   │   ├── App.tsx          ← Router
│   │   └── main.tsx         ← Entry point
│   ├── index.html
│   └── vite.config.ts
├── start_backend.bat        ← Start FastAPI (Windows)
├── start_frontend.bat       ← Start Vite dev server (Windows)
└── README.md
```

## Requirements

### Backend
- Python 3.11+ with the ResNet50 project virtual environment  
  (`ResNet50/.venv` must contain TensorFlow, Keras, FastAPI, uvicorn, Pillow, NumPy)
- Trained model checkpoints:
  - `ResNet50/models/clean_split_v1/resnet50.keras`
  - `MobileNetV2/models/clean_split_v1/mobilenetv2.keras`

### Frontend
- Node.js 18+ and npm

## Installation

### 1. Install backend dependencies (if not already installed)

```powershell
# From the Rice Diseases Classification root
& "ResNet50\.venv\Scripts\python.exe" -m pip install fastapi "uvicorn[standard]" python-multipart
```

### 2. Install frontend dependencies

```powershell
cd RiceVisionAI\frontend
npm install
```

## Running the Application

Open **two terminal windows**.

### Terminal 1 – Start the backend

```powershell
# From RiceVisionAI\
.\start_backend.bat
```

Or manually:

```powershell
$py = "ResNet50\.venv\Scripts\python.exe"
& $py -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload --app-dir "RiceVisionAI"
```

The API will be available at: `http://localhost:8000`  
Interactive API docs: `http://localhost:8000/docs`

### Terminal 2 – Start the frontend

```powershell
# From RiceVisionAI\
.\start_frontend.bat
```

Or manually:

```powershell
cd frontend
npm run dev
```

Open your browser at: **http://localhost:5173**

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Backend and model availability check |
| GET | `/api/classes` | Class names, descriptions, representative images |
| GET | `/api/models` | Model metadata, metrics, efficiency data |
| GET | `/api/metrics` | Per-class metrics and confusion matrices |
| POST | `/api/predict?model=resnet50` | Predict from uploaded image |
| POST | `/api/predict?model=mobilenetv2` | Predict from uploaded image |

### Predict Request

```
POST /api/predict?model=resnet50
Content-Type: multipart/form-data
Body: file = <image.jpg>
```

### Predict Response

```json
{
  "model": "resnet50",
  "predicted_class": "Healthy Rice Leaf",
  "predicted_index": 2,
  "confidence": 0.9812,
  "scores": {
    "Bacterial Leaf Blight": 0.0021,
    "Brown Spot": 0.0041,
    "Healthy Rice Leaf": 0.9812,
    ...
  },
  "inference_ms": 198.4,
  "disclaimer": "..."
}
```

## Models Used

| Model | Accuracy | Macro F1 | ROC-AUC | Size | Avg Inference |
|-------|----------|----------|---------|------|---------------|
| ResNet50 | 88.60% | 88.39% | 98.93% | 96.64 MB | ~197 ms |
| MobileNetV2 | 81.58% | 81.32% | 97.09% | 12.96 MB | ~105 ms |

Evaluated on 570 test images (Clean Split v1). Hardware: CPU (AMD64).

## Notes

- **First prediction may be slow** — models are loaded lazily on the first request. Subsequent predictions are faster.
- Images are processed in-memory. File system paths are never exposed to the browser.
- This is an **educational research prototype**, not a certified agricultural diagnostic system.
- Preprocessing exactly matches training code:
  - ResNet50: `tensorflow.keras.applications.resnet50.preprocess_input` (zero-centered BGR ImageNet means)
  - MobileNetV2: `tensorflow.keras.applications.mobilenet_v2.preprocess_input` (scaled to [−1, 1])

## Production Deployment (Option 2: Vercel + Render / Railway)

This setup deploys the frontend as a static SPA on **Vercel** and the FastAPI inference engine on **Render** (or **Railway**).

### Part 1: Deploy Backend (Render or Railway)

1. **Push your code to GitHub**:
   Ensure Git LFS is enabled for `.keras` files:
   ```powershell
   git lfs install
   git add .gitattributes
   git commit -m "Configure Git LFS"
   ```

2. **Create a Web Service on [Render.com](https://render.com/)**:
   - Choose **New Web Service** and connect your GitHub repository.
   - **Environment**: `Python 3`
   - **Root Directory**: Leave blank (repo root) or set to the project root.
   - **Build Command**:
     ```bash
     pip install -r RiceVisionAI/backend/requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn RiceVisionAI.backend.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type**: Minimum 1 GB RAM (Free or Starter tier).
   - **Environment Variables**:
     - `PYTHON_VERSION`: `3.11.9`
     - `ALLOWED_ORIGINS`: `https://your-frontend-domain.vercel.app` (or `*`)

3. Copy your live backend URL (e.g. `https://ricevision-backend.onrender.com`).

---

### Part 2: Deploy Frontend (Vercel)

1. Log into [Vercel.com](https://vercel.com/) and click **Add New > Project**.
2. Import your GitHub repository.
3. Configure project settings:
   - **Framework Preset**: `Vite`
   - **Root Directory**: `RiceVisionAI/frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. **Environment Variables**:
   - Name: `VITE_API_URL`
   - Value: `https://ricevision-backend.onrender.com` (your backend URL from Part 1, without trailing slash)
5. Click **Deploy**.

---

### Part 3: Test Live App

- Open your Vercel deployment URL.
- Test the **Image Prediction** tab by uploading a rice leaf image to confirm end-to-end inference between Vercel and your backend.

