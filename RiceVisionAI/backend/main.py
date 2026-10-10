"""
RiceVision AI – FastAPI Backend
Rice Leaf Disease Classification Research Dashboard
"""

import json
import os
import time
import uuid
import base64
import io
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image

# ─── Path constants ──────────────────────────────────────────────────────────

# Base data directory: allows overriding via DATA_ROOT environment variable
DEFAULT_BASE = Path(__file__).resolve().parent.parent.parent  # Rice Diseases Classification/
DATA_ROOT = Path(os.environ.get("DATA_ROOT", DEFAULT_BASE)).resolve()

RESNET_DIR = Path(os.environ.get("RESNET_DIR", DATA_ROOT / "ResNet50")).resolve()
MOBILENET_DIR = Path(os.environ.get("MOBILENET_DIR", DATA_ROOT / "MobileNetV2")).resolve()
DATASET_DIR = Path(os.environ.get("DATASET_DIR", RESNET_DIR / "dataset")).resolve()

RESNET_CHECKPOINT = Path(os.environ.get("RESNET_CHECKPOINT", RESNET_DIR / "models" / "clean_split_v1" / "resnet50.keras")).resolve()
MOBILENET_CHECKPOINT = Path(os.environ.get("MOBILENET_CHECKPOINT", MOBILENET_DIR / "models" / "clean_split_v1" / "mobilenetv2.keras")).resolve()

RESNET_METRICS = Path(os.environ.get("RESNET_METRICS", RESNET_DIR / "outputs" / "clean_split_v1" / "metrics.json")).resolve()
MOBILENET_METRICS = Path(os.environ.get("MOBILENET_METRICS", MOBILENET_DIR / "outputs" / "clean_split_v1" / "metrics.json")).resolve()

RESNET_EFFICIENCY = Path(os.environ.get("RESNET_EFFICIENCY", RESNET_DIR / "outputs" / "clean_split_v1" / "efficiency.json")).resolve()
MOBILENET_EFFICIENCY = Path(os.environ.get("MOBILENET_EFFICIENCY", MOBILENET_DIR / "outputs" / "clean_split_v1" / "efficiency.json")).resolve()

RESNET_REPORT = Path(os.environ.get("RESNET_REPORT", RESNET_DIR / "outputs" / "clean_split_v1" / "classification_report.csv")).resolve()
MOBILENET_REPORT = Path(os.environ.get("MOBILENET_REPORT", MOBILENET_DIR / "outputs" / "clean_split_v1" / "classification_report.csv")).resolve()

RESNET_CONFUSION = Path(os.environ.get("RESNET_CONFUSION", RESNET_DIR / "outputs" / "clean_split_v1" / "resnet50_confusion_matrix.csv")).resolve()
MOBILENET_CONFUSION = Path(os.environ.get("MOBILENET_CONFUSION", MOBILENET_DIR / "outputs" / "clean_split_v1" / "mobilenetv2_confusion_matrix.csv")).resolve()

CLASS_NAMES_PATH = Path(os.environ.get("CLASS_NAMES_PATH", RESNET_DIR / "models" / "clean_split_v1" / "class_names.json")).resolve()

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RiceVision AI",
    description="Rice Leaf Disease Classification – ResNet50 vs MobileNetV2",
    version="1.0.0",
)

# Parse allowed origins from environment variable or use local defaults
env_origins = os.environ.get("ALLOWED_ORIGINS", "")
if env_origins.strip() == "*":
    origins = ["*"]
elif env_origins.strip():
    origins = [orig.strip() for orig in env_origins.split(",") if orig.strip()]
else:
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=True if origins != ["*"] else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Lazy model cache ─────────────────────────────────────────────────────────

_models: dict = {}
_class_names: Optional[list] = None


def _get_class_names() -> list:
    global _class_names
    if _class_names is None:
        with open(CLASS_NAMES_PATH, "r") as f:
            _class_names = json.load(f)
    return _class_names


def _load_model(name: str):
    """Load and cache a model by name ('resnet50' or 'mobilenetv2')."""
    if name in _models:
        return _models[name]

    import tensorflow as tf

    if name == "resnet50":
        path = RESNET_CHECKPOINT
    elif name == "mobilenetv2":
        path = MOBILENET_CHECKPOINT
    else:
        raise ValueError(f"Unknown model: {name}")

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model = tf.keras.models.load_model(str(path))
    _models[name] = model
    return model


def _preprocess(img: Image.Image, model_name: str) -> np.ndarray:
    """Resize and preprocess a PIL Image for the given model."""
    if model_name == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    else:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def _load_csv_as_list(path: Path) -> list[dict]:
    """Read a CSV file and return list of row dicts."""
    import csv
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


# ─── Representative class images cache ───────────────────────────────────────

_class_images: Optional[dict] = None


def _get_class_images() -> dict:
    """Return a dict mapping class name → base64-encoded JPEG string."""
    global _class_images
    if _class_images is not None:
        return _class_images

    class_names = _get_class_names()
    result = {}
    for cls in class_names:
        class_dir = DATASET_DIR / cls
        if not class_dir.exists():
            result[cls] = None
            continue
        images = sorted(class_dir.glob("*.jpg"))
        # Pick a non-augmented image when possible
        real_images = [p for p in images if not p.name.startswith("aug_")]
        candidate = real_images[0] if real_images else (images[0] if images else None)
        if candidate is None:
            result[cls] = None
            continue
        with Image.open(candidate) as img:
            img_thumb = img.convert("RGB").resize((300, 300))
            buf = io.BytesIO()
            img_thumb.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            result[cls] = f"data:image/jpeg;base64,{b64}"
    _class_images = result
    return result


# ─── Routes ──────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "resnet50_available": RESNET_CHECKPOINT.exists(),
        "mobilenetv2_available": MOBILENET_CHECKPOINT.exists(),
        "resnet50_loaded": "resnet50" in _models,
        "mobilenetv2_loaded": "mobilenetv2" in _models,
    }


@app.get("/api/classes")
def get_classes():
    """Return class names, descriptions, and a representative image."""
    class_names = _get_class_names()
    class_images = _get_class_images()

    descriptions = {
        "Bacterial Leaf Blight": (
            "A serious bacterial disease caused by Xanthomonas oryzae pv. oryzae. "
            "Symptoms include water-soaked to yellowish stripes on leaf margins, "
            "eventually turning white or gray. Severe infections can cause wilting "
            "and sterility, significantly reducing yield."
        ),
        "Brown Spot": (
            "Caused by the fungus Cochliobolus miyabeanus. Characterized by small, "
            "round to oval spots with a brown center and yellow halo on leaves, "
            "glumes, and grains. Often associated with nutrient-deficient soils "
            "and can cause grain discoloration."
        ),
        "Healthy Rice Leaf": (
            "A normal, disease-free rice leaf showing uniform green coloration and "
            "intact tissue. Healthy leaves have no lesions, discoloration, or "
            "abnormal patterns. Correctly identifying healthy leaves is important "
            "to avoid unnecessary pesticide applications."
        ),
        "Leaf Blast": (
            "One of the most destructive rice diseases, caused by Magnaporthe oryzae. "
            "Produces distinctive diamond-shaped or spindle-shaped lesions with gray "
            "centers and brown borders. Can affect all above-ground plant parts "
            "and cause significant yield loss under favorable conditions."
        ),
        "Leaf Scald": (
            "Caused by Microdochium oryzae. Produces tan to brown scalded lesions "
            "typically beginning at leaf tips and margins. Lesions have an "
            "interdigitated, zonate pattern at their borders. Most prevalent "
            "under high humidity and temperature conditions."
        ),
        "Sheath Blight": (
            "Caused by Rhizoctonia solani. Affects the leaf sheath and blade, "
            "producing oval or elliptical lesions with a gray-white center and "
            "brown margin. Spreads rapidly under humid conditions and high plant "
            "density, often causing lodging and significant yield reduction."
        ),
    }

    return [
        {
            "name": cls,
            "index": i,
            "description": descriptions.get(cls, "No description available."),
            "image": class_images.get(cls),
        }
        for i, cls in enumerate(class_names)
    ]


@app.get("/api/models")
def get_models():
    """Return metadata for both models."""
    def _read_json(path: Path):
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    resnet_metrics = _read_json(RESNET_METRICS)
    mobilenet_metrics = _read_json(MOBILENET_METRICS)
    resnet_eff = _read_json(RESNET_EFFICIENCY)
    mobilenet_eff = _read_json(MOBILENET_EFFICIENCY)

    return {
        "resnet50": {
            "available": RESNET_CHECKPOINT.exists(),
            "metrics": resnet_metrics,
            "efficiency": resnet_eff,
        },
        "mobilenetv2": {
            "available": MOBILENET_CHECKPOINT.exists(),
            "metrics": mobilenet_metrics,
            "efficiency": mobilenet_eff,
        },
    }


@app.get("/api/metrics")
def get_metrics():
    """Return full per-class metrics and confusion matrices for both models."""
    import csv

    def read_csv(path: Path) -> list:
        if not path.exists():
            return []
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows

    def read_confusion(path: Path) -> dict:
        if not path.exists():
            return {}
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = None
            for row in reader:
                if headers is None:
                    headers = list(row.keys())
                rows.append({k: int(v) for k, v in row.items() if k != ""})
        # Return as {actual_class: {predicted_class: count}}
        result = {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            col_names = header[1:]  # skip first empty col
            for row in reader:
                actual = row[0]
                result[actual] = {col_names[i]: int(row[i + 1]) for i in range(len(col_names))}
        return result

    class_names = _get_class_names()

    def parse_report(path: Path) -> list:
        """Parse classification_report.csv → list of per-class dicts."""
        if not path.exists():
            return []
        out = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("", row.get("Unnamed: 0", ""))
                if label in class_names:
                    out.append({
                        "class": label,
                        "precision": float(row["precision"]),
                        "recall": float(row["recall"]),
                        "f1_score": float(row["f1-score"]),
                        "support": int(float(row["support"])),
                    })
        return out

    return {
        "resnet50": {
            "per_class": parse_report(RESNET_REPORT),
            "confusion_matrix": read_confusion(RESNET_CONFUSION),
        },
        "mobilenetv2": {
            "per_class": parse_report(MOBILENET_REPORT),
            "confusion_matrix": read_confusion(MOBILENET_CONFUSION),
        },
        "class_names": class_names,
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query("resnet50", description="Model to use: 'resnet50' or 'mobilenetv2'"),
):
    """Upload an image and receive a classification prediction."""
    if model not in ("resnet50", "mobilenetv2"):
        raise HTTPException(status_code=400, detail="model must be 'resnet50' or 'mobilenetv2'")

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are accepted.",
        )

    # Read and size-check upload (max 10 MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be smaller than 10 MB.")

    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image. Ensure it is a valid JPEG or PNG.")

    try:
        loaded_model = _load_model(model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    class_names = _get_class_names()

    arr = _preprocess(img, model)
    t0 = time.perf_counter()
    preds = loaded_model.predict(arr, verbose=0)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    scores = preds[0].tolist()
    top_idx = int(np.argmax(scores))

    return {
        "model": model,
        "predicted_class": class_names[top_idx],
        "predicted_index": top_idx,
        "confidence": scores[top_idx],
        "scores": {class_names[i]: scores[i] for i in range(len(class_names))},
        "inference_ms": round(elapsed_ms, 1),
        "disclaimer": (
            "Prediction scores are model outputs and are not necessarily calibrated "
            "probabilities. Results may be unreliable for images outside the training domain. "
            "This is an educational research prototype, not a certified agricultural diagnostic system."
        ),
    }

