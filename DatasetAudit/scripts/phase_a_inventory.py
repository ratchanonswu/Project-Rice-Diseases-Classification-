import os
import sys
import hashlib
import json
from pathlib import Path
import pandas as pd
from PIL import Image

DATASET_ROOT = Path(r"C:\Gitwork\Rice diseases\Rice Diseases Classification\ResNet50\dataset")
RESNET50_OUTPUTS = Path(r"C:\Gitwork\Rice diseases\Rice Diseases Classification\ResNet50\outputs")
MOBILENET_OUTPUTS = Path(r"C:\Gitwork\Rice diseases\Rice Diseases Classification\MobileNetV2\outputs")
AUDIT_DIR = Path(r"C:\Gitwork\Rice diseases\DatasetAudit")

EXPECTED_CLASSES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight"
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def inspect_dataset():
    print(f"Scanning dataset at {DATASET_ROOT}...")
    inventory = []
    corrupt_files = []
    
    for class_name in EXPECTED_CLASSES:
        class_dir = DATASET_ROOT / class_name
        if not class_dir.exists():
            print(f"Error: Missing class directory {class_dir}")
            continue
        
        all_files = list(class_dir.iterdir())
        for f in all_files:
            ext = f.suffix.lower()
            if ext not in VALID_EXTENSIONS:
                print(f"Unsupported file: {f}")
                continue
            
            size = f.stat().st_size
            if size == 0:
                print(f"Empty file: {f}")
                corrupt_files.append((str(f), "empty"))
                continue
            
            try:
                with Image.open(f) as img:
                    img.verify()
                with Image.open(f) as img:
                    w, h = img.size
                    fmt = img.format
                
                sha256 = compute_sha256(f)
                inventory.append({
                    "file_path": str(f.resolve()),
                    "filename": f.name,
                    "class_name": class_name,
                    "file_size": size,
                    "width": w,
                    "height": h,
                    "format": fmt,
                    "sha256": sha256,
                    "is_readable": True
                })
            except Exception as e:
                print(f"Corrupt file: {f} ({e})")
                corrupt_files.append((str(f), str(e)))
                inventory.append({
                    "file_path": str(f.resolve()),
                    "filename": f.name,
                    "class_name": class_name,
                    "file_size": size,
                    "width": None,
                    "height": None,
                    "format": None,
                    "sha256": None,
                    "is_readable": False
                })
    
    df_inv = pd.DataFrame(inventory)
    print(f"Total files scanned: {len(df_inv)}")
    print(f"Total readable files: {df_inv['is_readable'].sum()}")
    print("Class distribution:")
    print(df_inv["class_name"].value_counts())
    
    # Save baseline inventory
    reports_dir = AUDIT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    df_inv.to_csv(reports_dir / "dataset_inventory.csv", index=False)
    print(f"Saved baseline inventory to {reports_dir / 'dataset_inventory.csv'}")

    return df_inv

def inspect_existing_splits(df_inv: pd.DataFrame):
    print("\nInspecting existing split CSVs...")
    splits = {}
    for split_name in ["train", "validation", "test"]:
        csv_path = RESNET50_OUTPUTS / f"{split_name}.csv"
        df = pd.read_csv(csv_path)
        splits[split_name] = df
        print(f"{split_name}: {len(df)} rows. Columns: {list(df.columns)}")
        
        # Check path existence
        missing = [p for p in df["image_path"] if not os.path.exists(p)]
        print(f"  Missing paths: {len(missing)}")
    
    # Check if MobileNet split CSVs match ResNet50
    for split_name in ["train", "validation", "test"]:
        mb_csv = MOBILENET_OUTPUTS / f"{split_name}.csv"
        df_mb = pd.read_csv(mb_csv)
        diff = (df_mb["image_path"] != splits[split_name]["image_path"]).sum()
        print(f"MobileNet {split_name} path differences from ResNet50: {diff}")

if __name__ == "__main__":
    df_inv = inspect_dataset()
    inspect_existing_splits(df_inv)

