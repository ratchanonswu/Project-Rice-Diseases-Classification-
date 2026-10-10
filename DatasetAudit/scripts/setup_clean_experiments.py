import os
import shutil
import hashlib
from pathlib import Path
import pandas as pd

AUDIT_MANIFESTS = Path(r"C:\Gitwork\Rice diseases\DatasetAudit\manifests")
RESNET50_DIR = Path(r"C:\Gitwork\Rice diseases\Rice Diseases Classification\ResNet50")
MOBILENET_DIR = Path(r"C:\Gitwork\Rice diseases\Rice Diseases Classification\MobileNetV2")

def setup_clean_experiment_dirs():
    print("Setting up clean_split_v1 experiment directories...")
    
    for base_dir in [RESNET50_DIR, MOBILENET_DIR]:
        exp_out = base_dir / "outputs" / "clean_split_v1"
        exp_mod = base_dir / "models" / "clean_split_v1"
        
        exp_out.mkdir(parents=True, exist_ok=True)
        exp_mod.mkdir(parents=True, exist_ok=True)
        
        # Copy clean manifests into exp_out
        shutil.copy2(AUDIT_MANIFESTS / "clean_train.csv", exp_out / "train.csv")
        shutil.copy2(AUDIT_MANIFESTS / "clean_validation.csv", exp_out / "validation.csv")
        shutil.copy2(AUDIT_MANIFESTS / "clean_test.csv", exp_out / "test.csv")
        
        # Copy class_names.json into exp_mod and exp_out
        shutil.copy2(AUDIT_MANIFESTS / "class_names.json", exp_mod / "class_names.json")
        shutil.copy2(AUDIT_MANIFESTS / "class_names.json", exp_out / "class_names.json")
        
        print(f"Set up {exp_out} and {exp_mod}")
        
    # Verify copies
    for base_dir, name in [(RESNET50_DIR, "ResNet50"), (MOBILENET_DIR, "MobileNetV2")]:
        exp_out = base_dir / "outputs" / "clean_split_v1"
        for fname in ["train.csv", "validation.csv", "test.csv"]:
            src_hash = hashlib.sha256(open(AUDIT_MANIFESTS / f"clean_{fname}", "rb").read()).hexdigest()
            dst_hash = hashlib.sha256(open(exp_out / fname, "rb").read()).hexdigest()
            assert src_hash == dst_hash, f"Hash mismatch for {name} {fname}"
        print(f"Verified byte-exactness of manifests in {name}/outputs/clean_split_v1/")

if __name__ == "__main__":
    setup_clean_experiment_dirs()

