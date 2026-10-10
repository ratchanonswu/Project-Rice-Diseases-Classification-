"""
prepare_dataset.py - Dataset Validation and ResNet50 Split Manifest Import for MobileNetV2

This script:
1. Validates the local MobileNetV2 dataset structure and ensures all 6 target classes are present.
2. Inspects all image files, detects corrupt or unreadable files, and extracts image dimensions.
3. Generates outputs/dataset_summary.csv.
4. Checks for exact duplicate images using SHA-256 file hashes and reports duplicate groups.
5. Verifies and imports the exact split manifests from ResNet50 (outputs/train.csv, validation.csv, test.csv)
   to guarantee a fair, strictly identical baseline for comparative research.
6. Verifies that all referenced images exist and that there are zero overlapping paths across splits.
7. Validates per-split sample counts and class distributions against canonical class mappings.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# Canonical class list in strict fixed order - must match ResNet50 exactly
EXPECTED_CLASSES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight"
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_class_directories(dataset_dir: Path):
    """
    Scans dataset_dir and matches directories against EXPECTED_CLASSES (case-insensitive).
    Returns a dict mapping canonical class name -> Path to directory.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    found_dirs = {}
    subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    subdir_map = {d.name.strip().lower(): d for d in subdirs}

    for canonical in EXPECTED_CLASSES:
        canonical_lower = canonical.lower()
        if canonical_lower in subdir_map:
            found_dirs[canonical] = subdir_map[canonical_lower]
        else:
            # Check for whitespace-stripped match
            matched = None
            for key, path in subdir_map.items():
                if key.replace(" ", "") == canonical_lower.replace(" ", ""):
                    matched = path
                    break
            if matched:
                found_dirs[canonical] = matched

    missing_classes = [c for c in EXPECTED_CLASSES if c not in found_dirs]
    if missing_classes:
        available = [d.name for d in subdirs]
        raise ValueError(
            f"Dataset validation failed!\n"
            f"Missing required classes: {missing_classes}\n"
            f"Available directories in '{dataset_dir}': {available}\n"
            f"All six required classes must exist: {EXPECTED_CLASSES}"
        )

    return found_dirs


def calculate_file_hash(filepath: Path, block_size: int = 65536) -> str:
    """Computes SHA-256 hash for duplicate detection."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            sha256.update(block)
    return sha256.hexdigest()


def scan_and_summarize(dataset_dir: Path, output_dir: Path):
    """
    Inspects all images in the dataset, checks integrity, records dimensions,
    detects exact duplicates via hash, and produces outputs/dataset_summary.csv.
    """
    class_dirs = find_class_directories(dataset_dir)
    records = []
    summary_by_class = {}
    corrupted_count = 0
    hashes = {}  # hash -> list of paths

    print("=" * 65)
    print("STEP 1: SCANNING AND VALIDATING MOBILENETV2 DATASET")
    print("=" * 65)

    for class_name in EXPECTED_CLASSES:
        class_path = class_dirs[class_name]
        files = [
            f for f in class_path.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        ]

        class_widths = []
        class_heights = []
        class_corrupted = 0
        valid_files = []

        for f in files:
            try:
                with Image.open(f) as img:
                    img.verify()
                with Image.open(f) as img:
                    w, h = img.size
                    class_widths.append(w)
                    class_heights.append(h)
                    valid_files.append(f)
                    
                    # Compute hash for duplicate tracking
                    f_hash = calculate_file_hash(f)
                    hashes.setdefault(f_hash, []).append(str(f.resolve()))

                    records.append({
                        "image_path": str(f.resolve()),
                        "class_name": class_name,
                        "width": w,
                        "height": h,
                        "hash": f_hash,
                        "is_valid": True
                    })
            except Exception as e:
                class_corrupted += 1
                corrupted_count += 1
                print(f"[WARNING] Corrupt/unreadable file: {f} ({e})")
                records.append({
                    "image_path": str(f.resolve()),
                    "class_name": class_name,
                    "width": None,
                    "height": None,
                    "hash": None,
                    "is_valid": False
                })

        count = len(valid_files)
        min_dim = f"{min(class_widths)}x{min(class_heights)}" if class_widths else "N/A"
        max_dim = f"{max(class_widths)}x{max(class_heights)}" if class_widths else "N/A"
        avg_w = f"{np.mean(class_widths):.1f}" if class_widths else "N/A"
        avg_h = f"{np.mean(class_heights):.1f}" if class_heights else "N/A"

        summary_by_class[class_name] = {
            "image_count": count,
            "corrupted_count": class_corrupted,
            "min_dimensions": min_dim,
            "max_dimensions": max_dim,
            "avg_width": avg_w,
            "avg_height": avg_h,
        }
        print(f"  - {class_name:25s}: {count:4d} valid images (corrupt: {class_corrupted})")

    total_valid = sum(s["image_count"] for s in summary_by_class.values())

    # Build summary rows
    summary_rows = []
    for c in EXPECTED_CLASSES:
        cnt = summary_by_class[c]["image_count"]
        pct = (cnt / total_valid * 100.0) if total_valid > 0 else 0.0
        summary_rows.append({
            "class_name": c,
            "image_count": cnt,
            "percentage": f"{pct:.2f}%",
            "min_dimensions": summary_by_class[c]["min_dimensions"],
            "max_dimensions": summary_by_class[c]["max_dimensions"],
            "avg_dimensions": f"{summary_by_class[c]['avg_width']}x{summary_by_class[c]['avg_height']}",
            "corrupted_images": summary_by_class[c]["corrupted_count"]
        })

    summary_rows.append({
        "class_name": "TOTAL",
        "image_count": total_valid,
        "percentage": "100.00%",
        "min_dimensions": "-",
        "max_dimensions": "-",
        "avg_dimensions": "-",
        "corrupted_images": corrupted_count
    })

    df_summary = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "dataset_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[INFO] Dataset summary saved to: {summary_path}")

    # Report duplicates if any
    duplicates = {h: p for h, p in hashes.items() if len(p) > 1}
    if duplicates:
        print(f"[INFO] Found {len(duplicates)} hash-identical groups (total duplicate files: {sum(len(p)-1 for p in duplicates.values())})")
    else:
        print("[INFO] Duplicate scan: 0 exact duplicate images found.")

    return pd.DataFrame(records)


def import_and_verify_resnet_splits(resnet_dir: Path, output_dir: Path):
    """
    Imports ResNet50 splits to MobileNetV2 outputs/ while verifying:
    1. ResNet50 split files exist.
    2. Zero missing images.
    3. Zero overlap between Train, Validation, and Test sets.
    4. Canonical class indices [0..5] match EXPECTED_CLASSES.
    """
    print("\n" + "=" * 65)
    print("STEP 2: IMPORTING & VALIDATING RESNET50 SPLIT MANIFESTS")
    print("=" * 65)

    resnet_outputs = resnet_dir / "outputs"
    train_source = resnet_outputs / "train.csv"
    val_source = resnet_outputs / "validation.csv"
    test_source = resnet_outputs / "test.csv"

    for p in [train_source, val_source, test_source]:
        if not p.exists():
            raise FileNotFoundError(
                f"CRITICAL ERROR: Required ResNet50 split file not found: {p}\n"
                f"Cannot proceed without the identical ResNet50 split manifest!"
            )

    train_df = pd.read_csv(train_source)
    val_df = pd.read_csv(val_source)
    test_df = pd.read_csv(test_source)

    print(f"[INFO] Source Train samples:      {len(train_df)}")
    print(f"[INFO] Source Validation samples: {len(val_df)}")
    print(f"[INFO] Source Test samples:       {len(test_df)}")
    total_samples = len(train_df) + len(val_df) + len(test_df)
    print(f"[INFO] Total samples across splits: {total_samples}")

    # 1. Check overlap
    train_paths = set(train_df["image_path"])
    val_paths = set(val_df["image_path"])
    test_paths = set(test_df["image_path"])

    assert len(train_paths & val_paths) == 0, "ERROR: Overlap between Train and Validation sets!"
    assert len(train_paths & test_paths) == 0, "ERROR: Overlap between Train and Test sets!"
    assert len(val_paths & test_paths) == 0, "ERROR: Overlap between Validation and Test sets!"
    print("[INFO] Isolation Check PASSED: Zero overlap across Train, Val, and Test splits.")

    # 2. Check image existence
    all_paths = list(train_paths | val_paths | test_paths)
    missing = [p for p in all_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"CRITICAL ERROR: {len(missing)} images referenced in split manifests do not exist!\n"
            f"Sample missing: {missing[:5]}"
        )
    print(f"[INFO] Image Existence Check PASSED: All {len(all_paths)} referenced images exist on disk.")

    # 3. Check class indices mapping
    class_to_idx = {name: i for i, name in enumerate(EXPECTED_CLASSES)}
    for name, df_split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        for _, row in df_split.iterrows():
            expected_idx = class_to_idx[row["class_name"]]
            actual_idx = int(row["numeric_class_index"])
            if expected_idx != actual_idx:
                raise ValueError(
                    f"Class index mismatch in {name} split for {row['class_name']}: "
                    f"expected {expected_idx}, got {actual_idx}"
                )
    print("[INFO] Class Mapping Check PASSED: All class labels and numeric indices match canonical order.")

    # Save identical copies to MobileNetV2 outputs/
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dest = output_dir / "train.csv"
    val_dest = output_dir / "validation.csv"
    test_dest = output_dir / "test.csv"

    train_df.to_csv(train_dest, index=False)
    val_df.to_csv(val_dest, index=False)
    test_df.to_csv(test_dest, index=False)

    print(f"[SUCCESS] Successfully saved manifests to:")
    print(f"  - {train_dest}")
    print(f"  - {val_dest}")
    print(f"  - {test_dest}")

    # Print distribution table
    dist_check = pd.DataFrame({
        "Train": train_df["class_name"].value_counts(),
        "Val": val_df["class_name"].value_counts(),
        "Test": test_df["class_name"].value_counts(),
        "Total": train_df["class_name"].value_counts() + val_df["class_name"].value_counts() + test_df["class_name"].value_counts()
    }).loc[EXPECTED_CLASSES]
    print("\nPer-Class Distribution across splits:")
    print(dist_check)
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate dataset for MobileNetV2.")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to local dataset directory")
    parser.add_argument("--resnet_dir", type=str, default="../ResNet50", help="Path to ResNet50 project root")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs and splits")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = (project_root / args.dataset_dir).resolve()
    resnet_dir = (project_root / args.resnet_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    print(f"Project root:    {project_root}")
    print(f"Dataset dir:     {dataset_dir}")
    print(f"ResNet50 root:   {resnet_dir}")
    print(f"Output dir:      {output_dir}\n")

    scan_and_summarize(dataset_dir, output_dir)
    import_and_verify_resnet_splits(resnet_dir, output_dir)
    print("\n[SUCCESS] MobileNetV2 dataset preparation & ResNet50 split alignment complete!")


if __name__ == "__main__":
    main()

