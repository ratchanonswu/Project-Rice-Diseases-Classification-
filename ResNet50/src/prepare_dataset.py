"""
prepare_dataset.py - Dataset Verification, Summary Generation, and Stratified Splitting

This script:
1. Validates the dataset structure and ensures all 6 target classes are present.
2. Inspects all images, checks for corrupt/unreadable files, and extracts image dimensions.
3. Generates a dataset summary saved to outputs/dataset_summary.csv.
4. Performs a reproducible, stratified Train (75%) / Validation (10%) / Test (15%) split.
5. Saves outputs/train.csv, outputs/validation.csv, and outputs/test.csv.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Canonical class list in strict fixed order
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
            # Check for partial matches or common aliases
            matched = None
            for key, path in subdir_map.items():
                if key.replace(" ", "") == canonical_lower.replace(" ", ""):
                    matched = path
                    break
            if matched:
                found_dirs[canonical] = matched

    # Validate that all expected classes exist
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


def scan_and_summarize(dataset_dir: Path, output_dir: Path):
    """
    Inspects all images in the dataset, checks integrity, records dimensions,
    and produces outputs/dataset_summary.csv.
    """
    class_dirs = find_class_directories(dataset_dir)
    records = []
    summary_by_class = {}
    corrupted_count = 0

    print("=" * 60)
    print("SCANNING AND VALIDATING DATASET")
    print("=" * 60)

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
                # Re-open to read dimensions after verify()
                with Image.open(f) as img:
                    w, h = img.size
                    class_widths.append(w)
                    class_heights.append(h)
                    valid_files.append(f)
                    records.append({
                        "image_path": str(f.resolve()),
                        "relative_path": str(f.relative_to(dataset_dir.parent)).replace("\\", "/"),
                        "class_name": class_name,
                        "width": w,
                        "height": h,
                        "is_valid": True
                    })
            except Exception as e:
                class_corrupted += 1
                corrupted_count += 1
                print(f"[WARNING] Corrupt/unreadable file: {f} ({e})")
                records.append({
                    "image_path": str(f.resolve()),
                    "relative_path": str(f.relative_to(dataset_dir.parent)).replace("\\", "/"),
                    "class_name": class_name,
                    "width": None,
                    "height": None,
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

    # Add total row
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

    df_records = pd.DataFrame(records)
    valid_records = df_records[df_records["is_valid"]].copy()

    return valid_records


def create_stratified_split(
    df_valid: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.75,
    val_ratio: float = 0.10,
    test_ratio: float = 0.15
):
    """
    Creates a stratified split (75% Train, 10% Val, 15% Test) before any augmentation.
    Saves outputs/train.csv, outputs/validation.csv, outputs/test.csv.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    print("=" * 60)
    print(f"PERFORMING STRATIFIED SPLIT (Train: {train_ratio*100:.0f}%, Val: {val_ratio*100:.0f}%, Test: {test_ratio*100:.0f}%)")
    print(f"Random Seed: {seed}")
    print("=" * 60)

    class_to_idx = {name: idx for idx, name in enumerate(EXPECTED_CLASSES)}
    df_valid["numeric_class_index"] = df_valid["class_name"].map(class_to_idx)

    # First split off test set: test_ratio (15%)
    # Remaining: 85%
    train_val_df, test_df = train_test_split(
        df_valid,
        test_size=test_ratio,
        random_state=seed,
        stratify=df_valid["class_name"]
    )

    # Next split train and val from train_val_df
    # val_ratio out of remaining = val_ratio / (train_ratio + val_ratio) = 0.10 / 0.85
    val_rel_ratio = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_rel_ratio,
        random_state=seed,
        stratify=train_val_df["class_name"]
    )

    # Select standard columns
    columns = ["image_path", "class_name", "numeric_class_index"]
    train_out = train_df[columns].copy()
    val_out = val_df[columns].copy()
    test_out = test_df[columns].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "validation.csv"
    test_csv = output_dir / "test.csv"

    train_out.to_csv(train_csv, index=False)
    val_out.to_csv(val_csv, index=False)
    test_out.to_csv(test_csv, index=False)

    print(f"[INFO] Train split:      {len(train_out):4d} images ({len(train_out)/len(df_valid)*100:.2f}%) -> {train_csv}")
    print(f"[INFO] Validation split: {len(val_out):4d} images ({len(val_out)/len(df_valid)*100:.2f}%) -> {val_csv}")
    print(f"[INFO] Test split:       {len(test_out):4d} images ({len(test_out)/len(df_valid)*100:.2f}%) -> {test_csv}")

    # Check distribution consistency
    dist_check = pd.DataFrame({
        "Train": train_out["class_name"].value_counts(),
        "Val": val_out["class_name"].value_counts(),
        "Test": test_out["class_name"].value_counts()
    }).loc[EXPECTED_CLASSES]
    print("\nPer-Class Distribution across splits:")
    print(dist_check)

    return train_out, val_out, test_out


def main():
    parser = argparse.ArgumentParser(description="Prepare, validate, and split rice leaf dataset.")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to raw dataset folder")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save summaries and splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--train_ratio", type=float, default=0.75, help="Proportion of training set (default: 0.75)")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Proportion of validation set (default: 0.10)")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Proportion of test set (default: 0.15)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    valid_records = scan_and_summarize(dataset_dir, output_dir)
    create_stratified_split(
        valid_records,
        output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    print("\n[SUCCESS] Dataset preparation complete.")


if __name__ == "__main__":
    main()

