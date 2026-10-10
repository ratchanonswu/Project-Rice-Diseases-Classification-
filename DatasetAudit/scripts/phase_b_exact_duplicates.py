import os
import sys
import hashlib
from pathlib import Path
import pandas as pd

AUDIT_DIR = Path(r"C:\Gitwork\Rice diseases\DatasetAudit")
RESNET50_OUTPUTS = Path(r"C:\Gitwork\Rice diseases\Rice Diseases Classification\ResNet50\outputs")

def analyze_exact_duplicates():
    df_inv = pd.read_csv(AUDIT_DIR / "reports" / "dataset_inventory.csv")
    
    # Load splits
    train_df = pd.read_csv(RESNET50_OUTPUTS / "train.csv")
    train_df["split"] = "train"
    val_df = pd.read_csv(RESNET50_OUTPUTS / "validation.csv")
    val_df["split"] = "validation"
    test_df = pd.read_csv(RESNET50_OUTPUTS / "test.csv")
    test_df["split"] = "test"
    
    all_splits = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Merge split info into inventory
    df_merged = df_inv.merge(
        all_splits[["image_path", "split", "numeric_class_index"]],
        left_on="file_path",
        right_on="image_path",
        how="left"
    )
    
    # Group by sha256
    hash_counts = df_merged["sha256"].value_counts()
    dup_hashes = hash_counts[hash_counts > 1].index.tolist()
    
    print(f"Total unique hashes: {len(hash_counts)}")
    print(f"Duplicate hash groups: {len(dup_hashes)}")
    
    dup_records = []
    cross_split_records = []
    label_conflict_records = []
    
    group_id = 0
    for h in sorted(dup_hashes):
        group_id += 1
        group_df = df_merged[df_merged["sha256"] == h].sort_values("file_path")
        classes = group_df["class_name"].unique()
        splits_in_group = group_df["split"].unique()
        labels_agree = (len(classes) == 1)
        
        # Check label conflict
        if not labels_agree:
            for _, row in group_df.iterrows():
                label_conflict_records.append({
                    "group_id": f"EXACT_DUP_{group_id:03d}",
                    "sha256": h,
                    "file_path": row["file_path"],
                    "class_name": row["class_name"],
                    "split": row["split"],
                    "conflicting_classes": ";".join(classes)
                })
        
        # Check cross-split leakage
        if len(splits_in_group) > 1:
            for _, row in group_df.iterrows():
                cross_split_records.append({
                    "group_id": f"EXACT_DUP_{group_id:03d}",
                    "sha256": h,
                    "file_path": row["file_path"],
                    "class_name": row["class_name"],
                    "split": row["split"],
                    "splits_involved": ";".join(sorted(splits_in_group))
                })
        
        # Determine recommendation
        # Deterministic representative: first path lexicographically
        rep_path = group_df.iloc[0]["file_path"]
        for _, row in group_df.iterrows():
            is_rep = (row["file_path"] == rep_path)
            if not labels_agree:
                rec = "EXCLUDE_LABEL_CONFLICT"
                reason = f"Label conflict across classes: {';'.join(classes)}. Manual review required."
            else:
                if is_rep:
                    rec = "RETAIN_AS_REPRESENTATIVE"
                    reason = "Deterministic representative (first path lexicographically)."
                else:
                    rec = "EXCLUDE_DUPLICATE_COPY"
                    reason = f"Exact byte-identical duplicate of representative {Path(rep_path).name}."
            
            dup_records.append({
                "group_id": f"EXACT_DUP_{group_id:03d}",
                "file_path": row["file_path"],
                "filename": row["filename"],
                "class_name": row["class_name"],
                "sha256": h,
                "current_split": row["split"],
                "labels_agree": labels_agree,
                "is_representative": is_rep,
                "recommended_handling": rec,
                "reason": reason
            })
            
    df_dup = pd.DataFrame(dup_records)
    reports_dir = AUDIT_DIR / "reports"
    df_dup.to_csv(reports_dir / "exact_duplicate_groups.csv", index=False)
    
    df_cross = pd.DataFrame(cross_split_records)
    df_cross.to_csv(reports_dir / "cross_split_leakage.csv", index=False)
    
    df_conflict = pd.DataFrame(label_conflict_records)
    df_conflict.to_csv(reports_dir / "label_conflicts.csv", index=False)
    
    print("\n--- EXACT DUPLICATE SUMMARY ---")
    print(f"Total duplicate groups: {len(dup_hashes)}")
    print(f"Total duplicate files involved: {len(df_dup)}")
    print(f"Label conflicts found: {len(label_conflict_records)}")
    
    # Specific breakdowns
    within_class = 0
    across_class = 0
    within_train = 0
    within_val = 0
    within_test = 0
    train_val = 0
    train_test = 0
    val_test = 0
    
    for h in dup_hashes:
        gdf = df_merged[df_merged["sha256"] == h]
        if len(gdf["class_name"].unique()) == 1:
            within_class += 1
        else:
            across_class += 1
            
        splits_set = set(gdf["split"].dropna())
        if splits_set == {"train"}:
            within_train += 1
        elif splits_set == {"validation"}:
            within_val += 1
        elif splits_set == {"test"}:
            within_test += 1
        
        if "train" in splits_set and "validation" in splits_set:
            train_val += 1
        if "train" in splits_set and "test" in splits_set:
            train_test += 1
        if "validation" in splits_set and "test" in splits_set:
            val_test += 1
            
    print(f"Groups within same class: {within_class}")
    print(f"Groups across different classes: {across_class}")
    print(f"Groups strictly within Train: {within_train}")
    print(f"Groups strictly within Validation: {within_val}")
    print(f"Groups strictly within Test: {within_test}")
    print(f"Groups crossing Train & Validation: {train_val}")
    print(f"Groups crossing Train & Test: {train_test}")
    print(f"Groups crossing Validation & Test: {val_test}")

if __name__ == "__main__":
    analyze_exact_duplicates()

