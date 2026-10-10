import os
import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image

AUDIT_DIR = Path(r"C:\Gitwork\Rice diseases\DatasetAudit")
MANIFESTS_DIR = AUDIT_DIR / "manifests"
REPORTS_DIR = AUDIT_DIR / "reports"

EXPECTED_CLASSES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight"
]

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def validate_clean_splits():
    print("=" * 70)
    print("PHASE F: INDEPENDENT CLEAN DATASET VALIDATION PASS")
    print("=" * 70)
    
    validation_results = {
        "overall_status": "PENDING",
        "checks": {}
    }
    
    # 1. Load manifests
    train_path = MANIFESTS_DIR / "clean_train.csv"
    val_path = MANIFESTS_DIR / "clean_validation.csv"
    test_path = MANIFESTS_DIR / "clean_test.csv"
    
    assert train_path.exists(), "clean_train.csv missing"
    assert val_path.exists(), "clean_validation.csv missing"
    assert test_path.exists(), "clean_test.csv missing"
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    n_total = n_train + n_val + n_test
    
    print(f"Loaded manifests: Train={n_train}, Val={n_val}, Test={n_test}, Total={n_total}")
    validation_results["split_counts"] = {
        "train": n_train,
        "validation": n_val,
        "test": n_test,
        "total": n_total,
        "train_pct": round(n_train / n_total * 100, 2),
        "val_pct": round(n_val / n_total * 100, 2),
        "test_pct": round(n_test / n_total * 100, 2)
    }
    
    # Check 1: Schema consistency
    expected_cols = ["image_path", "class_name", "numeric_class_index", "group_id"]
    schema_ok = (list(train_df.columns) == expected_cols and 
                 list(val_df.columns) == expected_cols and 
                 list(test_df.columns) == expected_cols)
    print(f"Check 1 - Schema consistency across manifests: {'PASS' if schema_ok else 'FAIL'}")
    assert schema_ok, f"Schema mismatch: {train_df.columns}, {val_df.columns}, {test_df.columns}"
    validation_results["checks"]["schema_consistency"] = "PASS"
    
    # Check 2: Physical image existence and readability
    missing_paths = []
    unreadable_paths = []
    all_dfs = [("train", train_df), ("val", val_df), ("test", test_df)]
    for name, df in all_dfs:
        for p in df["image_path"]:
            if not os.path.exists(p):
                missing_paths.append((name, p))
            else:
                try:
                    with Image.open(p) as img:
                        img.verify()
                except Exception as e:
                    unreadable_paths.append((name, p, str(e)))
                    
    print(f"Check 2 - Physical file existence: {len(missing_paths)} missing. {'PASS' if len(missing_paths) == 0 else 'FAIL'}")
    print(f"Check 3 - Image readability & corruption check: {len(unreadable_paths)} unreadable. {'PASS' if len(unreadable_paths) == 0 else 'FAIL'}")
    assert len(missing_paths) == 0, f"Missing files found: {missing_paths}"
    assert len(unreadable_paths) == 0, f"Corrupt files found: {unreadable_paths}"
    validation_results["checks"]["physical_file_existence"] = "PASS"
    validation_results["checks"]["image_readability"] = "PASS"
    
    # Check 4: Duplicate paths within manifests
    dup_train = train_df["image_path"].duplicated().sum()
    dup_val = val_df["image_path"].duplicated().sum()
    dup_test = test_df["image_path"].duplicated().sum()
    print(f"Check 4 - Duplicate paths within manifests: Train={dup_train}, Val={dup_val}, Test={dup_test}. {'PASS' if (dup_train + dup_val + dup_test) == 0 else 'FAIL'}")
    assert dup_train == 0 and dup_val == 0 and dup_test == 0, "Duplicate paths found inside manifest!"
    validation_results["checks"]["internal_path_uniqueness"] = "PASS"
    
    # Check 5: Path overlap between splits
    train_paths = set(train_df["image_path"])
    val_paths = set(val_df["image_path"])
    test_paths = set(test_df["image_path"])
    
    ov_tr_va = len(train_paths & val_paths)
    ov_tr_te = len(train_paths & test_paths)
    ov_va_te = len(val_paths & test_paths)
    print(f"Check 5 - Path overlap across splits: Train-Val={ov_tr_va}, Train-Test={ov_tr_te}, Val-Test={ov_va_te}. {'PASS' if (ov_tr_va + ov_tr_te + ov_va_te) == 0 else 'FAIL'}")
    assert ov_tr_va == 0 and ov_tr_te == 0 and ov_va_te == 0, "Cross-split path overlap detected!"
    validation_results["checks"]["cross_split_path_overlap"] = "PASS"
    
    # Check 6: Exact SHA-256 byte hash overlap across splits
    print("Computing SHA-256 hashes across clean splits for independent verification...")
    train_hashes = {compute_sha256(Path(p)): p for p in train_paths}
    val_hashes = {compute_sha256(Path(p)): p for p in val_paths}
    test_hashes = {compute_sha256(Path(p)): p for p in test_paths}
    
    hash_ov_tr_va = set(train_hashes.keys()) & set(val_hashes.keys())
    hash_ov_tr_te = set(train_hashes.keys()) & set(test_hashes.keys())
    hash_ov_va_te = set(val_hashes.keys()) & set(test_hashes.keys())
    
    print(f"Check 6 - Exact SHA-256 hash overlap across splits: Train-Val={len(hash_ov_tr_va)}, Train-Test={len(hash_ov_tr_te)}, Val-Test={len(hash_ov_va_te)}. {'PASS' if (len(hash_ov_tr_va) + len(hash_ov_tr_te) + len(hash_ov_va_te)) == 0 else 'FAIL'}")
    assert len(hash_ov_tr_va) == 0 and len(hash_ov_tr_te) == 0 and len(hash_ov_va_te) == 0, "FATAL: Exact duplicate hash crossed splits!"
    validation_results["checks"]["exact_hash_leakage"] = "PASS"
    
    # Check 7: Confirmed near-duplicate group overlap across splits
    train_groups = set(train_df["group_id"])
    val_groups = set(val_df["group_id"])
    test_groups = set(test_df["group_id"])
    
    grp_ov_tr_va = train_groups & val_groups
    grp_ov_tr_te = train_groups & test_groups
    grp_ov_va_te = val_groups & test_groups
    print(f"Check 7 - Group ID overlap across splits: Train-Val={len(grp_ov_tr_va)}, Train-Test={len(grp_ov_tr_te)}, Val-Test={len(grp_ov_va_te)}. {'PASS' if (len(grp_ov_tr_va) + len(grp_ov_tr_te) + len(grp_ov_va_te)) == 0 else 'FAIL'}")
    assert len(grp_ov_tr_va) == 0 and len(grp_ov_tr_te) == 0 and len(grp_ov_va_te) == 0, "FATAL: Near-duplicate group crossed splits!"
    validation_results["checks"]["group_leakage"] = "PASS"
    
    # Check 8: Label conflicts and representation of all 6 classes
    all_clean = pd.concat([train_df, val_df, test_df], ignore_index=True)
    per_class_summary = {}
    for c in EXPECTED_CLASSES:
        c_tr = (train_df["class_name"] == c).sum()
        c_va = (val_df["class_name"] == c).sum()
        c_te = (test_df["class_name"] == c).sum()
        c_tot = (all_clean["class_name"] == c).sum()
        per_class_summary[c] = {"train": int(c_tr), "validation": int(c_va), "test": int(c_te), "total": int(c_tot)}
        assert c_tr > 0 and c_va > 0 and c_te > 0, f"Class {c} is missing from one of the splits!"
    print(f"Check 8 - All six classes represented in every split: PASS")
    validation_results["checks"]["all_six_classes_represented"] = "PASS"
    validation_results["per_class_counts"] = per_class_summary
    
    # Check 9: Numeric index mapping consistency
    idx_check = all_clean.groupby("class_name")["numeric_class_index"].unique()
    for c, idxs in idx_check.items():
        assert len(idxs) == 1 and idxs[0] == EXPECTED_CLASSES.index(c), f"Index mismatch for {c}"
    print("Check 9 - Canonical class index mapping ([0..5]): PASS")
    validation_results["checks"]["class_index_mapping"] = "PASS"
    
    # Check 10: Excluded images traceability
    df_excl = pd.read_csv(REPORTS_DIR / "excluded_images.csv")
    assert len(df_excl) == 31, f"Expected 31 excluded images, found {len(df_excl)}"
    # Verify none of the excluded images appear in the clean manifests
    excl_set = set(df_excl["file_path"])
    clean_set = set(all_clean["image_path"])
    leak_excl = excl_set & clean_set
    print(f"Check 10 - Excluded files excluded from clean manifests: {len(leak_excl)} leaked. {'PASS' if len(leak_excl) == 0 else 'FAIL'}")
    assert len(leak_excl) == 0, "Excluded image leaked into clean manifest!"
    validation_results["checks"]["excluded_images_isolation"] = "PASS"
    
    # Check 11: Total source images accounted for
    total_raw_accounted = len(all_clean) + len(df_excl)
    print(f"Check 11 - Total source images accounted for: {total_raw_accounted} / 3829. {'PASS' if total_raw_accounted == 3829 else 'FAIL'}")
    assert total_raw_accounted == 3829, f"Mismatch in total accounted images: {total_raw_accounted} != 3829"
    validation_results["checks"]["total_inventory_conservation"] = "PASS"
    
    validation_results["overall_status"] = "ALL_CHECKS_PASSED"
    print("\n" + "=" * 70)
    print("FINAL VALIDATION RESULT: ALL CHECKS PASSED PERFECTLY")
    print("=" * 70)
    
    return validation_results

if __name__ == "__main__":
    res = validate_clean_splits()
    print(json.dumps(res, indent=2))

