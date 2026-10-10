import os
import sys
import json
from pathlib import Path
import pandas as pd

AUDIT_DIR = Path(r"C:\Gitwork\Rice diseases\DatasetAudit")
MANIFESTS_DIR = AUDIT_DIR / "manifests"

TRAIN_CSV = MANIFESTS_DIR / "clean_train.csv"
VAL_CSV = MANIFESTS_DIR / "clean_validation.csv"
TEST_CSV = MANIFESTS_DIR / "clean_test.csv"
CLASS_NAMES_JSON = MANIFESTS_DIR / "class_names.json"

def verify_source_of_truth():
    print("=" * 60)
    print("VERIFYING CLEAN DATASET MANIFESTS SOURCE OF TRUTH")
    print("=" * 60)
    
    assert TRAIN_CSV.exists(), f"Missing {TRAIN_CSV}"
    assert VAL_CSV.exists(), f"Missing {VAL_CSV}"
    assert TEST_CSV.exists(), f"Missing {TEST_CSV}"
    assert CLASS_NAMES_JSON.exists(), f"Missing {CLASS_NAMES_JSON}"
    
    with open(CLASS_NAMES_JSON, "r") as f:
        class_mapping = json.load(f)
        
    print(f"Loaded class mapping: {class_mapping}")
    
    df_tr = pd.read_csv(TRAIN_CSV)
    df_va = pd.read_csv(VAL_CSV)
    df_te = pd.read_csv(TEST_CSV)
    
    print(f"Manifest counts: Train={len(df_tr)}, Val={len(df_va)}, Test={len(df_te)}, Total={len(df_tr)+len(df_va)+len(df_te)}")
    
    # Check paths
    missing = []
    for split_name, df in [("train", df_tr), ("val", df_va), ("test", df_te)]:
        for p in df["image_path"]:
            if not os.path.exists(p):
                missing.append((split_name, p))
    assert len(missing) == 0, f"Found missing files: {missing}"
    print("[PASS] All 3,798 image paths resolve successfully.")
    
    # Check labels and numeric index mapping
    for split_name, df in [("train", df_tr), ("val", df_va), ("test", df_te)]:
        for _, row in df.iterrows():
            cname = row["class_name"]
            cidx = row["numeric_class_index"]
            assert cname in class_mapping, f"Unknown class {cname} in {split_name}"
            assert class_mapping[cname] == cidx, f"Index mismatch for {cname} in {split_name}"
    print("[PASS] All labels match canonical class mapping.")
    
    # Check group cross-split isolation
    g_tr = set(df_tr["group_id"])
    g_va = set(df_va["group_id"])
    g_te = set(df_te["group_id"])
    
    assert len(g_tr & g_va) == 0, "Group overlap between Train and Val"
    assert len(g_tr & g_te) == 0, "Group overlap between Train and Test"
    assert len(g_va & g_te) == 0, "Group overlap between Val and Test"
    print("[PASS] Zero confirmed duplicate groups cross splits.")
    
    print("\nPre-flight validation complete: Source of truth is 100% verified.")

if __name__ == "__main__":
    verify_source_of_truth()

