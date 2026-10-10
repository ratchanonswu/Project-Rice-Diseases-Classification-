import os
import sys
import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image

AUDIT_DIR = Path(r"C:\Gitwork\Rice diseases\DatasetAudit")
REPORTS_DIR = AUDIT_DIR / "reports"
MANIFESTS_DIR = AUDIT_DIR / "manifests"
RESNET50_OUTPUTS = Path(r"C:\Gitwork\Rice diseases\Rice Diseases Classification\ResNet50\outputs")

EXPECTED_CLASSES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight"
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(EXPECTED_CLASSES)}

def generate_unified_dataset_artifacts(seed=42):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load inventories and audit tables
    df_inv = pd.read_csv(REPORTS_DIR / "dataset_inventory.csv")
    df_exact = pd.read_csv(REPORTS_DIR / "exact_duplicate_groups.csv")
    df_near = pd.read_csv(REPORTS_DIR / "near_duplicate_candidates.csv")
    
    # Load original split info
    orig_train = pd.read_csv(RESNET50_OUTPUTS / "train.csv")
    orig_train["orig_split"] = "train"
    orig_val = pd.read_csv(RESNET50_OUTPUTS / "validation.csv")
    orig_val["orig_split"] = "validation"
    orig_test = pd.read_csv(RESNET50_OUTPUTS / "test.csv")
    orig_test["orig_split"] = "test"
    orig_all = pd.concat([orig_train, orig_val, orig_test], ignore_index=True)
    
    # Merge orig_split into inventory
    df_inv = df_inv.merge(orig_all[["image_path", "orig_split"]], left_on="file_path", right_on="image_path", how="left")
    
    # 2. Build complete connected components across all 3,829 images
    # Edges from 31 exact duplicate groups (SHA-256 byte-identical)
    exact_edges = []
    for gid, group in df_exact.groupby("group_id"):
        paths = sorted(group["file_path"].tolist())
        if len(paths) == 2:
            exact_edges.append((paths[0], paths[1]))
            
    # Edges from 125 confirmed near-duplicate pairs
    confirmed_near = df_near[df_near["status"] == "CONFIRMED_NEAR_DUPLICATE"]
    near_edges = []
    for _, row in confirmed_near.iterrows():
        near_edges.append((row["file_path_a"], row["file_path_b"]))
        
    print(f"Exact duplicate edges: {len(exact_edges)}")
    print(f"Confirmed near-duplicate edges: {len(near_edges)}")
    
    # Unified graph adjacency list
    adj = defaultdict(set)
    for u, v in exact_edges:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in near_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    all_paths = sorted(df_inv["file_path"].tolist())
    visited = set()
    components = []
    
    for p in all_paths:
        if p not in visited:
            comp = []
            queue = [p]
            visited.add(p)
            while queue:
                curr = queue.pop()
                comp.append(curr)
                for nbr in sorted(adj[curr]):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            components.append(sorted(comp))
            
    print(f"Total connected components: {len(components)}")
    
    # 3. Assign stable Group IDs GRP_0001 to GRP_3736
    path_to_group = {}
    path_to_class = dict(zip(df_inv["file_path"], df_inv["class_name"]))
    path_to_hash = dict(zip(df_inv["file_path"], df_inv["sha256"]))
    
    groups_by_class = defaultdict(list)
    group_comp_map = {}
    
    for idx, comp in enumerate(components, 1):
        gid = f"GRP_{idx:04d}"
        group_comp_map[gid] = comp
        cname = path_to_class[comp[0]]
        for p in comp:
            path_to_group[p] = gid
        groups_by_class[cname].append((gid, comp))
        
    # 4. Intra-group exact deduplication:
    # Within each connected component, find any files sharing the exact same sha256.
    # Retain the lexicographically smallest path as representative, exclude redundant copies.
    rep_paths = set()
    excl_records = []
    
    for gid, comp in group_comp_map.items():
        hash_to_paths = defaultdict(list)
        for p in comp:
            h = path_to_hash[p]
            hash_to_paths[h].append(p)
            
        for h, paths in hash_to_paths.items():
            paths.sort()
            rep = paths[0]
            rep_paths.add(rep)
            if len(paths) > 1:
                for duplicate in paths[1:]:
                    excl_records.append({
                        "file_path": duplicate,
                        "filename": Path(duplicate).name,
                        "class_name": path_to_class[duplicate],
                        "sha256": h,
                        "original_split": df_inv[df_inv["file_path"] == duplicate]["orig_split"].values[0],
                        "exclusion_reason": "EXACT_BYTE_DUPLICATE",
                        "group_id": gid,
                        "retained_representative_path": rep,
                        "retained_representative_filename": Path(rep).name
                    })
                    
    df_excluded = pd.DataFrame(excl_records)
    df_excluded.to_csv(REPORTS_DIR / "excluded_images.csv", index=False)
    print(f"Total exact duplicate files excluded: {len(df_excluded)}")
    assert len(df_excluded) == 31, f"Expected 31 excluded exact duplicates, got {len(df_excluded)}"
    
    excl_set = set(df_excluded["file_path"])
    
    # 5. Group-aware stratified split with seed=42
    # Clean members per group
    group_clean_members = {}
    for gid, comp in group_comp_map.items():
        clean_m = [p for p in comp if p not in excl_set]
        group_clean_members[gid] = clean_m
        
    rng = random.Random(seed)
    group_split_map = {}
    assigned_splits = {}
    
    split_counts_by_class = {c: {"train": 0, "validation": 0, "test": 0, "total": 0} for c in EXPECTED_CLASSES}
    
    for cname in EXPECTED_CLASSES:
        c_groups = groups_by_class[cname]
        c_groups.sort(key=lambda x: x[0])
        rng.shuffle(c_groups)
        
        # Total clean images in class
        total_clean_class = sum(len(group_clean_members[gid]) for gid, _ in c_groups)
        split_counts_by_class[cname]["total"] = total_clean_class
        
        target_test = int(round(0.15 * total_clean_class))
        target_val = int(round(0.10 * total_clean_class))
        
        curr_test = 0
        curr_val = 0
        
        for gid, _ in c_groups:
            k = len(group_clean_members[gid])
            if curr_test + k <= target_test or (curr_test < target_test and abs(curr_test + k - target_test) <= abs(curr_test - target_test)):
                chosen_split = "test"
                curr_test += k
            elif curr_val + k <= target_val or (curr_val < target_val and abs(curr_val + k - target_val) <= abs(curr_val - target_val)):
                chosen_split = "validation"
                curr_val += k
            else:
                chosen_split = "train"
                
            group_split_map[gid] = chosen_split
            split_counts_by_class[cname][chosen_split] += k
            for p in group_clean_members[gid]:
                assigned_splits[p] = chosen_split
                
    # 6. Save duplicate_groups.csv (all 3,829 files)
    group_lookup_rows = []
    for gid, comp in sorted(group_comp_map.items()):
        cname = path_to_class[comp[0]]
        assigned_sp = group_split_map[gid]
        for p in comp:
            is_excl = p in excl_set
            group_lookup_rows.append({
                "group_id": gid,
                "file_path": p,
                "filename": Path(p).name,
                "class_name": cname,
                "sha256": path_to_hash[p],
                "group_total_raw_files": len(comp),
                "group_clean_files": len(group_clean_members[gid]),
                "is_multi_image_group": (len(comp) > 1),
                "is_exact_duplicate_copy": is_excl,
                "is_clean_representative": (not is_excl),
                "assigned_split": assigned_sp if not is_excl else "EXCLUDED"
            })
    df_group_lookup = pd.DataFrame(group_lookup_rows)
    df_group_lookup.to_csv(MANIFESTS_DIR / "duplicate_groups.csv", index=False)
    print(f"Saved master group lookup (3,829 files) to manifests/duplicate_groups.csv")
    
    # 7. Generate clean split manifests
    clean_inv = df_inv[~df_inv["file_path"].isin(excl_set)].copy()
    clean_inv["group_id"] = clean_inv["file_path"].map(path_to_group)
    clean_inv["split"] = clean_inv["file_path"].map(assigned_splits)
    clean_inv["numeric_class_index"] = clean_inv["class_name"].map(CLASS_TO_IDX)
    
    manifest_cols = ["image_path", "class_name", "numeric_class_index", "group_id"]
    for sp_name, filename in [("train", "clean_train.csv"), ("validation", "clean_validation.csv"), ("test", "clean_test.csv")]:
        sp_df = clean_inv[clean_inv["split"] == sp_name].copy()
        if "image_path" in sp_df.columns:
            sp_df = sp_df.drop(columns=["image_path"])
        sp_df = sp_df.rename(columns={"file_path": "image_path"})
        sp_df = sp_df.sort_values(["numeric_class_index", "image_path"])
        out_csv = MANIFESTS_DIR / filename
        sp_df[manifest_cols].to_csv(out_csv, index=False)
        print(f"Saved {len(sp_df)} rows to manifests/{filename}")
        
    # 8. Save class_names.json
    with open(MANIFESTS_DIR / "class_names.json", "w") as f:
        json.dump(CLASS_TO_IDX, f, indent=2)
        
    # 9. Save all_images_audit.csv
    all_audit_rows = []
    for _, row in df_inv.iterrows():
        p = row["file_path"]
        is_excl = p in excl_set
        gid = path_to_group[p]
        if is_excl:
            audit_status = "EXCLUDED_EXACT_DUPLICATE"
            clean_split = "NONE"
        else:
            clean_split = assigned_splits[p]
            audit_status = f"CLEAN_{clean_split.upper()}"
            
        all_audit_rows.append({
            "file_path": p,
            "filename": row["filename"],
            "class_name": row["class_name"],
            "numeric_class_index": CLASS_TO_IDX[row["class_name"]],
            "file_size": row["file_size"],
            "width": row["width"],
            "height": row["height"],
            "format": row["format"],
            "sha256": row["sha256"],
            "original_split": row["orig_split"],
            "clean_status": audit_status,
            "clean_split": clean_split,
            "group_id": gid
        })
    df_all_audit = pd.DataFrame(all_audit_rows).sort_values(["numeric_class_index", "file_path"])
    df_all_audit.to_csv(MANIFESTS_DIR / "all_images_audit.csv", index=False)
    print(f"Saved all 3829 images audit to manifests/all_images_audit.csv")
    
    # 10. Save split_summary.csv
    summary_rows = []
    total_clean = len(clean_inv)
    for c in EXPECTED_CLASSES:
        c_stats = split_counts_by_class[c]
        c_tot = c_stats["total"]
        summary_rows.append({
            "class_name": c,
            "original_total": len(df_inv[df_inv["class_name"] == c]),
            "excluded_duplicates": len(df_excluded[df_excluded["class_name"] == c]),
            "clean_total": c_tot,
            "train_count": c_stats["train"],
            "train_pct": f"{c_stats['train'] / c_tot * 100:.2f}%",
            "val_count": c_stats["validation"],
            "val_pct": f"{c_stats['validation'] / c_tot * 100:.2f}%",
            "test_count": c_stats["test"],
            "test_pct": f"{c_stats['test'] / c_tot * 100:.2f}%"
        })
        
    tot_train = sum(split_counts_by_class[c]["train"] for c in EXPECTED_CLASSES)
    tot_val = sum(split_counts_by_class[c]["validation"] for c in EXPECTED_CLASSES)
    tot_test = sum(split_counts_by_class[c]["test"] for c in EXPECTED_CLASSES)
    
    summary_rows.append({
        "class_name": "TOTAL",
        "original_total": len(df_inv),
        "excluded_duplicates": len(df_excluded),
        "clean_total": total_clean,
        "train_count": tot_train,
        "train_pct": f"{tot_train / total_clean * 100:.2f}%",
        "val_count": tot_val,
        "val_pct": f"{tot_val / total_clean * 100:.2f}%",
        "test_count": tot_test,
        "test_pct": f"{tot_test / total_clean * 100:.2f}%"
    })
    pd.DataFrame(summary_rows).to_csv(REPORTS_DIR / "split_summary.csv", index=False)
    print(f"Saved split summary to reports/split_summary.csv")

if __name__ == "__main__":
    generate_unified_dataset_artifacts(seed=42)
