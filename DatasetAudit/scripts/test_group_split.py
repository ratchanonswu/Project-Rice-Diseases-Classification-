import os
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

AUDIT_DIR = Path(r"C:\Gitwork\Rice diseases\DatasetAudit")
EXPECTED_CLASSES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight"
]

def simulate_group_split(seed=42):
    df_inv = pd.read_csv(AUDIT_DIR / "reports" / "dataset_inventory.csv")
    df_exact = pd.read_csv(AUDIT_DIR / "reports" / "exact_duplicate_groups.csv")
    df_near = pd.read_csv(AUDIT_DIR / "reports" / "near_duplicate_candidates.csv")
    
    # 1. Exact duplicates: exclude non-representatives
    excl_files = set(df_exact[~df_exact["is_representative"]]["file_path"])
    
    # Clean images
    clean_inv = df_inv[~df_inv["file_path"].isin(excl_files)].copy()
    
    # 2. Build groups from confirmed near duplicates
    confirmed_near = df_near[df_near["status"] == "CONFIRMED_NEAR_DUPLICATE"]
    adj = defaultdict(set)
    for _, r in confirmed_near.iterrows():
        p1 = r["file_path_a"]
        p2 = r["file_path_b"]
        if p1 not in excl_files and p2 not in excl_files:
            adj[p1].add(p2)
            adj[p2].add(p1)
            
    visited = set()
    components = []
    
    # Sort paths for deterministic grouping
    for p in sorted(clean_inv["file_path"]):
        if p not in visited:
            comp = []
            queue = [p]
            visited.add(p)
            while queue:
                curr = queue.pop()
                comp.append(curr)
                for neighbor in sorted(adj[curr]):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(sorted(comp))
            
    # Assign group IDs
    path_to_group = {}
    path_to_class = dict(zip(clean_inv["file_path"], clean_inv["class_name"]))
    
    groups_by_class = defaultdict(list)
    for idx, comp in enumerate(components, 1):
        gid = f"GRP_{idx:04d}"
        cname = path_to_class[comp[0]]
        for p in comp:
            path_to_group[p] = gid
        groups_by_class[cname].append((gid, comp))
        
    print(f"Total groups: {len(components)}")
    
    # Group-aware stratified split per class with seed=42
    rng = random.Random(seed)
    
    train_paths = []
    val_paths = []
    test_paths = []
    
    for cname in EXPECTED_CLASSES:
        c_groups = groups_by_class[cname]
        # Deterministic shuffle with seed
        # Sort first, then shuffle
        c_groups.sort(key=lambda x: x[0])
        rng.shuffle(c_groups)
        
        total_class_images = sum(len(comp) for _, comp in c_groups)
        target_test = int(round(0.15 * total_class_images))
        target_val = int(round(0.10 * total_class_images))
        
        curr_test = 0
        curr_val = 0
        
        c_test_groups = []
        c_val_groups = []
        c_train_groups = []
        
        for gid, comp in c_groups:
            k = len(comp)
            # Greedily assign to test if test needs it
            if curr_test + k <= target_test or (curr_test < target_test and abs(curr_test + k - target_test) <= abs(curr_test - target_test)):
                c_test_groups.append((gid, comp))
                curr_test += k
            elif curr_val + k <= target_val or (curr_val < target_val and abs(curr_val + k - target_val) <= abs(curr_val - target_val)):
                c_val_groups.append((gid, comp))
                curr_val += k
            else:
                c_train_groups.append((gid, comp))
                
        for _, comp in c_train_groups:
            train_paths.extend(comp)
        for _, comp in c_val_groups:
            val_paths.extend(comp)
        for _, comp in c_test_groups:
            test_paths.extend(comp)
            
        print(f"Class {cname:25s}: Total={total_class_images:4d} | Train={len([p for _, c in c_train_groups for p in c]):4d} | Val={curr_val:3d} | Test={curr_test:3d}")
        
    print("\nOVERALL SPLIT COUNTS:")
    n_tot = len(clean_inv)
    print(f"Train: {len(train_paths):4d} ({len(train_paths)/n_tot*100:.2f}%)")
    print(f"Val:   {len(val_paths):4d} ({len(val_paths)/n_tot*100:.2f}%)")
    print(f"Test:  {len(test_paths):4d} ({len(test_paths)/n_tot*100:.2f}%)")
    print(f"Total: {n_tot:4d}")
    
    # Check overlaps
    s_tr, s_va, s_te = set(train_paths), set(val_paths), set(test_paths)
    print(f"Path overlap Train & Val: {len(s_tr & s_va)}")
    print(f"Path overlap Train & Test: {len(s_tr & s_te)}")
    print(f"Path overlap Val & Test: {len(s_va & s_te)}")
    
    # Check group overlaps
    g_tr = {path_to_group[p] for p in train_paths}
    g_va = {path_to_group[p] for p in val_paths}
    g_te = {path_to_group[p] for p in test_paths}
    print(f"Group overlap Train & Val: {len(g_tr & g_va)}")
    print(f"Group overlap Train & Test: {len(g_tr & g_te)}")
    print(f"Group overlap Val & Test: {len(g_va & g_te)}")

if __name__ == "__main__":
    simulate_group_split()

