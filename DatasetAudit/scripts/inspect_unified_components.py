import os
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

AUDIT_DIR = Path(r"C:\Gitwork\Rice diseases\DatasetAudit")
REPORTS_DIR = AUDIT_DIR / "reports"
MANIFESTS_DIR = AUDIT_DIR / "manifests"

def build_full_connected_components():
    df_inv = pd.read_csv(REPORTS_DIR / "dataset_inventory.csv")
    df_exact = pd.read_csv(REPORTS_DIR / "exact_duplicate_groups.csv")
    df_near = pd.read_csv(REPORTS_DIR / "near_duplicate_candidates.csv")
    
    # 1. Edges from exact duplicates (31 pairs)
    exact_edges = []
    for gid, group in df_exact.groupby("group_id"):
        paths = group["file_path"].tolist()
        if len(paths) == 2:
            exact_edges.append((paths[0], paths[1]))
            
    # 2. Edges from confirmed near-duplicates (125 pairs)
    confirmed_near = df_near[df_near["status"] == "CONFIRMED_NEAR_DUPLICATE"]
    near_edges = []
    for _, row in confirmed_near.iterrows():
        near_edges.append((row["file_path_a"], row["file_path_b"]))
        
    print(f"Exact duplicate edges: {len(exact_edges)}")
    print(f"Confirmed near-duplicate edges: {len(near_edges)}")
    
    # Build unified graph
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
            
    print(f"Total images: {len(all_paths)}")
    print(f"Total connected components: {len(components)}")
    
    singletons = [c for c in components if len(c) == 1]
    multi = [c for c in components if len(c) > 1]
    print(f"Singleton groups (1 image): {len(singletons)}")
    print(f"Multi-image groups (>1 image): {len(multi)}")
    print(f"Images in multi-image groups: {sum(len(c) for c in multi)}")
    
    # Check class consistency in every component
    path_to_class = dict(zip(df_inv["file_path"], df_inv["class_name"]))
    class_conflicts = 0
    for idx, c in enumerate(components, 1):
        classes = {path_to_class[p] for p in c}
        if len(classes) > 1:
            class_conflicts += 1
            print(f"Group {idx} has class conflict: {classes}")
            
    print(f"Class conflicts across all components: {class_conflicts}")
    
    # Analyze multi-image groups
    # For each multi-image group, how many exact copies vs how many distinct hashes?
    path_to_hash = dict(zip(df_inv["file_path"], df_inv["sha256"]))
    group_summary = []
    for idx, comp in enumerate(multi, 1):
        cname = path_to_class[comp[0]]
        hashes = {path_to_hash[p] for p in comp}
        n_exact_redundant = len(comp) - len(hashes)
        group_summary.append({
            "group_id": f"GRP_{idx:04d}",
            "class_name": cname,
            "total_files": len(comp),
            "unique_hashes": len(hashes),
            "redundant_exact_copies": n_exact_redundant,
            "filenames": [Path(p).name for p in comp]
        })
    df_group_summary = pd.DataFrame(group_summary)
    print("\nMulti-image group sizes distribution:")
    print(df_group_summary["total_files"].value_counts().sort_index())
    print(f"Total redundant exact copies in multi-image groups: {df_group_summary['redundant_exact_copies'].sum()}")

if __name__ == "__main__":
    build_full_connected_components()

