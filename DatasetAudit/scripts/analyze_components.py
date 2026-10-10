import os
from pathlib import Path
import pandas as pd
import numpy as np

AUDIT_DIR = Path(r"C:\Gitwork\Rice diseases\DatasetAudit")

def analyze_groups():
    df_inv = pd.read_csv(AUDIT_DIR / "reports" / "dataset_inventory.csv")
    df_exact = pd.read_csv(AUDIT_DIR / "reports" / "exact_duplicate_groups.csv")
    df_near = pd.read_csv(AUDIT_DIR / "reports" / "near_duplicate_candidates.csv")
    
    # Confirmed near duplicates
    confirmed_near = df_near[df_near["status"] == "CONFIRMED_NEAR_DUPLICATE"]
    
    # Build graph of connections
    # We want to identify groups
    # First: exact duplicates. 
    # Exact duplicates: 31 groups of 2 images each (62 images).
    # In each exact duplicate group, 1 is representative, 1 is excluded.
    rep_files = set(df_exact[df_exact["is_representative"]]["file_path"])
    excl_files = set(df_exact[~df_exact["is_representative"]]["file_path"])
    
    print(f"Total raw images: {len(df_inv)}")
    print(f"Exact duplicate excluded copies: {len(excl_files)}")
    print(f"Clean candidate images after exact deduplication: {len(df_inv) - len(excl_files)}")
    
    # What about confirmed near-duplicates?
    # Are any of the confirmed near-duplicates recompressed identical copies that should also be deduplicated?
    # Let's check pairs with pHash == 0 and MAE < 1.0 (like Sheath Blight (19) vs (54), and (91) vs (92))
    zero_dist = df_near[(df_near["phash_distance"] == 0) & (df_near["mae"] < 1.0)]
    print(f"Recompressed copies with pHash == 0 and MAE < 1.0: {len(zero_dist)} pairs")
    for _, r in zero_dist.iterrows():
        print(f"  {Path(r['file_path_a']).name} <-> {Path(r['file_path_b']).name}")
        
    # Adjacency list for grouping
    # Nodes are all clean candidate images (or all images)
    from collections import defaultdict
    adj = defaultdict(set)
    
    # Add confirmed near-duplicate edges
    for _, r in confirmed_near.iterrows():
        p1 = r["file_path_a"]
        p2 = r["file_path_b"]
        adj[p1].add(p2)
        adj[p2].add(p1)
        
    # Find connected components among clean images
    visited = set()
    components = []
    
    clean_images = [p for p in df_inv["file_path"] if p not in excl_files]
    
    for p in clean_images:
        if p not in visited:
            comp = []
            queue = [p]
            visited.add(p)
            while queue:
                curr = queue.pop()
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in excl_files and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)
            
    singletons = [c for c in components if len(c) == 1]
    multi = [c for c in components if len(c) > 1]
    print(f"\nConnected components among {len(clean_images)} clean images:")
    print(f"Total components (groups): {len(components)}")
    print(f"Singleton groups (1 image): {len(singletons)}")
    print(f"Multi-image groups (>1 image): {len(multi)}")
    print(f"Total images in multi-image groups: {sum(len(c) for c in multi)}")
    
    # Group size distribution
    sizes = [len(c) for c in multi]
    print(f"Multi-image group sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.2f}")
    
    # Check if all images in each component have the same class
    path_to_class = dict(zip(df_inv["file_path"], df_inv["class_name"]))
    class_mismatches = 0
    for c in components:
        classes = {path_to_class[p] for p in c}
        if len(classes) > 1:
            class_mismatches += 1
            print(f"Class mismatch in component: {classes}")
    print(f"Component class mismatches: {class_mismatches}")

if __name__ == "__main__":
    analyze_groups()

