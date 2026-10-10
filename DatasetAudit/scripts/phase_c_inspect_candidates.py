import os
import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import scipy.fftpack

AUDIT_DIR = Path(r"C:\Gitwork\Rice diseases\DatasetAudit")
RESNET50_OUTPUTS = Path(r"C:\Gitwork\Rice diseases\Rice Diseases Classification\ResNet50\outputs")

def compute_phash(img_path, hash_size=8, highfreq_factor=4):
    img = Image.open(img_path).convert("L")
    img_size = hash_size * highfreq_factor
    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(img, dtype=float)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return diff.flatten()

def inspect_candidates():
    df_inv = pd.read_csv(AUDIT_DIR / "reports" / "dataset_inventory.csv")
    
    # Load splits
    train_df = pd.read_csv(RESNET50_OUTPUTS / "train.csv")
    train_df["split"] = "train"
    val_df = pd.read_csv(RESNET50_OUTPUTS / "validation.csv")
    val_df["split"] = "validation"
    test_df = pd.read_csv(RESNET50_OUTPUTS / "test.csv")
    test_df["split"] = "test"
    all_splits = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df_merged = df_inv.merge(all_splits[["image_path", "split"]], left_on="file_path", right_on="image_path", how="left")
    
    phashes = [compute_phash(p) for p in df_merged["file_path"]]
    P = np.array(phashes, dtype=np.int8)
    p_sums = P.sum(axis=1, keepdims=True)
    dist_matrix = p_sums + p_sums.T - 2 * (P @ P.T)
    
    # Check all pairs with dist <= 6
    i_idx, j_idx = np.where(np.triu(dist_matrix <= 6, k=1))
    
    pairs = []
    for i, j in zip(i_idx, j_idx):
        d = dist_matrix[i, j]
        row_i = df_merged.iloc[i]
        row_j = df_merged.iloc[j]
        is_exact = (row_i["sha256"] == row_j["sha256"])
        same_class = (row_i["class_name"] == row_j["class_name"])
        same_split = (row_i["split"] == row_j["split"])
        
        pairs.append({
            "idx_a": i,
            "idx_b": j,
            "phash_distance": int(d),
            "is_exact_sha256": is_exact,
            "same_class": same_class,
            "class_a": row_i["class_name"],
            "class_b": row_j["class_name"],
            "same_split": same_split,
            "split_a": row_i["split"],
            "split_b": row_j["split"],
            "file_a": row_i["file_path"],
            "file_b": row_j["file_path"],
            "filename_a": row_i["filename"],
            "filename_b": row_j["filename"]
        })
        
    df_pairs = pd.DataFrame(pairs)
    print(f"Total pairs with pHash distance <= 6: {len(df_pairs)}")
    print(f"Exact duplicates (sha256 match): {df_pairs['is_exact_sha256'].sum()}")
    print(f"Near duplicate candidates (differing sha256): {(~df_pairs['is_exact_sha256']).sum()}")
    
    print("\nBreakdown of near duplicate candidates by pHash distance:")
    print(df_pairs[~df_pairs['is_exact_sha256']]['phash_distance'].value_counts().sort_index())
    
    print("\nCheck if any candidates have conflicting classes:")
    conflicts = df_pairs[~df_pairs['same_class']]
    print(f"Conflicting class pairs (dist <= 6): {len(conflicts)}")
    if len(conflicts) > 0:
        print(conflicts[["phash_distance", "class_a", "class_b", "filename_a", "filename_b"]])
        
    print("\nNon-exact pairs with dist == 0:")
    zero_dist_non_exact = df_pairs[(df_pairs['phash_distance'] == 0) & (~df_pairs['is_exact_sha256'])]
    for _, r in zero_dist_non_exact.iterrows():
        print(f"  [{r['class_a']}] {r['filename_a']} <-> {r['filename_b']} (split: {r['split_a']} vs {r['split_b']})")
        
    print("\nNon-exact pairs with dist == 2:")
    two_dist_non_exact = df_pairs[(df_pairs['phash_distance'] == 2) & (~df_pairs['is_exact_sha256'])]
    print(f"Total: {len(two_dist_non_exact)}")
    for _, r in two_dist_non_exact.head(15).iterrows():
        print(f"  [{r['class_a']}] {r['filename_a']} <-> {r['filename_b']} (splits: {r['split_a']} vs {r['split_b']})")
        
    # Check cross-split near duplicates for dist <= 4
    cross_split_near = df_pairs[(~df_pairs['is_exact_sha256']) & (~df_pairs['same_split']) & (df_pairs['phash_distance'] <= 4)]
    print(f"\nCross-split near duplicates with dist <= 4: {len(cross_split_near)}")
    for _, r in cross_split_near.iterrows():
        print(f"  Dist {r['phash_distance']}: [{r['class_a']}] {r['filename_a']} ({r['split_a']}) <-> {r['filename_b']} ({r['split_b']})")

if __name__ == "__main__":
    inspect_candidates()

