import os
import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import scipy.fftpack

DATASET_ROOT = Path(r"C:\Gitwork\Rice diseases\Rice Diseases Classification\ResNet50\dataset")
AUDIT_DIR = Path(r"C:\Gitwork\Rice diseases\DatasetAudit")

def compute_phash(img_path, hash_size=8, highfreq_factor=4):
    img = Image.open(img_path).convert("L")
    img_size = hash_size * highfreq_factor
    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(img, dtype=float)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    # Convert to 64-bit uint
    return diff.flatten()

def compute_dhash(img_path, hash_size=8):
    img = Image.open(img_path).convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(img)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten()

def test_hashes():
    df_inv = pd.read_csv(AUDIT_DIR / "reports" / "dataset_inventory.csv")
    print(f"Loaded {len(df_inv)} images.")
    
    phashes = []
    dhashes = []
    for idx, row in df_inv.iterrows():
        p = compute_phash(row["file_path"])
        d = compute_dhash(row["file_path"])
        phashes.append(p)
        dhashes.append(d)
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(df_inv)} images...")
            
    P = np.array(phashes) # (3829, 64) boolean
    D = np.array(dhashes)
    
    print("Computing pairwise pHash distances...")
    # P is bool. We can compute hamming distance efficiently:
    # dist(i, j) = sum(P[i] != P[j])
    # For all pairs: P @ (1 - P).T ... or popcount
    # Since P is (3829, 64), pack into uint64
    P_packed = np.packbits(P, axis=1).view(np.uint64).flatten()
    
    # Let's inspect distance histogram within and across classes
    # We can do this block-by-block or full matrix:
    # With 3829, 3829 * 3829 * 1 byte is 14.6 MB! That fits easily in memory.
    P_int = P.astype(np.int8)
    # dist = sum(P_i != P_j) = 64 - sum(P_i == P_j)
    # Using matrix multiplication: (P_i - P_j)^2 = P_i^2 + P_j^2 - 2 P_i P_j = P_i + P_j - 2 (P @ P.T)
    # For binary 0/1: sum(p_i != p_j) = sum(p_i) + sum(p_j) - 2 * (P @ P.T)
    p_sums = P_int.sum(axis=1, keepdims=True) # (3829, 1)
    dot = P_int @ P_int.T # (3829, 3829)
    dist_matrix = p_sums + p_sums.T - 2 * dot # (3829, 3829)
    
    print(f"Dist matrix computed: shape {dist_matrix.shape}")
    
    # Check upper triangle
    i_idx, j_idx = np.triu_indices(len(df_inv), k=1)
    dists = dist_matrix[i_idx, j_idx]
    
    print("pHash distance distribution:")
    for d_thresh in range(0, 16):
        count = (dists == d_thresh).sum()
        cum_count = (dists <= d_thresh).sum()
        print(f"  dist == {d_thresh:2d}: {count:5d} pairs  (cumulative <= {d_thresh:2d}: {cum_count:5d} pairs)")

if __name__ == "__main__":
    test_hashes()

