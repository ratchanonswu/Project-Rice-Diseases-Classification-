import os
import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import scipy.fftpack

AUDIT_DIR = Path(r"C:\Gitwork\Rice diseases\DatasetAudit")
RESNET50_OUTPUTS = Path(r"C:\Gitwork\Rice diseases\Rice Diseases Classification\ResNet50\outputs")
CONTACT_DIR = AUDIT_DIR / "review" / "near_duplicate_contact_sheets"
CONTACT_DIR.mkdir(parents=True, exist_ok=True)

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

def compute_image_metrics(path1, path2):
    """Computes pixel-level difference metrics between two images after normalizing size."""
    try:
        im1 = Image.open(path1).convert("RGB")
        im2 = Image.open(path2).convert("RGB")
        
        # Check original size
        orig_size_match = (im1.size == im2.size)
        
        # Resize to standard 256x256 for pixel comparison
        im1_norm = np.asarray(im1.resize((256, 256), Image.Resampling.BILINEAR), dtype=float)
        im2_norm = np.asarray(im2.resize((256, 256), Image.Resampling.BILINEAR), dtype=float)
        
        mae = np.mean(np.abs(im1_norm - im2_norm))
        rmse = np.sqrt(np.mean((im1_norm - im2_norm) ** 2))
        max_diff = np.max(np.abs(im1_norm - im2_norm))
        
        return {
            "orig_size_match": orig_size_match,
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "max_diff": round(float(max_diff), 2)
        }
    except Exception as e:
        return {"orig_size_match": False, "mae": 999.0, "rmse": 999.0, "max_diff": 999.0}

def create_contact_sheet(pair_id, file_a, file_b, info_str, out_path):
    try:
        im1 = Image.open(file_a).convert("RGB").resize((300, 300))
        im2 = Image.open(file_b).convert("RGB").resize((300, 300))
        
        # Canvas: 650 wide x 380 high
        canvas = Image.new("RGB", (650, 380), color=(240, 240, 240))
        canvas.paste(im1, (20, 50))
        canvas.paste(im2, (330, 50))
        
        draw = ImageDraw.Draw(canvas)
        # Header text
        draw.text((20, 10), f"Candidate Pair: {pair_id} | {info_str}", fill=(0, 0, 0))
        draw.text((20, 355), Path(file_a).name[:40], fill=(0, 0, 0))
        draw.text((330, 355), Path(file_b).name[:40], fill=(0, 0, 0))
        
        canvas.save(out_path)
    except Exception as e:
        print(f"Error making contact sheet: {e}")

def main():
    df_inv = pd.read_csv(AUDIT_DIR / "reports" / "dataset_inventory.csv")
    train_df = pd.read_csv(RESNET50_OUTPUTS / "train.csv")
    train_df["split"] = "train"
    val_df = pd.read_csv(RESNET50_OUTPUTS / "validation.csv")
    val_df["split"] = "validation"
    test_df = pd.read_csv(RESNET50_OUTPUTS / "test.csv")
    test_df["split"] = "test"
    all_splits = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df_merged = df_inv.merge(all_splits[["image_path", "split"]], left_on="file_path", right_on="image_path", how="left")
    
    print("Computing pHashes...")
    phashes = [compute_phash(p) for p in df_merged["file_path"]]
    P = np.array(phashes, dtype=np.int8)
    p_sums = P.sum(axis=1, keepdims=True)
    dist_matrix = p_sums + p_sums.T - 2 * (P @ P.T)
    
    # We inspect candidate pairs:
    # All pairs with pHash distance <= 6 that are NOT exact sha256 matches
    i_idx, j_idx = np.where(np.triu((dist_matrix <= 6), k=1))
    
    candidate_records = []
    pair_counter = 0
    
    for i, j in zip(i_idx, j_idx):
        d = int(dist_matrix[i, j])
        row_i = df_merged.iloc[i]
        row_j = df_merged.iloc[j]
        is_exact = (row_i["sha256"] == row_j["sha256"])
        if is_exact:
            continue  # Exact duplicates handled in Phase B
            
        pair_counter += 1
        metrics = compute_image_metrics(row_i["file_path"], row_j["file_path"])
        
        # Categorize
        name_a = row_i["filename"]
        name_b = row_j["filename"]
        
        # Heuristics for relationship
        # 1. Nearly identical (d == 0 and MAE < 5.0)
        # 2. Burst sequence (similar filename timestamps like 202310... within seconds or (N) vs (N+1))
        # 3. Augmentation pair (aug_0_...)
        # 4. Merely similar appearance
        
        is_burst = False
        # Check if timestamps or numbering indicate capture sequence
        base_a = name_a.replace(" - Copy", "").replace("(0)", "").split(".")[0]
        base_b = name_b.replace(" - Copy", "").replace("(0)", "").split(".")[0]
        
        if (d == 0 and metrics["mae"] < 10.0):
            relation_type = "VISUALLY_IDENTICAL_OR_RECOMPRESSED"
            confidence = "HIGH"
            status = "CONFIRMED_NEAR_DUPLICATE"
        elif metrics["mae"] < 25.0 and d <= 4:
            relation_type = "CAPTURE_SEQUENCE_BURST"
            confidence = "HIGH"
            status = "CONFIRMED_NEAR_DUPLICATE"
        elif "aug_" in name_a or "aug_" in name_b:
            relation_type = "AUGMENTED_OR_SYNTHETIC_DERIVATIVE"
            confidence = "MEDIUM"
            status = "REVIEW_REQUIRED"
        elif d <= 2:
            relation_type = "HIGHLY_SIMILAR_FRAMING"
            confidence = "MEDIUM"
            status = "REVIEW_REQUIRED"
        else:
            relation_type = "SIMILAR_SYMPTOMS_ONLY"
            confidence = "LOW"
            status = "REVIEW_REQUIRED"
            
        pair_id = f"NEAR_DUP_PAIR_{pair_counter:03d}"
        
        # Generate contact sheet for d <= 4
        if d <= 4:
            sheet_name = f"{pair_id}_dist{d}_{row_i['class_name'].replace(' ', '_')}.jpg"
            out_sheet = CONTACT_DIR / sheet_name
            info = f"d={d}, MAE={metrics['mae']}, {row_i['class_name']}"
            create_contact_sheet(pair_id, row_i["file_path"], row_j["file_path"], info, out_sheet)
        
        candidate_records.append({
            "candidate_id": pair_id,
            "phash_distance": d,
            "similarity_method": "pHash_64bit_DCT",
            "class_name_a": row_i["class_name"],
            "class_name_b": row_j["class_name"],
            "labels_agree": (row_i["class_name"] == row_j["class_name"]),
            "file_path_a": row_i["file_path"],
            "file_path_b": row_j["file_path"],
            "split_a": row_i["split"],
            "split_b": row_j["split"],
            "cross_split": (row_i["split"] != row_j["split"]),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "orig_size_match": metrics["orig_size_match"],
            "relation_type": relation_type,
            "confidence": confidence,
            "status": status,
            "recommended_handling": "GROUP_TO_SINGLE_SPLIT" if status == "CONFIRMED_NEAR_DUPLICATE" else "PRESERVE_AND_REVIEW",
            "reason": f"pHash dist={d}, MAE={metrics['mae']}. Classified as {relation_type}."
        })
        
    df_candidates = pd.DataFrame(candidate_records)
    reports_dir = AUDIT_DIR / "reports"
    df_candidates.to_csv(reports_dir / "near_duplicate_candidates.csv", index=False)
    
    print(f"Total candidate pairs analyzed (pHash dist <= 6, differing sha256): {len(df_candidates)}")
    print("\nCandidates by status:")
    print(df_candidates["status"].value_counts())
    print("\nCandidates by relation_type:")
    print(df_candidates["relation_type"].value_counts())
    print("\nCross-split near duplicates:")
    print(df_candidates.groupby(["status", "cross_split"]).size())

if __name__ == "__main__":
    main()

