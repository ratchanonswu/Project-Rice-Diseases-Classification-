import os
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

AUDIT_DIR = Path(r"C:\Gitwork\Rice diseases\DatasetAudit")

df_cand = pd.read_csv(AUDIT_DIR / "reports" / "near_duplicate_candidates.csv")
zero_d = df_cand[df_cand["phash_distance"] == 0]

print(f"Total candidate pairs with pHash == 0: {len(zero_d)}")
for idx, r in zero_d.iterrows():
    f1, f2 = r["file_path_a"], r["file_path_b"]
    im1 = Image.open(f1)
    im2 = Image.open(f2)
    s1, s2 = os.path.getsize(f1), os.path.getsize(f2)
    print(f"\nPair {r['candidate_id']}:")
    print(f"  A: {Path(f1).name} | Size: {s1} bytes | Res: {im1.size} | Mode: {im1.mode}")
    print(f"  B: {Path(f2).name} | Size: {s2} bytes | Res: {im2.size} | Mode: {im2.mode}")
    print(f"  Class: {r['class_name_a']} | Splits: {r['split_a']} vs {r['split_b']}")
    print(f"  MAE: {r['mae']} | RMSE: {r['rmse']}")

