# Rice Leaf Disease Dataset Deduplication, Leakage Prevention & Audit Report

**Date of Audit:** October 10, 2026  
**Auditor:** Senior Machine Learning Data Engineer  
**Output Directory:** `C:\Gitwork\Rice diseases\DatasetAudit`  
**Target Comparison:** Fair ResNet50 vs MobileNetV2 Rice Leaf Disease Classification  
**Status:** Audit Completed — Independent Validation: **PASS (11/11 Checks)**

---

## 1. Executive Summary

This audit establishes a leak-proof, reproducible, and verifiable dataset split across the six rice leaf disease classes. The original dataset suffered from raw duplicate file proliferation (`IMG_... - Copy.jpg`, duplicate sequential numbering) that historically caused data leakage between Train, Validation, and Test partitions.

### Key Metrics Summary
- **Initial Raw Images Scanned:** 3,829 images (100% valid and readable; 0 corrupt or empty files).
- **Exact Duplicate Groups (SHA-256):** 31 groups (62 total files; 31 redundant duplicate files).
- **Label Conflict Groups (Exact & Near Duplicate):** 0 groups (100% intra-group label agreement).
- **Confirmed Near-Duplicate Groups:** 41 multi-image groups (103 images) comprising burst captures and recompressed twin images.
- **Unresolved Candidate Pairs Flagged for Review:** 79 candidate pairs (flagged as `REVIEW_REQUIRED`, kept intact in their respective classes).
- **Excluded Images from Clean Experiment:** 31 redundant duplicate files (all original files remain strictly preserved on disk).
- **Final Clean Dataset Size:** 3,798 images.
- **Final Split Distribution:**
  - **Train:** 2,847 images (**74.96%**, target ~75%)
  - **Validation:** 381 images (**10.03%**, target ~10%)
  - **Test:** 570 images (**15.01%**, target ~15%)
- **Data Leakage in Clean Split:**
  - Path overlap: **0**
  - Exact SHA-256 hash overlap: **0**
  - Confirmed near-duplicate group overlap: **0**
- **Final Audit Verdict:** **READY FOR EXPERIMENTAL ADOPTION** (Full compliance with leakage-prevention criteria).

---

## 2. Before-and-After Dataset Statistics

### Class-by-Class Distribution Comparison

| Canonical Class Name | Canonical Index | Raw Source Count | Excluded Duplicates | Clean Dataset Total | Clean Train (74.96%) | Clean Val (10.03%) | Clean Test (15.01%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Bacterial Leaf Blight** | 0 | 636 | 0 | 636 | 477 | 64 | 95 |
| **Brown Spot** | 1 | 646 | 9 | 637 | 477 | 64 | 96 |
| **Healthy Rice Leaf** | 2 | 653 | 3 | 650 | 487 | 65 | 98 |
| **Leaf Blast** | 3 | 634 | 15 | 619 | 464 | 62 | 93 |
| **Leaf Scald** | 4 | 628 | 0 | 628 | 471 | 63 | 94 |
| **Sheath Blight** | 5 | 632 | 4 | 628 | 471 | 63 | 94 |
| **TOTAL** | — | **3,829** | **31** | **3,798** | **2,847** | **381** | **570** |

*Note:* Bacterial Leaf Blight and Leaf Scald contained zero exact byte-duplicates in the raw folder. Leaf Blast contained the highest number of redundant exact copies (15 copies), followed by Brown Spot (9 copies), Sheath Blight (4 copies), and Healthy Rice Leaf (3 copies).

---

## 3. Exact Duplicate Findings (SHA-256 Analysis)

Exact duplicate detection was performed using full-file SHA-256 byte hashing across all 3,829 images.

### Findings Breakdown
- **Total Unique Hashes:** 3,798
- **Duplicate Hash Groups:** 31 groups (each group consists of exactly 2 identical files; 62 files total).
- **Label Agreement:** 31 / 31 groups (100%) share identical class labels.
- **Historical Partitioning of Duplicates:**
  - **Strictly within Train:** 15 groups
  - **Strictly within Validation:** 2 groups
  - **Strictly within Test:** 0 groups
  - **Crossing Train and Validation:** 5 groups (Data leakage in legacy split)
  - **Crossing Train and Test:** 9 groups (Test contamination in legacy split)
  - **Crossing Validation and Test:** 0 groups

### The 9 Legacy Train $\leftrightarrow$ Test Contamination Pairs
1. `IMG_20231018_143841 - Copy.jpg` (Train) $\leftrightarrow$ `IMG_20231018_143841.jpg` (Test) [Leaf Blast]
2. `Brown_spot (132).jpg` (Train) $\leftrightarrow$ `Brown_spot (131).jpg` (Test) [Brown Spot]
3. `IMG_20231018_143750.jpg` (Train) $\leftrightarrow$ `IMG_20231018_143750 - Copy.jpg` (Test) [Leaf Blast]
4. `IMG_20231018_143844 - Copy.jpg` (Train) $\leftrightarrow$ `IMG_20231018_143844.jpg` (Test) [Leaf Blast]
5. `IMG_20231018_143839.jpg` (Train) $\leftrightarrow$ `IMG_20231018_143839 - Copy.jpg` (Test) [Leaf Blast]
6. `IMG_20231018_143855.jpg` (Train) $\leftrightarrow$ `IMG_20231018_143855 - Copy.jpg` (Test) [Leaf Blast]
7. `IMG_20231006_163221.jpg` (Train) $\leftrightarrow$ `IMG_20231006_163221 - Copy.jpg` (Test) [Healthy Rice Leaf]
8. `Healthy_rice_leaf (7).jpg` (Train) $\leftrightarrow$ `Healthy_rice_leaf (6).jpg` (Test) [Healthy Rice Leaf]
9. `Brown_spot (134).jpg` (Train) $\leftrightarrow$ `Brown_spot (133).jpg` (Test) [Brown Spot]

### Deduplication Policy & Handling
To avoid synthetic inflation and bias toward duplicated images:
1. For each of the 31 exact duplicate groups, exactly **one representative** was retained deterministically (the file with the lexicographically smallest path).
2. The remaining 31 duplicate copies were **excluded from the clean manifests** and cataloged in `reports/excluded_images.csv`.
3. **Zero original files were deleted, moved, or renamed.**

---

## 4. Near-Duplicate Findings (Perceptual Hash & Pixel Distance)

Perceptual duplicate detection was executed using standard 64-bit DCT perceptual hashing (`pHash`) on normalized 32×32 grayscale projections, complemented by normalized 256×256 RGB Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

### Methodology and Parameter Selection
- **Hash Size:** 8×8 low-frequency DCT coefficients (excluding DC term median thresholding).
- **Hamming Distance Matrix:** Evaluated across all $\frac{3,829 \times 3,828}{2} = 7,328,706$ image pairs.
- **Candidate Screening Threshold:** Hamming Distance $\le 6$.
- **Pair Categorization Logic:**
  - **Visually Identical / Recompressed Copies:** Distance $= 0$ and $\text{MAE} < 10.0$ (differing only by JPEG compression artifacts or EXIF metadata).
  - **Capture Sequence / Burst Shots:** Consecutive timestamps or sequential photo numbering representing the identical physical leaf under slight camera movement/lighting shifts ($\text{MAE} < 25.0$ and $\text{Distance} \le 4$).
  - **Augmented Derivatives:** Files prefixed with `aug_` paired with raw or other augmented images.
  - **Similar Symptoms / Different Leaves:** Distance $\le 6$ with high MAE/RMSE representing distinct plant samples sharing typical lesion patterns.

### Candidate Near-Duplicate Statistics (204 Pairs with Differing SHA-256)
- **Hamming Distance Distribution:**
  - Distance $= 0$: 10 pairs (non-identical bytes, but identical perceptual hash)
  - Distance $= 2$: 50 pairs
  - Distance $= 4$: 72 pairs
  - Distance $= 6$: 72 pairs
- **Confirmed Near-Duplicates:** 125 pairs (118 capture sequence bursts, 7 recompressed twins).
- **Unresolved / Review Required:** 79 pairs (52 symptom similarities on different leaves, 27 augmented derivatives).
- **Legacy Cross-Split Near-Duplicate Leakage:** 52 confirmed near-duplicate pairs crossed between Train, Validation, and Test in the original experiment.

### Connected Component Grouping
All 125 confirmed near-duplicate pairs were resolved into **41 multi-image groups** (containing 103 clean images, ranging from 2 to 11 images per cluster).
- In the new clean split, **each group is treated as an indivisible atom** assigned strictly to a single split partition via group-aware stratified splitting.
- Visual contact sheets for all candidate pairs with Distance $\le 4$ were generated and saved to `review/near_duplicate_contact_sheets/`.

---

## 5. Label Conflict Findings

- **Exact Duplicate Label Conflicts:** **0**
- **Near-Duplicate Label Conflicts (Distance $\le 6$):** **0**
- **Conclusion:** There are no ambiguous or conflicting class assignments in the dataset. All duplicate groups belong unambiguously to a single canonical class.

---

## 6. Split Integrity Verification (Phase F Pass/Fail)

An independent validation script (`scripts/phase_f_validate_splits.py`) evaluated the generated manifests against 11 strict criteria:

| Check # | Verification Criterion | Result | Status |
|---|---|---|:---:|
| 1 | Manifest Schema Consistency (`image_path`, `class_name`, `numeric_class_index`, `group_id`) | Identical across all 3 CSVs | **PASS** |
| 2 | Physical File Existence on Disk | 3,798 / 3,798 paths exist | **PASS** |
| 3 | Image Readability & Corruption Integrity | 0 unreadable / corrupt files | **PASS** |
| 4 | Internal Path Uniqueness within Manifests | 0 duplicate rows per CSV | **PASS** |
| 5 | Cross-Split Image Path Overlap | $\text{Tr} \cap \text{Val} = 0, \text{Tr} \cap \text{Test} = 0, \text{Val} \cap \text{Test} = 0$ | **PASS** |
| 6 | Cross-Split Exact SHA-256 Hash Overlap | 0 shared byte hashes between splits | **PASS** |
| 7 | Cross-Split Near-Duplicate Group Leakage | 0 shared group IDs between splits | **PASS** |
| 8 | Full Representation of All Six Classes | All 6 classes present in Train, Val, Test | **PASS** |
| 9 | Canonical Class Index Mapping (`[0..5]`) | Strictly matched to canonical order | **PASS** |
| 10 | Excluded Images Isolation | 0 excluded images leaked into clean split | **PASS** |
| 11 | Complete Inventory Accounting | $3,798 + 31 = 3,829$ (100% accounted) | **PASS** |

---

## 7. Reproducibility Details

To ensure exact reproducibility across machines and future audits:
- **Random Seed:** `42`
- **Python Version:** `3.11.9`
- **Core Dependencies:**
  - `numpy`: 2.4.6
  - `pandas`: 3.0.6
  - `pillow`: 12.3.0
  - `scipy`: 1.17.1
  - `scikit-learn`: 1.9.1
- **Hashing Algorithms:**
  - Exact: SHA-256 (64 KB buffered binary read)
  - Perceptual: 64-bit DCT pHash (`scipy.fftpack.dct` 2D orthogonal DCT on 32×32 LANCZOS resized grayscale, median threshold on 8×8 low-frequency block)
- **Pipeline Scripts:**
  - `scripts/phase_a_inventory.py` (Inventory scan)
  - `scripts/phase_b_exact_duplicates.py` (Exact duplicate analysis)
  - `scripts/phase_c_deep_inspect.py` (pHash, MAE/RMSE, and contact sheets)
  - `scripts/generate_clean_artifacts.py` (Deterministic group-aware splitting)
  - `scripts/phase_f_validate_splits.py` (Independent audit verification)

---

## 8. Excluded Images Traceability

A total of 31 duplicate files were excluded from the clean experimental manifests. Every exclusion is cataloged in `reports/excluded_images.csv` with its SHA-256 hash, original split, and corresponding retained representative image path.

### Summary by Class:
- **Leaf Blast (15 excluded):** Redundant `IMG_... - Copy.jpg` files.
- **Brown Spot (9 excluded):** Duplicated sequential files (e.g. `Brown_spot (132).jpg`, `Brown_spot (134).jpg`).
- **Sheath Blight (4 excluded):** `IMG_... - Copy.jpg` and duplicate sequence files.
- **Healthy Rice Leaf (3 excluded):** `Healthy_rice_leaf (7).jpg`, `Healthy_rice_leaf (9).jpg`, `IMG_20231006_163221 - Copy.jpg`.
- **Bacterial Leaf Blight (0 excluded):** No exact byte duplicates present.
- **Leaf Scald (0 excluded):** No exact byte duplicates present.

> **CRITICAL ASSURANCE:** None of the 31 excluded original image files were deleted, moved, or altered. They remain completely preserved at their original locations on disk.

---

## 9. Limitations & Disclosures

1. **Unresolved Candidate Pairs (79 pairs):** 79 pairs with $4 \le \text{pHash Distance} \le 6$ exhibited structural similarity but lacked proof of common capture origin (e.g. different lesion locations on different leaves, or synthetic augmentations). Rather than risking unnecessary data loss, these were retained in their natural classes and documented in `reports/manual_review_required.csv`.
2. **Approximate Split Proportions:** Due to atomic group-aware clustering (where all images in a 2–11 image burst sequence are held together in one split partition), the split proportions slightly deviate from exact 75.00% / 10.00% / 15.00%:
   - Train: **74.96%** (2,847 images, deviation: $-0.04\%$)
   - Validation: **10.03%** (381 images, deviation: $+0.03\%$)
   - Test: **15.01%** (570 images, deviation: $+0.01\%$)
   This negligible variation was necessary to ensure 100% prevention of near-duplicate leakage.

---

## 10. Final Recommendation

The clean dataset manifests generated in `DatasetAudit\manifests`:
- `clean_train.csv` (2,847 images)
- `clean_validation.csv` (381 images)
- `clean_test.csv` (570 images)

are **fully validated, free of cross-split data leakage, completely reproducible, and mathematically aligned**. They are ready to be adopted for a definitive, scientifically rigorous ResNet50 vs MobileNetV2 head-to-head comparison whenever the research team chooses to do so.

