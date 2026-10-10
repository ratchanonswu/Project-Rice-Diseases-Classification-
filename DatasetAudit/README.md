# DatasetAudit: Rice Leaf Disease Dataset Deduplication & Leakage Prevention

This directory contains the audit artifacts, reports, and clean split manifests for the six-class rice leaf disease dataset used in the ResNet50 vs MobileNetV2 benchmark.

## Directory Structure

```text
DatasetAudit/
├── reports/
│   ├── dataset_inventory.csv          # Baseline inventory of all 3,829 images
│   ├── exact_duplicate_groups.csv     # 31 exact SHA-256 duplicate groups (62 files)
│   ├── near_duplicate_candidates.csv  # 204 near-duplicate candidate pairs (pHash <= 6)
│   ├── cross_split_leakage.csv        # Detailed audit of historical cross-split leakage
│   ├── label_conflicts.csv            # Audit of class label conflicts (0 found)
│   ├── excluded_images.csv            # 31 excluded redundant duplicate files & reasons
│   ├── manual_review_required.csv     # 79 candidate pairs flagged for manual review
│   ├── split_summary.csv              # Before-and-after split distribution comparison
│   ├── audit_report.md                # Full narrative research audit report
│   └── audit_summary.json             # Machine-readable audit summary
├── manifests/
│   ├── clean_train.csv                # Clean training split (2,847 images, 74.96%)
│   ├── clean_validation.csv           # Clean validation split (381 images, 10.03%)
│   ├── clean_test.csv                 # Clean test split (570 images, 15.01%)
│   ├── all_images_audit.csv           # Master audit log for all 3,829 raw images
│   ├── duplicate_groups.csv           # Master group mapping (3,736 groups)
│   └── class_names.json               # Canonical class-to-index mapping
├── review/
│   └── near_duplicate_contact_sheets/ # Visual comparison contact sheets (dist <= 4)
├── scripts/
│   ├── phase_a_inventory.py           # Phase A inventory & baseline scan
│   ├── phase_b_exact_duplicates.py    # Phase B SHA-256 exact duplicate detection
│   ├── phase_c_deep_inspect.py        # Phase C pHash, MAE/RMSE & contact sheets
│   ├── generate_clean_artifacts.py    # Phase D/E group-aware split generator
│   └── phase_f_validate_splits.py     # Phase F independent validation suite
└── README.md                          # This documentation file
```

## How to Adopt the Clean Manifests

To train or re-evaluate ResNet50 and MobileNetV2 on the clean, leakage-free dataset:

1. Backup existing manifests in `ResNet50\outputs\` and `MobileNetV2\outputs\` (if desired).
2. Copy or point to the clean manifests:
   - `clean_train.csv` $\rightarrow$ `train.csv`
   - `clean_validation.csv` $\rightarrow$ `validation.csv`
   - `clean_test.csv` $\rightarrow$ `test.csv`
3. Execute standard training or evaluation pipelines without code modifications.

## Integrity Guarantees

- **Zero Path Overlap:** $\text{Train} \cap \text{Val} = 0, \text{Train} \cap \text{Test} = 0, \text{Val} \cap \text{Test} = 0$.
- **Zero Exact SHA-256 Leakage:** Exact duplicate files never cross split boundaries.
- **Zero Near-Duplicate Group Leakage:** Multi-image burst capture sequences are assigned atomically to single splits.
- **Original Data Unchanged:** 100% of original image files remain untouched on disk.

