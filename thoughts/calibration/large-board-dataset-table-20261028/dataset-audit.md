# Larger-board dataset audit — 2026-10-28

All eight retained JPEG and decoded-pixel hashes match the manifest and are unique. All sources and annotations are 3840 × 2160; every corner file matches its 54-corner manifest record. Every grid is finite, in bounds and has consistent nonzero cell winding.

| View | Median pitch px | Minimum pitch px | Contrast proxy | Laplacian variance proxy |
|---|---:|---:|---:|---:|
| view-001 | 71.64 | 66.93 | 109.0 | 558.1 |
| view-002 | 75.19 | 71.73 | 170.0 | 357.1 |
| view-003 | 80.02 | 73.05 | 151.0 | 526.8 |
| view-004 | 77.17 | 71.61 | 167.0 | 436.5 |
| view-005 | 69.64 | 42.00 | 174.0 | 876.9 |
| view-006 | 74.49 | 62.55 | 201.0 | 922.7 |
| view-007 | 76.30 | 62.78 | 147.0 | 782.3 |
| view-008 | 77.56 | 71.07 | 138.0 | 511.4 |

The montage shows center, left/right and upper/lower coverage, several perspective views and a near-front-facing portrait view 8. View 5 has only about 21 source pixels of estimated pattern margin at the image edge; its minimum pitch is 42 px. Other images have larger margins.

The board has visible page seams. These images cannot verify physical flatness, precise square dimensions, or alignment of assembled pages. Preserve both physical verification fields as pending. Exploratory fitting may test numerical consistency using nominal geometry; no loadable calibration is justified by this integrity audit.

The contrast and sharpness proxies include the checkerboard edges and printing; values cannot rank actual corner accuracy across differing poses. The full-resolution originals and annotated images remain unchanged.

Reproduce with `/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/offline-large-board-dataset-table-20261028.py --stage dataset`. Inspect `dataset-montage.jpg`; see JSON for exact inputs, checks and limitations.
