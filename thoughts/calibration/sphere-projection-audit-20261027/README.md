# Fixed-sphere projection audit — 2026-10-27

Offline saved-image diagnostic only. No camera was accessed, no new measurement or fixture was used, and no production behavior changed.

## Method

Each individually qualified source-space outline was fit as an ellipse. With the frozen 1280×720 zero-radial camera matrix, calibrated conic geometry was applied under the fixed 28.575 mm ball radius. The same fixed center plane is used for every ball; the report contains table X/Y only. One common rigid camera-to-table transform was fit from candidate assignments and the nine existing user-confirmed X/Y references, with user-facing Y-down restored at output.

The solver is deterministic and bounded. On the full nine-point fit it examined **912** initial candidate combinations across the four highest-area anchor quadrilaterals, then alternated nearest candidate assignment and proper rigid alignment. This is a bounded heuristic, not an exhaustive search across every nine-ball candidate combination; the JSON records every distinct converged assignment found, plus every individual candidate including unqualified/rejected fits.

## Results

Qualified outline candidates per reference: `{"bottom-black-second-left-ball": 3, "bottom-yellow-second-right-ball": 4, "far-left-yellow-middle-row-ball": 6, "far-right-purple-middle-row-ball": 2, "interior-green-second-right-middle-row-ball": 3, "interior-purple-second-left-middle-row-ball": 4, "middle-row-center-ball": 4, "top-orange-solid-second-left-ball": 2, "top-red-striped-second-right-ball": 8}`.

Best all-nine training X/Y RMS: **15.208 mm**; maximum per-point X/Y residual: **31.839 mm**.

The all-nine pose selected a candidate whose ellipse center was nearest the saved segmentation-ellipse center in **1/9** cases. That fixed center file is an alternate same-image algorithm, not physical truth.

Existing image-center homography baseline LOO RMS: **8.706 mm** (recorded report: 8.71 mm). It uses nine segmentation-derived centers and has the same unknown reference uncertainty.

For comparison, the sphere-conic pose produces an **oracle-best-candidate lower bound** with LOO X/Y RMS **18.652 mm**, mean **13.950 mm**, median **7.416 mm**, p95 **37.123 mm**, max **40.426 mm**. This is not an automatic localization result: for each held-out position, it scores all candidate projections against that position and selects the lowest-error candidate after seeing the held-out reference.

The existing segmentation-center homography has LOO X/Y RMS **8.706 mm** on the same positions. Even the oracle sphere result is worse in this diagnostic. No global infeasibility claim is made because the assignment search is heuristic.

| Held-out position | Oracle fit ID for lower bound | Oracle X/Y error (mm) | Homography X/Y error (mm) |
|---|---:|---:|---:|
| middle-row-center-ball | 49 | 5.234 | 9.150 |
| far-right-purple-middle-row-ball | 67 | 12.502 | 2.857 |
| far-left-yellow-middle-row-ball | 39 | 6.856 | 12.275 |
| interior-purple-second-left-middle-row-ball | 5 | 5.316 | 2.585 |
| interior-green-second-right-middle-row-ball | 3 | 4.225 | 2.747 |
| top-orange-solid-second-left-ball | 25 | 32.168 | 10.334 |
| top-red-striped-second-right-ball | 24 | 7.416 | 7.806 |
| bottom-black-second-left-ball | 67 | 11.408 | 13.982 |
| bottom-yellow-second-right-ball | 70 | 40.426 | 7.879 |

## Interpretation and limits

This experiment tests whether one camera pose and the fixed-size sphere constraint can make competing measured-ball outlines globally consistent with the nine known table positions. It does not establish the physical silhouette or center, because the refiner rejects the observations as ambiguous and the fitted candidates are generated from the same image evidence. The physical measurements have no stated uncertainty, and the frozen checkerboard intrinsics remain weakly identified. Consequently, a low training or leave-one-out residual cannot be reported as millimeter accuracy or used to activate a production measurement.

The sphere LOO values are an optimistic oracle lower bound: each held-out point's known physical position is used only after pose fitting to score all that point's outlines and retain the least residual. The method has not shown how to choose that boundary without the physical answer. The prior homography instead uses one fixed segmentation-derived pixel center per ball. Both compare against the same user-reported points and share uncertain references, so neither establishes accuracy.

Synthetic tests exercise exact tangent-cone recovery, 200-degree arcs, a 0.15-pixel contour noise level, table edges, and a competing contour. A small contour perturbation produces several millimeters of X/Y error in this geometry, showing how outline errors can affect the result. These ideal synthetic cases do not characterize the saved RGB edge error.

The complete JSON includes every refiner fit and rejection, ellipse/conic quality metrics, selected associations, every distinct converged assignment found by the bounded heuristic, each candidate's projected table X/Y under each leave-one-out pose, all X/Y residuals, input hashes, and runtime versions. It contains no height, depth, or out-of-plane residual values.

## Reproduction

```sh
python thoughts/calibration/offline-sphere-projection-audit-20261027.py --self-test
python thoughts/calibration/offline-sphere-projection-audit-20261027.py
```

The audit requires NumPy and OpenCV Python. The current saved environment is `/tmp/billiards-checkerboard-env/bin/python`.
