# Playing-area rectangle homography audit — 2026-10-27

Offline saved-image diagnostic only. No new physical measurement, jig, or camera capture was used. No production mapping was changed.

## Method and edge identity

The four image boundaries are Huber-fit to all existing cushion-straightness samples from the measured 1280×720 image. Top, bottom, left, and right each have nine smoothing/search-band variants; both the weak inner bottom transition and the nearby outer seam are retained. The physical nose is not independently identified by these brightness transitions. The nine ball references never fit or select a boundary.

A cloth-surface mapping is formed from the line intersections and the verified 2336.8 × 1155.7 mm rectangle. The saved segmentation centers correspond to balls with one known common radius. For the ball-center diagnostic, the same fixed radius offset is applied uniformly using the frozen checkerboard K; no individual height or depth is estimated, and only X/Y is reported.

## Nominal diagnostic

The pre-existing nominal edge extraction (σ=1.0, band shift 0) gives X/Y RMS **22.039 mm** against the nine physical references, with median **20.615 mm** and maximum **29.976 mm**. These references have unknown uncertainty. This is a report-only comparison; it does not establish the physical nose or accuracy.

| Boundary line | Samples | Line RMS (px) | 95th percentile (px) | Maximum (px) |
|---|---:|---:|---:|---:|
| top | 341 | 1.011 | 1.249 | 7.446 |
| bottom_nose | 319 | 0.911 | 1.391 | 2.003 |
| left | 123 | 0.136 | 0.275 | 0.329 |
| right | 137 | 2.151 | 5.068 | 5.277 |

The nominal four-corner homography and frozen checkerboard intrinsics do not reproduce one another exactly: the fixed-radius pose decomposition misses the four extracted corners by **2.17 px RMS** (max **3.05 px**). This limits confidence in projecting from the cloth boundary to the common ball-center plane.

| Ball | Mapped X (mm) | Mapped Y (mm) | Diagnostic residual (mm) |
|---|---:|---:|---:|
| middle-row-center-ball | 1158.76 | 590.82 | 20.61 |
| far-right-purple-middle-row-ball | 2299.20 | 592.36 | 19.85 |
| far-left-yellow-middle-row-ball | 5.07 | 593.27 | 29.98 |
| interior-purple-second-left-middle-row-ball | 561.10 | 596.18 | 23.89 |
| interior-green-second-right-middle-row-ball | 1740.12 | 591.07 | 17.50 |
| top-orange-solid-second-left-ball | 553.79 | 40.43 | 18.76 |
| top-red-striped-second-right-ball | 1741.82 | 40.18 | 12.42 |
| bottom-black-second-left-ball | 567.95 | 1152.64 | 25.52 |
| bottom-yellow-second-right-ball | 1743.41 | 1151.86 | 24.89 |

The existing segmentation-center homography fitted to those same nine references has training RMS **3.893 mm** and leave-one-out RMS **8.706 mm**. The nominal rectangle map is independent of the ball references, but the physical edge is uncertain; the two numbers have different fitting protocols and neither is independent accuracy evidence.

## Edge and corner sensitivity

The audit evaluates all 6,561 combinations of the nine top, bottom, left, and right extraction variants for each lower-edge family (13,122 mappings total). Each candidate mapping, its four extrapolated image corners, and all nine reference-error magnitudes are retained in `summary.json`.

| Lower edge choice | Valid mappings | Maximum corner variation from median (px), TL/TR/BR/BL | Maximum per-ball X/Y spread from median (mm) | Diagnostic RMS range (mm) |
|---|---:|---|---:|---|
| bottom_nose | 6561 / 6561 | 0.05, 11.18, 4.64, 0.77 | 23.81 | 19.40–22.12 |
| bottom_seam_alternative | 6561 / 6561 | 0.05, 11.18, 5.13, 0.18 | 23.30 | 33.76–41.84 |

Do not choose the lowest-error edge family. The seam is an explicit alternative transition, not the intended cushion nose. The wide variation is evidence that the current images do not identify the physical boundary tightly enough for a reliable metric map.

## Interpretation and limits

This is a geometric sensitivity experiment, not a qualified table calibration. The sampled lines are brightness/shadow transitions with no surveyed labels; the right side is notably jumpy, the lower inner line is weak, and corner points must be extrapolated around pockets. The frozen-intrinsics decomposition's nominal surface-corner reconstruction error is reported in the JSON and is another check on the fixed-radius projection correction. The saved segmentation ellipse centers are not independently verified projections of physical sphere centers.

The nine user-provided positions were applied only after all line candidates and mappings were computed. Their measurement uncertainty is unknown, so residuals cannot diagnose whether the measurements or the map are wrong. No metric asset, precise coordinates, or guidance behavior is activated.

## Reproduction

```sh
python thoughts/calibration/offline-playing-area-homography-audit-20261027.py --self-test
python thoughts/calibration/offline-playing-area-homography-audit-20261027.py
```

The runner requires NumPy and OpenCV Python; the saved calibration environment is `/tmp/billiards-checkerboard-env/bin/python`.
