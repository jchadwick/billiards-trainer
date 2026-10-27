# Static table-feature registration audit — 2026-10-27

Offline saved-image registration only. No camera or new physical measurement was used. No production behavior changed.

## Coordinate and reference scope

The report contains pixel coordinates only, normalized to the 1280×720 frame. The 3840×2160 framing image is reduced exactly 3× before comparison. The inspected measurement, feasibility, corpus, and framing records contain the 2336.8 × 1155.7 mm playing-area dimensions and nine ball positions, but no exact physical X/Y coordinates for the visible rail dots or cloth markers. No spacing is assumed from table conventions. No metric table mapping is emitted.

Two high-contrast circular markers are visible on the cloth plane in the framing-before/after pair. They are covered by balls in the measured and alternate grid scenes, so they are recorded as occluded rather than inferred. White sights on the upper/lower rails are used for registration; side-rail sights and cloth markers are held out from that fit because they are on other physical planes.

## Results

### measured_to_alternate_720

Fit 6 upper/lower rail sights. Their training RMS is **0.051 px**. Held-out upper/lower rail transfer RMS is **0.209 px** (max 0.317 px).
Held-out side-face sight transfer RMS is **0.463 px** (max 0.847 px).

Held-out cloth marker `cloth_marker_1` status: `not_comparable`.
Held-out cloth marker `cloth_marker_2` status: `not_comparable`.

### framing_before_to_after_720_equivalent

Fit 6 upper/lower rail sights. Their training RMS is **0.010 px**. Held-out upper/lower rail transfer RMS is **0.094 px** (max 0.146 px).
Held-out side-face sight transfer RMS is **0.196 px** (max 0.299 px).

Held-out cloth marker `cloth_marker_1` transfers with **1.360 px** error; physical table coordinates are not recorded.
Held-out cloth marker `cloth_marker_2` transfers with **0.004 px** error; physical table coordinates are not recorded.

## Annotations

Every detected rail sight and cloth marker is listed with its image ID, plane class, refined pixel center, detector settings, and source hashes in `summary.json`. Balls are excluded from registration by construction: only seeded rail-sight neighborhoods and the two static cloth-marker neighborhoods are annotated.

## Interpretation and limits

A low pixel transfer error shows that static landmarks can register these saved frames consistently. It does not establish absolute table X/Y, physical ball-center accuracy, or that all points share one physical plane. The exact metric coordinates of the table dots are not present in the inspected project records, despite the user saying they had been measured precisely. Raised-rail sight points cannot substitute for cloth-plane calibration without their physical plane and coordinates.

The measured-to-alternate registration uses only upper/lower rail sights for fitting. It never transfers the measured arrangement's ball positions. The 720p/4K pair adds held-out vertical-side rail sights and visible cloth markers to test cross-plane pixel transfer; the prior 4K perimeter SIFT report is not duplicated as a metric calibration.

## Reproduction

```sh
python thoughts/calibration/offline-table-feature-registration-audit-20261027.py --self-test
python thoughts/calibration/offline-table-feature-registration-audit-20261027.py
```

The audit requires NumPy and OpenCV Python. The saved environment is `/tmp/billiards-checkerboard-env/bin/python`.
