# Mark-grid calibration audit

This is an offline, diagnostics-only experiment. No production logic or camera settings changed.

## Decision

The saved rail dots strongly resemble the WPA 8-foot sight lattice, and the saved cloth marks are visually consistent with the head/foot spots. The standard supplies a useful candidate coordinate grid. However, WPA explicitly cautions that the competition specification may not apply to home tables, the user's measured width is 0.5 in smaller than the nominal 8-foot width, and the saved checkerboard intrinsics are not yet qualified. The measured results below therefore test feasibility but do not yet establish a production metric map.

The key check is not a same-plane rail homography: it decomposes that homography with saved checkerboard intrinsics, estimates one common rail-cap-to-cloth plane separation from the two visible cloth spots, and checks each spot against the other.

## Main measurements

- Rail sight offset used: 93.6625 mm; symmetric sensitivity allowance ±3.1750 mm (published specification text shows +3.175 mm, not a symmetric tolerance).
- Checkerboard-derived rail-to-cloth separation: 52.831 mm.
- Two-spot joint reprojection RMS: 1.341 px; max 1.388 px.
- Applying the rail-plane homography directly at those cloth XY positions misses the spots by 11.00, 10.57 px (RMS 10.786 px).
- Single-spot-to-other-spot checks:
  - Marker 1 estimates h=58.391 mm; predicts marker 2 with 2.296 px residual.
  - Marker 2 estimates h=46.257 mm; predicts marker 1 with 2.481 px residual.

## Rail-grid fit and independent checks

- `measured_720`: 10 long-rail marks; fit RMS 0.572 px; leave-one-sight-out RMS 0.986 px (max 1.564 px).
  - Held-out left rail marks: matched 2 expected quarter positions; assigned residuals 4.65, 4.30 px; unassigned detection None.
  - Held-out right rail marks: matched 3 expected quarter positions; assigned residuals 5.89, 5.46, 5.91 px; unassigned detection {'feature_id': 'right_rail_vertical_face_3', 'observed_px': [1267.1904761904761, 476.2857142857143]}.
- `framing_before_720`: 10 long-rail marks; fit RMS 0.588 px; leave-one-sight-out RMS 1.013 px (max 1.583 px).
  - Held-out left rail marks: matched 2 expected quarter positions; assigned residuals 4.59, 4.28 px; unassigned detection None.
  - Held-out right rail marks: matched 3 expected quarter positions; assigned residuals 6.04, 5.49, 6.05 px; unassigned detection {'feature_id': 'right_rail_vertical_face_3', 'observed_px': [1267.0, 476.375]}.
- An additional bottom-rail dot visible under glare was not in the seeded annotation set: the long-rail-only map predicts the 5/8 sight at (783.87, 690.45) px; visual center estimate (784.0, 690.0) px differs by 0.47 px. This is a qualitative confirmation, not a scored precise holdout.

## Ball-reference diagnostic only

The nine saved measured-ball references were not used to fit any parameter. Mapping coarse manually annotated box midpoints to the fixed ball-center-height plane gives radial error RMS 11.286 mm, p95 17.070 mm, max 17.615 mm. This is not an accuracy result: the physical reference uncertainties are unknown and the box midpoint is not a validated sphere center. See the per-ball rows and intrinsics sensitivity in `summary.json`.

## Interpretation and next step

The user can avoid measuring round balls: rail sights and marked head/foot spots can establish a candidate grid if this table follows the standard. The minimum next calibration capture is a larger rigid-backed checkerboard at the unchanged camera zoom/focus/resolution, moved through sharp views spanning the image center and corners with tilts about both axes. The saved rail/cloth-mark framing already supplies the plane bridge; use it as the held-out table-specific check after recalibrating intrinsics. Do not ship the map until the calibration model is stable across held-out views and the observed sight lattice/cloth-spot identities agree.

A full-table print is not required. A target roughly twice the present linear size (for example, a 20 × 14 in pattern with 2 in squares) would improve corner precision while remaining portable; several deliberate poses across center/corners are more useful than another small target view at the image center. No ball jig or round-ball measurements are needed.

## Source references

- WPA Rules of Play (effective 2025-09-15), §2.1 defines sights at 1/8 of length and 1/4 of width and identifies head/foot spots at string intersections: https://wpapool.com/wp-content/uploads/2025/10/2025.09.15-WPA-Rules-NP.pdf
- WPA Recommended Equipment Specifications, §6 gives the 8-foot 11.5-inch sight spacing and a 3 11/16-inch sight-to-cushion-nose offset with +3.175 mm in the published text; the specification says it may not apply to home-use tables: https://www.wpapool.com/wp-content/uploads/2024/01/RECOMMENDED-EQUIPMENT-SPECIFICATIONS.pdf

## Reproduction

Run `offline-mark-grid-calibration-audit-20261027.py`. It writes `summary.json` and this report. Synthetic homography recovery and fixed-plane ray round-trip checks must pass.
