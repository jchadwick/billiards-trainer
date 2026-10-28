# Saved-table transfer under larger-board optics — 2026-10-28

All registration uses ten existing long-rail white sight neighborhoods at 720-equivalent scale. Six fixed sights fit an affine transform and four outer sights are held out; when a sight is not detected, the remaining matching observations retain the same split. The additional direct-coordinate and corner checks constrain the unchanged-framing hypothesis. An affine transform is sufficient for this check and avoids an unstable projective estimate when only one lower-rail training sight remains. Green/red points in the registration montage denote fitting/held-out sights.

| New frame | Reference | Direct displacement RMS / max px720 | Held-out transfer RMS px720 |
|---|---|---:|---:|
| view-001 | framing_after_4k | 0.151 / 0.273 | 0.155 |
| view-001 | measured_720 | 0.108 / 0.233 | 0.119 |
| view-001 | framing_before_720 | 0.153 / 0.250 | 0.150 |
| view-002 | framing_after_4k | 0.222 / 0.373 | 0.233 |
| view-002 | measured_720 | 0.179 / 0.345 | 0.200 |
| view-002 | framing_before_720 | 0.210 / 0.332 | 0.211 |
| view-003 | framing_after_4k | 0.174 / 0.335 | 0.135 |
| view-003 | measured_720 | 0.144 / 0.257 | 0.135 |
| view-003 | framing_before_720 | 0.172 / 0.350 | 0.110 |
| view-004 | framing_after_4k | 0.156 / 0.335 | 0.189 |
| view-004 | measured_720 | 0.140 / 0.257 | 0.188 |
| view-004 | framing_before_720 | 0.154 / 0.350 | 0.174 |
| view-005 | framing_after_4k | 0.075 / 0.140 | 0.113 |
| view-005 | measured_720 | 0.106 / 0.204 | 0.127 |
| view-005 | framing_before_720 | 0.073 / 0.140 | 0.107 |
| view-006 | framing_after_4k | 0.144 / 0.273 | 0.198 |
| view-006 | measured_720 | 0.133 / 0.257 | 0.250 |
| view-006 | framing_before_720 | 0.145 / 0.257 | 0.161 |
| view-007 | framing_after_4k | 0.111 / 0.190 | 0.118 |
| view-007 | measured_720 | 0.124 / 0.210 | 0.151 |
| view-007 | framing_before_720 | 0.110 / 0.190 | 0.116 |
| view-008 | framing_after_4k | 0.113 / 0.170 | 0.175 |
| view-008 | measured_720 | 0.134 / 0.217 | 0.127 |
| view-008 | framing_before_720 | 0.099 / 0.170 | 0.174 |

Approximate same-framing support: **True**. The fitted affine transforms and individual correspondences are retained in JSON. Intrinsics are scaled by exactly 1/3 only for this diagnostic comparison; no arbitrary rail-plane projective transform is applied to a camera matrix.

## Mark-grid comparison

Original and source-relocalized (0.30-pitch corner windows) optical candidates each undistort the same saved pixel observations with their own K and D, fit the standard fractional rail grid, decompose the resulting ideal-image homography, and fit a common rail-to-cloth nuisance separation using the two existing cloth spots. Held-out prediction fits the separation from one spot and predicts the other; scores below are re-distorted back to the original observed pixel domain. The separation fit minimizes ideal-image pixel error.

| Model | Rail fit RMS px720 | Rail leave-one-out RMS px720 | Cloth joint RMS px720 | Spot 1→2 / 2→1 px720 |
|---|---:|---:|---:|---:|
| zero | 0.588 | 1.013 | 2.415 | 4.509 / 4.329 |
| k1 | 4.406 | 8.152 | 1.043 | 0.994 / 1.194 |
| k1k2 | 3.279 | 6.105 | 0.794 | 1.018 / 1.127 |
| relocalized-0.3/zero | 0.588 | 1.013 | 2.489 | 4.647 / 4.458 |
| relocalized-0.3/k1 | 4.516 | 8.352 | 1.060 | 1.022 / 1.247 |
| relocalized-0.3/k1k2 | 3.233 | 6.023 | 0.801 | 1.066 / 1.179 |

## Sensitivity and interpretation

Every usable full fit and withheld-board fit is evaluated at the nominal sight offset and ±3.175 mm. A separate full-fit-only variant substitutes the published 46-inch width to expose reliance on a standard-table assumption. It does not replace the user’s 45.5-inch measurement. JSON retains each residual and inferred rail/cloth separation; none is an application Z estimate.

- zero: across withheld-board optics and offset variants, one-spot prediction errors range 0.160–13.618 px720. Nominal joint rail/cloth nuisance separation is 62.183 mm.
- k1: across withheld-board optics and offset variants, one-spot prediction errors range 0.933–4.785 px720. Nominal joint rail/cloth nuisance separation is 74.821 mm.
- k1k2: across withheld-board optics and offset variants, one-spot prediction errors range 0.781–6.028 px720. Nominal joint rail/cloth nuisance separation is 66.997 mm.
- relocalized-0.3/zero: across withheld-board optics and offset variants, one-spot prediction errors range 0.056–13.235 px720. Nominal joint rail/cloth nuisance separation is 62.329 mm.
- relocalized-0.3/k1: across withheld-board optics and offset variants, one-spot prediction errors range 0.897–4.621 px720. Nominal joint rail/cloth nuisance separation is 74.976 mm.
- relocalized-0.3/k1k2: across withheld-board optics and offset variants, one-spot prediction errors range 0.838–6.895 px720. Nominal joint rail/cloth nuisance separation is 66.272 mm.

## Frozen-candidate coordinate sensitivity

The following envelopes convert the original rejected source centers through the provisional maps. They do not rerun the refiner under new optics, select an outline, qualify coordinates, or estimate ball height. Each aiming iteration is kept separate. The shared known-radius projection offset is constant for every ball.

| Ball | Aiming iteration | Nominal full-model envelope mm | Full/fold/offset envelope mm |
|---|---:|---:|---:|
| bottom-black-second-left-ball | 1 | 9.028 | 23.698 |
| bottom-yellow-second-right-ball | 1 | 14.819 | 28.526 |
| far-left-yellow-middle-row-ball | 1 | 41.291 | 52.894 |
| far-right-purple-middle-row-ball | 1 | 38.717 | 57.224 |
| interior-green-second-right-middle-row-ball | 1 | 4.604 | 20.166 |
| interior-green-second-right-middle-row-ball | 2 | 4.877 | 20.389 |
| interior-purple-second-left-middle-row-ball | 1 | 9.166 | 23.256 |
| middle-row-center-ball | 1 | 9.448 | 22.671 |
| top-orange-solid-second-left-ball | 1 | 7.156 | 19.514 |
| top-red-striped-second-right-ball | 1 | 11.457 | 24.847 |

These envelopes expose uncertainty in the coordinate transform as well as differences among frozen candidates. They are not confidence bounds or physical accuracy estimates. Unknown board geometry, home-table layout and reference uncertainty remain outside the envelope.

A low checkerboard score alone cannot qualify the table map. Compare the rail/felt behavior above with the optical holdouts. Standard sight identities, unknown home-table offsets, two cloth spots and unverified target construction prevent independent full-table millimeter accuracy claims. Any improvement or degradation is diagnostic rather than permission to activate a model.

No ball positions are fitted from references and no production position is emitted. Conditional rejected-candidate coordinates are sensitivity diagnostics only. The source images and original annotations are read-only. Reproduce with `/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/offline-large-board-dataset-table-20261028.py --stage table`; run `--self-test` for synthetic topology, registration and distortion/scale consistency.
