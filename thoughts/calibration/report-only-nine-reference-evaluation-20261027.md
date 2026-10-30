# Report-only evaluation of the nine measured ball positions — 2026-10-27

## Result

No diagnostic ball-center mapping was fit. The final production-seed refiner
accepted **0 of the 9** manually proposed observations that have matching
physical references. All nine reject as `ambiguous_boundary`. The fitting
solver was therefore not run; there are no training residuals,
leave-one-position-out predictions, or accuracy statistics to report. No
calibration asset was emitted.

This is the result required by the fail-closed procedure when the selected
estimator supplies no usable image-center correspondences. It is not evidence
that the physical measurements are wrong.

## Inputs and coordinate convention

- Image: `table-grid-20261026-114241.jpg`, 1280 × 720, SHA-256
  `06ef40f52e0f93db17601585c4ed1d014895aaadcaf6b04c6abbd6e986d57e7b`.
- Proposal and image-center attempt: final Task D refiner revision 2, source
  SHA-256
  `602266310962d8a4703a264515c272dd024c731d1b45edc56ef04a5d2bfb1b82`.
- Detector boxes were not available for this frozen image. The proposals are
  the saved manual annotation-derived boxes recorded in the corpus; they are
  not detector output or exact center labels.
- The image intrinsics are the frozen exploratory zero-radial model, not a
  qualified runtime camera profile.
- Physical references use an upper-left origin at the cushion-nose
  intersection, +X image-right, +Y image-down, and millimeters. The playing
  area is 2336.8 × 1155.7 mm. Coordinates below are copied from the saved
  measurement record in this convention; they have not been converted to a
  different production origin.
- Each reference is marked complete in the measurement record, but its
  uncertainty is `null`; no numerical uncertainty is asserted here.

## Per-reference outcome

| Ball reference | Physical X/Y from upper-left (mm) | Refiner outcome | Qualified candidates |
|---|---:|---|---:|
| top-orange-solid-second-left-ball | (568.325, 28.575) | `ambiguous_boundary` | 2 |
| top-red-striped-second-right-ball | (1746.250, 28.575) | `ambiguous_boundary` | 8 |
| far-left-yellow-middle-row-ball | (28.575, 574.675) | `ambiguous_boundary` | 6 |
| interior-purple-second-left-middle-row-ball | (571.500, 574.675) | `ambiguous_boundary` | 4 |
| middle-row-center-ball | (1171.575, 574.675) | `ambiguous_boundary` | 4 |
| interior-green-second-right-middle-row-ball | (1746.250, 574.675) | `ambiguous_boundary` | 2 |
| far-right-purple-middle-row-ball | (2308.225, 574.675) | `ambiguous_boundary` | 2 |
| bottom-black-second-left-ball | (568.325, 1127.125) | `ambiguous_boundary` | 3 |
| bottom-yellow-second-right-ball | (1746.250, 1127.125) | `ambiguous_boundary` | 4 |

“Qualified candidates” counts individually qualified converged fit candidates
before the final ambiguity decision. None is a selected observation. The
per-case trace and outline evidence is retained in
[`converged-candidate-audit-20261027/`](converged-candidate-audit-20261027/).

## Why no mapping was fit

The accepted mapping requires image-center observations selected by the same
estimator revision that production will use. Substituting proposal midpoints,
centers accepted under non-production RANSAC seeds, older-estimator centers,
or unselected candidates would make the fit circular or inconsistent with the
production measurement contract. The separate seed audit found five accepted
outputs under four alternate seeds, but did not establish which result was
correct; those outputs are excluded.

Even if centers had been accepted, the current production calibration utility
requires at least twelve correspondences. The existing nine measurements do
not meet that activation contract. The nine points could support a
report-only diagnostic solve after valid centers exist, but that condition is
not met here. Do not weaken the minimum or manufacture additional
correspondences to make an asset possible.

## Next investigation

Continue with the existing saved images and traces. Determine why multiple
distinct boundary families pass individual quality gates and why accepted
fits vary with their deterministic starting seed. Keep the production seed,
measurement gates, raw-box contract, and unavailable-on-rejection behavior
unchanged while diagnosing it. Compare any proposed estimator change against
the frozen measured and reserved scenes before deciding whether it merits a
separate revision. This report requires no new camera capture, jig, or user
measurement.

The Intel GPU JPEG decoder finding remains independently adoptable; this
report does not affect that discovery. Target FPS and complete-pipeline
responsiveness remain unqualified.
