# Offline nine-position feasibility analysis

This is exploratory analysis, not an activated calibration. Six unmeasured pocket-adjacent balls are excluded; no further pocket measurements are required now. Corner and pocket accuracy remains unverified.

## Method

Seeds select 49×49 source-RGB regions. Local cloth chromaticity is estimated from the outer annulus; chromaticity difference or substantial darkness selects ball pixels. A connected component nearest the seed supplies outer-boundary pixels after flood-filling internal stripe/highlight holes and taking a convex envelope to bridge stripe gaps connected to the exterior. Envelope segments can bridge missing arcs and should not be mistaken for independently observed edges. An iteratively reweighted conic fit estimates ellipse centers without forcing circles. The annotated image shows the actual selected boundaries and centers. Three segmentation thresholds probe sensitivity, not physical uncertainty. Shadows, rail contact, stripes, JPEG edges and nonelliptic silhouettes can bias these centers. The source resolution is used throughout.

Models compare an 8-parameter homography with a homography plus one image-centered radial coordinate correction. Image center and normalization are fixed rather than estimated. The radial coefficient is searched over [-0.18, 0.18]; each leave-one-position-out fold fits only its eight training positions. This is a restricted 2D empirical model, not a lens-intrinsics calibration. No Z model is used.

## Results

| Model | Training RMS mm | Leave-one-out RMS mm | Median mm | Max mm |
|---|---:|---:|---:|---:|
| homography | 3.89 | 8.71 | 7.88 | 13.98 |
| homography_plus_one_radial_parameter | 3.76 | 13.05 | 9.64 | 22.38 |

| Ball | Fitted pixel center | Threshold sensitivity px | H held-out mm | Radial held-out mm |
|---|---|---:|---:|---:|
| 1: middle-row-center-ball | 643.39, 361.29 | 0.09 | 9.15 | 9.17 |
| 2: far-right-purple-middle-row-ball | 1199.25, 362.61 | 0.37 | 2.86 | 20.84 |
| 3: far-left-yellow-middle-row-ball | 95.31, 361.89 | 0.08 | 12.27 | 22.38 |
| 4: interior-purple-second-left-middle-row-ball | 357.71, 363.55 | 0.22 | 2.58 | 2.45 |
| 5: interior-green-second-right-middle-row-ball | 924.97, 361.70 | 0.27 | 2.75 | 2.73 |
| 6: top-orange-solid-second-left-ball | 352.44, 96.57 | 0.17 | 10.33 | 9.64 |
| 7: top-red-striped-second-right-ball | 927.81, 93.48 | 0.07 | 7.81 | 11.01 |
| 8: bottom-black-second-left-ball | 362.70, 627.40 | 1.21 | 13.98 | 13.98 |
| 9: bottom-yellow-second-right-ball | 924.54, 631.11 | 0.77 | 7.88 | 9.50 |

## Interpretation and limits

The extra radial parameter does not earn its complexity: its held-out RMS is worse despite slightly lower training error. Keep the homography as the comparison baseline; do not activate either model from this experiment. Bottom black has the largest segmentation-threshold center sensitivity (see the sensitivity table), and left/top/bottom held-out discrepancies deserve review without automatic removal. All recorded inch-to-mm conversions pass consistency checks; that cannot establish that physical measurements are correct.

Held-out discrepancies mix image localization, physical placement/measurement error, table geometry and model inadequacy; they are not verified camera accuracy. Training errors are not validation. Measurement uncertainty is unknown; shared table scale and rail-contact assumptions create correlated errors. Eighth-inch reporting granularity does not establish a numerical error bound. No point was automatically removed. Large per-point errors should prompt a repeat measurement and image capture, not automatic rejection.

Nine points occupy only three rows. The extreme corner/pocket regions lie outside their convex hull, and some leave-one-out folds extrapolate at edge positions. An unconstrained thin-plate spline could interpolate the nine training points but is unsupported by this dataset. Independent off-grid validation and repeated frames are needed before selecting or deploying a nonlinear model. Fixed physical diameter is a modeling input, not established by these center fits; a center mapping alone does not guarantee circular rectified silhouettes.

Next: review the annotated contours, repeat any anomalous measured positions, use one movable ball for independent accessible off-grid center/edge and interior corner-quadrant positions safely away from pockets, and accept a more complex mapping only if it improves held-out error reliably. Pocket measurements can remain excluded.

## Reproduction

Run this file with the bundled Python runtime (NumPy and Pillow). No production dependencies or configuration are modified. JSON conversion checks and synthetic homography and radial-model recovery pass. Raw results include model matrices, fold coefficients and boundary-fit diagnostics.
