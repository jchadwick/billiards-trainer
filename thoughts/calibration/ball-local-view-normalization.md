# Local-view circle normalization - offline test

The fixed physical ball diameter remains 57.15 mm. This experiment changes only local 2D image coordinates. No sphere depth/height is estimated, no application Z output is produced, and no correction is activated.

## Method

Frozen zero-radial intrinsics: fx=696.17, fy=695.36, cx=644.18, cy=349.67. A Rodrigues rotation maps the coarse ball image ray onto a virtual optical axis; H=Kpatch R K^-1 maps contour pixels into a local virtual patch. A three-parameter robust circle is fitted and its center projected back. Three aiming iterations refine the patch direction. Pixel radii may differ between patches; physical diameter does not change. No full-frame image warp is performed.

Segmented convex-envelope points and previously selected gradient/ellipse-consensus points are reused from the source-image analysis. This isolates fitting constraints; it does not repair bad edge selection or supply an independent center truth. White stripes, rail shadows, and bridged missing arcs remain problematic.

## Source-image comparison

| Ball | Raw segment aspect | Local segment aspect | Segment center shift px | Raw gradient aspect | Local gradient aspect | Gradient center shift px |
|---|---:|---:|---:|---:|---:|---:|
| top-left-pocket-ball | 1.658 | 1.326 | 0.131 | 1.314 | 1.078 | 0.263 |
| top-orange-solid-second-left-ball | 1.156 | 1.026 | 0.151 | 1.176 | 1.029 | 0.095 |
| top-middle-pocket-ball | 1.215 | 1.143 | 0.330 | 1.078 | 1.016 | 0.134 |
| top-red-striped-second-right-ball | 1.141 | 1.029 | 0.180 | 1.172 | 1.027 | 0.190 |
| top-right-cue-pocket-ball | 1.097 | 1.214 | 0.171 | 1.328 | 1.002 | 0.229 |
| far-left-yellow-middle-row-ball | 1.267 | 1.053 | 0.408 | 1.281 | 1.012 | 0.175 |
| interior-purple-second-left-middle-row-ball | 1.060 | 1.027 | 0.124 | 1.364 | 1.273 | 1.081 |
| middle-row-center-ball | 1.013 | 1.012 | 0.023 | 1.274 | 1.277 | 0.809 |
| interior-green-second-right-middle-row-ball | 1.103 | 1.191 | 0.130 | 2.039 | 1.895 | 3.150 |
| far-right-purple-middle-row-ball | 1.252 | 1.042 | 0.502 | 1.336 | 1.045 | 0.277 |
| bottom-left-pocket-ball | 1.189 | 1.133 | 0.470 | 1.322 | 1.011 | 0.218 |
| bottom-black-second-left-ball | 1.008 | 1.147 | 0.715 | 1.268 | 1.106 | 0.653 |
| bottom-middle-pocket-ball | 1.113 | 1.037 | 0.310 | 1.363 | 1.369 | 0.868 |
| bottom-yellow-second-right-ball | 1.076 | 1.181 | 0.884 | 1.178 | 1.023 | 0.165 |
| bottom-right-pocket-ball | 1.188 | 1.169 | 0.304 | 1.414 | 1.075 | 0.518 |

## Actual source-pixel patches

[Original versus local-view crop montage](ball-local-view-normalization-crops.png) shows left yellow, center red control, and right purple, each on the same 65 x 65 native-pixel canvas enlarged exactly 3x by nearest-neighbor for inspection. The local patch is inverse-sampled from source RGB through the camera rotation homography with the same 696 px virtual focal scale for every ball. There is no circular mask, diameter equalization, or per-ball resizing. Neighbor clutter and shadows remain visible. Center directions come from the fitted source observations and remain uncertain.

## Nine-point center diagnostics

Each method is evaluated with its own simple H and leave-one-position-out refitting. These discrepancies include unknown physical measurement/contact error and are not verified mm accuracy. Pocket balls remain outside physical ground truth.

| Fit | Training RMS mm | Held-out RMS mm |
|---|---:|---:|
| segmentation_free_ellipse | 3.970 | 8.948 |
| segmentation_local_circle | 3.776 | 8.471 |
| gradient_free_ellipse | 3.811 | 8.008 |
| gradient_local_circle | 4.305 | 8.460 |

## Exact synthetic sphere-contour check

Synthetic generation uses an arbitrary sphere radius/range only to construct exact tangent rays and known projected center. The tested estimator receives pixels and K only. At four image positions, noiseless local normalization circularizes the exact contour and recovers its known center within 0.00001 px. With independently seeded 0.3 px contour noise, errors remain below 0.15 px in these four tests. This verifies transformation/inversion and limited recovery, not real-image physical accuracy. JSON retains every test and shape ratio.

## Limits and practical conclusion

Clean off-axis ball outlines become closer to circular in local coordinates, consistent with ordinary viewing geometry. This supports an image-domain local circularity prior without changing physical diameter. The fitted-center differences and held-out results determine whether it actually improves localization here; stronger shape constraints cannot fix contours contaminated by stripe, shadow, cushion, or pocket edges. Near-center points should change little. Review the source-image cyan segmentation-constrained circle projections versus orange gradient-constrained projections.

The top-left pocket segmentation remains a known highlight-fragment failure and is retained/flagged rather than promoted to a valid circle. Bad contour fits may normalize poorly or have large center shifts; successful circle fitting is not an acceptance criterion by itself. Corner/pocket physical accuracy remains unverified. No new measurement is requested. Local normalization is a bounded candidate for future robust ROI localization, not a reason to activate global radial lens warping.

## Verification

Exact synthetic tangent-contour recovery with and without pixel noise, finite transforms, image-homography inversion round trips, original source dimensions and JSON output are verified by the script. No production file, dependency, calibration state or existing report is changed.
