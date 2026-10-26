# Ball outline and center consistency - original source image

No production correction is activated. All balls retain one fixed 57.15 mm physical diameter. Apparent ellipse diameters below describe source pixels only. Source: `table-grid-20261026-114241.jpg`; original nine physical measurements are used only with that associated snapshot.

## Methods

Three chromaticity thresholds segment each source-RGB ROI against locally estimated cloth. A convex envelope bridges internal stripe gaps; its ellipse center is compared with the segmented bounding-box midpoint. These are segmentation-derived boxes, not archived YOLO detector boxes; this experiment does not directly measure current detector midpoint bias. This can also bridge invalid shadow/rail arcs and is not a true-center estimator by definition. Independently, channel-vector Sobel gradients are sampled radially over 8-22 px from coarse ROI seeds. RANSAC-like six-point ellipse hypotheses select consensus edges, then refit. This explicitly allows failure or disagreement. Dropping each quadrant probes arc sensitivity; this is not a statistical confidence interval. No upsampling or learned precision is assumed.

## Source-image findings

| Ball | Apparent ellipse diameters px | Major angle deg | Bbox/ellipse offset px | Threshold sensitivity px | Gradient/seg offset px | Gradient arc sensitivity px |
|---|---|---:|---:|---:|---:|---:|
| 1: top-left-pocket-ball | 7.5 x 12.5 | 13 | 0.39 | 11.50 | 9.90 | 1.04 |
| 2: top-orange-solid-second-left-ball | 29.8 x 34.4 | 36 | 0.44 | 0.18 | 0.35 | 0.18 |
| 3: top-middle-pocket-ball | 32.8 x 39.8 | 91 | 0.98 | 0.22 | 4.56 | 0.20 |
| 4: top-red-striped-second-right-ball | 29.7 x 33.9 | 144 | 0.19 | 0.12 | 0.17 | 0.17 |
| 5: top-right-cue-pocket-ball | 37.2 x 40.8 | 162 | 0.73 | 0.24 | 3.23 | 0.11 |
| 6: far-left-yellow-middle-row-ball | 30.0 x 38.0 | 5 | 0.34 | 0.06 | 0.39 | 0.43 |
| 7: interior-purple-second-left-middle-row-ball | 27.9 x 29.6 | 4 | 0.22 | 0.25 | 2.31 | 0.86 |
| 8: middle-row-center-ball | 27.4 x 27.7 | 30 | 0.23 | 0.03 | 2.82 | 0.45 |
| 9: interior-green-second-right-middle-row-ball | 26.6 x 29.4 | 91 | 0.19 | 0.32 | 0.69 | 1.68 |
| 10: far-right-purple-middle-row-ball | 28.9 x 36.2 | 177 | 1.08 | 0.41 | 1.97 | 0.29 |
| 11: bottom-left-pocket-ball | 36.1 x 42.9 | 145 | 0.40 | 0.30 | 4.04 | 0.28 |
| 12: bottom-black-second-left-ball | 29.4 x 29.7 | 99 | 0.91 | 1.94 | 1.49 | 0.23 |
| 13: bottom-middle-pocket-ball | 34.7 x 38.7 | 95 | 1.32 | 0.26 | 1.59 | 0.34 |
| 14: bottom-yellow-second-right-ball | 32.6 x 35.1 | 176 | 1.05 | 0.17 | 1.40 | 0.11 |
| 15: bottom-right-pocket-ball | 36.7 x 43.6 | 13 | 0.61 | 0.51 | 7.27 | 0.77 |

Off-axis diagnostic uses frozen optical principal point (644.18,349.67) and approximate focal length 696 px. JSON records major-axis alignment with the radial image direction, radius and sqrt(1+(radius/f)^2), a small-sphere apparent-aspect approximation only. This characterizes a 2D image pattern without estimating ball height or changing physical diameter; segmentation/occlusion can confound it. Near-circular angles are unreliable.

## Nine-point homography diagnostic

Each method gets its own simple homography fit to the same nine measured positions; leave-one-position-out refits only the other eight. These are discrepancy diagnostics against unknown-uncertainty measurements, not accuracy estimates. They cannot select a physically correct center method without independent validation. The baseline segmentation-ellipse H also converts inter-method pixel shifts to illustrative table-distance shifts in the JSON; that conversion is not physical truth.

| Center method | Training RMS mm | Held-out RMS mm |
|---|---:|---:|
| segmentation_bbox | 3.56 | 7.90 |
| segmentation_ellipse | 3.97 | 8.95 |
| gradient_ellipse | 3.81 | 8.01 |

| Measured ball | Bbox held-out mm | Seg ellipse held-out mm | Gradient held-out mm |
|---|---:|---:|---:|
| top-orange-solid-second-left-ball | 8.61 | 10.69 | 7.97 |
| top-red-striped-second-right-ball | 9.44 | 7.45 | 10.86 |
| far-left-yellow-middle-row-ball | 8.94 | 13.10 | 9.43 |
| interior-purple-second-left-middle-row-ball | 2.53 | 2.62 | 3.38 |
| middle-row-center-ball | 8.64 | 9.07 | 8.75 |
| interior-green-second-right-middle-row-ball | 3.12 | 2.92 | 2.53 |
| far-right-purple-middle-row-ball | 5.88 | 2.16 | 5.75 |
| bottom-black-second-left-ball | 10.75 | 14.64 | 8.66 |
| bottom-yellow-second-right-ball | 8.78 | 7.92 | 10.21 |

## Interpretation

Visual review finds concrete failure modes: top-left pocket segmentation selected a tiny internal highlight/fragment (7.5 x 12.5 px ellipse), not the whole ball, with 11.5 px threshold sensitivity. This result is retained and flagged as a localization failure. Top-middle and corner balls show 3-10 px inter-method disagreements and can be contaminated by pocket/shadow edges. Even the measured striped middle balls can move 2-3 px under independent gradient fitting, consistent with internal-edge ambiguity. Bottom black threshold sensitivity is about 1.94 px and its segmented-ellipse held-out discrepancy is 14.64 mm; this deserves review but is not automatically discarded.

For a clean radial-pattern example, the far-left measured yellow ellipse aspect is about 1.27 and major-axis direction about 5 degrees from horizontal, close to the image-radial direction; its small-sphere apparent-aspect approximation is about 1.27. The far-right measured purple ellipse aspect is about 1.25 and its major axis is nearly horizontal. These observations are compatible with a fixed-diameter round ball having position-dependent apparent shape; they do not justify changing physical radius or estimating ball Z. Green stripe segmentation and near-circular balls provide less reliable orientation evidence.

The earlier 8.71 mm homography held-out RMS was obtained with another robust conic implementation on convex-envelope segmentations. Differences here quantify method dependence rather than a new physical truth. A 1 px center shift maps to roughly 2 mm at this image scale, so several pixels of shadow/rail ambiguity can explain a material portion of an 8-14 mm discrepancy. It cannot establish that all error is localization: physical measurements, contact assumptions, and the global map also contribute. No difficult ball is discarded.

Near-round ellipse angles are unstable and should not be interpreted as meaningful orientation. Rail-contact silhouettes and corner-pocket shadows remain difficult; the six unmeasured pocket balls are diagnostic only. White cue surfaces and white stripes require outer-contour reasoning rather than treating white patches or numbers as complete balls. Gradient methods can reduce color dependence but may instead lock onto stripes, shadow borders, or cushion edges. Review cyan/orange disagreement and green inliers in the annotated original image.

The synthetic test confirms only limited known-center recovery on an ideal ellipse with a shadow and stripe. Real JPEG anti-aliasing, rail contact, and partial silhouettes remain unvalidated. At approximately 13 px radius, subpixel fitted output is not proof of subpixel correctness.

Recommendation: retain native-resolution ROIs, publish raw bbox midpoint separately from refined outer-boundary observation, use robust arc selection and uncertainty/occlusion gating, and validate localization separately from XY calibration. Keep physical diameter fixed in table coordinates. Do not add radial image warping to force these raw outlines into circles. More reliable boundary evidence and independent validation should precede production integration.

## Verification

JSON decoding, nine physical-coordinate inch conversions, source dimensions, finite fits, and a synthetic known-center/shadow/stripe test are checked by the script. No physical measurement, production file, configuration, or dependency is changed.
