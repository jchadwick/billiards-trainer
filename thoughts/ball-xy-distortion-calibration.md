# Ball X/Y localization with the existing camera

Date: 2026-10-25
Status: investigated design; not implemented or physically validated.

## Scope and verification first

Treat balls as equal-diameter objects on a 2D table coordinate grid. Publish
accurate centers in millimeters and one configured physical diameter. Do not
estimate Z. Keep the existing camera. Follow `project-memory.md`.

Before changing production geometry, record independently measured ball-center
X/Y positions, including central, edge, corner, rail-adjacent, and touching-ball
cases. Reserve entire physical positions for validation, not just different
frames of training positions. Record measurement uncertainty. Report median,
95th-percentile and maximum center error, regional bias, stationary jitter,
failure/coverage rates, and added latency. Accuracy thresholds must be
configuration, and cannot be claimed achieved without physical measurements.
Reuse the physical-ground-truth support in `perception/src/temporal_evaluation.rs`.

## Existing implementation

- `perception/src/calibration.rs` solves a single homography from four corners.
- `perception/src/geometry.rs` maps detector box midpoints with that homography.
- `perception/src/inference/runtime.rs` and the v1 detection-batch contract require
  `center_image_px` to be the bounding-box midpoint.
- Full-resolution RGB frames remain available in the detector path; region
  refinement need not use the smaller model-input image.
- Raw/rectified image-space metadata exists, but capture currently emits raw
  frames. A legacy distortion-correction setting does not establish that a
  working lens correction stage is present.

## Recommended design

Use raw-image ball proposals, source-resolution boundary refinement, and a
calibrated nonlinear 2D center map. Do not require full-frame image warping.

### 1. Measure the actual ball-center mapping

Use a measured placement jig or template to place the same ball at known X/Y
centers across the playable ball-center region. A template must locate the
center physically; manually clicking apparent centers is not independent
ground truth. Include points as close to cushions as ball centers can legally
be, with extra samples near corners. An initial 7-by-5 layout is an experiment,
not a production constant or guaranteed sufficient sample count. Add samples
where independent validation reveals spatial residuals. Record several
stationary frames and multiple ball appearances/orientations to separate
localization noise from systematic distortion.

Use exactly the same center estimator for calibration and production. If that
estimator changes, reevaluate and generally refit the center map. Hold camera
mount, capture mode, crop, zoom, and resolution fixed and bind the calibration
to those settings.

### 2. Refine source-image observations

Keep the existing detector for proposals. In a padded full-resolution region,
use cloth/background and boundary evidence to isolate the outer ball contour.
Fit a robust rotated ellipse/local boundary model; exclude shadows, internal
number/stripe edges, and highlights. Fit quality must account for visible arc
coverage, boundary residual, competing edges, and rail/ball occlusion. An
ellipse is a local approximation, not an exact fisheye model. If systematic
boundary residuals persist, use a calibrated local contour template or a
segmentation model rather than pretending the ellipse is exact.

Retain subpixel coordinates. Output a separately versioned refined observation
and uncertainty; preserve the existing bounding-box-midpoint contract. Do not
force circular raw-image fits or allow smoothing to conceal systematic bias.
Reject/flag ambiguous observations rather than inventing precise centers.

### 3. Fit the 2D calibration

For a refined raw-image feature center (u,v), use:

    (X,Y) = H(u,v) + residual(u,v)

Here H is the coarse image-to-table homography and residual is a smooth,
regularized two-component field fitted from measured ball-center errors.
Start with a regularized thin-plate spline; compare against the homography
baseline using held-out physical positions and select complexity/smoothing by
validation. Avoid unconstrained high-degree global polynomials. Reject maps
with folds, singularities, or abrupt local scale changes; preserve the base
map's orientation. Restrict precision claims to the sampled domain. Fit offline
and store coefficients plus provenance; evaluate the small map in Rust.

This directly calibrates the actual ball-center observation to physical X/Y.
No Z estimate or full-frame undistortion is required. Lens calibration remains
an optional base correction if independent tests show it improves results.

### 4. Represent physical shape consistently

Coaching and the canonical table view use a circle at (X,Y) with the configured
physical diameter. A raw-camera overlay uses its observed contour/ellipse.
The center map alone does not guarantee circular corrected silhouettes. If
predicted camera footprints are needed, calibrate a separate smooth appearance
field from the reference ball's outlines and validate its contour alignment.
Do not call the configured diameter an independently measured diameter result.

### 5. Integrate and qualify

Keep capture-to-frame identity, raw image dimensions, calibration revision,
localizer revision, and crop/profile metadata attached to observations. Update
tracking to consume refined centers consistently. Add contract and numerical
tests for calibration serialization, revision mismatch, out-of-domain samples,
folds, synthetic distorted grids, and missing/uncertain refined observations.
Run relevant Rust/browser checks when implementing their respective modules.

Implement one module at a time: physical benchmark and calibration/localization
in perception first, followed by any separate browser visualization task.
Deployment qualification requires new independently measured positions and
ball appearances, including difficult edge cases. A screenshot cannot supply
the calibration samples or demonstrate millimeter accuracy.

## Primary references

- OpenCV fisheye camera model and point/image correction:
  https://docs.opencv.org/4.13.0/db/d58/group__calib3d__fisheye.html
- OpenCV regularized thin-plate-spline transformer:
  https://docs.opencv.org/3.4.11/dc/d18/classcv_1_1ThinPlateSplineShapeTransformer.html

The references establish available methods; the combined design above is a
project-specific recommendation, not an experimentally demonstrated result.
