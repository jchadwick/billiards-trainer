# Native-resolution 2D ball localization: implementation record

Last updated: 2026-10-27

Review and immediate next action: `review-and-next-step-20261027.md`.
The controlled saved-image audit demonstrates acceptance dependence on
detection index for seven of nine measured balls. A new saved-image audit
finds that rail sights form a strong candidate table grid and the two visible
cloth dots are consistent with the table's centerline spots. The grid is not
yet a qualified metric calibration because the optics and rail-to-cloth plane
transfer remain uncertain. Stabilize the estimator before FPS tuning or
calibration activation.

## Project requirements

- The table is a 2D X/Y plane. Do not estimate or emit Z, ball height, or depth.
- Every ball is a regulation 57.15 mm diameter. Its physical diameter does not
  vary with image position.
- Verified playing-area dimensions are 2336.8 × 1155.7 mm.
- Preserve raw detector boxes and their bounding-box midpoint. A displayed
  coarse detection is not a precise ball-center measurement.
- Publish physical X/Y only when a frame-matched native-pixel fit is accepted
  and the corresponding ball-center calibration is valid for that exact camera,
  optics, refiner, image geometry, table revision, and image space.
- Prefer continuous, fresh measurements at a lower rate over a higher rate
  that drops work or repeatedly makes positions unavailable. Performance
  numbers are desired benchmarks to approach, not release choices to maximize
  at the expense of stability.
- Preserve independently verified findings (for example GPU decoding or a
  stable camera rate) even if the broader localization experiment is discarded.

## Verified independent discovery

The target is `192.168.1.31`, with UC40M UVC camera identity and Intel UHD 630.
The camera provides 3840 × 2160 MJPEG; the installed `vaapijpegdec` successfully
decoded full-resolution frames while the application was running. A bounded
probe decoded 100 frames in 5.044 seconds (about 19.8 frames/s) with roughly
64% of one CPU core. This establishes a useful decoder result only: it does
not measure the complete application, native-crop refinement, Coral load,
viewer path, or long-run memory.

The stream is currently configured for 3840 × 2160 at 20 FPS. The application
configuration was temporarily changed to 5 FPS for a short source-cadence
check and then restored byte-for-byte. The target reported online 3840 × 2160,
desired 5 FPS and captured 5 FPS in all six samples over 18 seconds; after
restoration it reported desired/captured 20 FPS. This establishes stable
camera delivery at 5 FPS only, not end-to-end localization availability.
Details are recorded in `stream-5fps-check.md`; the original 4K change and
rollback information remain in `stream-4k-change.md`.

The exact saved 1280 × 720 checkerboard matrix and frozen 4K framing check are
in `checkerboard-eight-view-baseline.json` and `stream-4k-framing.md`. The 4K
test found similar framing under a direct 3× coordinate scale, but did not
establish identical intrinsics, table millimeter accuracy, or ball-center
accuracy. Do not activate a scaled profile as ground truth without binding it
to a verified camera profile and qualifying its effect.

## Table-mark grid evidence

The saved-image audits are in `mark-grid-calibration-audit-20261027/` and
`table-feature-registration-audit-20261027/`. Ten visible long-rail sights fit
the expected one-eighth table-length lattice at 0.57 px RMS, with 0.99 px
leave-one-sight-out RMS. A glare-obscured lower sight visually agrees with the
predicted 5/8 position, but is qualitative only. Side-rail annotations miss
their expected quarter positions by roughly 4–6 px and remain excluded from
the fit. Two cloth dots appear at approximately 23 and 69 inches along the
long axis and 22 3/4 inches across; this is consistent with the head/foot
centerline spots, but their identities are not confirmed by the image alone.

This supports using the rail marks as a candidate grid tied to the user's
92 × 45 1/2 inch playing-area dimensions. It does not yet produce a production
metric map: a direct rail-plane projection misses the two cloth dots by
10.57–11.00 px. A single shared rail-to-cloth plane separation fits them at
1.34 px joint RMS, but fitting either dot alone predicts the other only within
2.30–2.48 px, and the two implied separations differ by 12.13 mm. The
checkerboard-derived intrinsics and the cushion-nose sight offset also need
qualification. These are saved-image diagnostics; no new table or ball
measurements and no live camera access were used.

The next useful user-side input, if further optics calibration is needed, is a
set of original JPEGs of the larger checkerboard described in
`checkerboard-20x14in-capture.md`. No jig or ball repositioning measurements
are needed. Do not activate table X/Y from the candidate grid until the
checkerboard and rail-to-cloth transfer checks are stable.

## Current implementation on `codex/native-ball-localization`

- The GStreamer capture probe parses MJPEG, drops stale compressed frames
  before decode, uses `vaapijpegdec`, converts its native YUY2 output to RGB,
  and bounds app-sink buffering. It does not silently use software decode.
- The Rust local-view refiner works on borrowed native RGB, returns explicit
  fit/reject evidence per proposal, and has bounded sample counts, work time,
  ROI size, and scratch allocation. Its ambiguity check now requires a spatial
  cluster of plausible competing RANSAC fits; one alternate sample no longer
  causes rejection by itself. Offline checks accepted 4/9
  manually seeded crops from the 720p measured scene and 3/3 crops from a
  separate saved 4K scene. These are not physical accuracy or live-availability
  results; details and limitations are in
  `offline-refiner-nine-position-check.md`. A later, differently constructed
  all-ball proposal set accepts 5/9 measured balls. Holding its boxes fixed and
  varying only detection index changes acceptance for 7/9. Raw hypothesis
  ambiguity, mixed diagnostic iterations, and inconsistent center-distance
  coordinate spaces need correction; the three-vote change is not a qualified
  solution. Exact audit results are in `offline-refiner-identity-check.json`.
- The separate ball-center calibration solver fits normalized homographies
  from at least 12 distributed correspondences, reports every residual, and
  validates the asset against runtime context.
- The live pipeline carries refinement results with source-frame identity,
  resolves one image center for tracking, and maps only accepted fits through
  the independent center calibration. A missing/rejected fit keeps the coarse
  image detection visible but withholds precise table X/Y.
- Replay uses the same native-pixel crop refiner, frame-bound measurement
  resolver, and tracker API when the matching optics and center-calibration
  assets are supplied. Without those exact assets it retains coarse image
  detections and emits no precise table X/Y.
- Live timing records keep model inference, crop refinement, measurement
  resolution, and total frame age separate so target FPS comparisons can
  prioritize sustained fresh availability.
- The new calibration command is
  `billiards-perception-ball-center-calibrate`; its input contract is
  `perception/contracts/calibration/v1/ball-center-calibration-input.schema.json`.
  The output is created without overwriting an existing asset and embeds all
  fit residuals. Training residuals are not independent accuracy evidence.

## Physical references without a jig

The user declined building a positioning jig. Do not ask for one or request
measurements of the pocket-adjacent balls. The project already contains nine
user-confirmed ball-center positions, including rail contacts, in
`table-grid-20261026-114241-measurements.json`; those coordinates apply only to
the matching `table-grid-20261026-114241.jpg` arrangement. Use them with the
new refiner for a provisional ball-center map. The user identifies the rail
sights and felt dots as the precisely measured table marks; the current saved
audit derives their candidate regular grid from the visible lattice and the
known table dimensions, rather than requesting new physical measurements.

Known table reference marks help establish table geometry. Verify the exact
physical reference represented by each mark; do not assume all marks are
interchangeable with ball-center correspondences. Keep the ball-center mapping
separate. The existing nine-point exploratory fit used
segmentation-derived centers and had 8.71 mm leave-one-out RMS error; physical
measurement uncertainty is unknown, so this is neither a result for the new
refiner nor evidence of release-level accuracy. Re-evaluate the same original
frame with the production refiner and report residuals and rejected
observations honestly.

The production calibration utility currently requires 12 correspondences, so
it cannot emit an asset from the existing nine. Nine-point diagnostic mapping
and residual analysis remain possible; resolve this contract explicitly before
activation rather than duplicating points or silently lowering its minimum.

The user's existing rail and felt dots are the intended table references;
their individual numeric readings are not recorded in the repository. The
candidate regular grid and specified playing-area dimensions allow offline
analysis to continue without asking the user to remeasure them. Do not request
new tape measurements over round balls, a jig, or pocket-ball coordinates.
Independent held-out accuracy and availability remain unverified. If the existing dots,
rail contacts, and saved nine-position scene cannot provide independent
validation, report that limit rather than claiming the desired accuracy.

## Remaining qualification

The existing mark-grid audit is reproduced; its candidate rail lattice is
not yet a qualified metric map. If the existing checkerboard views are
insufficient, the next user-side input is original JPEGs of the larger
rigid-backed target in `checkerboard-20x14in-capture.md`. No live table-camera
access, jig, new ball measurements, or pocket-ball coordinates are needed.
Estimator stability remains required before production localization or FPS
selection and can be fixed from saved images while the table calibration is
being prepared.

1. Correct detection-index sensitivity in the estimator using the saved
   corpus, following `review-and-next-step-20261027.md`.
2. Calibrate intrinsics with fixed camera settings, sharp center/corner
   coverage, and held-out tilted checkerboard views.
3. Refit the rail-to-cloth transfer using the saved felt dots as held-out
   references and the measured 92 × 45 1/2 inch playing area. Do not treat the
   current provisional transfer as calibration.
4. Verify and select the native source geometry in the live capture request
   without changing the Coral input tensor shape. The repository default has
   no optics or center-calibration asset configured, so precise X/Y correctly
   remains unavailable until qualified assets are supplied.
5. Benchmark 3840 × 2160, 2560 × 1440, and 1920 × 1080 with camera rates
   5, 7.5, 10, 15, and 20 FPS on the target. Include actual source delivery,
   decoded/analyzed/published/dropped counts, detector/refiner age, motion
   invalidation, CPU/GPU use, and memory trend. Select the most responsive
   configuration that keeps measurements consistently available; do not
   choose a nominal FPS that falls behind. Treat desired latency/accuracy
   metrics as goals; prefer continuous available positions at a lower rate to
   frequent drops or unavailable measurements at a higher rate.
6. Qualify the exact optics profile and local-view fit on real source crops,
   including highlights, stripe edges, shadows, cushion contact, touching
   balls, and partial contours.
7. Add operator acquisition/export for accepted image centers paired with the
   existing measured references; do not require a jig or new pocket-ball
   measurements.
8. Run whatever independent held-out physical X/Y validation the known dots,
   rail contacts, and saved positions support. Report actual available
   reference coverage and accepted-measurement rate alongside residuals.
9. Only then deploy a selected single configuration. The old deployment and
   the separate GPU discovery can each be retained or rolled back
   independently.
