# Native-resolution 2D ball localization: implementation record

Last updated: 2026-10-27

Review and immediate next action: `review-and-next-step-20261027.md`.
The controlled saved-image audit demonstrates acceptance dependence on
detection index for seven of nine measured balls. Stabilize the estimator
before further FPS tuning or calibration activation.

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
new refiner for a provisional ball-center map, and use the precisely measured
table dots and rail geometry where their physical coordinates are known.

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

Existing references are enough to continue development without further user
measurements. Independent held-out accuracy and availability are still
unverified. If existing dots, rail contacts, and the saved nine-position scene
cannot provide independent validation, report that limitation instead of
requiring a fixture or claiming the desired accuracy.

## Remaining qualification

The ordered immediate estimator task is recorded in
`review-and-next-step-20261027.md`; complete that task before target-device
benchmarking below.

1. Verify and select the native source geometry in the live capture request
   without changing the Coral input tensor shape. The repository default has
   no optics or center-calibration asset configured, so precise X/Y correctly
   remains unavailable until qualified assets are supplied.
2. Benchmark 3840 × 2160, 2560 × 1440, and 1920 × 1080 with camera rates
   5, 7.5, 10, 15, and 20 FPS on the target. Include actual source delivery,
   decoded/analyzed/published/dropped counts, detector/refiner age, motion
   invalidation, CPU/GPU use, and memory trend. Select the most responsive
   configuration that keeps measurements consistently available; do not
   choose a nominal FPS that falls behind. Treat desired latency/accuracy
   metrics as goals; prefer continuous available positions at a lower rate to
   frequent drops or unavailable measurements at a higher rate.
3. Qualify the exact optics profile and local-view fit on real source crops,
   including highlights, stripe edges, shadows, cushion contact, touching
   balls, and partial contours.
4. Add operator acquisition/export for accepted image centers paired with the
   existing measured references; do not require a jig or new pocket-ball
   measurements.
5. Run whatever independent held-out physical X/Y validation the known dots,
   rail contacts, and saved positions support. Report actual available
   reference coverage and accepted-measurement rate alongside residuals.
6. Only then deploy a selected single configuration. The old deployment and
   the separate GPU discovery can each be retained or rolled back
   independently.
