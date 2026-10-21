# Vision performance implementation notes

## Workflow and runtime behavior

The vision path now has an opt-in phase scheduler for recovery, idle, aiming,
limited motion, and many-moving-ball operation. It schedules motion checks,
complete ball-map verification, cue detection, and detailed tracking
independently. Motion immediately invalidates aim eligibility. On motion onset,
integration freezes the last valid aim result as an immutable comparison and
holds it through limited motion, many-ball motion, recovery, and settlement.
Actual paths use source observation timestamps and preserve visible gaps when
tracking is skipped, stale, or lost. A later valid aiming result replaces the
comparison; a calibration revision hides the old spatial overlay.

The scheduler requires calibrated, repeated map reconciliation after startup,
uncertainty, motion, or calibration changes. Configurable entry/exit thresholds,
hysteresis, dwell, observation-age limits, and moving-ball thresholds guard
transitions. Missing detections do not prove settlement. The motion monitor
normalizes illumination, checks calibrated table regions and local ball
neighborhoods, and commits a stable reference only after quiet evidence and a
verified map. The stationary camera assumption is explicit; camera or geometry
changes invalidate the reference.

Shared-memory source sequence and publication timestamps reach downstream
status, separately from monotonic component observation times. Publication age
does not claim sensor exposure time. The frame queue now replaces stale queued
work with the latest available frame and counts drops. Tracker expiry and
reacquisition use elapsed observation time. Explicit frame/view cache identities
replace sampled-content caching. Ball coordinates are restored from detector
resize and crop views before filtering and canonical conversion. Optional
calibrated table cropping affects ball inference only; cue inference retains
full-frame coverage, and color classification uses the aligned unmasked view.

## Benchmark fixture and method

The repository contains eight labeled full-resolution still images with 52 ball
centers/radii; 26 labels are in the configured edge margin. They support
stationary detection matching and edge-ball checks. They do not label cue angle,
per-frame ball identity, collision events, or aiming outcomes.

Video replays available during implementation:

- `test_data/demo2_short.mp4`: 1280×720, approximately 39 s.
- `test_data/demo2.mp4`: 1280×720, approximately 57 s.
- `test_data/demo.mp4`: 1280×720, approximately 65 s.
- `test_data/demo3.mp4`: 3840×2160, approximately 270 s.

The benchmark runner compares native analysis and a conventional 1280×720
analysis view, cadence multipliers, and full-frame versus calibrated-table ball
crops. Cue inference remains full frame. It reports per-phase inference counts,
latency, processing CPU, moving-ball observation gaps, recovery time, cue-angle
stability where observations exist, and labeled static ball/edge metrics. It
does not change configuration defaults.

## Measurements

On the eight labeled stills, each resolution/crop candidate matched all 52
annotated centers, including all 26 edge balls. Native full-frame detection had
3.98 px mean and 10.57 px p95 center error; native table crop had 4.24 px mean
and 11.47 px p95; 720p full-frame had 3.89 px mean and 9.31 px p95; 720p table
crop had 3.88 px mean and 10.20 px p95. Edge-ball mean errors were 3.96, 4.26,
3.93, and 3.93 px respectively. All candidates returned 56 detections for the
52 labels (precision 0.929, recall 1.0, F1 0.963). These small differences do
not establish a reliable aiming benefit for any reduced view.

A 120-frame replay of `demo3.mp4` sampled only startup/recovery and a short idle
segment, so it provides no moving-ball gaps or cue stability. On a 120-frame
segment of `demo2_short.mp4` beginning at 4 s, full-frame processing had 41.91 ms
overall p50 and 58.11 ms p95, 0.033 s p95 moving-ball gaps with one 0.967 s
maximum gap, and 0.50 s mean recovery to a trusted map. Calibrated-table ball
crops had 14.05 ms p50 and 54.77 ms p95, 0.033 s p95 and maximum observed gap,
and 0.40 s mean recovery. The crop run entered many-moving mode for 48 frames
and made only two ball-inference attempts there; the full-frame run had 45
limited-motion frames and 44 ball-inference attempts. Both runs had only 9–16
aiming frames, with measured cue-angle standard deviations of 26.37° and
26.68°. These values expose substantial cue jitter and do not meet a defensible
aiming-stability target.

The same clip was also tested at native resolution with full-frame ball
inference and interval multipliers 0.5, 1, and 2 (smaller is more frequent).
The multipliers produced overall p50/p95 processing times of 50.93/55.58 ms,
47.10/58.89 ms, and 0.11/48.72 ms, respectively. The x2 run used less work in
the many-moving phase, observed a 0.067 s p95/maximum moving-ball gap, and took
0.67 s on average to regain a trusted map. The x0.5 and x1 runs each had a
0.033 s p95 gap, but their maximum gaps were 0.833 s and 0.967 s. These short
segments produce different transitions and moving-ball samples at each
multiplier, so the lower x2 median is not evidence of preserved shot coverage.
No frequency is selected from this single replay.

The short replay candidates follow different phase paths and produce different
detector outputs, so these are exploratory observations rather than paired
causal comparisons. A 4K startup replay compared native and 720p analysis views,
but contained no labeled motion or usable cue-angle stability sample. Labels do
not cover live aiming, ball identity through motion, collision opportunities,
publication freshness, hands/occlusion, manual repositioning, or recovery.
Tracker IDs were not available for stationary jitter on these segments. A cue
angle parser regression was fixed after an earlier exploratory run; the values
above come from the corrected parser and show that the remaining cue stability
problem is not explained by that array-shape bug. No resolution, crop, or
cadence default is selected. The checked-in configuration leaves phase
scheduling disabled and uses native resolution with full-frame ball and cue
views. Reduced profiles stay opt-in until repeatable labeled replay data covers
the missing measures and the cue-angle jitter is addressed.

## Verification record

- Phase scheduler, motion monitor, integration, tracker timing, coordinate/cache,
  cue-angle array-shape, prediction lifecycle, queue freshness, and benchmark
  tests: 64 passed.
- Hardware-free backend suite (`not slow and not hardware`): 849 passed.
- Frontend unit/integration suite: 16 passed.
- Changed frontend modules produced no TypeScript diagnostics in the filtered
  typecheck. The repository-wide typecheck reports 89 errors, none in the
  changed frontend files.
- Black and isort checks pass for the changed Python implementation and tests.
- Architecture decision records pass the bundled ADR validator.
- Repository-wide backend coverage is 28.93%, below its configured 80% gate;
  the hardware-free tests pass, but the gate includes many untested API,
  capture, and calibration modules outside this feature.

## Follow-up validation

Collect labeled aiming and motion sequences for native, 720p, full-frame and
calibrated-crop candidates at multiple processing frequencies. Include stationary
position jitter and accuracy, cue-angle error/stability, aim-assist latency,
limited-motion continuity, missed collision opportunities, publication age,
frame drops, and recovery with edge and clustered balls. Add labeled hands,
lighting changes, manual moves, missing/pocketed balls, source restarts, and
calibration changes. Use repeated runs to select a profile only when the
performance gain exceeds measurement repeatability without reducing aiming
accuracy or useful shot observations.
