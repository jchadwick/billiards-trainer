# ADR-0001: Phase-aware vision scheduling and prediction lifecycle

- Status: accepted
- Date: 2026-10-18
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

Vision capture already runs out of process, while inference and integration can
consume frames more slowly than capture. Processing every component on every
frame increases latency and allows queued work to become stale. The application
also needs reliable ball maps before aim assistance, useful observations during
shots, and a clear comparison between a pre-shot aim prediction and actual
motion. The camera is stationary during a session, but hands, occlusions,
lighting changes, dropped frames, and manually moved balls can make observations
uncertain.

## Decision

Make vision the authority for phase and observation validity. Schedule ball
verification, cue observation, motion monitoring, and moving-ball tracking by
phase. Use elapsed observation timestamps and explicit skipped/failed status so
intentional inference skips do not count as missed detections. Require sustained
quiet evidence and repeated successful map reconciliation before declaring the
table settled.

Carry the shared-memory source sequence and publication timestamp through
vision and downstream status; publication age is distinct from local monotonic
observation gaps and is not represented as camera exposure time. Process the
latest queued frame and record frames discarded under overload. Keep ball and
cue schedules independent, sharing explicit source-frame/view identities when
they use the same pixels. Restore crop offsets and analysis resize factors before
source-coordinate filtering and canonical conversion. Optional calibrated table
crops apply to ball inference only; cue inference retains full-frame coverage.

Permit new aim predictions only when the map is trusted and cue evidence is
fresh. At motion onset, retain an immutable snapshot of the last valid prediction
as a comparison reference. Do not recalculate or overwrite that snapshot during
motion or recovery. Show measured actual observations against it, preserving
gaps when observation continuity is unavailable. A new valid aiming prediction
replaces the comparison; calibration changes invalidate its spatial overlay.

Choose per-phase resolution and processing frequencies from representative
measurements. Keep the scheduler and reduced-resolution/crop profiles opt-in
until representative labeled recordings validate aiming accuracy, cue-angle
stability, moving-ball continuity, collision opportunities, and recovery. The
720p candidate is a benchmark input, not a selected default.

## Consequences

### Positive

- Avoids repeated inference while keeping aiming and shot observations timely.
- Prevents uncertain or stale observations from triggering new aim guidance.
- Gives users a stable pre-shot reference for comparing observed ball motion.

### Negative

- Phase tracking, observation metadata, and frozen prediction state cross module
  boundaries and require compatibility coverage.
- Many-ball mode intentionally sacrifices detailed identity and collision
  tracking; comparison paths can therefore contain visible gaps.
- Source publication timestamps provide frame-freshness context, but do not
  claim sensor exposure timing that the capture interface does not supply.
- Resolution and crop measurements on short unlabeled clips are exploratory;
  they do not justify changing production analysis defaults.

## Alternatives considered

### Run every detector at a fixed rate

This continues to spend inference on stable scenes and does not distinguish
trusted aiming from uncertain motion or recovery.

## Verification

- Deterministic replay tests cover phase transitions, skipped versus failed
  observations, settlement safeguards, and timestamp gaps.
- Prediction lifecycle tests verify immutable freeze, motion-time gating,
  retained comparison during recovery, and replacement on the next valid aim.
- Representative benchmark artifacts report accuracy and latency by phase and
  resolution.
- The benchmark tool reports labeled still-image ball and edge-ball accuracy
  separately from short replay measurements, and leaves defaults unchanged.
- Architecture decision validation checks this record's required sections.
