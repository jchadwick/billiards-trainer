# ADR-0035: Version temporal localization evaluation data

- Status: accepted
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

Current benchmark ground truth annotates isolated still images. It can measure
ball precision, recall, center error, and radius error, but it cannot establish
tracking continuity, time to first localization, observation age, cue quality
over a sequence, or behavior across calibration and source-session changes.
Those are mandatory accuracy gates for the primary localization path, which
must remain reliable before optional prediction or ball-number identity work.

The recorded-input benchmark already owns replay pacing and Python phase-profile
measurements. Rust replay emits different artifacts, and hardware capture is not
yet qualified on macOS or Linux. A second benchmark harness would split scoring
semantics; treating a video presentation timestamp as a camera exposure time
would overstate freshness evidence. Temporal labels are not yet available, so
metric code can be verified with deterministic fixtures but production accuracy
cannot be claimed until held-out sequences are annotated.

Annotation authoring needs a reproducible path from video-review tooling into
this evaluator. CVAT's video tracks can display interpolated positions between
keyframes; those displayed positions are useful for authoring but are not
independent ground truth unless a reviewer checks them. A conversion tool must
therefore preserve exactly which frames were reviewed, use the source video's
presentation timeline, and bind the resulting labels to the exact source bytes.
The CVAT export does not provide a cryptographic identity for its source video,
so the workflow must retain an operator-recorded source hash in the task-creation
record.

## Decision

Extend the existing recorded-input benchmark with a versioned, engine-neutral
temporal annotation contract and evaluator. Version 1 annotations identify the
sequence by input hash and image geometry; provide strictly ordered source frame
indices and presentation timestamps; label each ball as visible, occluded, or
removed; assign stable ground-truth track identities; record visible ball
centers and radii; label cue visibility and visible cue geometry; and mark
calibration and expected source-session epochs. Invalid schema versions,
duplicate frame indices, non-monotonic timestamps, non-finite coordinates,
duplicate per-frame identities, and video identity/geometry mismatches fail
closed.

Calibration revisions are compared by their recorded numeric values. Source
session identifiers may be generated independently by each engine run, so
session correctness is compared by boundary transitions rather than string
equality between annotation and runtime IDs.

The benchmark converts each engine's result into the same per-frame evaluation
shape. Visible-ball localization uses one-to-one global minimum-cost spatial
matching within an explicit configurable fraction of annotated radius. Only a
new observation sourced from the current frame counts as fresh detection;
cached/reused estimates remain separately visible and cannot inflate detection
throughput or freshness. Missing engine frames count as unavailable results,
not successful empty observations. Report detection precision/recall, center
error, per-track identity switches and reacquisition after occlusion, first
detection delay, observation-age percentiles and configured freshness-budget
misses. Score cue presence separately from available geometry metrics. The
Rust scene contract currently supports cue center error but does not publish
cue endpoints or angle. Report calibration value mismatches and session
boundary transitions independently.

Recorded-video presentation time is always labeled as replay time. This evaluator
does not claim camera-exposure age, calibrated millimeter accuracy, physical
collision behavior, or production suitability unless the corresponding ground
truth and clock evidence are present. Rust output must be normalized through
this same evaluator before cross-engine accuracy comparisons; synthetic fixtures
verify metric behavior only.

The CVAT video importer accepts only CVAT Video 1.1 interpolation exports,
requires an explicit list of fully reviewed zero-based frame ordinals, and
emits labels only where each active track has an explicit keyframe. It rejects
reviewed frames whose active state would be inferred from CVAT interpolation.
Ball tracks use stable CVAT track IDs as localization identities; those IDs do
not represent numbered-ball identity. Ball centers and radii are derived from
reviewed bounding boxes using the documented mean half-extent rule. A visible
cue is a two-point polyline ordered tip then butt. `ffprobe` best-effort frame
presentation timestamps are normalized to the first video frame and recorded
as replay time, never as camera exposure time. Optional epoch events annotate
expected calibration and source-session transitions. The output records hashes
of the video, CVAT export, reviewed-frame list and optional epoch sidecar, as
well as the selected source-video stream index. Import requires the caller to
provide the SHA-256 recorded for the video submitted to CVAT; the importer
checks it against the bytes being converted. CVAT's declared task frame count
and original geometry must match the selected source stream, rejecting filtered
or partial exports whose frame mapping is ambiguous. Both configured labels
must be present in task metadata. If no ball tracks exist, conversion fails
unless the caller explicitly enables the empty-ball-sequence mode for a
reviewed empty-table clip. CVAT's video export and XML track fields follow the
[CVAT Video 1.1 format](https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/).

## Consequences

### Positive

- Localization, tracking, cue, and freshness evidence share one scoring
  definition across runtime engines, reducing false comparisons caused by
  duplicate benchmark implementations.
- Cached state, missing source frames, occlusion, and new source sessions remain
  explicit in the measurements instead of being mistaken for new observations.

### Negative

- Temporal annotation is a new dataset maintenance cost and requires careful
  handling of occlusion, track identity, and camera/session boundaries.
- The first implementation can prove scoring correctness without proving model
  accuracy; no temporal production recommendation is valid until held-out labels
  exist.
- Review completeness remains a human responsibility. The importer can verify
  that a frame was attested and that track geometry is an explicit keyframe, but
  it cannot prove that every visible ball or cue was annotated correctly.
- `ffprobe` supplies the video's presentation timeline, not camera exposure or
  live sensor timestamps. Freshness metrics based on this dataset remain replay
  metrics.
- CVAT does not embed the original source hash in its XML. The CLI's expected
  hash is an operator attestation captured when the task is created; the tool
  verifies the later supplied bytes against that record but cannot inspect the
  remote CVAT upload itself.

## Alternatives considered

### Keep accuracy evaluation limited to still images

Rejected because isolated frames cannot detect identity switches, delayed first
observations, stale cached positions, or reacquisition failures—the central
risks for near-real-time table localization.

### Create a separate Rust benchmark

Rejected because it would duplicate annotation and matching semantics across
engines. The existing Python benchmark remains the acceptance harness; Rust
results are adapted into its engine-neutral frame schema.

## Verification

- JSON Schema and evaluator tests reject malformed, duplicated, unordered,
  non-finite, or incompatible sequence records.
- Deterministic sequences verify one-to-one matching, misses on missing frames,
  track switches, occlusion/reacquisition, cue endpoint/angle error, freshness
  age, and calibration/session boundary handling.
- A Python replay with temporal annotations reports labeled metrics and retains
  still-image and runtime metrics; without labels, the temporal result is
  explicitly unavailable.
- The Rust scene-snapshot adapter validates the shared scene schema, maps the
  one-based replay frame ID to the annotation index, and uses annotation
  presentation time instead of Rust monotonic ticks.
- Rust and Python normalized records pass through the same scorer. Only current
  source-frame measurements in raw image coordinates count as detections;
  cached tracks and non-observation scene updates do not.
- The command-line Rust evaluator binds labels to the exact video hash and
  geometry. Cross-engine ranking remains unavailable until held-out temporal
  annotations are available and exercised.
- `billiards-temporal-import-cvat` and
  `scripts/import-cvat-temporal-annotations.sh` convert only explicitly
  reviewed CVAT keyframes. Rust tests cover PTS normalization, offsets, stable
  identities, ball lifecycle, cue endpoint ordering, epoch events, invalid
  geometry, and rejection of unreviewed interpolation. Additional checks cover
  source SHA-256, CVAT task frame count and geometry, configured labels,
  empty-ball opt-in, selected-stream provenance, input bounds, `ffprobe`
  timeout/output limits, and atomic no-clobber output. Before retirement, the
  Python reference passed 21 tests; Rust and Python then produced semantically
  identical documents from the same generated three-frame video using the
  installed `ffprobe`. The Python importer and its tests have been retired.
- `python3 /Users/jchadwick/.agents/skills/manage-architecture-decisions/scripts/validate_adrs.py docs/architecture/decisions`
  validates the decision register.

## Implementation tracking

The backend temporal evaluator, Python replay adapter, Rust scene-snapshot
adapter, and Rust CVAT Video 1.1 import tool are implemented. The importer
requires a separate fully reviewed frame list and never promotes interpolated
track positions to ground truth. Dataset creation/review and held-out
temporal/physical labels remain follow-up work. The existing still dataset and
recorded performance audit are preserved.
