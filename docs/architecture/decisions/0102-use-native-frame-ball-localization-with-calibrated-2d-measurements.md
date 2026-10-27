# ADR-0102: Use native frame ball localization with calibrated 2D measurements

- Status: accepted
- Date: 2026-10-26
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The camera delivers a wide-angle image in which equal-diameter balls have
position-dependent apparent outlines. The detector's bounding-box midpoint
and a homography derived from the playing surface do not establish the ball
center accuracy required for coaching. The source can deliver 4K frames, but
the existing HTTP-MJPEG path reduces the frame before a high-resolution ball
measurement can be made. The target has an Intel GPU with a working VA-API
JPEG decoder and a Coral accelerator running the existing detector model.

Calibration experiments found no demonstrated improvement from the tested
full-frame radial correction. Local viewing-angle normalization made several
edge-ball outlines closer to circular, but has not established physical X/Y
accuracy. Measurement uncertainty, rail and pocket occlusions, processing age,
and capacity of retained native-resolution frames must remain explicit.

## Decision

Use one production ball-localization pipeline. For HTTP-MJPEG input on the
target, decode complete JPEG frames with the Intel VA-API decoder, bound the
pending compressed-frame queue to the latest frame, and retain source
resolution in an owned RGB frame. Keep the existing Coral model input shape;
perform bounded local image-angle normalization and outer-boundary refinement
on native-resolution ball regions in the existing single-flight detector
worker. Do not silently fall back to software JPEG decoding when the required
hardware decoder is unavailable. Select the operating resolution and camera
rate through target-device qualification rather than at runtime.

Preserve `DetectionBatchV1` as the raw detector contract: its center remains
the bounding-box midpoint. Carry refinement separately in a frame-scoped
measurement sidecar bound to detection identity, source geometry, session,
configuration, optics, and refiner revisions. Resolve a single image center
per observation and use that same measurement for tracking and physical
coordinate conversion. A stale, mismatched, rejected, or timed-out
refinement may retain a coarse image detection, but cannot produce a precise
table coordinate or enable precise guidance.

Maintain separate, versioned camera optics, table geometry and visibility,
and ball-center mapping assets. Fit the ball-center mapping from verified
physical X/Y correspondences produced with the same refiner revision. Bind
each asset to its capture profile and reject incompatible geometry or
revisions. A playing-surface homography alone is not evidence of accurate
ball centers, and previous-resolution calibration must not be silently
promoted to a new profile. Continue to honor the frame pairing and table
visibility decisions in ADR-0100 and ADR-0099.

The application model remains two-dimensional with a configured common
physical ball diameter. The refiner uses image coordinates and intrinsics and
does not estimate ball height or depth. Physical diameter must not be inferred
from a position-dependent image footprint. The selected implementation must
be qualified against independent physical accuracy and reliable availability.
Latency and FPS targets are desired benchmarks, not reasons to accept dropped
work or repeated unavailable positions. Include low-rate candidates such as
5 FPS; prefer a lower camera rate when it gives consistently fresh
measurements over a higher rate that frequently falls behind. Report missed
targets clearly. There are no selectable experimental localization modes or
parallel production localization implementations.

Verified discoveries are independently adoptable. A qualified GPU decoder,
camera capability result, or stable FPS setting can be preserved and
incorporated even if the remaining localization branch is discarded.

## Follow-up — 2026-10-27: estimator stability before qualification

The accepted architecture is the selected direction, not a declaration that
the current implementation is qualified. The
[saved-image identity audit](../../../thoughts/calibration/offline-refiner-identity-check.json)
holds pixels, proposal geometry, intrinsics, thresholds, and work limits fixed
while varying detection index. Seven of the nine measured-ball cases change
acceptance. The current implementation therefore fails the required
invariance to incidental detection identity. These repeated trials on one
frame do not measure independent placements or live availability. The
[experiment review](../../../thoughts/calibration/review-and-next-step-20261027.md)
records the inputs, limitations, related findings, and ordered next task.

Complete saved-image estimator stability work before further FPS tuning,
calibration activation, or rollout. With sufficient per-proposal and batch
budget and below capacity, identical pixels and proposals must produce the
same selected center and acceptance regardless of detection index or ordering.
Under explicit deadline or capacity exhaustion, confidence-order scheduling
may change which observations finish; report those outcomes separately from
geometric rejection. Compare genuinely distinct, converged boundary solutions
in common source-image coordinates; raw sampled circle guesses alone do not
establish competing boundaries. Every diagnostic value must identify the
fit and aiming iteration it describes. Retain rejection for genuinely
ambiguous or unsupported observations; widening thresholds to force a desired
acceptance count is not qualification.

Use the existing saved corpus for this immediate work. No positioning jig or
additional user measurements are required. Separate detection-identity
variation, candidate-start variation, and proposal perturbation in audits so
each result has an interpretable cause. Reserve saved scenes from tuning for
additional contour checks, respecting their optics and arrangement provenance.
These scenes do not supply independent physical X/Y truth merely by being
held out.

The production calibration utility currently requires at least twelve
correspondences, while the matching measured scene supplies nine. Those nine
support diagnostic mapping and residual analysis, but cannot currently produce
an asset through that utility. Resolve the contract explicitly before asset
activation; do not invent or duplicate correspondences, silently lower the
minimum, or treat table-reference marks as ball-center truth without verifying
their reference definitions. This record does not change the minimum or claim
that the available physical references establish the desired accuracy.

After estimator stability is established, resume complete target-pipeline
qualification initially at native 4K and 5 FPS. Select the final configuration
from sustained fresh precise availability, latency, memory, and motion response.
The short source-only 5 FPS check is not a selected production rate. Preserve
the independently verified Intel GPU decoding result throughout this work.

## Follow-up — 2026-10-27: candidate ambiguity blocks saved-image calibration

The detection-identity defect is corrected: with sufficient budget, the
revision-2 estimator returns semantically identical results for all 15 frozen
manual proposals under detection identities 0–15 and reversed proposal order.
This establishes repeatability for those fixed inputs, not physical accuracy.

The version-2 candidate audit at
`thoughts/calibration/converged-candidate-audit-20261027/` shows 0/15 precise
centers accepted, including 0/9 proposals with matching physical references.
Fourteen proposals reject for `ambiguous_boundary`; one pocket proposal
rejects for `insufficient_boundary`. Thirteen observations have multiple
individually qualified but geometrically distinct converged fits. Another
ambiguous observation has no qualified fit after bounded reselection. Search
was not truncated and no trace events were dropped. At this image width, the
configured center and same-boundary outline tolerances scale to 0.5 source px.
These settings remain conservative algorithm limits, not measured accuracy.

Therefore the current nine physical references cannot produce a ball-center
map through the selected estimator: none of their observations resolves to a
center. Do not generate an asset from rejected candidates, annotation-derived
box centers, or coordinates from an older estimator. Keep Task F report-only
unless a later, separately justified estimator revision accepts reproducible
centers and its correspondence/uncertainty limits are documented. The planned
saved-image sensitivity work with proposal perturbations, fitting seeds, and
reserved scenes is recorded in
`thoughts/calibration/robustness-audit-20261027/`. Preserve every rejection
and do not weaken gates merely to raise the accepted count. No jig, new user
measurements, camera access, or physical-accuracy claim is needed for this
diagnostic continuation.

## Follow-up — 2026-10-27: proposal, seed, and saved-scene sensitivity

The controlled robustness audit at
`thoughts/calibration/robustness-audit-20261027/` used refiner revision 2 and
the planned 96 radial samples. Across 150 independent X/Y proposal-center
perturbations and 150 width/height perturbations on the measured saved image,
one upward-shifted box accepted and no size perturbation accepted. Across 240
seed/proposal combinations, the selected production seed 716 accepted 0/15;
four alternative seeds produced five accepted outputs for three proposals.
The same purple-ball proposal accepted under three seeds yielded centers
spanning 0.682 source px. These results make proposal and fitting-start
sensitivity explicit; they do not identify which output is correct. Keep the
production seed fixed and do not select another seed to increase availability.

Reviewed coarse manual boxes on the previously inspected alternate 720p scene
and saved native 4K scene all reject: 0/15 and 0/3 accepted respectively.
Their optics models remain exploratory/derived diagnostics and neither scene
has physical center references. This is contour behavior only, not accuracy
generalization.

The same 15-proposal batch on a Darwin arm64 development host timed out all
observations at the initial 25 ms batch / 10 ms per-proposal budget; a
1000/1000 ms run completed in about 293 ms. This result is not target-device
performance, but it shows the initial bound cannot be treated as a release
readiness claim. Optimize or qualify the selected schedule while preserving
freshness semantics; do not reuse stale positions to hide missed work.

For the unchanged production seed and original nine measured proposals, zero
image centers are accepted. The nine-reference mapping remains ineligible;
no diagnostic center should be substituted for the selected production
measurement. Continue by investigating candidate family/seed sensitivity and
by resolving the fitting workload before any complete-pipeline rate selection.


## Consequences

### Positive

- Native pixels are available to refine small off-axis ball observations
  without requiring model inference at source resolution.
- GPU JPEG decode and Coral inference use separate existing accelerators,
  while bounded native crops avoid another full-frame processing queue.
- Raw detector provenance remains distinct from the chosen physical
  measurement, and precise guidance remains unavailable when evidence is
  insufficient.

### Negative

- Native-resolution RGB frames, decoder surfaces, model tensors, and worker
  scratch buffers consume more memory and processing bandwidth than the
  existing reduced-frame path.
- The HTTP-MJPEG target now requires a qualified VA-API decoder and device
  access. Hardware, geometry, or focus changes require requalification.
- A ball-center map and independent reference measurements are required;
  rounder image outlines alone cannot prove coaching accuracy.

## Alternatives considered

### Globally undistort the full frame

The measured checkerboard and cushion evidence do not justify the tested
radial corrections, which made straight edges bow. A global warp also adds
frame-wide work without establishing ball-center accuracy.

### Use detector box midpoints as physical ball centers

Proposal midpoints provide useful coarse detections, but their bias and the
appearance of ball boundaries vary across the image. They remain available
as raw observations rather than being presented as qualified precision.

### Keep selectable legacy, shadow, and refined pipelines

Parallel production paths multiply calibration, compatibility, and
qualification states. Compare the previous and replacement builds offline,
then release one qualified implementation or discard the experiment.

## Verification

- `make test` covers image geometry, sidecar identity, tracker and calibration
  compatibility, guidance gating, and browser frame pairing.
- `make test-rust-camera` covers negotiated GPU JPEG decoding, bounded frame
  ownership, dropping, reconnect, and shutdown on supported target hardware.
- `make lint`, `make format-check`, `make config-check`, and `make test-deploy`
  validate code, configuration, and deployment behavior.
- The frozen identity audit and its archived research harness at
  `thoughts/calibration/offline-refiner-identity-check.rs` supply a reproducible
  detection-index baseline. Regression checks must verify unchanged results
  for identical pixels and proposals under both index and proposal-order
  changes with sufficient budget and below capacity. Exhausted-budget and
  capacity scheduling tests must verify explicit outcomes and association
  rather than an invariant completed subset. The recorded 2026-10-27 audit is
  failing, not a passed qualification
  check; its current harness varies index only.
- Controlled saved-image audits perturb proposal center and size separately
  from fitting starts, export boundaries and competing converged fits, and
  check that diagnostics refer to the correct fit and iteration. Known-center
  synthetic cases verify transforms and center recovery; reserved saved scenes
  check contour behavior without being presented as physical ground truth.
- Target-device capture and replay audits measure actual frame cadence,
  decode/conversion and refinement latency, stale results, and bounded memory
  for candidate resolution and FPS combinations. Select the rate that keeps
  measurements available and responsive; numeric performance targets are
  goals to approach, not hard gates that override stability.
- Independently measured ball-center placements across the table establish
  error and acceptance rates, including difficult boundary conditions. The
  limited initial manual measurements are diagnostic evidence, not release
  qualification.
