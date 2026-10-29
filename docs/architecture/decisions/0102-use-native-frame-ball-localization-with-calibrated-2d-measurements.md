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

## Follow-up — 2026-10-27: use existing table marks as a candidate grid

The user clarified that the precisely measured table marks are the sights on
the rails and dots on the cloth, and authorized using those existing marks to
build the table grid. No ball measurements, pocket-ball coordinates, jig, or
live camera read is needed for the current saved-image analysis. The
[registration audit](../../../thoughts/calibration/table-feature-registration-audit-20261027/README.md)
and [mark-grid audit](../../../thoughts/calibration/mark-grid-calibration-audit-20261027/README.md)
retain the input hashes, coordinates, assumptions, and full numeric output.

Ten long-rail marks fit a one-eighth table-length lattice with 0.57 px RMS and
0.99 px leave-one-sight-out RMS. This agrees with the
[WPA Rules of Play](https://wpapool.com/wp-content/uploads/2026/01/2026.01.02-WPA-Rules.pdf),
which define rail sights at one-eighth length and one-quarter width intervals.
The standard is a candidate reference only: WPA equipment specifications
warn that competition measurements may not apply to home tables, and this
table's measured 45 1/2-inch width differs from the nominal 46-inch 8-foot
width. Use the user's 92 × 45 1/2-inch playing-area dimensions for metric
coordinates rather than silently substituting nominal table dimensions.

Two visible cloth dots appear near 23 and 69 inches along the long axis and
22 3/4 inches across, consistent with centerline head/foot spots; the image
does not independently confirm those identities. Side-rail sight annotations
are about 4–6 px from the expected quarter positions and are excluded from
the fit. A direct rail-plane projection misses the two cloth dots by 10.57–11
px. A shared rail-to-cloth separation gives 1.34 px joint RMS, but
single-marker cross-validation is 2.30–2.48 px and the two inferred
separations differ by 12.13 mm. Therefore the candidate lattice is useful for
continued calibration, but it is not yet a production metric map and must not
be activated for ball X/Y.

The [WPA Recommended Equipment Specifications](https://www.wpapool.com/wp-content/uploads/2024/01/RECOMMENDED-EQUIPMENT-SPECIFICATIONS.pdf)
gives a sight offset from the cushion nose with source notation `+3.175 mm`;
this is not a symmetric tolerance. The
audit uses ±3.175 mm only as a sensitivity range, not as a claim about this
table's conformity. Qualify camera intrinsics with the larger checkerboard
and re-check the rail-to-cloth bridge before using the grid for coordinates.
The provided 20 × 14-inch target guide needs only original target JPEGs if
that optics step is needed; it does not require new table or ball measurements.

## Follow-up — 2026-10-28: reopen architecture and validate observations first

The user explicitly permits revisiting every prior architecture decision to
solve accurate reliable table X/Y. This ADR records the current experiment;
it does not constrain a replacement to local normalization, stochastic circle
search, a unique-outline contract, a particular inference backend or the
existing calibration eligibility count. Supersede consequential choices when
a replacement is justified; preserve the evidence for past decisions.

The eight new native4K larger-board captures are now evaluated in
`thoughts/calibration/large-board-optics-20261028/` and
`thoughts/calibration/large-board-dataset-table-20261028/`. Integrity and
corner-count checks pass, but several view005 coordinates miss visible
checker intersections, one by about 20 px. Source-only wider-window
relocalization removes those gross errors without camera-model seeding or
mutating the original manifest. Its k1 whole-view holdout RMS is 1.591 px,
versus 2.083 px on raw observations. Neither result verifies assembled-board
shape or calibration accuracy.

The new radial fits do not generalize to the older prospective small-board
view009: old frozen zero scores .2279 px, versus .2680 px for the new
relocalized k1 candidate under approximate resolution transfer. The old
nine-view aggregate contains eight training images and is not independent
validation. Relocalized k1 also worsens rail-grid leave-one-sight-out RMS from
1.013 to 8.352 px720 while improving the two cloth checks. Target construction,
observation bias, approximate transfer and home-table mark assumptions remain
confounded. Selecting the lowest larger-board residual would be unsupported.

The center/outline audit at
`thoughts/calibration/center-outline-audit-20261028/` finds zero of thirty
multi-fit aiming iterations with stable centers and only outline disagreement.
Removing the outline gate alone is therefore not selected. Standard-seed
measurements remain unavailable for all nine referenced balls. Conditional
map sensitivity is larger than candidate-center variation but establishes no
true position-error bound. Neither the user's measurements nor any competing
image outline is disproved by these diagnostics.

**Decision:** this phase selects a discriminating experiment, rather than an
unsupported replacement ball estimator. Prepare source-pixel-validated corner
observations, then score two native4K images of the previously verified small
single-piece rigid target at opposite image edges against frozen old/new
models. Fit board pose only, not intrinsic parameters. The complete input,
validation, rejection, test and interpretation specification is in
`thoughts/calibration/next-localization-decision-20261028.md`. This tests whether
an independent target supports a common optical correction before choosing a
new localization architecture. The two views cannot qualify full-table X/Y.

The table is covered. All work in this evaluation used saved files only.
Before future camera access explain the needed two-view check and wait for
the user's confirmation that the table is uncovered. No jig, new round-ball
measurements, runtime flags, production calibration or deployment is introduced.
GPU decoding remains independently useful; future complete-pipeline comparisons
start at 4K/5 FPS and prioritize continuous fresh measurements.

Verification: meaningful synthetic tests for all three new audit tools passed;
18 optics, six dataset/table and two center-report artifacts reproduce exactly
in separate directories. The input hash audit validates 47 references and all
44 pre-existing calibration raster images remain unchanged. Runtime source and
configuration are unchanged from the reconciled evaluation baseline. See
`thoughts/calibration/evaluation-verification-20261028.json`. The benefits are a
proven observation defect and reproducible limits; the cost is that spatial
qualification remains incomplete and a small independent target test is still
needed. An ADR or passing research test is not evidence of physical accuracy.

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

## Follow-up — 2026-10-29: inference placement revised

ADR-0104 replaces the Coral assignment with explicit Intel GPU inference after
actual target measurements showed 69.68 ms GPU versus 1551.04 ms Coral inference
p95 on saved pixels. Native Rust execution on GPU.0 has also been verified.
Other native-frame and measurement contracts remain valid.
