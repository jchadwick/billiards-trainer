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
1000/1000 ms run completed in 287.6 ms and a 10000/10000 ms run completed in
289.4 ms. This result is not target-device performance, but it shows the
initial bound cannot be treated as a release readiness claim. Optimize or
qualify the selected schedule while preserving freshness semantics; do not
reuse stale positions to hide missed work.

For the unchanged production seed and original nine measured proposals, zero
image centers are accepted. The nine-reference mapping remains ineligible;
no diagnostic center should be substituted for the selected production
measurement. Continue by investigating candidate family/seed sensitivity and
by resolving the fitting workload before any complete-pipeline rate selection.

## Follow-up — 2026-10-27: nine-reference report is no-fit

The report-only Task F evaluation is recorded in
`thoughts/calibration/report-only-nine-reference-evaluation-20261027.md`.
With revision 2 and production seed 716, all nine measured proposals reject
as `ambiguous_boundary`; no diagnostic homography was fit and no asset was
emitted. No bounding-box midpoint, older-estimator center, or alternative-seed
center was substituted. The reference coordinates remain in their saved
upper-left, image-right/image-down convention; their uncertainty remains
unknown. The existing production utility's twelve-correspondence eligibility
contract is unchanged.

The six variants that happened to pass during the sensitivity audit were
visually reviewed, but their seed/proposal sensitivity and lack of independent
center truth prevent their use as calibration correspondences. Next work stays
on the saved images: explain the competing boundary families and make the
selection stable before resuming target FPS qualification. Performance goals
remain tradeoffs; fresh consistent measurements at a lower rate are preferred
to a higher rate with repeated drops or unavailable positions. The separately
verified Intel GPU decoder result remains independently adoptable regardless
of whether the localization experiment branch is retained.

## Follow-up — 2026-10-27: sampling audit confirms non-monotonic RANSAC outcomes

To identify the seed effect, audit schema 3 now records the three candidate
indices and rays that generated each valid RANSAC hypothesis. This trace-only
addition does not change refiner revision 2 or production selection.

The first-iteration audit is in
`thoughts/calibration/sampling-audit-20261027/`; the normal three-step aiming
audit is in `thoughts/calibration/full-aiming-sampling-audit-20261027/`. With
seed 716 on the saved measured-purple proposal, the isolated first iteration
rejects as ambiguous at 128 and 256 RANSAC iterations, accepts at 512, then
rejects at 1024. The 512 case does not survive normal aiming: its first pass
accepts and moves the patch axis; its second pass yields four individually
qualified, distinct fits and rejects with no center. At 128, 256, and 1024 the
normal flow rejects in its first pass. This rules out “increase the iteration
count” as a stable correction on this image.

The seed 716 first-pass 128-iteration trace has four qualified fits with
supports 95, 92, 92, and 90, low source residuals, and pairwise outline
disagreements of 0.76–1.38 source px. They use different mixes of the three
retained per-ray gradient-strength ranks. A rank is not an outer-silhouette
label. Seed 11 also accepts despite an individually qualified 90-support
alternative because 90/95 falls just below the configured 95% competing-fit
support threshold. Alternate-seed accepted centers differ; no independent
evidence selects one as correct. Keep the production seed, iteration count,
candidate cap, and gates unchanged. These are saved-image diagnostics, not
accuracy or target-performance evidence.

## Follow-up — 2026-10-27: candidate-cap audit leaves ambiguity unresolved

The saved-image cap audit is recorded in
`thoughts/calibration/candidate-cap-audit-20261027/`. It varies only the
per-ray candidate cap across 1, 2, 4, 6, 8, and production 16 for purple,
orange-solid, and red-striped proposals from the measured 720p scene. All 18
runs reject as `ambiguous_boundary`. At cap 16 each search is untruncated and
contains 4, 2, and 8 individually qualified distinct fits respectively. Thus,
for these crops, increasing the cap does not reveal a unique fit or explain
the rejection as cap truncation. This does not establish which fit follows the
true ball silhouette.

Keep production cap, RANSAC seed/count, and quality gates unchanged. Stop
parameter sweeps aimed at increasing acceptance. The next justified estimator
investigation is to inspect actual source-pixel appearance and candidate traces
for one repeatable cue that separates the outer ball/background boundary from
internal, stripe, highlight, or shadow transitions, then test that hypothesis
on the measured scene and reserved 720p/4K scenes. If those pixels do not
support a stable distinction, preserve fail-closed localization and identify
what independent evidence would be needed. No calibration activation or
camera/FPS tuning is justified yet. The separately verified GPU decoder
remains independently adoptable.

## Follow-up — 2026-10-27: pixel profiles do not identify a unique contour

The saved-image contour/color audit is recorded in
`thoughts/calibration/contour-color-audit-20261027/`. It covers nine measured
720p proposals, 15 reserved alternate-720p crops, and three reserved 4K crops
at the unchanged revision-2 production settings. All 27 observations reject.
The traces contain 91 individually qualified fits and 7,891 sampled inlier-ray
RGB profiles; 83 fits use all three gradient-strength ranks. Inside/outside
RGB summaries overlap across candidate families, and visual review shows
nearby outlines following the same apparent rim. No repeatable simple rank or
color cue identifies the physical silhouette. Do not promote a candidate from
these profiles or weaken fail-closed selection.

The fisheye-model audit is recorded in
`thoughts/calibration/fisheye-optics-audit-20261027/`. The model had 0/8
admissible leave-one-view-out folds, prospective view-009 parity RMS of
1.3266 px versus 0.2279 px for the frozen Brown-zero model, and a transformed
cushion diagnostic with much greater bow. Its flag/skew behavior was also
unresolved. It is rejected as a correction candidate; this does not prove zero
residual lens distortion.

Existing ball-outline metrics give a plausible geometric explanation for
some apparent elongation: the far-left yellow gradient ellipse aspect ratio
is 1.314 against a 1.312 small-sphere off-axis approximation, and the far-right
purple ellipse is about 1.25 against about 1.27. These annotations are not
independent true-boundary labels. The fixed-sphere conic audit is recorded in
`thoughts/calibration/sphere-projection-audit-20261027/`. Its synthetic
tangent-cone inversion passes under ideal contours, but the saved measured
scene does not support a useful selector: 36 qualified outlines across nine
balls produce 15.208 mm all-nine training X/Y RMS, while leave-one-out
oracle-best-candidate RMS is 18.652 mm. The oracle sees the held-out physical
position to choose an outline and is only a lower bound. The existing
segmentation-center homography reports 8.706 mm on the same references. The
bounded search is heuristic, not exhaustive, and user-reference uncertainty is
unknown. Do not activate this method or treat these consistency scores as
accuracy. This rejects the current candidate-selection experiment, not the
ideal conic geometry.

The static-feature registration audit is recorded in
`thoughts/calibration/table-feature-registration-audit-20261027/`. Held-out
upper/lower rail sights transfer with 0.209 px RMS between measured and
alternate 720p scenes, and 0.094 px RMS between the retained 720p and 4K
framing pair after 3× normalization. Two cloth-marker transfers on the framing
pair are 1.360 px and 0.004 px; that sample is too small to infer a
cross-plane correction. All results are pixel registration only. The project
does not record exact physical coordinates for the dots, so these results do
not establish a metric mapping.

The report-only X/Y homography audit used the known playing-area rectangle and
existing cushion-nose line samples, with all nine ball positions held out.
Boundary choice, line extrapolation, and table-coordinate sensitivity are
recorded; it did not produce a usable metric map. Do not transfer ball
positions between arrangements or activate a calibration from these
diagnostics alone.

The rectangle-map audit is recorded in
`thoughts/calibration/playing-area-homography-audit-20261027/`. Its nominal
X/Y RMS against the nine held-out ball references is 22.039 mm; across 13,122
line/corner variants the largest per-ball coordinate spread is 23.814 mm.
Using the nearby lower seam instead of the inner transition yields 33.758 to
41.840 mm RMS. The sampled visible edges do not establish the physical cushion
nose, and corners are extrapolated around pockets. Reject this map as a
production candidate.

### Follow-up — 2026-10-27: rail and felt marks define a provisional grid

The user clarified that the table's rail and felt dots are the physical table
marks, and asked us to derive the grid from them. The project has no separate
survey file listing each mark's XY, but official pool-table definitions
provide a checkable candidate layout. [WPA Rules of Play §2.1](https://wpapool.com/wp-content/uploads/2025/10/2025.09.15-WPA-Rules-NP.pdf)
places rail sights at 1/8-length and 1/4-width fractions of the cushion-nose
rectangle, and defines the head, center, and foot spots on the long centerline.
The [WPA 8-foot equipment specification](https://www.wpapool.com/wp-content/uploads/2024/01/RECOMMENDED-EQUIPMENT-SPECIFICATIONS.pdf)
gives 11.5-inch sight spacing and a sight
center 3 11/16 inches from the cushion nose, with +3.175 mm in the published
text; it also warns that
the competition specification may not apply to home tables. The 92-inch
measured length matches this sight interval. The user's 45.5-inch width is
below the nominal 46 inches, so keep the user's dimensions and verify the
standard pattern against this specific table. This resolves the earlier
missing-coordinate blocker at the candidate-grid level; exact table-specific
mark offsets and the resulting camera map remain unverified.

The saved-image audit at
`thoughts/calibration/mark-grid-calibration-audit-20261027/` finds six upper
rail sights at 1/8, 2/8, 3/8, 5/8, 6/8, and 7/8; four clear lower sights at
2/8, 3/8, 6/8, and 7/8; and a glare-obscured fifth lower sight visually
consistent with 5/8. Ten long-rail points fit with 0.57–0.59 px RMS and
0.99–1.01 px leave-one-sight-out RMS. Side-rail sight labels are ambiguous
and have 4–6 px residuals against the assumed quarter-width layout; they are
not used to force the fit.

Two visible cloth marks are consistent with head and foot spots at x=23 and
69 inches, y=22.75 inches in the user's 92 x 45.5 inch table coordinates. A
single homography across rail and cloth planes is invalid: the rail-plane map
misses the two cloth points by 10.57–11.00 px. Decomposing the rail-plane
homography with the saved checkerboard intrinsics and estimating a common
rail-cap-to-cloth separation improves the joint spot fit to 1.341 px RMS
(52.831 mm estimated separation). Fitting from either spot and predicting
the other leaves 2.296–2.481 px residual, with 12.134 mm spread between the
two inferred separations. This demonstrates a plausible geometric bridge,
not a qualified one; camera intrinsics, spot localization, home-table
construction, and the rail-height estimate remain uncertain.

The nine user-measured ball positions were used only as held-out diagnostics.
Mapped detector-box midpoints show 11.286 mm RMS, 17.070 mm p95, and 17.615 mm
maximum radial residual. This is not ball-center accuracy: box midpoints are
not qualified centers and physical-reference uncertainty is unknown. Do not
emit a production table map or calibrate from those residuals.

The next step is a larger checkerboard at the same camera resolution, zoom,
and focus, with sharp views across image center and corners and tilts about
both axes. The print-ready 20 x 14 inch active pattern has 2-inch squares and
9 x 6 internal corners; details are in
`thoughts/calibration/checkerboard-20x14in-capture.md` and
`output/pdf/ball-localization-checkerboard-20x14in.pdf`. Re-estimate camera
intrinsics, then repeat the plane bridge while holding each saved cloth spot
out in turn. Do not ask the user for new ball measurements or a positioning
jig. The calibration uses a fixed 57.15 mm sphere-center plane only as a
projection constant when needed; the application remains X/Y-only and does
not estimate or publish ball height/depth.


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
- `thoughts/calibration/offline-mark-grid-calibration-audit-20261027.py`
  reproduces the saved-image rail-grid fit, held-out sight/spot checks, and
  synthetic geometry round trips. Its `summary.json` records source hashes and
  per-point errors. A passing synthetic test or low fit residual is not a
  qualified production calibration; require the larger-target intrinsic
  capture and successful held-out rail/cloth transfer before activating a
  table map.
- Independently measured ball-center placements across the table establish
  error and acceptance rates, including difficult boundary conditions. The
  limited initial manual measurements are diagnostic evidence, not release
  qualification.

## Follow-up — 2026-10-29: implementation defects corrected

ADR-0103 records the deterministic geometry, complete sampling ROI, exact-frame
contour propagation and overdetermined calibration validation now implemented.
The zero-acceptance and twelve-point statements above describe earlier revisions;
they are preserved as history, not current blockers. Hardware assignments and
source-frame ownership in this ADR remain in effect.

## Follow-up — 2026-10-29: inference placement revised

ADR-0104 replaces the Coral assignment with explicit Intel GPU inference after
actual target measurements showed 69.68 ms GPU versus 1551.04 ms Coral inference
p95 on saved pixels. Other native-frame and measurement contracts remain valid.

## Follow-up — 2026-10-29: saved replay improves availability, not accuracy

The R10 saved-image qualification in
`thoughts/calibration/refiner-r10-validation-20261029/` shows that changing the
maximum center-spread setting from 1.5 to 1.6 reference pixels accepts one
previously rejected far-corner proposal while retaining seven agreeing seed
fits. Paired runs on 15 actual Intel GPU proposals, 18 proposal-seed variants,
and 297 synthetic known-center cases produced no status, reason, center, or
outline changes between those settings. The replay and exact-box evidence are
preserved with their inputs and limits in that folder. This is an operational
availability correction for the observed proposal, not a bound on center error
or physical X/Y error.

The 60-second target replay used the same saved 1280 × 720 image enlarged to
3840 × 2160 and repeatedly delivered at 5 FPS. It kept all 15 positions fresh
through 600 public-state samples and completed motion checks for all 304
captured frames. It does not establish behavior on native-4K camera pixels,
moving balls, exposure changes, or independent physical references. The current
nine-point table map still has unknown input uncertainty and is not qualified
for accurate coaching. No production millimeter-accuracy claim or calibration
activation follows from this result.
