# Implement accurate 2D ball localization from native camera frames

Execution continuation, 2026-10-27. This document is intended to be executable
by an implementer without the conversation history. Read ADR-0102 and
`review-and-next-step-20261027.md` first. The latter contains the experiment
review; this document defines tasks, deliverables, checks, and commits.

## 1. Verification strategy and completion definition

The next milestone is a reproducible, stable native-pixel estimator on saved
inputs, followed by an honest nine-reference mapping diagnostic. Preserve
independently useful capture improvements on `main`. Do not call the current
estimator qualified because it builds, its unit tests pass, or balls look round.

Verification proceeds in this order:

1. Preserve the original failing experiment, exact inputs, and implementation.
2. Correct diagnostic attribution and prove detection-identity invariance.
3. Compare converged boundary solutions in common source-image coordinates.
4. Verify controlled crop and fitting-start sensitivity on saved scenes.
5. Evaluate the nine matching physical references with the finalized estimator.
6. Qualify target decode with saved JPEGs, then measure the complete pipeline
   separately when spatial estimation is stable.

Required correctness properties are finite/bounded work, correct frame and
asset identity, accurate diagnostics, deterministic geometry for identical
inputs with sufficient budget, and no fabricated physical coordinates.
Numerical FPS, latency, and physical-accuracy benchmarks are desired targets.
Report actual results and their limits; do not trade continuous reliable
measurements for a higher nominal rate or force acceptance by weakening checks.

Deliverables for the saved-input milestone:

- Buildable corrective commits, each with meaningful regression coverage.
- Frozen corpus manifest and versioned offline audit outputs/overlays.
- A before/after table for all annotated balls, including every rejection.
- Separate results for identity changes, proposal perturbations, and fitting
  starts; include reserved-scene results separately from tuning scenes.
- A nine-position diagnostic mapping report with per-point and held-out errors.
- A clear list of remaining spatial, temporal, and physical-reference limits.

## Execution ledger — 2026-10-27

Continue implementation on `codex/native-ball-localization`. The following
experiment checkpoints are committed locally and remain unmerged:

| Commit | Scope | Verification / result |
|---|---|---|
| `191f9ac` | Per-iteration quality attribution and bounded audit-trace API. | Focused localization tests, strict Clippy, formatting, and the perception crate's full all-target suite passed. |
| `cb71d5f` | Detection index no longer changes the sampling seed; added identity and reversed-batch regression. | The same complete checks passed; the regression compares centers, statuses, reasons, quality, and outlines for IDs 0–15. |
| `df93696` | Frozen-image identity audit source, JSON results, and review summary. | All 15 saved manual proposals are semantically identical across IDs 0–15. Three of nine measured proposals accept; six reject. |
| `99f6918` | Bounded, fixed-seed, converged candidate search and source-space qualification; synchronized schema/defaults and refiner revision 2. | 383 library tests and all perception targets passed serially; strict Clippy, formatting, and `make config-check` passed. The frozen audit at the final 96 radial samples accepts 0/15; this is a conservative rejection result, not accuracy proof. |

The audit is repeatability evidence only. The boxes are annotation-derived, not
YOLO output, and the physical reference uncertainty remains unknown. Current
acceptance is not evidence of correct image centers or millimeter accuracy.

Open work at this checkpoint (before the Task D audit below):

- Task B has stage-aware audit events and bounded trace memory, but exported
  multi-case traces and visual overlays still need to be produced after the
  converged-candidate event format is settled.
- Task D is the active implementation: compare geometrically diverse,
  independently qualified converged boundaries, correct center-shift units,
  synchronize settings/schema/revisions, and diagnose the six measured
  rejections without loosening qualification gates.
- Saved-image perturbation, nine-reference mapping, and streaming qualification
  remain later tasks; no physical accuracy or live responsiveness is claimed.

## Execution update — 2026-10-27

Task D implementation is committed locally as `99f6918` and has passed the
perception crate's complete single-threaded all-target suite, strict Clippy,
format checks, and `make config-check`. Its final implementation audit is saved at
`thoughts/calibration/converged-candidate-audit-20261027/` and reproduced by
`thoughts/calibration/offline-refiner-converged-candidate-audit-20261027.rs`.

Task D's current result differs from the earlier Task C identity audit:

- The Task C run on the pre-Task D selection behavior accepted 3/9 measured
  proposals but changed acceptance for 7/9 when only detection identity
  changed. The regression now checks identities 0–15 and reversed proposal
  order with fixed geometry; the frozen rerun was semantically identical.
- Task D revision 2 has bounded, traceable candidate comparison in common
  source-image coordinates. On the frozen 15 manual proposals, 0/15 precise
  centers are accepted, including 0/9 with matching physical references.
  Fourteen are `ambiguous_boundary`; one pocket proposal is
  `insufficient_boundary`. Thirteen have at least two individually qualified
  fits, with at least one alternative distinct from the top-ranked candidate;
  the bottom-right pocket proposal has no qualified fit after bounded
  reselection. There is no candidate-cap truncation or trace
  loss. This is evidence for conservative rejection, not a claim that the
  candidates or annotations reveal the true center.
- The radial-sampling default was checked against the approved plan and
  corrected from 48 to 96 in both typed Rust defaults and checked-in JSON.
  The saved-image audit was rerun with that final value; its 15 traces and
  overlays are the version recorded in the report.
- Standalone full-source and enlarged per-ball SVG overlays were rendered and
  visually inspected. Complete per-case JSON traces retain every candidate
  evaluation, quality record, and rejection stage.
- The nine-reference mapping is currently report-only and cannot use any
  accepted refiner centers. Do not build or emit a ball-center asset from
  these rejected observations or from bbox midpoints.

Task E's proposal-center, proposal-size, fitting-seed, reserved-scene, and
work-budget audits are complete. Its full evidence package is recorded under
`thoughts/calibration/robustness-audit-20261027/`, with exact source/input
hashes, all reserved-scene traces, and the six visually reviewed accepted
variant traces. The results show proposal/seed sensitivity and no accepted
results on either reserved scene; they do not qualify physical accuracy.

Task F is complete as a report-only no-fit evaluation in
`thoughts/calibration/report-only-nine-reference-evaluation-20261027.md`.
Revision 2 with production seed 716 accepted 0/9 observations, all rejected
for `ambiguous_boundary`, so the diagnostic solver was not run and no map was
emitted. Alternate-seed centers, box midpoints, and older-estimator centers
were excluded. The current twelve-correspondence production contract is
unchanged; no new measurement, jig, or camera capture was added.

### Follow-up — RANSAC sample provenance and non-monotonic search

Added sampled candidate indices and their rays to the bounded estimator audit
trace (audit schema 3); the production estimator and refiner revision 2 did
not change. The integration test confirms each recorded triplet maps back to
the exact previously traced boundary candidates.

The first-iteration sampling audit and its second, normal-aiming phase are in
`thoughts/calibration/sampling-audit-20261027/` and
`thoughts/calibration/full-aiming-sampling-audit-20261027/`. On the measured
purple proposal, seed 716 produces four individually qualified fits at 128
iterations and rejects as ambiguous. At 512, one isolated aiming iteration
accepts; 1024 rejects again. With the normal three-step aiming flow, 512
reaches a second pass, finds four further qualified, distinct fits, and
rejects with no center. Counts 128, 256, and 1024 also reject in the normal
flow. Alternate seeds 2, 10, and 11 accept different first-iteration centers;
the seed-11 alternative at 90/95 support falls just below the configured 95%
plausibility gate. These outputs do not establish which center is correct.

The recorded inliers for each production-seed fit use a different mix of the
three retained per-ray gradient-strength ranks. That is consistent with the
current RGB edge evidence presenting several nearby transitions; rank is not
a semantic outer-edge label. Keep the production seed, 128-iteration setting,
per-ray candidate cap, and all quality gates unchanged. Do not select a
seed/count because it accepts. The sampling timings include diagnostic traces
on the Darwin host and do not qualify the Intel target.

The per-ray candidate-cap audit is complete in
`thoughts/calibration/candidate-cap-audit-20261027/`. Across caps 1, 2, 4, 6,
8, and 16 on purple, orange-solid, and red-striped measured-ball proposals,
all 18 runs reject as `ambiguous_boundary`. At production cap 16 all three
searches are untruncated and retain 4, 2, and 8 fully qualified distinct fits
respectively. This rules out a too-small representative cap as the explanation
for these three rejections; it does not establish that every generated fit is
a true ball boundary. Do not continue sweeping cap values, RANSAC counts, or
seeds, and do not loosen quality gates to raise acceptance.

Next step: inspect the saved candidate and source-pixel traces for a visual
feature that distinguishes the outer ball/background contour from nearby
internal, stripe, highlight, or shadow transitions. State one falsifiable
selection hypothesis, then test it offline on the measured image and the
reserved 720p/4K scenes, reporting accepted and rejected cases. Keep production
behavior unchanged until one cue is repeatable across those scenes and can be
checked against available references without using the same image evidence
both to choose and validate the center. If no stable cue is supported, preserve
fail-closed behavior and document what evidence is missing. No saved-image
work to date establishes physical accuracy, a selected camera FPS, or live
pipeline availability.

## 2. Fixed requirements and working rules

- Table output is X/Y only. Playing area is 2336.8 × 1155.7 mm; all balls have
  physical diameter 57.15 mm. Do not estimate height or variable physical size.
- Keep the Coral model and its [1,1280,1280,3] float32 input unchanged.
- Use Intel `vaapijpegdec` for the qualified target JPEG path. Its YUY2 output
  may be converted to RGB. Do not force NV12, add GPU scaling, move inference
  to OpenVINO, add silent CPU fallback, or claim zero-copy processing.
- Maintain one production localization implementation. Offline audit controls
  do not select alternative production algorithms.
- Keep raw `DetectionBatchV1` boxes and their exact midpoint contract.
  Tracking and physical conversion consume the same resolved measurement.
- Invalid/stale/mismatched/unsupported observations cannot enable precise
  guidance. Preserve a visible coarse detection when applicable.
- Initial work uses saved pixels only. No jig, new manual ball measurements,
  camera snapshots, or pocket-ball coordinates are needed.
- Modify one top-level module at a time. Give each worker exclusive file
  ownership; tell workers others are active and prohibit reverting their edits.
- Keep research material under `thoughts/`. Keep production settings typed,
  configurable, validated, and synchronized with schemas/defaults/fixtures.
- Existing user authorization for routine reversible work persists. Do not
  repeatedly stop for confirmation of already authorized implementation work.

## 3. Preserve these discoveries with their actual evidence scope

| Discovery | What the evidence supports | What it does not establish |
|---|---|---|
| Native 4K MJPEG | Retained native JPEG, driver/source observations, and `stream-4k-change.md` establish the available detail and observed 20 FPS source delivery. | Complete-application 4K throughput, lower physical error, or qualified calibration. |
| Intel JPEG decoder | The installed `vaapijpegdec` decoded actual native JPEG pixels; the tested `vajpegdec` failed negotiation. The new saved-frame qualification retains command, plugin output, negotiated caps, hashes, exit status, and exactly 24,883,200 RGB output bytes. | GPU scaling, zero-copy RGB, whole-application latency, or sustained delivery. |
| Earlier GPU benchmark | Status/review records retain the observed 100 frames/5.044 seconds/about 64% of one CPU core. Preserve as a dated observation; original raw probe output was not found. | A raw persisted benchmark trace or reproduction of the full application. Do not reconstruct missing measurements. |
| 5 FPS source check | Six samples over about 18 seconds reported desired/captured 5 FPS; configuration was restored. | Selected production FPS, end-to-end availability, motion responsiveness, or long-run behavior. |
| 4K framing | Retained before/after images and registration support approximately direct 3× image-coordinate scaling. | Identical optics, blanket calibration promotion, or 3× physical accuracy. |
| Tested radial correction | Recorded holdouts showed no improvement; several rail features bowed more under those candidates. Preserve these negative findings. | Zero lens distortion or a universal rejection of every possible optics model. |
| Local viewing normalization | Some clean off-axis outlines become more circular. | Correct outer-boundary selection or demonstrated improvement in millimeter X/Y. |
| Refiner defect | The frozen identity audit changes acceptance for 7/9 measured crops when only detection index changes. Three complete repeats matched exactly. | Live availability percentages or independent physical-placement statistics. |
| Physical references | Nine user-confirmed positions apply to one retained image. Older leave-one-out discrepancies are around 8 mm RMS, with mixed unknown causes. | Proof the user measured incorrectly, independent 2 mm accuracy, or corner/pocket accuracy. |

Preserve failed experiments, assumptions, corrections, and data provenance as
carefully as successful findings. Do not overwrite old JSON with improved
results. Distinguish recorded observations, repeated controlled experiments,
and independently qualified application behavior in every report.

## 4. Repository preservation and promotion

Starting state was branch `codex/native-ball-localization`, base
`e9fb9a0f8402c1d5dd69657b0273d6fbc46b5e20`, with dirty experimental perception
changes. A recovery checkpoint must include tracked binary patches, separate
staged/unstaged patches, untracked files, base/branch, and hashes. A normal
`git diff` alone misses new Rust modules and research artifacts.

The initial buildable experimental source is now preserved at
`e595fe255291dd9e820295743fface33f8eae538`, explicitly unqualified. See
`proven-findings-promotion-record.json` for which preservation/promotion tasks
were completed during plan preparation; do not repeat completed commits.

Resume estimator implementation on `codex/native-ball-localization`, not on
main: main intentionally lacks the unqualified estimator modules. Start from
the checkpoint and preservation commits, verify a clean working tree, and
bring committed main into the experiment branch before the next code task.
If the helper conflicts, preserve main's qualified helper and the experimental
library changes; do not resolve by replacing an entire module from one branch.
Verify that integration with camera-feature tests. Bringing main into the
experiment branch does not authorize merging the experiment back into main.

Before each commit inspect the exact staged paths and full staged diff. Avoid
blanket staging, resets, stashes that hide unrelated work, destructive cleanup,
or accidentally committing model binaries and machine-local configuration.
Never include tokens, stream client IDs/keys, `.env` backups, SSH material, or
unfiltered service state. Preserve sanitized facts rather than credentials.

Already tracked at the starting base: calibration JPEGs, nine-position
measurements, checkerboard sources/corners/manifest, frozen optical models,
rail diagnostics, local-normalization prototypes, 4K framing evidence, and
`stream-4k-change.md`. Reference these files rather than duplicating them.

Commit groups:

The first four groups are completed: main has capture commit `3e67991`,
evidence commit `1739867`, and architecture commit `5787b56`; the experiment
branch has checkpoint `e595fe2`. Their actual order differs from the grouping
below. See the promotion record for exact SHAs and checks; do not re-extract
or recommit these groups. The following descriptions preserve their scope.

1. **Completed evidence on main:** `docs(calibration): preserve native-frame experiment findings`.
   Include this plan, corrected status/review notes, the frozen identity audit
   JSON and exact harness, short 5 FPS record, and saved-frame GPU qualification.
   Add an evidence index with source hashes, provenance, and limitations.
2. **Completed architecture on main:** `docs(architecture): record estimator stability requirements`.
   Include ADR-0102 and validate records. Accepted architecture does not mean
   the current implementation is qualified. Clarify that geometry/order
   invariance applies with sufficient budget and below capacity; legitimate
   deadline/capacity scheduling effects must be reported separately.
3. **Completed independent capture on main:** `perf(capture): decode MJPEG with bounded Intel VA-API`.
   Extract and verify only the independent helper change in
   `perception/src/bin/gstreamer_capture_probe.rs`, including its tests.
   Do not sweep in `capture_runtime.rs` or `capture_pipeline.rs`; their current
   dirty changes include localization interfaces. Do not include the new
   calibration binary merely because its Cargo registration is adjacent.
   Qualify this group against a clean main checkout and the saved JPEG target
   result. Record separately which behavior was fixture-tested versus tested
   on target hardware.
4. **Completed experimental checkpoint on the experiment branch:**
   `chore(perception): preserve unqualified native-localization checkpoint`.
   The checkpoint preserved the coherent Rust/config/test changes after
   baseline checks, including the original experimental GPU helper. Reconcile
   that helper deliberately with the qualified main version when branches are
   brought together. Do not merge this checkpoint into main as a qualified
   localization feature.
5. **Subsequent corrections:** commit diagnostics, identity independence,
   converged-boundary handling, corpus audits, and physical mapping diagnostics
   in the task boundaries below. Each code commit includes its regression
   coverage and necessary configuration/revision changes and remains buildable.

Keep an actual promotion record with commit SHAs, files, verification commands,
results, known limits, and whether the change is on local main or published.
Do not push unrelated existing local history merely to publish these commits.
Do not merge the entire experiment branch to obtain a small proven finding.

## 5. Task A — Freeze the corpus and executable failing baseline

Ownership: research records under `thoughts/calibration/`; change this module
before returning to perception code.

Create `native-localization-corpus-v1.json` with, per source:

- Stable image ID, repository path, source SHA-256, dimensions, arrangement ID.
- Intrinsics provenance and exact matrix; flag derived 4K matrices explicitly.
- Proposal origin and stored boxes. Manual/annotation proposals are never
  labelled detector output. Preserve known contaminated estimates.
- Which physical references apply, their coordinate convention, and their
  unknown uncertainty. Keep the six pocket cases outside physical truth.
- Intended use: baseline/tuning, reserved image check, optics-only, or GPU-only.

Use these retained sources:

- `table-grid-20261026-114241.jpg`: 15 annotated balls, nine matching measured
  positions. Existing proposal construction is centered at segmentation bbox
  midpoint with square side `max(gradient ellipse diameters) + 4 px`.
- `table-grid-2026-10-26.jpg`: separate arrangement; image diagnostics only.
- `checkerboard-session/view-001.jpg` through `view-009.jpg`: optics provenance;
  visible balls need separately reviewed proposals and occlusion labels.
  Balls were moved by view 006. Original nine X/Y values do not transfer.
- `stream-4k-verification/before.jpg` and `after.jpg`: matched framing evidence;
  the latter is the only saved native 4K scene and contains three balls.

Freeze the tuning/reserved split before further changes: use the measured
grid for diagnosis; reserve the alternate grid and native 4K three-ball scene
from further threshold tuning. Both scenes have already been inspected, and
the 4K scene was previously tested; these are not blind or previously unseen
holdouts. They are additional contour checks, not independent
physical accuracy references. Do not count upscaled or downscaled copies as
new independent images. Do not treat annotated overlay PNGs as raw inputs.

Archive the original failing audit untouched. It specifies exact source and
code hashes, K, defaults, and diagnostic budgets (5000 ms batch, 400 ms per
proposal). These generous budgets isolate fitting behavior and are not
production performance settings. Preserve a code checkpoint matching its
refiner hash so the old result remains reproducible after implementation changes.

Task completion: every input is hash-addressed; a clean research run reproduces
the recorded counts; no physical coordinates migrate between arrangements.
Commit: `docs(calibration): freeze native-localization corpus and baseline`.

## 6. Task B — Make diagnostics describe the actual fit

Ownership: `perception/`, primarily `src/ball_localization.rs`, its tests, and
the standalone offline audit entry point.

1. Inspect `BallLocalizationQuality`, `refine_one`, and its aiming loop.
   Reset iteration-local values before each iteration. Residual, inliers,
   spread, and foreground values not computed in the failing iteration must
   remain absent or explicitly attributed to an earlier fit.
2. Give diagnostic records iteration ID, candidate/fit ID, stage, and reason.
   The compact runtime summary describes the final accepted fit or failed
   iteration. Do not present earlier residuals as final rejection evidence.
3. Add a bounded audit observer to the same estimator logic, used by an offline
   auditor. Record candidate edges/rays, raw hypothesis summaries, converged
   solutions, selected inliers, projected centers/outlines, and rejection stage.
   Tracing must not alter candidate ordering or mathematical results.
4. Bound retained trace size by validated sample/iteration/hypothesis ceilings.
   Write JSON and overlays as each image completes; do not retain unbounded
   frame histories, serialize every sampled pixel, or clone full RGB frames.
5. Give persisted research traces their own schema/version. Preserve
   `DetectionBatchV1` unchanged. Do not add a production localization mode or UI.

Regression: iteration 1 computes residual/spread, iteration 2 rejects before
those stages; the final summary cannot misattribute iteration-1 values.
Also test early rejection, zero detections, accepted final summaries, bounded
trace output, and equal geometry with/without an audit observer.

Completion: existing cases yield interpretable per-iteration overlays and
summaries, with no new geometry claims. Commit:
`fix(perception): attribute ball-fit diagnostics to their actual iteration`.

## 7. Task C — Remove detection identity from geometric computation

Ownership: `perception/src/ball_localization.rs` and associated tests.

Replace `config.seed ^ proposal.detection_index` with a deterministic sampling
schedule controlled by the configured estimator seed and independent of
detection index, batch position, frame/session IDs, and incidental labels.
Do not replace the seed with an ID-derived hash. Resolve equal scores using
stable geometric/support tie breaks. Preserve detection indices only for
association and output identity.

Tests and audits:

- Fixed pixels/box/config, every index 0–15: center, status, reason, outline,
  and semantic diagnostics match after ignoring association IDs and timings.
- Proposal permutations with sufficient budget and below capacity: associate
  results by observation and compare geometry exactly.
- Different legitimate image/box inputs can still produce different results.
- Overload: keep confidence-order scheduling, explicit `DeadlineExceeded`
  and `ProposalLimitExceeded`, and correct association. Order can affect which
  work finishes under a shared deadline; do not mislabel scheduling as a
  geometric ambiguity or promise an invariant completed subset under overload.
- Repeat the frozen identity experiment using new result filenames/source
  hashes. Removing index dependence proves repeatability, not accuracy or
  insensitivity to the configured fitting seed.

Bump `BALL_REFINER_REVISION` when center/acceptance semantics change; update
profile/resolver fixtures and reject assets bound to older semantics. Do not
silently reuse old ball-center assets.

Completion: no identity-only acceptance changes with adequate budget. Commit:
`fix(perception): make ball fitting independent of detection identity`.

## 8. Task D — Compare supported converged boundaries

Ownership: `perception/`, especially `ball_localization.rs`, profile revisions,
configuration defaults/schema/validators, and relevant fixtures.

The current `has_competing_fit_cluster` counts raw triplet circles before IRLS.
Replace this mechanism; changing the vote count or selecting a lucky seed is
not the planned solution.

Algorithm requirements:

1. Keep bounded radial candidate generation and deterministic bounded RANSAC.
   Preserve multiple candidates per ray and outer/background evidence.
2. Rank plausible hypotheses by supported rays and normalized residual with
   deterministic ties. Do not favor fewer points merely for a lower summed
   squared error. Retain a bounded diverse set of candidates so near-duplicate
   guesses do not exhaust every refinement slot.
3. Refine each retained representative with IRLS and bounded inlier reselection.
   Each loop checks both proposal and batch deadlines. Initial configurable
   engineering limits: at most 16 refined representatives per aiming iteration
   and at most 3 inlier-reselection rounds per representative. Validate absolute
   ceilings and record these as initial work bounds, not accuracy-derived facts.
4. Evaluate converged candidates for supported arcs, largest gap, background
   agreement, outer-boundary evidence, radius validity, residual, total shift,
   foreground support, and omitted-quadrant sensitivity.
5. Project centers and sampled outlines into source-image coordinates before
   comparing alternatives. Center distance, outline disagreement, and quadrant
   sensitivity need source-pixel units and resolution scaling. Virtual-patch
   radius is not a physical radius or a source-image radius.
6. Cluster candidates converging to the same boundary. Several raw guesses in
   that basin are not separate evidence of ambiguity. Distinct well-supported
   competing boundaries remain grounds for rejection. If same-center contours
   have materially different outlines, preserve the outline uncertainty rather
   than declaring shape established merely from center agreement.
7. A candidate can be accepted only through the common qualification logic.
   Do not return an unqualified runner-up because the first candidate failed.

Use the audit traces to determine whether known rejects are sampling jitter,
wrong boundaries, coarse-proposal errors, missing evidence, or genuinely
competing contours. If the candidate cap appears to miss a solution, compare
against a higher bounded offline reference run; revise the work bound with
recorded evidence. Do not add unlimited work or tune each ball separately.

Configuration/revision work belongs in the same coherent commit:

- Remove obsolete `minimum_competing_fit_votes` from Rust, defaults, schema,
  validation, tests, and documentation; do not preserve an unused control for
  an unshipped experiment.
- Add only parameters the selected algorithm consumes, including bounded
  representative/reselection counts and any separately meaningful source-pixel
  outline threshold. Every parameter has defaults and validation.
- Update `ball_localization_profile.rs`, refiner revision, and dependent
  profile/resolver fixtures. Retain exact optics/refiner/frame compatibility.
- Do not change physical dimensions or `DetectionBatchV1` to fit image errors.

Meaningful regressions: noisy raw hypotheses converging to one known contour;
two genuinely distinct plausible contours; highlights/stripes/shadows;
rail-contact partial arcs; unsupported arcs; off-axis source-coordinate
distance; deterministic ties; deadline inside alternate refinement; bounded
allocations; known synthetic centers and inverse transforms.

Completion: explain each changed real-image acceptance using trace evidence;
retain genuine failures. Commit:
`fix(perception): compare converged ball boundaries in source coordinates`.

## 9. Task E — Run controlled saved-image robustness audits

Ownership: first perception audit tooling/tests; then research artifacts in a
separate task. The auditor invokes the production estimator, not a copied fit.

Freeze one common parameter set before scoring reserved scenes. Perform these
independent experiments; never change index and crop together:

| Experiment | Fixed | Varied | Report |
|---|---|---|---|
| Identity | Pixels, box, seed, K, budget | Index 0–15; adequate-budget permutations | Exact semantic equality; scheduling results separate |
| Center sensitivity | Pixels, box extent, index, seed, K | X offset then Y offset: −2, −1, 0, +1, +2 source pixels at 1280 width; scale offsets with width | Acceptance, reasons, source-center spread, outlines |
| Size sensitivity | Pixels, center, index, seed, K | Width then height scale: 0.90, 0.95, 1.00, 1.05, 1.10 | Same metrics; label boxes that no longer contain the observed silhouette |
| Fitting starts | Pixels, box, K, budget | 16 recorded config seeds | Distribution of accepted centers and reasons; no production seed switching |
| Scene generalization | Selected parameters | Reserved saved images and reviewed proposals | Per-case behavior and source-profile limitations |
| Work budget | Same geometric inputs | Explicit bounded runtime settings | Deadline versus geometric rejection, timing distributions |

These perturbation values are declared research workloads, not hidden
production constants or certified error margins. Keep axes separate before
any combined stress test. Repeated trials on a still are not new placements.

Export full-source overlays plus enlarged native crops showing observed edges,
selected inliers, qualified alternatives, projected outline, center, and reason.
Label manual proposals and known segmentation-fragment errors explicitly.
Do not treat another image-fitting method's center as physical ground truth.

Summaries must include all 15 original cases, the nine measured cases, the six
unmeasured pocket cases, accepted and rejected counts, and per-case changes.
Investigate every failure among the nine. Do not remove difficult cases from
the denominator or accept them merely to reach nine. Report failures on the
three-ball native 4K image without claiming full-table 4K coverage.

Completion: the identity defect is gone, sensitivity is measured, accepted
contours are visually reviewed, and unresolved failures are explicitly named.
Commit research outputs separately:
`docs(calibration): report frozen-corpus localization robustness`.

## 10. Task F — Re-evaluate physical X/Y with the existing references

Ownership: research tooling/records first. Production calibration changes,
if justified later, are a separate perception task and ADR update.

Use only the nine coordinates associated with
`table-grid-20261026-114241-measurements.json` and its exact JPEG. Extract
accepted centers with the final estimator revision. Preserve rejected cases
and never substitute bbox midpoints or older-method centers into the new fit.

The measurements use an upper-left origin with +X image-right and +Y
image-down; production numerical coordinates use a table-center origin.
Verify the selected profile's corner0 and axis orientation before conversion.
If it matches the measured convention, compute `x_center = x_upper_left - W/2`
and `y_center = y_upper_left - H/2` using configured physical dimensions. For
this table the offsets are (1168.4, 577.85) mm. If the profile corner order
differs, map axes explicitly first. Serialize the origin, directions, units,
and conversion in the report; do not compare mixed coordinate conventions.

Fit a normalized diagnostic homography using a report-only research solver or
extracted numerical core, without invoking or weakening the production
eligibility validator. Never emit a loadable calibration asset from this
diagnostic. Include rank/conditioning checks and report
every training residual, every leave-one-position-out prediction where the
remaining points are sufficient, spatial coverage/extrapolation, and the
number of accepted points. If too few/spatially degenerate points remain,
report that outcome instead of manufacturing a map. Keep uncertainty unknown
where the physical reference uncertainty was never independently established.

Compare against existing 8.71 mm baseline as a diagnostic comparison with
different estimators and possibly different accepted subsets. For a direct
comparison also report a common-point subset; disclose the selection effect.
Do not equate lower training residual with better held-out accuracy. No spline,
additional radial coefficient, or more complex map is planned from these nine.

The current production calibrator requires at least 12 points in solver
defaults, policy validation, and input schema. Keep the initial nine-point
report non-activatable. Resolving production eligibility requires an explicit
follow-up decision: either existing reference evidence eventually satisfies
the current contract, or replace that contract with justified redundancy,
coverage, conditioning, and qualification requirements in coordinated code,
schema, tests, and ADR changes. Merely permitting nine points cannot establish
independent accuracy. Do not turn this mismatch into a demand for a jig or
duplicate observations to reach 12.

Completion: reproducible per-point diagnostic result and explicit activation
status. Commit: `docs(calibration): evaluate revised estimator against nine references`.

## 11. Task G — Further qualify capture independently

Independent capture commit `3e67991` is already on main. Its 15 focused
helper tests, 709 camera-feature tests, strict camera-feature Clippy, and
format checks passed; the target separately decoded a saved native JPEG.
The original 100-frame GPU observation has summary evidence only. Do not
re-extract or recommit the helper. Spatial-localization failure does not
invalidate the independently qualified GPU discovery.

Use the saved native JPEG for target hardware checks. Retain source hash,
exact command, GStreamer/plugin versions, render-device access, decoder and
output caps, RGB byte count, exit status, and logs. The retained
`gpu-saved-frame-qualification-20261027.json` establishes one successful saved
frame decode. Process startup time from that run is not camera FPS or sustained
decode throughput.

For subsequent changes, retain regression coverage for multipart JPEG parsing,
latest-only compressed buffering before decode, slow consumers,
malformed/truncated data, encoded-size limits, disconnect/error propagation,
reconnect through the supervisor, and shutdown. Test-only decoder injection
must not become a production backend selector or fallback. Record new results
independently and do not merge the entire experimental branch to obtain them.

## 12. Task H — Later complete-pipeline responsiveness and release

Begin only after estimator stability and calibration eligibility are resolved.
Saved stills alone cannot qualify motion response, temporal availability, or
capture-to-publication age. A later complete-stream workload must measure them;
do not claim it was performed by the saved-JPEG checks.

Start at native 3840 × 2160, 5 FPS. Keep Coral input unchanged. Retain a small
motion grid and existing cue/ball scheduling unless measurements justify an
explicit configuration change. Keep focus/zoom fixed and record exposure/gain.
Select the next candidate adaptively rather than blindly running a large matrix:

- If 4K/5 FPS is reliable and motion response benefits from more samples, test
  7.5 or 10 FPS; raise further only with useful measured benefit.
- If 4K saturates conversion, memory, or crop budgets, compare 2560 × 1440 and
  1920 × 1080 at low cadence while measuring the corresponding localization
  quality and matching profiles. Do not silently scale old calibrations.
- Prefer consistent fresh measurements. Report desired-target misses; do not
  change freshness semantics or reuse stale coordinates to improve availability.

For each sustained run record decoded/analyzed/published/dropped counts;
geometric versus timeout/profile rejection; precise availability and unavailable
episode durations; model/refinement/resolve/receipt-to-result age; motion-to-
guidance invalidation; CPU/GPU use; leased RGB frames, decoder surfaces, model
tensors and scratch memory; and memory trend. State denominators and workload.
Begin with 10-minute comparative runs; use a longer soak for a selected candidate.
One processing instance owns the target at a time.

Keep a previous known deployment and use the existing PID-aware bundle update
workflow. Do not assume the named application systemd service owns the process.
Ship one selected configuration only with compatible assets and clearly stated
measured accuracy/availability. Existing desired 2 mm p95/5 mm maximum targets
are not claims supported by current references. Preserve independently useful
main commits even if the localization continuation is not ready to ship.

## 13. Verification commands and commit discipline

Run focused meaningful regressions after each perception correction; use
actual test names and package paths from the current repository. At coherent
estimator boundaries run:

```text
make test-rust
make lint-rust
make format-check
make config-check
```

After worker/runtime/profile integration, and before promoting that complete
implementation, run the repository-required suite:

```text
make test
make test-rust-camera
make lint
make format-check
make config-check
make test-deploy
make test-tooling
```

`test-rust-camera` needs host GStreamer development support; passing it does
not prove the target GPU was used. Validate ADRs with
`/Users/jchadwick/.agents/skills/manage-architecture-decisions/scripts/validate_adrs.py`.
Use an available equivalent skill path if the executing machine differs.
Run `git diff --check`, inspect staged changes, and record verification before
each commit. Do not add a deliberately red default test commit; include its
regression with the fix. Re-run broader checks when new code/failures justify
them, rather than repeating unchanged suites indefinitely.

Each handoff/commit record states: current branch/SHA, owned files, change and
reason, exact checks/results, artifact locations/hashes, observed limitations,
and the next dependent task. Preserve user workspace changes and clean only
temporary files created by this work.

## 14. Executor completion report

Report the promoted main commits separately from the unqualified experiment
checkpoint and subsequent corrections. State whether main is local/published.
Provide the identity-invariance result, all-case acceptance/rejection table,
controlled-sensitivity results, reserved-scene observations, nine-point mapping
diagnostics, and remaining calibration/physical-reference limits.

If later target qualification occurred, also state selected processing geometry,
camera FPS, Coral shape, Intel GPU workload, measured latency, sustained precise
availability, unavailable-episode durations, motion response, and memory trend.
If it did not occur, say so. Do not substitute software-test success, pixel fit
residuals, rounder outlines, or short source cadence for those measurements.
