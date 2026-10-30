# Native-resolution 2D ball localization: status and implementation record

Last updated: 2026-10-30

## Current merged status

The user selected improvement over perfection and authorized merging the
localization implementation into `main`. The merge includes the deterministic
native-pixel ball-center refiner, center-calibration path, frame-bound tracking
integration, scheduler and capture changes, and the accepted ADR follow-ups.
Coral is retired from live production inference; the live path uses OpenVINO on
the Intel GPU. These changes are committed locally but have not been pushed or
deployed.

The best recent continuity evidence is the R10 saved-pipeline replay: 3840 ×
2160 processing at 5 FPS, 600/600 fresh public-state samples, and all 15 balls
with precise positions in each sample. The input was one static 720p image
upscaled to 4K, and the exact source snapshot that produced the recorded run
is unavailable. This supports improving availability on that fixture; it does
not establish physical X/Y accuracy or prove the exact checked-in refiner
reproduces the run.

The nine-point diagnostic mapping has about 8.8 mm RMS and 11.1 mm maximum
error, with unknown reference uncertainty. Independent physical position
accuracy, native-camera edge/corner performance, and shot-coaching response
remain unqualified. These are limits to characterize while improving the
product, not reasons to withhold this user-authorized code merge.

## Prior saved-image result — 2026-10-29, refiner revision 6

Production fixes are committed through `af1f9055`; exact-frame frontend contour
rendering is also committed. Frozen actual detector boxes accept **13/15**, with
**9/9** measured references accepted. No new capture or physical work was used.
The previous 0-acceptance results below remain historical evidence.

A complete CPU ONNX Rust replay of the measured frozen image at 5 source FPS
produced 12 precise table centers and source contours on all 11 scheduled detector
calls; 9 other source frames were intentional phase-schedule skips, with zero
overload skips, obsolete results or inference deadline misses. Detector backend
proposal differences mean this is not the same experiment as the 13/15 PyTorch
proposal audit, and neither is target Coral numerical qualification. Repeated
still frames prove integration/repeatability, not independent accuracy.

Default no-trace refiner replay p95 on the development Mac: 103.51 ms for the
15-ball measured scene, 99.35 ms for 15 alternate proposals, 23.38 ms for the
three-ball native4K scene. Budgets are 150 ms/batch and 30 ms/proposal, configurable.
These figures exclude inference and do not select target FPS.

The two pocket failures implicate a single felt-background appearance assumption:
visible silhouette sections against dark pocket hardware are discarded. Research
is continuing on bounded local-background evidence while preserving actual
geometric qualification. Extra fit iterations alone did not restore acceptance.

Revision-bound diagnostic X/Y mappings and full replay records are preserved in
`runtime-localization-20261029/`. Unknown physical-reference uncertainty and
extrapolated corners remain explicit; no independent 2 mm claim is justified.
Target SSH was unreachable during this continuation, so nothing was deployed.

---

## Implementation record and historical evidence

Last updated: 2026-10-28

The notes below preserve the sequence of implementation and research results.
Earlier rejection counts and suggested experiments are historical; the current
merged implementation and latest saved-image findings are summarized above and
in `refiner-r10-validation-20261029/r10-scheduler-and-corner-availability.md`.
The broader qualification checklist remains in
`native-ball-localization-next-steps-plan.md`.

## Historical correction direction — 2026-10-29

The user explicitly directs continuing with all existing saved data and no
further checkerboard or other camera images. This supersedes the earlier
suggestion of two new independent-target captures. Physical capture is not a
prerequisite to this software work. Runtime correction now replaces stochastic
boundary-hypothesis generation with bounded deterministic cyclic paths, retaining
full pixel qualification and aiming. The earlier offline-only path comparison
is not a production acceptance result. Actual detector replay supplies genuine
15-ball proposals for the measured image; the unchanged refiner rejects 0/15
accepted. New code must be evaluated against those inputs, not only manual boxes.
The twelve-reference eligibility rule is also under review for replacement by
justified rank/coverage checks, without manufacturing physical accuracy claims.

## Offline evaluation completed — 2026-10-28

The eight-image dataset, optics, table-transfer and center/outline audits are
complete. See `next-localization-decision-20261028.md` for the selected next
discriminating experiment and its implementation/validation specification.

- Source hashes, 54-corner counts and grid topology pass, but view 005 has a
  demonstrated corner-location error of approximately 20 source pixels.
  Source-only relocalization fixes the gross outliers in a separate record.
- Original whole-view holdout RMS is 2.504/2.083/2.089 px for zero/k1/k1k2;
  relocalized k1 reaches 1.591 px. Neither target construction nor these
  optical candidates is qualified. On the old prospective view009, old zero
  scores .2279 px versus .2680 for new relocalized k1 under approximate scale
  transfer. New radial fits also degrade the rail-lattice checks.
- Thirty multi-candidate aiming iterations across 33 saved cases contain no
  center-stable/outline-only disagreement. Dropping the outline check alone
  is unsupported. Conditional table-map sensitivity exceeds within-candidate
  center spread; neither is independently measured accuracy.
- Current production-seed acceptance remains 0/9 measured and 0/18 additional
  saved proposals. No map, runtime setting, deployment or camera change was
  made. The evaluation does not prove that the user's ball measurements were
  wrong.
- Next: prepare validated corner observations, then independently score two
  native4K views of the earlier verified small rigid target with frozen optics.
  This is a discriminating experiment, not a selected replacement estimator.
  Every architectural decision remains open to revision on evidence.

The table remains covered. No camera access occurred in this evaluation.
Before either future capture, explain its purpose and wait for confirmation
that the table is uncovered. The user has no physical task during this phase.

## Current executed state — 2026-10-27

- The final revision-2 audit on 15 manual proposals from the measured 720p
  frame accepts **0/15** centers at production seed 716. Fourteen reject as
  `ambiguous_boundary`; one pocket proposal has insufficient boundary
  evidence. The nine proposals with physical references all reject, so no
  report-only homography was fit and no calibration asset was emitted.
- Controlled saved-image sensitivity work found 1 accepted result across 150
  proposal-center perturbations, 0/150 size perturbations, and 5 accepted
  results across 240 seed/proposal runs. The selected production seed accepted
  0/15; alternate-seed acceptance is not evidence of accuracy. The 15-box
  batch timed out all proposals at the initial 25/10 ms limits on the Darwin
  development host, while relaxed limits took about 0.29 s. These timings do
  not qualify the Intel target.
- All 18 manually proposed crops on the reserved 720p and 4K scenes reject.
  These images lack qualified optics profiles and physical X/Y references.
- Audit schema 3 records the sampled boundary-candidate indices and rays for
  each valid RANSAC hypothesis. The first-iteration and normal-aiming saved
  audits show RANSAC acceptance is non-monotonic: seed 716 accepts at 512
  iterations for one isolated pass, but the normal aiming flow rejects with
  no center. Do not change the production seed, count, candidate cap, or
  gates to increase acceptance.
- The independently verified Intel GPU JPEG-decoding result remains
  independently adoptable. Complete pipeline accuracy, target FPS, latency,
  memory trend, and sustained precise availability remain unqualified.

The per-ray candidate-cap question is now tested. All 18 cap/proposal runs
reject as ambiguous; at the production cap of 16 the candidate search is
untruncated for all three crops and still finds multiple qualified boundary
fits. Do not spend more effort raising the cap or changing the RANSAC count,
seed, or acceptance gates to make these crops pass. The next useful saved-image
pixel audit has also been completed: across 27 measured/reserved proposals,
91 qualified fits and 7,891 inlier-ray RGB profiles, neither transition rank
nor simple inside/outside color statistics consistently distinguishes the
physical silhouette. Preserve the ambiguous result. The fisheye-specific
model audit failed to produce an admissible calibration. The fixed-sphere
conic audit also failed to improve table consistency: its best all-nine X/Y
training RMS was 15.21 mm and its held-out oracle candidate score was 18.65 mm
RMS, compared with 8.71 mm for the existing segmentation-center homography.
The oracle uses each held-out answer to choose its candidate, so it is not an
automatic result. Preserve the synthetic math check, but do not activate this
selector. Static table-feature registration passes relative image-transfer
checks, and the report-only rectangle map is too sensitive to the ambiguous
cushion edge for metric localization. The new mark-grid audit shows that the
rail sights and cloth spots provide a candidate physical coordinate grid;
however, checkerboard intrinsics and the rail-to-cloth plane bridge are not
yet qualified. No ball measurements or new table-mark measurements are
needed.

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

## Implementation details now merged into local `main`

- The GStreamer capture probe parses MJPEG, drops stale compressed frames
  before decode, uses `vaapijpegdec`, converts its native YUY2 output to RGB,
  and bounds app-sink buffering. It does not silently use software decode.
- The Rust local-view refiner works on borrowed native RGB, returns explicit
  fit/reject evidence per proposal, and has bounded sample counts, work time,
  ROI size, and scratch allocation. The revision-2 selection was invariant to detection identity and proposal
  order for its frozen audit inputs, but at that historical revision accepted
  0/15 measured-scene proposals and 0/9 proposals with physical references. The 18-run cap audit found that cap 16 is untruncated on all
  three tested crops and still leaves 4, 2, and 8 individually qualified
  boundary fits for the purple, orange, and striped balls respectively; every
  cap/proposal run remains ambiguous. Earlier 4/9, 3/3, and 5/9 acceptance
  counts are historical results from prior estimator or proposal constructions
  and must not be presented as current qualification. Details are in
  `report-only-nine-reference-evaluation-20261027.md`,
  `robustness-audit-20261027/`, the sampling audit folders, and
  `candidate-cap-audit-20261027/`.
  A separate pixel-level audit of nine measured 720p proposals and 18 reserved
  720p/4K proposals found no repeatable simple RGB or gradient-rank rule among
  91 qualified fits. The report and source-pixel profiles are in
  `contour-color-audit-20261027/`; they do not establish which outline is the
  physical silhouette.
- The existing checkerboard/ellipse evidence is consistent with off-axis
  perspective shaping a sphere's apparent image ellipse: the measured far-left
  yellow ball has a gradient-ellipse aspect ratio of 1.314 versus a 1.312
  small-sphere perspective approximation; the far-right purple ball is about
  1.25 versus about 1.27. This does not prove the lens has no residual
  distortion. A new OpenCV fisheye-model audit found 0/8 admissible held-out
  folds; the prospective ninth-view parity RMS was 1.3266 px versus 0.2279 px
  for the frozen Brown-zero model. The test model also greatly increased
  straight-edge bow and had unresolved flag/skew behavior. It is rejected as a
  correction candidate, not evidence that the lens is perfectly pinhole.
  Details are in `fisheye-optics-audit-20261027/`.
- The fixed-sphere conic audit is in `sphere-projection-audit-20261027/`.
  Ideal synthetic tests recover table X/Y to under 0.001 mm, but 0.15 px
  contour noise produces up to 0.935 mm error. On the saved scene, 36 qualified
  outlines across nine balls yield 15.21 mm all-nine training X/Y RMS; the
  held-out oracle candidate score is 18.65 mm RMS versus 8.71 mm for the
  existing segmentation-center homography. The oracle sees each held-out
  position when choosing among candidates, and the bounded search is not
  exhaustive. This rejects the tested selector as an improvement, not the
  ideal sphere-conic geometry. The user-provided reference uncertainty remains
  unknown; no height/depth output or production change is made.
- The static table-feature registration audit is in
  `table-feature-registration-audit-20261027/`. Six upper/lower rail sights
  produce 0.209 px held-out RMS between the measured and alternate 720p scenes
  and 0.094 px between the before/after framing images after 3× normalization.
  Held-out side-rail sights measure 0.463 px and 0.196 px RMS respectively.
  Two cloth markers are visible only in the framing pair; their cross-plane
  transfers are 1.360 px and 0.004 px, too few to infer a plane correction.
  All reported positions are pixels. Exact physical dot XY is absent from the
  inspected records, so this is relative registration evidence only.
- The report-only playing-area rectangle audit is in
  `playing-area-homography-audit-20261027/`. Its synthetic homography and fixed
  center-plane round trips pass, but the nominal X/Y residual against the nine
  ball references is 22.04 mm RMS. Across 13,122 line/corner variants, the
  maximum per-ball coordinate spread is about 23.8 mm; using the nearby lower
  seam instead of the inner nose transition worsens the residual range to
  33.76–41.84 mm RMS. The sampled brightness edges do not identify the
  physical nose reliably, and the ball-reference uncertainty is unknown. No
  metric asset or production change is justified.
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
the matching `table-grid-20261026-114241.jpg` arrangement. The current refiner
rejects all nine matching image observations, so they cannot yet be used to
fit even a diagnostic ball-center map. Use the precisely measured table dots
and rail geometry only for the physical references they actually represent.

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

The user clarified that the table's rail and felt dots are the physical grid.
The offline mark-grid audit at
`mark-grid-calibration-audit-20261027/` found that ten visible long-rail
sights match the WPA 8-foot eighth-spacing layout with 0.57–0.59 px fit RMS
and 0.99–1.01 px leave-one-sight-out RMS. The 92-inch table length matches the
11.5-inch sight interval; the 45.5-inch measured width differs from the
nominal 46-inch WPA value, and WPA warns that its competition specification
may not apply to home tables. Treat the standard as a candidate lattice and
the user's dimensions as authoritative.

The two visible cloth dots match the head/foot spots at x=23 and 69 inches,
y=22.75 inches. A rail-plane map alone misses them by roughly 11 px. The
checkerboard-intrinsics plane-aware fit estimates a 52.831 mm rail-cap/cloth
separation and gives 1.341 px joint error, but single-spot cross-checks remain
2.296–2.481 px and yield rail-plane estimates 12.134 mm apart. Side-rail
sight annotations are ambiguous; their 4–6 px residuals are excluded from
the fit. The nine saved ball references remain diagnostics only. Coarse box
midpoints give 11.286 mm RMS after mapping, not qualified ball-center
accuracy.

## Remaining qualification

The replacement localization implementation is now part of local `main`, as
a deliberate improvement over the previous path. It has not been deployed to
the target device. The latest saved 60-second replay reported fresh positions
for all 15 balls in 600/600 state samples with no detector, cue, overload, or
obsolete-result failures. That replay repeated a static 720p image upscaled to
4K; its exact refiner source snapshot is missing, so treat it as encouraging
continuity evidence rather than a reproducible accuracy result for the current
checkout.

Physical X/Y accuracy remains uncertain. The nine existing references have
unknown measurement uncertainty; their diagnostic mapping reports about
8.8 mm RMS and 11.1 mm maximum error. No independent held-out corner or pocket
accuracy qualification has been completed. The saved replay's 5 FPS cadence
does not select a live camera rate or establish motion-coaching behavior. The
user does not need to provide another checkerboard image, build a jig, or repeat
ball measurements for this implementation merge.
