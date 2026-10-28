# Localization decision after the larger-board evaluation — 2026-10-28

## Decision

The saved evidence does not select a qualified replacement ball-localization
algorithm or a production calibration. The selected next step is a small frozen-model test on an independently
checked single-piece target, using explicitly validated corner observations. This is the discriminating
experiment allowed by the approved evaluation plan. No new camera image was
requested or fetched during this evaluation; the table can remain covered.

All architectural choices remain revisable. In particular, unique-outline
acceptance, stochastic fitting, local normalization, global image correction,
inference placement and the twelve-reference contract are not immutable user
requirements. Preserve fixed physical diameter and X/Y output, reliable fresh
measurements, and independently verified GPU decoding.

## What changed our understanding

The eight original images are distinct, intact native 4K sources with correct
54-corner counts and consistent grid ordering. However, view 005 contains
visibly mislocalized intersections. Source-only refinement moves one by
19.98 pixels. A small-window refinement can converge without moving a badly
initialized corner; counting corners did not verify their accuracy. The raw
manifest remains unchanged. This is a demonstrated software observation
failure, not evidence that the user measured balls incorrectly.

Original whole-view holdout RMS is 2.504/2.083/2.089 px for zero/k1/k1k2.
A separately labeled source-only wider-window diagnostic reduces k1 RMS to
about 1.591 px and removes the approximately 20 px worst error. These are
camera reprojection diagnostics, not table millimeter accuracy. Wider-window
agreement alone cannot establish the physical intersection or target flatness.

The added radial terms help the larger board but do not improve the old
prospective small-board validation view: the old frozen zero model scores
0.2279 px RMS there, compared with about 0.2680 px for the source-relocalized
new k1 fit. The old aggregate includes eight training images and must not be
represented as independent validation. Approximate framing transfer is
supported by fixed rail sights but remains an additional assumption.

The corrected k1 candidate also increases rail-lattice leave-one-sight-out RMS
from 1.013 to 8.352 px at 720-equivalent scale, while improving the two cloth
spot checks. That disagreement cannot establish which target/standard/optical
assumption is wrong. The assembled board's flatness and dimensions, standard
sight offsets on this home table, and exact independent ball coordinates are
not all verified.

Thirty multi-candidate aiming iterations across 33 baseline cases contain no
case where centers stay within the current tolerance while only outlines
disagree. Dropping outline rejection alone is unsupported. Existing measured
candidate centers span about 1.19–3.71 mm through the old diagnostic map;
changing plausible optics and sight offsets produces much larger conditional
coordinate envelopes. These are sensitivity results, not accuracy bounds.
The current refiner remains a no-fit result on the nine physical references.

## Discriminating experiment: software and interfaces

Create a versioned, report-only validated-observation input alongside the raw
capture manifest. Its record contains source hash, detector/refiner revision,
geometry, initial corners, every neighborhood refinement, per-corner movement,
review status and rejection reasons. Preserve all original inputs immutably.
Do not change runtime measurement APIs, activate optics, or pick a new ball
estimator during this experiment.

Use the existing image detector for initial corners and the demonstrated
source-only cornerSubPix operation with half-windows equal to the floor of
.15, .22 and .30 times the view's median adjacent pitch, clamped to at least
one pixel. Retain the original five-pixel-window result for comparison. Use
40 iterations and .001-pixel convergence epsilon as in the original utility.
These declared parameters define a bounded research workload; they are not
production accuracy tolerances. Do not seed or snap corners from optical-model
predictions. Verify one-to-one identities, consistent cell winding and finite
in-bounds coordinates. Render native-pixel overlays of all initial/refined
corners and verify every corner against the visible checker intersection
before designating a view usable for the frozen test. If window results
disagree or the pixel evidence is ambiguous, withhold that view's score and
retain the reason; do not select a window by its camera reprojection score.

This deliberately small test uses human review of observed pixels to avoid
pretending that multi-window agreement proves accuracy. It does not introduce
an untested saddle detector or relax the ball refiner's qualification gates.
A later automated extractor is a separate implementation decision if the
validated corner evidence demonstrates a repeatable rule.

Before new captures, test the wrapper on synthetic projected intersections,
large initial offsets and neighboring-corner attraction, as well as the exact
saved view-005 failure. Ensure rejected/ambiguous observations remain visible,
all neighborhoods and raw records remain immutable, and the frozen evaluator
cannot invoke intrinsic fitting. Repeat the existing self-tests and source
hash checks. Save every original and revised score in separate artifacts.

## Smallest subsequent physical observation

After that software validation, the next independent check requires only two
native-4K images of the previously verified 10 × 7-inch rigid-backed target,
one toward the left edge and one toward the right, each fully visible with a
clear out-of-plane tilt. Reconfirm that it is that verified single-piece target;
if unavailable, stop and specify a single-sheet substitute before asking for
any picture. No tiled enlargement, ball jig, or ball-position measurement is
required. Keep camera focus, zoom, crop and resolution fixed.

Freeze the old zero model and all new original/relocalized candidate coefficients
before capture. Score each new view with parity-split pose estimation only;
fit no intrinsic coefficients or target geometry to these observations. Report
each corner and candidate, including failures. Consistent advantage on both
independent views and agreement with fixed-table geometry is required before
using an optical candidate for subsequent localization experiments. If the
checks disagree, record the unresolved cause rather than pick the smallest
training score. Two views discriminate candidates; they do not qualify full-table
X/Y or establish independent 2 mm accuracy.

Before either capture, explain the two-view purpose and obtain confirmation
that the table is uncovered. Until then, perform saved-image software work only.

## Preservation and later release

See the large-board dataset/table, large-board optics and center-outline audit
directories for reproducible full results, original and derived data, exact
commands, hashes and limitations. Promote reviewed research to local main in
separate commits; keep unqualified runtime code on its experiment branch.
No deployment, camera-setting change, live qualification or push is part of
this phase. Begin eventual complete-pipeline comparisons at 4K/5 FPS after
spatial feasibility, measuring availability episodes and freshness as well as
throughput. Performance targets remain desired benchmarks.
