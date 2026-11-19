# Ball identification experiments

## Decision

None of the three candidates meets the acceptance rule. Keep ball identity
unknown in production until a candidate demonstrates useful coverage with no
accepted-label errors on independent camera sessions. Detection and its public
API remain unchanged.

The manual labels below were confirmed by the user before any candidate was
trained or scored. The frozen corpus includes both full 16-ball video frames,
both grids with ball 10 absent, and the mostly white 14 ball in checkerboard
view 006.

## Frozen corpus

[`corpus-v1.json`](corpus-v1.json) contains 23 source scenes: 21 stills and two
frames from `test_data/demo3.mp4`. It records each source hash, the two exact
video frame indices and timestamps, the 161 manually marked ball regions and
identities, empty scenes, capture groups, and confirmation provenance. Eight
scenes have no visible balls. Source images and video remain at their existing
paths; they are not duplicated here.

The video frames are frame 0 at 0.0 seconds and frame 900 at 30.0 seconds. Both
come from one recording and one camera session. They are a useful held-out
appearance shift from the still images, but they are not independent validation
sessions and are too far apart to test stationary evidence accumulation.

Hashes for the frozen inputs and evaluator:

| Artifact | SHA-256 |
| --- | --- |
| `corpus-v1.json` | `71b9cc6b65290b915a2712775cb7879d2f6c3cb13a8787ad728de4eb43608573` |
| `detector-proposals.json` | `390efbac9a85b1d882727e345723eb8df87d6f272c5342aaef36ddab51eb9386` |
| `evaluate_candidates.py` | `612b576e898febf992889307546700024c3597485af522b2b7443af7ee034677` |
| `demo3.mp4` | `0329ca2aa917f968bf4f0f55c51fd5af67124c90ec1dc8eede7674c013c8a555` |

Detector proposals were generated once and reused for every approach. Across
the complete corpus, the detector matched 154 of 161 manual balls, missed 7,
and produced 45 unmatched proposals. On the two video frames it matched 29 of
32, missed 3, and produced 2 unmatched proposals. Missed detections are counted
separately from identity abstentions and identity errors.

All three approaches trained only on the 129 still-image crops. They were
scored first on the 32 manually marked video crops and separately on the 29
detector crops matched to those labels. The same manual positions, proposal
matches, and acceptance metrics were used for all candidates. The 45 unmatched
proposal crops were also classified to expose false identity assignments.

## Approaches and results

Coverage is the share of evaluated balls assigned a numbered identity or
`cue`; accepted-label errors are counted only among those assignments. `Unknown`
is an abstention. The detector column uses matched detector crops, so a
detection miss cannot lower its identification accuracy.

| Candidate | Manual video crops | Matched detector crops | Unmatched proposals assigned an identity | CPU inference time per crop* | Peak experiment RSS* |
| --- | --- | --- | --- | ---: | ---: |
| Learned Lab color + spatial pattern, conservative product score | 3/32 accepted, 0 wrong (9.4%) | 3/29, 0 wrong (10.3%) | 7/45 | 1.42 ms | 628 MB |
| Same model, minimum component score | 6/32, 1 wrong: 2→4 (18.8%) | 6/29, 1 wrong (20.7%) | 7/45 | 1.40 ms | 662 MB |
| MobileNetV3 Small, separate color/pattern heads | 1/32, 1 wrong: 1→9 (3.1%) | 0/29 (0%) | 0/45 | 5.68 ms | 1,491 MB |
| Learned appearance + number-template corroboration + conservative fusion | 1/32, 0 wrong (3.1%) | 1/29, 0 wrong (3.4%) | 0/45 | 1.10 ms | 860 MB |

The first candidate is the only one with more than one accepted holdout label
and zero observed errors, but it identifies only three of 32 manual crops and
four of 29 detector crops. That is too little coverage to count as a useful
result. The more permissive feature score accepts more balls but makes a
visible 2→4 error. The network makes a confident 1→9 error after its still-only
calibration set achieved 21/34 correct accepted labels with no observed
calibration errors. The corrected fusion candidate abstains on nearly every
video ball; an earlier confidence rule accepted 14 wrong labels among 20
assignments before its calibration bug was fixed.

The fusion helper rejects repeated near-identical views in a synthetic check
and never multiplies confidence by the number of repeated observations. There
is no reviewed within-five-second sequence for a stationary physical ball in
this corpus, so temporal settling delay and stale-result behavior were not
measured.

The feature model's grouped still-only calibration selected a cutoff with zero
observed OOF errors at 50/129 accepted crops (product) or 51/129 (minimum
component score). Its three non-empty capture groups were held out as groups;
no fold was skipped for missing color or pattern classes. MobileNet used 95
still crops for training and one moved-checkerboard group of 34 crops for
calibration. That calibration split accepted 21/34 with zero errors, but did
not transfer to the video frames. Fusion's grouped calibration found only
three eligible identities; it could not calibrate support for blue, green,
maroon, white, or cue appearance. Training-only grouped validation also showed
appearance confusion: green 6 versus black 8 for the feature model,
color/pattern ambiguity in cue crops, and substantial error among the fusion
model's accepted detector-region crops.
Across the 45 unmatched proposal crops from all scenes, the feature candidate
assigned identities to 7 under either confidence score; the network and
fusion candidate assigned none. These are diagnostic results, not a reason to
trust a candidate that has almost no held-out coverage.

The experiments measured crop-classifier time and whole-process peak memory on
the development Mac. Those measurements include Python, image libraries,
training data, and evaluation crops. They are not model-only estimates and do
not predict the target computer's runtime. No target benchmark was run because
no candidate qualified for deployment; the running application was not
interrupted.

## Recommendation

Do not integrate any identity output yet. The main failure is transfer from
the reviewed still-image setups to video appearance, combined with too little
evidence to calibrate useful abstention. More threshold tuning against these
two video frames would leak the evaluation labels and would not solve the
session-level gap.

For the next experiment, collect reviewed crops from the intended live camera
and lighting across several independent sessions. Include cue balls, every
numbered ball (especially 10), white-facing stripes, changing ball rotation,
touching balls, pocket shadows, partial occlusion, bad detector crops, and
empty scenes. Hold out whole sessions. Add short stationary sequences so the
fusion candidate can be measured for settling time, duplicate-view handling,
and stale-result rejection. Revisit the 100 ms worker budget and target
CPU/memory only after one candidate achieves useful zero-error coverage on
those held-out sessions.

## Reproduction and artifacts

Run the shared evaluator from any of the three attached experiment worktrees,
pointing `--root` to the main checkout so it can read ignored source images and
video. For example:

```bash
python thoughts/ball-identification/evaluate_candidates.py \
  --root /Users/jchadwick/repos/jchadwick/billiards-trainer \
  --approach thoughts/ball-identification/approach_features.py \
  --output thoughts/ball-identification/results.json
```

The three isolated implementations and their self-checks remain in attached
worktrees:

- Feature model: `/Users/jchadwick/.codex/worktrees/ball-id-features/billiards-trainer`
- MobileNetV3: `/Users/jchadwick/.codex/worktrees/ball-id-mobilenet/billiards-trainer`
- Fusion: `/Users/jchadwick/.codex/worktrees/ball-id-fusion/billiards-trainer`

Machine-readable reports and the exact settings used for the main comparisons
are in [`results/`](results/). The initial fusion failure report is retained
there as a diagnostic. None of the prototype code is connected to the live
detection path.

Research that informed the experiments:

- [Pool Aid: Computer Vision for Billiards](https://kastner.ucsd.edu/wp-content/uploads/sites/5/2014/03/admin/pool-aid.pdf) describes overlapping appearance and number visibility challenges in pool-ball vision.
- [TorchVision MobileNetV3 Small](https://docs.pytorch.org/vision/2.0/models/mobilenetv3.html) and the [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial) support the compact learned-model candidate.
- [Selective Classification for Deep Neural Networks: A Survey](https://arxiv.org/abs/2401.12708) motivates reporting error against coverage instead of accuracy alone.

## Retired implementation cleanup

The failed OpenCV ball-color rules, brightness thresholds, and classification
switch have been removed from the live config, versioned defaults, configuration
schema, and saved config snapshots. The old ignored Python bytecode cache was
deleted. No runtime code or fallback invokes the old classifier. Table-cloth
color filtering remains because it serves localization and is not a ball
identity rule.
