# Temporal label review plan for `demo2_short.mp4`

This is a source-frame selection aid for an initial temporal accuracy check,
not a label file or accuracy result. A reviewer must inspect the source video
and explicitly mark each reviewed frame before any labels are scored.

## Bound source

- Video: `test_data/demo2_short.mp4`
- SHA-256: `4e09e30c49fed7e134f43f276c629394c209757e0440958a51ab148f12f6e388`
- Stream: zero-based video stream 0, 1280×720, 1,177 source frames
- Timeline: zero-based source frame index and presentation time normalized to
  the first video frame; these are replay times, not camera exposure times.
- Candidate frame indices are zero-based and selected without consulting
  detector output. CVAT frame indices use the same zero-based source timeline.

## Candidate source frames

The original nine-frame proposal was selected because the replay had emitted a
scene at each index. That conditions the sample on successful detector output
and excludes dropped source frames, so those nine indices must not be used as
the accuracy sample.

The replacement is two frozen source-only cohorts:

1. [`candidate-clipwide-source-frame-indices.json`](candidate-clipwide-source-frame-indices.json)
   contains one reproducible pseudo-random source frame from each of 40 equal
   contiguous strata. For stratum `k` from 0 through 39, let
   `start=floor(k*1177/40)` and `end=floor((k+1)*1177/40)`. Select
   `start + (H mod (end-start))`, where `H` is the first eight bytes of
   `SHA256("<video-sha>:uniform-stratum:<k>")` interpreted as an unsigned
   big-endian integer. The source-video hash above seeds this selection.
2. [`candidate-motion-burst-source-frame-indices.json`](candidate-motion-burst-source-frame-indices.json)
   contains every source frame from 270 through 306 inclusive. Source-only
   inspection shows the table still in the aiming phase near frame 270, rapid
   ball displacement around frame 282, and a settled post-shot layout around
   frame 300.

Together there are 75 unique indices; frames 281 and 305 appear in both
cohorts. Keep the clip-wide sample and motion burst as separate accuracy
reports so the intentionally oversampled event does not distort the clip-wide
result. Neither JSON file asserts that its frames have been reviewed: add an
index to a final reviewed-frame file only after a person checks the complete
source frame and its tracks.

## Review rules

1. Annotate from the original video only. Do not inspect `scenes.jsonl` or
   `detections.jsonl` until both CVAT labels and final reviewed-frame lists are
   complete; this keeps labels and sample selection independent of model
   output. Include a candidate even when replay produced no scene there;
   missing source-frame output is part of the freshness result.
2. In CVAT, use the `ball` and `cue` labels and stable track IDs. At each
   candidate frame, add an explicit keyframe for every active track. Mark
   visible, occluded, and removed balls deliberately; do not let CVAT
   interpolation stand in for a reviewed observation.
3. Add a frame index to the importer's reviewed-frame JSON list only after a
   reviewer has checked the entire frame, including table edges, pockets,
   cue visibility, partial occlusion, and any balls that are no longer on the
   table. The current list is only a set of candidates.
4. This first pass is version-1 pixel-space evidence. It can measure image
   localization, recall/precision, continuity, cue presence, and replay
   freshness. It cannot establish physical table-coordinate error in
   millimeters without independently measured table geometry and ball
   positions. Ball-number identity is outside this model's two-class
   `cue`/`ball` output.

After the review, import the same CVAT Video 1.1 export separately for each
cohort with `scripts/import-cvat-temporal-annotations.sh`, passing the exact
video hash, the CVAT XML, and that cohort's final reviewed-frame JSON array.
Score each imported annotation file against the existing replay snapshots with
`scripts/evaluate-temporal-localization.sh`, keeping the clip-wide result and
motion-burst stress result separate. The current replay report remains
explicitly accuracy-unavailable until that human review and scoring are done.

The 40-frame clip-wide sample and 37-frame event burst are diagnostic evidence
from one video only. They cannot support a production recommendation without
held-out sequences covering varied motion, collisions,
occlusion/reacquisition, camera interruptions, and calibration changes.
