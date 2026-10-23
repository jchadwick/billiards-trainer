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
- Candidate frames below were chosen from the recorded-video replay output so
  each has a corresponding produced scene snapshot. Scene IDs are one-based;
  CVAT reviewed-frame indices are zero-based.

## Candidate source frames

The original nine-frame proposal was selected because the replay had emitted a
scene at each index. That conditions the sample on successful detector output
and excludes dropped source frames, so those nine indices must not be used as
the accuracy sample.

The replacement list is frozen independently of detector output in
[`candidate-source-frame-indices.json`](candidate-source-frame-indices.json).
It has 66 zero-based source indices: a 2-second systematic sample across the
whole clip, including its last frame, plus a 5 Hz sample from source frames 240
through 540 (about 8–18 seconds) to cover the visible aiming, shot, movement,
occlusion, and settlement sequence. The dense interval was chosen by inspecting
a source-video contact sheet only; it was not filtered against replay scene or
detection records. The importer JSON is still only a candidate list: add an
index to the final reviewed-frame file only after a person checks the complete
source frame and its tracks.

## Review rules

1. Annotate from the original video only. Do not inspect `scenes.jsonl` or
   `detections.jsonl` until the CVAT labels and reviewed-frame list are
   complete; this keeps the labels and sample selection independent of model
   output. Include candidate indices even when the replay did not produce a
   scene there; missing source-frame output is part of the freshness result.
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

After the review, convert the CVAT Video 1.1 export with
`scripts/import-cvat-temporal-annotations.sh`, passing the exact video hash,
the CVAT XML, and the final reviewed-frame JSON array. Then score the imported
labels against the existing replay snapshots with
`scripts/evaluate-temporal-localization.sh`. The current replay report remains
explicitly accuracy-unavailable until that human review and scoring are done.

The 66-frame list is a diagnostic sample from one video only. It cannot support
a production recommendation without held-out sequences covering varied motion,
collisions, occlusion/reacquisition, camera interruptions, and calibration
changes.
