# Temporal label review plan for `demo2_short.mp4`

This is a frame-selection aid for an initial temporal accuracy check, not a
label file or accuracy result. A reviewer must inspect the source video and
explicitly mark every listed frame reviewed before any labels are scored.

## Bound source

- Video: `test_data/demo2_short.mp4`
- SHA-256: `4e09e30c49fed7e134f43f276c629394c209757e0440958a51ab148f12f6e388`
- Stream: zero-based video stream 0, 1280×720, 1,177 source frames
- Timeline: zero-based source frame index and presentation time normalized to
  the first video frame; these are replay times, not camera exposure times.
- Candidate frames below were chosen from the recorded-video replay output so
  each has a corresponding produced scene snapshot. Scene IDs are one-based;
  CVAT reviewed-frame indices are zero-based.

| Source frame index | Replay time (s) | Scene ID | Review context |
| ---: | ---: | --- | --- |
| 0 | 0.000000 | `frame-000001` | Initial settled table layout |
| 150 | 5.000000 | `frame-000151` | Player approaches and begins aiming |
| 240 | 8.000000 | `frame-000241` | Pre-shot / cue near the central cluster |
| 300 | 10.000000 | `frame-000301` | Shot and early movement |
| 356 | 11.866666 | `frame-000357` | Post-contact ball movement |
| 415 | 13.833333 | `frame-000416` | Later movement and partial player occlusion |
| 488 | 16.266666 | `frame-000489` | Settlement transition |
| 600 | 20.000000 | `frame-000601` | Settled post-shot layout |
| 1140 | 38.033333 | `frame-001141` | Late settled layout |

## Review rules

1. Annotate from the original video only. Do not inspect `scenes.jsonl` or
   `detections.jsonl` until CVAT labels are complete; this keeps the labels
   independent of the model output being evaluated.
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

Nine reviewed frames are a diagnostic sample only. A production recommendation
still requires held-out sequences covering varied motion, collisions,
occlusion/reacquisition, camera interruptions, and calibration changes.
