# Native OpenVINO target smoke: findings and limits

This is a short target-machine integration smoke, not a steady-state performance
baseline. It verifies that the native OpenVINO inference executable can run
inside the backend container and that state reaches a headless subscriber
through the frontend WebSocket route.

## Evidence

- The API used the native binary with SHA-256
  `01e22295cd157b572b96b2858e2f7f673e8dd4fe10df36ca1394558d9b9b0f0c`. Its
  one-image native OpenVINO CPU report records 48.5 ms decode, 70.0 ms
  preprocessing, 182.1 ms inference, 0.4 ms decode/NMS, and 301.0 ms total
  image-pipeline time. This is one 3840×2160 JPEG and is only a profiling clue.
- The corrected API receiver captured 17 complete, correlated messages: 12
  fresh ball observations, 4 unique cached/failed source-frame updates, and one
  repeated publication. It counted 16 unique frame updates and 5 source-frame
  gaps. The observed frame IDs span `frame-000010` through `frame-000030`.
- All received state messages carried 13 ball entries, but ball-position status
  was `cached` on five messages. The one-shot replay had already advanced before
  the collector attached; all 17 messages arrived in the first 4.94 seconds of
  a 25.01-second measurement window. The reported 0.48 fresh observations/s
  therefore includes idle time and must not be used as a sustained detector
  rate.
- Source timestamps use a Rust monotonic clock identity that the receiver could
  not verify as comparable. Publication-to-receipt latency, detector-stage
  timing, and accuracy against labels are unavailable. No temporal annotations
  were part of this run.
- The raw generated report predates the corrected freshness and unavailable-Git
  metadata semantics. `recount.json` reprocesses its bounded client event trace
  with the corrected fresh-versus-cached definition. The container has no Git
  checkout, so its revision and dirty-worktree identity are unknown; the empty
  digest in the original report does not indicate a clean worktree.

## Ranked follow-up

1. **Make replay and measurement start together.** The API starts a finite replay
   during backend startup, so a later subscriber misses early frames and then
   spends most of its configured window idle. Add a subscriber-ready start
   barrier or run replay and receiver through the existing synchronized replay
   harness. Validate that each measured repetition covers the full input and
   that the source remains active across the measurement interval.
2. **Separate fresh detection from retained state everywhere.** The collector
   now counts only explicitly observed ball positions in the primary unique
   observation rate. Keep cached, failed, skipped, and unknown-freshness updates
   separate in both reports and comparisons. The target trace had 5 cached
   messages and 5 source-frame gaps; repeat against paced motion footage to
   determine whether those indicate expected scheduler behavior, inference
   misses, or transport loss.
3. **Profile native preprocessing and inference on a representative replay.**
   The one-image sample spends about 39% of measured image-pipeline time in
   decode plus preprocessing and about 60% in inference. Compare those stages
   over repeated full-table runs before investigating copies, input resize, or
   OpenVINO thread settings; keep detector accuracy and ball-position error as
   acceptance gates.
4. **Keep identity and calibration quality as a separate gate.** This smoke
   reports detections as cue/ball classes; it does not verify numbered-ball
   identity, calibrated table coordinates, temporal continuity, or cue quality.
   Add temporal labels and calibration-ground-truth scenes before recommending
   classifier, tracker, or geometry changes.

No architecture or production optimization is recommended from this smoke.
