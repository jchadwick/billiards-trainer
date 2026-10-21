# Live replay: reporting, tracks, and cue detection

## Findings

- The live state payload reports `source_timestamp_kind=shared_memory_publication`. Sampled frame publication age was about 60–102 ms, with 8 observed tracks in the inspected state. This is evidence against a large delay from frame publication to the state message, but it does not measure MJPEG-to-WebSocket synchronization or total detector/render latency.
- The tracker can publish extrapolated positions. Live View now separates observed balls from predicted tracks, so a predicted position is visible as a tracker estimate instead of looking like a fresh detector result. The inspected replay also showed fewer tracks than visible balls in some frames; tracking prediction does not explain every miss.
- “Cue detection” means detecting the cue stick. The configured two-class YOLO model recognizes generic `ball` and `cue` objects; it does not classify ball numbers or cue-ball identity. The experimental OpenCV number heuristic is disabled by default, and the sampled WebSocket payload reported `number: null` for all balls.
- Cue detection has genuine false negatives in the replay, including visible-stick frames where the model returns no cue. Separately, cue angle estimation rejected near-horizontal Hough segments as table rails and selected a short hand/finger edge. The estimator now clusters aligned segments and extends the fitted shaft axis to the model bounding box. On replay frame 900, the estimated direction changed from a 43 px finger edge at 25.35° to the shaft axis across the cue box at 5.30°; the rendered line was visually checked against that frame.
- Cue freshness handling also clears cached geometry when a scheduled cue inference misses or exceeds the configured freshness window. The active prerecorded-video configuration has phase scheduling disabled, so that specific stale-cache path was not exercised in this live run.

## Runtime verification

The API was restarted to load the cue-axis change while the prerecorded video publisher and frontend stayed running. The live page reconnected, displayed the selected Practice mode, showed 8 observed balls, reported cue-stick detections when available, and rendered the cue line. The sampled state carried a source frame number and a 75 ms publication age; a later sample had no cue, consistent with the detector's intermittent recall.

No test suite was run. Verification used the running prerecorded-video system and a manual detector comparison on the recorded source frame.
