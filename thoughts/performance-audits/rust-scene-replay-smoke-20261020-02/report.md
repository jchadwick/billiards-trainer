# Rust detector-to-scene replay smoke

This one-run check exercises the real ONNX detector, source-time tracker,
scene-snapshot v1 serialization, and the retained Python equivalence scorer.
The input is ten varied still images presented on a synthetic 30 fps schedule;
it is not a temporal video or live camera measurement.

The completed run processed 9 of 10 scheduled images. The latest-work mailbox
replaced one pending input, and the processed source IDs expose that gap. All
nine detection batches and scene snapshots passed their shared JSON Schemas and
matched by engine, session, and frame identity. The nine scenes carried 284
ball measurements. Calibration was unavailable, so every table-space position
remained null.

The retained Python scorer matched all 57 detections in the processed frames,
with no unmatched detections and a maximum center difference of 0.032512 px.
This verifies Rust/Python model-output parity for this subset; it does not
measure whether those boxes are physically accurate.

Mean image decode took 40.64 ms, preprocessing 9.71 ms, inference 42.26 ms,
and decode/NMS 0.089 ms. Mean end-to-end image processing was 92.70 ms. Scene
tracking averaged 0.0730 ms and scene JSONL serialization/write averaged
0.0614 ms per processed image. Detection-batch export totaled 0.436 ms
(0.0484 ms per processed image). These are single-run observations with
synchronous file sinks; they do not meet the repeat count or duration of the
full performance-audit plan.

The scheduled source rate was 30 fps, but every one of the ten synthetic source
deadlines was missed. File readiness averaged 24.34 fps, processed throughput
was 17.58 fps, and source-arrival-to-inference-complete latency was
75.93/100.39/100.39 ms at p50/p95/p99 over nine samples. The dominant measured
costs were file decoding and inference, not scene association or scene output.
This does not establish camera-to-client freshness.

An initial run capped tracks at 24 and stopped with `ActiveTrackCapacityExceeded`.
The completed run used an explicit bounded capacity of 300. The image corpus
contains different table states rather than consecutive observations, so this
failure and the successful run say nothing about an appropriate live track cap
or association quality.

The record contains no table-plane calibration for this camera. The replay did
not test millimeter accuracy, temporal continuity, occlusion/reacquisition,
numbered-ball identity, TCP receipt, browser rendering, or platform parity on
Linux. Scene JSONL is flushed synchronously and can affect the measured loop.
The next useful evidence is a temporally labeled clip with a real applied table
calibration, followed by a receiver-backed run using the planned warm-up and
repetition durations.

Artifacts:

- [Machine-readable runtime report](report.json)
- [Run provenance and input hashes](run-metadata.json)
- [Detection batches](detection-batches.jsonl)
- [Tracked scene snapshots](scene-snapshots.jsonl)
- [Cross-runtime equivalence report](equivalence/equivalence-20261020T162306.322661Z.md)
- [Cross-runtime equivalence data](equivalence/equivalence-20261020T162306.322661Z.json)
