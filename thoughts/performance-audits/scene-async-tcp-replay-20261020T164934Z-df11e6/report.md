# Bounded asynchronous scene TCP replay smoke

Historical run from before ADR-0025: this replay used the temporary launch-token handshake, which has been removed. The recorded measurements remain transport and newest-only scheduling evidence with the real CPU ONNX detector; token-related counters describe only that old protocol. It is not a live-camera performance or accuracy qualification.

- Run: `scene-async-tcp-replay-20261020T164934Z-df11e6`
- Revision: `e748c0f99e8e8c089633aa282154a05697adb243` (dirty-worktree identity recorded in run metadata)
- Input: 10 recorded still images, paced at 30 fps
- Scene snapshots: Rust queued 10; it completed 10 local socket writes; Python accepted 10
- Transport loss: 0 Rust pending replacements, 0 write failures; Python queue replacements 9, gaps 0, rejects 0, authentication failures 0
- Processing: 2.29 fps; 10 of 10 30 fps deadlines missed
- Mean stage time: decode 363.8 ms, preprocess 358.3 ms, inference 40.8 ms, scene tracking 0.222 ms, scene encode/queue 0.447 ms
- Observation freshness proxy: source-to-inference-complete p50/p95/p99 573.7/730.1/730.1 ms; these are synthetic replay timestamps, not exposure-to-state latency
- One-way transport latency: unavailable because Rust and Python monotonic clock epochs are unqualified

The receiver accepted every generated snapshot; its 9 queue replacements show that no consumer drained the one-slot queue during this smoke. The short run identifies decode and preprocessing as larger measured stage costs than scene encode/queue, but it does not establish a controlled transport delta or justify an optimization. There was no physical calibration, temporal label set, Core/API mapping, HTTP/WebSocket publication, or physical-position ground truth.

Artifacts: [run metadata](run-metadata.json), [Rust report](rust-report.json), [receiver report](receiver-report.json), [raw detection batches](detection-batches.jsonl). The launch token is not retained.
