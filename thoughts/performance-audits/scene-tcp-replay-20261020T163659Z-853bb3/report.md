# Historical Rust scene TCP replay smoke

This run predates ADR-0025 and used a temporary launch-token handshake that
has since been removed. Keep its measured transport and inference results only
as historical evidence; its authentication counters and token metadata do not
describe the current anonymous, version-only stream.

This run verifies the Rust replay executable's scene-snapshot v1 TCP output against the Python gateway ingress. It is transport interoperability evidence, not a real-time performance result.

- Run: `scene-tcp-replay-20261020T163659Z-853bb3`
- Rust model/provider: ONNX Runtime CPU (`CPUExecutionProvider`)
- Input: 10 recorded still images, paced at 30 fps
- Scene snapshots: Rust reported 10; Python accepted 10
- Ordering/loss: 0 sequence gaps, 0 queue replacements, 0 rejected messages, 0 authentication failures
- Rust connection setup: 0.881 ms; mean scene tracking: 0.209 ms; mean synchronous encode/write: 0.421 ms
- Replay performance: 2.30 processed fps; all 10 synthetic 30 fps deadlines were missed
- One-way latency: unavailable because cross-process monotonic clock epochs have not been qualified

The process used the debug binary and synchronously wrote frames during recorded replay. No camera, live provider lifecycle, Core/API compatibility adapter, WebSocket publication, table calibration, temporal labels, or physical-position ground truth was exercised. The scene sender is not ready to serve the live provider path; a bounded reconnecting writer and application supervision remain required.

Artifacts: [run metadata](run-metadata.json), [Rust report](rust-report.json), [receiver receipts](receiver-report.json), [detection batches](detection-batches.jsonl). The launch token is not retained.
