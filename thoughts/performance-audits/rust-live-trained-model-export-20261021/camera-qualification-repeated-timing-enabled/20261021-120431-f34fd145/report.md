# Rust live camera qualification

Run: `20261021-120431-f34fd145`

This is an opt-in local qualification/smoke measurement. It is not a detector-accuracy result or production-readiness decision.
Camera controls use the camera driver's source defaults; this runner does not write exposure, gain, focus, brightness, or other camera controls.
No camera image/frame bytes are recorded. Bounded state-only traces contain detected locations and identity/status metadata.
The source timestamp is labeled read-completion only. Camera exposure time and source-to-state latency are unavailable until source timing is separately qualified.
Ball precision/recall, position error, temporal continuity, cue quality, and freshness accuracy require synchronized ground-truth annotations and remain unavailable here.

| Repetition | Status | States/s | Unique observations/s | Repeated | Source gaps | Out of order | State receipt interval p95 (ms) | Observed balls | Identified balls |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | complete | 19.45 | 19.45 | 0 | 1266 | 0 | 54.297 | 9168 | 0 |
| 2 | complete | 19.31 | 19.31 | 0 | 1283 | 0 | 54.240 | 8906 | 0 |
| 3 | complete | 19.00 | 19.00 | 0 | 1318 | 0 | 56.122 | 9096 | 0 |

## Rust pipeline timing (opt-in)

Durations are measured within the Rust process and correlated to received states by engine, session, frame, and transport sequence. They do not establish camera-exposure-to-client latency.

| Repetition | Timing status | Matched frames | Preprocess p95 (ms) | Inference p95 (ms) | Decode/NMS p95 (ms) | Scene socket write p95 (ms) | Pressure frames | Evicted tracks |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | complete | 2334 | 6.643 | 47.669 | 0.091 | 0.034 | 0 | 0 |
Association-pressure totals for repetition 1: complete; full process including warm-up=0 pressure frames, 0 evicted tracks.
| 2 | complete | 2317 | 6.696 | 47.521 | 0.091 | 0.043 | 0 | 0 |
Association-pressure totals for repetition 2: complete; full process including warm-up=0 pressure frames, 0 evicted tracks.
| 3 | complete | 2280 | 6.799 | 49.137 | 0.091 | 0.043 | 0 | 0 |
Association-pressure totals for repetition 3: complete; full process including warm-up=0 pressure frames, 0 evicted tracks.

Capture selector: `gstreamer-device-v1:b8004454b9c7061a1f3958b66edd6725fa7b9f22ca38345087aff9d5d7a957f4`.
Capture profile: `1920x1080@30 NV12` to `1920x1080 RGB8`.
Camera-control policy: `{'mode': 'source_defaults'}`.
State-event exporter: accepted=6931, written=6931, dropped=0, sampled=0, export errors=0.
Pipeline-timing event exporter: written=17455, dropped=0, sampled=0, export errors=0; trace: `thoughts/performance-audits/rust-live-trained-model-export-20261021/camera-qualification-repeated-timing-enabled/20261021-120431-f34fd145/rust-live-pipeline-timing-events/d3bc523d-9df2-4b67-9a88-f19871827a5e/events.jsonl`.

State-only event trace: `thoughts/performance-audits/rust-live-trained-model-export-20261021/camera-qualification-repeated-timing-enabled/20261021-120431-f34fd145/rust-live-state-events/567471ad-61ea-41a2-9ad1-664acde2c00a/events.jsonl`. Frame image persistence: `none; only bounded detected-state metadata is retained`.
