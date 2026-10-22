# Rust live camera qualification

Run: `20261021-115242-99f7149f`

This is an opt-in local qualification/smoke measurement. It is not a detector-accuracy result or production-readiness decision.
Camera controls use the camera driver's source defaults; this runner does not write exposure, gain, focus, brightness, or other camera controls.
No camera image/frame bytes are recorded. Bounded state-only traces contain detected locations and identity/status metadata.
The source timestamp is labeled read-completion only. Camera exposure time and source-to-state latency are unavailable until source timing is separately qualified.
Ball precision/recall, position error, temporal continuity, cue quality, and freshness accuracy require synchronized ground-truth annotations and remain unavailable here.

| Repetition | Status | States/s | Unique observations/s | Repeated | Source gaps | Out of order | State receipt interval p95 (ms) | Observed balls | Identified balls |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | complete | 19.61 | 19.61 | 0 | 1246 | 0 | 53.169 | 11717 | 0 |

## Rust pipeline timing (opt-in)

Durations are measured within the Rust process and correlated to received states by engine, session, frame, and transport sequence. They do not establish camera-exposure-to-client latency.

| Repetition | Timing status | Matched frames | Preprocess p95 (ms) | Inference p95 (ms) | Decode/NMS p95 (ms) | Scene socket write p95 (ms) | Pressure frames | Evicted tracks |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | complete | 2353 | 6.696 | 46.471 | 0.092 | 0.039 | 0 | 0 |
Association-pressure totals for repetition 1: complete; full process including warm-up=0 pressure frames, 0 evicted tracks.

Capture selector: `gstreamer-device-v1:b8004454b9c7061a1f3958b66edd6725fa7b9f22ca38345087aff9d5d7a957f4`.
Capture profile: `1920x1080@30 NV12` to `1920x1080 RGB8`.
Camera-control policy: `{'mode': 'source_defaults'}`.
State-event exporter: accepted=2353, written=2353, dropped=0, sampled=0, export errors=0.
Pipeline-timing event exporter: written=5880, dropped=0, sampled=0, export errors=0; trace: `thoughts/performance-audits/rust-live-trained-model-export-20261021/camera-qualification-timing-enabled/20261021-115242-99f7149f/rust-live-pipeline-timing-events/612f4fed-2e73-4118-bda5-b7be8390a2e7/events.jsonl`.

State-only event trace: `thoughts/performance-audits/rust-live-trained-model-export-20261021/camera-qualification-timing-enabled/20261021-115242-99f7149f/rust-live-state-events/64b3054d-a15a-4aec-98e3-ff9780698cb3/events.jsonl`. Frame image persistence: `none; only bounded detected-state metadata is retained`.
