# Rust live camera qualification

Run: `20261021-112641-e4f87a86`

This is an opt-in local qualification/smoke measurement. It is not a detector-accuracy result or production-readiness decision.
Camera controls use the camera driver's source defaults; this runner does not write exposure, gain, focus, brightness, or other camera controls.
No camera image/frame bytes are recorded. Bounded state-only traces contain detected locations and identity/status metadata.
The source timestamp is labeled read-completion only. Camera exposure time and source-to-state latency are unavailable until source timing is separately qualified.
Ball precision/recall, position error, temporal continuity, cue quality, and freshness accuracy require synchronized ground-truth annotations and remain unavailable here.

| Repetition | States/s | Unique observations/s | Repeated | Source gaps | Out of order | State receipt interval p95 (ms) | Observed balls | Identified balls |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.20 | 20.20 | 0 | 49 | 0 | 52.637 | 535 | 0 |

## Rust pipeline timing (opt-in)

Durations are measured within the Rust process and correlated to received states by engine, session, frame, and transport sequence. They do not establish camera-exposure-to-client latency.

| Repetition | Timing status | Matched frames | Preprocess p95 (ms) | Inference p95 (ms) | Decode/NMS p95 (ms) | Scene socket write p95 (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | partial | 0 | unavailable | unavailable | unavailable | unavailable |

Capture selector: `gstreamer-device-v1:b8004454b9c7061a1f3958b66edd6725fa7b9f22ca38345087aff9d5d7a957f4`.
Capture profile: `1920x1080@30 NV12` to `1920x1080 RGB8`.
Camera-control policy: `{'mode': 'source_defaults'}`.
State-event exporter: accepted=101, written=101, dropped=0, sampled=0, export errors=0.
Pipeline-timing event exporter: written=0, dropped=0, sampled=0, export errors=0; trace: `thoughts/performance-audits/rust-live-trained-model-export-20261021/camera-qualification-timing-enabled/20261021-112641-e4f87a86/rust-live-pipeline-timing-events/91af5ebb-0223-4fd1-9986-ea1f6b612c8d/events.jsonl`.

State-only event trace: `thoughts/performance-audits/rust-live-trained-model-export-20261021/camera-qualification-timing-enabled/20261021-112641-e4f87a86/rust-live-state-events/fea238eb-c87b-4d07-ac34-241ae9f1c8a5/events.jsonl`. Frame image persistence: `none; only bounded detected-state metadata is retained`.
