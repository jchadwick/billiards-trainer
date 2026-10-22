# Rust live camera qualification

Run: `20261021-112757-7070c042`

This is an opt-in local qualification/smoke measurement. It is not a detector-accuracy result or production-readiness decision.
Camera controls use the camera driver's source defaults; this runner does not write exposure, gain, focus, brightness, or other camera controls.
No camera image/frame bytes are recorded. Bounded state-only traces contain detected locations and identity/status metadata.
The source timestamp is labeled read-completion only. Camera exposure time and source-to-state latency are unavailable until source timing is separately qualified.
Ball precision/recall, position error, temporal continuity, cue quality, and freshness accuracy require synchronized ground-truth annotations and remain unavailable here.

| Repetition | States/s | Unique observations/s | Repeated | Source gaps | Out of order | State receipt interval p95 (ms) | Observed balls | Identified balls |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.80 | 19.80 | 0 | 50 | 0 | 53.029 | 574 | 0 |

## Rust pipeline timing (opt-in)

Durations are measured within the Rust process and correlated to received states by engine, session, frame, and transport sequence. They do not establish camera-exposure-to-client latency.

| Repetition | Timing status | Matched frames | Preprocess p95 (ms) | Inference p95 (ms) | Decode/NMS p95 (ms) | Scene socket write p95 (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | complete | 99 | 6.656 | 46.351 | 0.091 | 0.039 |

Capture selector: `gstreamer-device-v1:b8004454b9c7061a1f3958b66edd6725fa7b9f22ca38345087aff9d5d7a957f4`.
Capture profile: `1920x1080@30 NV12` to `1920x1080 RGB8`.
Camera-control policy: `{'mode': 'source_defaults'}`.
State-event exporter: accepted=99, written=99, dropped=0, sampled=0, export errors=0.
Pipeline-timing event exporter: written=284, dropped=0, sampled=0, export errors=0; trace: `thoughts/performance-audits/rust-live-trained-model-export-20261021/camera-qualification-timing-enabled/20261021-112757-7070c042/rust-live-pipeline-timing-events/31a756fa-951f-446d-9de7-0bef293de5aa/events.jsonl`.

State-only event trace: `thoughts/performance-audits/rust-live-trained-model-export-20261021/camera-qualification-timing-enabled/20261021-112757-7070c042/rust-live-state-events/a60b3401-d7ba-4992-bdbf-2bd016184345/events.jsonl`. Frame image persistence: `none; only bounded detected-state metadata is retained`.
