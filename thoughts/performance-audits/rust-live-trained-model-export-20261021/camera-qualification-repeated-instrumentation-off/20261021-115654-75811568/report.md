# Rust live camera qualification

Run: `20261021-115654-75811568`

This is an opt-in local qualification/smoke measurement. It is not a detector-accuracy result or production-readiness decision.
Camera controls use the camera driver's source defaults; this runner does not write exposure, gain, focus, brightness, or other camera controls.
No camera image/frame bytes are recorded. Bounded state-only traces contain detected locations and identity/status metadata.
The source timestamp is labeled read-completion only. Camera exposure time and source-to-state latency are unavailable until source timing is separately qualified.
Ball precision/recall, position error, temporal continuity, cue quality, and freshness accuracy require synchronized ground-truth annotations and remain unavailable here.

| Repetition | Status | States/s | Unique observations/s | Repeated | Source gaps | Out of order | State receipt interval p95 (ms) | Observed balls | Identified balls |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | complete | 19.40 | 19.40 | 0 | 1272 | 0 | 54.614 | 11435 | 0 |
| 2 | complete | 19.18 | 19.18 | 0 | 1298 | 0 | 55.586 | 10912 | 0 |
| 3 | complete | 19.66 | 19.66 | 0 | 1239 | 0 | 52.912 | 9411 | 0 |

Capture selector: `gstreamer-device-v1:b8004454b9c7061a1f3958b66edd6725fa7b9f22ca38345087aff9d5d7a957f4`.
Capture profile: `1920x1080@30 NV12` to `1920x1080 RGB8`.
Camera-control policy: `{'mode': 'source_defaults'}`.
State-event exporter: accepted=6989, written=6989, dropped=0, sampled=0, export errors=0.
Pipeline-timing event exporter: disabled.

State-only event trace: `thoughts/performance-audits/rust-live-trained-model-export-20261021/camera-qualification-repeated-instrumentation-off/20261021-115654-75811568/rust-live-state-events/6f7eae13-e9d6-4c37-8f64-c6df84cde18d/events.jsonl`. Frame image persistence: `none; only bounded detected-state metadata is retained`.
