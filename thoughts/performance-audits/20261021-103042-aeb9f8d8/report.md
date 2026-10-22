# Rust live camera qualification

Run: `20261021-103042-aeb9f8d8`

This is an opt-in local qualification/smoke measurement. It is not a detector-accuracy result or production-readiness decision.
Camera controls use the camera driver's source defaults; this runner does not write exposure, gain, focus, brightness, or other camera controls.
No camera image/frame bytes are recorded. Bounded state-only traces contain detected locations and identity/status metadata.
The source timestamp is labeled read-completion only. Camera exposure time and source-to-state latency are unavailable until source timing is separately qualified.
Ball precision/recall, position error, temporal continuity, cue quality, and freshness accuracy require synchronized ground-truth annotations and remain unavailable here.

| Repetition | States/s | Unique observations/s | Repeated | Source gaps | Out of order | State receipt interval p95 (ms) | Observed balls | Identified balls |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29.99 | 29.99 | 0 | 0 | 0 | 37.228 | 150 | 0 |

Capture selector: `gstreamer-device-v1:b8004454b9c7061a1f3958b66edd6725fa7b9f22ca38345087aff9d5d7a957f4`.
Capture profile: `1920x1080@30 NV12` to `1920x1080 RGB8`.
Camera-control policy: `{'mode': 'source_defaults'}`.
State-event exporter: accepted=150, written=150, dropped=0, sampled=0, export errors=0.

State-only event trace: `/Users/jchadwick/repos/jchadwick/billiards-trainer/thoughts/performance-audits/20261021-103042-aeb9f8d8/rust-live-state-events/d09c9dac-527f-4cd4-8d9c-6ecc38e3c03c/events.jsonl`. Frame image persistence: `none; only bounded detected-state metadata is retained`.
