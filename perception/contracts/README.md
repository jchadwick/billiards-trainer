# Runtime contracts

The files in this directory define the JSON contracts used by the Rust server,
live detector, replay/evaluation tools, and browser client. Contract schemas and
fixtures are covered by Rust tests; the web client tests validate its decoding
and rendering behavior.

| Area | Contract | Runtime use |
| --- | --- | --- |
| Configuration | `configuration/v1/` | Server defaults, validation, and persistence |
| Capture request | `capture-settings/v1/`, `v2/`, `v3/` | Source configuration; the live application emits v3 |
| Capture acknowledgement | `capture-session-ack/v1/`, `v2/` | Source-readiness handshake; live capture uses v2 |
| Scene snapshots | `scene-snapshot.schema.json` | Detector-to-gateway state, observation identity, and source timing |
| Detection batches | `detection-batch/v1/` | Per-frame inference output used by replay/evaluation tools |
| Public state | `public-state/` | HTTP/WebSocket response consumed by the browser client |
| Calibration | `calibration/v1/` | Image-to-table-plane transformation and provenance |
| Temporal labels | `temporal-localization-annotations-v1.schema.json` and v2 | Reviewed video labels for localization evaluation |
| Model export | `model-export/v1/` | ONNX lineage checked by the Rust model loader |

Each versioned directory contains the schema, fixtures, and any focused
contract notes. Existing versions remain available wherever current Rust
parsers or tests exercise them; a new version is required when a wire shape or
its meaning changes incompatibly.
