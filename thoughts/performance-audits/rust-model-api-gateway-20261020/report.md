# Real-model Rust API gateway smoke

This run verifies one bounded path: the real Rust ONNX executable processed a
recorded table image, published tracked scene snapshots over the version-only
loopback protocol, and made them available to the existing anonymous HTTP and
WebSocket clients.

The opt-in gateway test passed as part of the focused lifecycle/no-auth suite:
**4 passed in 5.67 seconds**. The real-model case repeated the same still image
eight times at a synthetic 5 Hz so the finite replay continued after the
clients attached. HTTP returned a non-empty ball collection with source
observations. The WebSocket returned state with source observations, and its
first ball position matched HTTP. The API contract test also confirmed the
OpenAPI schema contains no authentication routes or security schemes and CORS
does not allow credentials.

The binary was rebuilt from current Rust source before this run. Before the
rebuild, the ignored local release artifact still expected the rejected
credential handshake. The source and rebuilt executable now use a version-only
hello/ready exchange, and this run completed without an authentication
environment variable. This is why executable freshness is part of this
verification, not merely source inspection.

The model has only generic `cue` and `ball` classes. This still-image smoke has
no temporal annotations or applied table calibration. It proves
inference-to-client wiring only; it does not measure physical detection
accuracy, numbered-ball identity, table-coordinate error, camera-to-client
latency, or live capture behavior. The ONNX file is in `/tmp`, so the optional
test requires callers to provide asset paths and skips when those paths are not
configured.

## Provenance

- Run date: 2026-10-20 UTC.
- Repository HEAD: `e748c0f99e8e8c089633aa282154a05697adb243`; the worktree was dirty.
- Host: macOS arm64; Python 3.11.14; ONNX Runtime 1.31.0.
- Rust executable: `perception/target/release/billiards-perception-infer`,
  SHA-256 `d24ffa1d18e193cd17859772b7a800a29d7399a8d06fa4eb13fa68d21ee7a0c6`.
- ONNX export: `/tmp/billiards-rust-tcp.Ij3QEK/billiards-yolov8n.onnx`,
  SHA-256 `2444f8007934aa6a17584856b3049ebd1c1a6861c174408d7c3d549b41b45648`.
- Source checkpoint: `models/billiards-yolov8n.pt`,
  SHA-256 `f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`.
- Input still: `backend/vision/test_data/full_table.png`,
  SHA-256 `f1e1773a9c46e36d20a4af1cd3793053114ddac70a7e8273f4fc1bf679bd1f1b`.

The opt-in integration test reads four paths from
`BILLIARDS_RUST_REPLAY_AUDIT_EXECUTABLE`, `BILLIARDS_RUST_REPLAY_AUDIT_ONNX`,
`BILLIARDS_RUST_REPLAY_AUDIT_CHECKPOINT`, and
`BILLIARDS_RUST_REPLAY_AUDIT_INPUT`. It does not accept or read credentials.

## Verification

- Focused real-model lifecycle and anonymous API contract suite: 4 passed.
- Full backend suite without coverage: 1,400 passed, 2 skipped.
- Rust release tests: 87 passed; `cargo fmt -- --check` and strict release
  Clippy passed.
- Ruff, Black, and mypy passed on the changed Python files.
- ADR validator passed for all 27 records.
- The coverage-instrumented full suite passed the same 1,400 tests and skipped
  2 model-dependent cases, then exited on the configured threshold: 51.36%
  against 80%. This gateway smoke does not close that shortfall.
