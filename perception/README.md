# Rust runtime and perception

This crate contains the complete supported server-side application runtime.
`billiards-server` serves the browser client and owns live-process lifecycle;
`billiards-perception-live` captures frames, runs the configured ONNX detector,
tracks balls, and sends bounded scene updates to the gateway. The GStreamer
helper owns platform capture interaction when that source is selected.

The current detector reports generic ball locations and cue status. It does
not identify numbered balls or produce trajectory predictions. State delivery
is freshness-first: capture and publication buffers are bounded, pending frames
can be replaced by newer frames, and clients may miss intermediate states.

Important implementation areas:

- `capture*` and `gstreamer_capture_source.rs`: source selection, ownership,
  frame limits, and latest-frame handoff.
- `inference/`: verified model loading, preprocessing, inference, decoding, and
  recorded-video replay.
- `tracking.rs`, `public_state.rs`, and `server.rs`: observation identity,
  public-state conversion, and best-effort delivery.
- `configuration_authority.rs` and `contracts/`: validated runtime settings and
  versioned wire contracts.
- `live_audit*`, `performance_audit*`, and `temporal_evaluation.rs`: bounded
  timing, delivery, replay, and reviewed-label evaluation tools.

Build and verify through the root Makefile. The current workflows are described
in the [architecture](../docs/ARCHITECTURE.md) and
[performance audit](../docs/PERFORMANCE_AUDIT.md) guides. Rust unit, contract,
fixture, camera-feature, and browser tests are the supported regression suite.
