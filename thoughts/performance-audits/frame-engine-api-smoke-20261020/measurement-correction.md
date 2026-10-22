# Correction to the first frame-engine smoke

The first frame-engine smoke's `measurement_session_init_ms` value in
`audit-report.json` is not comparable with `provider_probe_session_init_ms`.
It included manifest parsing and repeated ONNX/checkpoint hashing because the
new frame-engine constructor performed provenance verification during the
measured interval. The provider probe only measured ONNX Runtime session
creation and signature validation.

The timing boundary is now corrected: `run_audit` reuses its verified model
contract, and its measurement-session timer covers only ONNX Runtime session
creation and signature validation. The repeated run measured 11.591 ms versus
16.549 ms for the provider probe. See the [corrected smoke report](../frame-engine-api-smoke-20261020-r2/report.md)
and [machine-readable result](../frame-engine-api-smoke-20261020-r2/audit-report.json).

The first run's per-image decode, preprocessing, inference, and postprocessing
measurements do not include that session-initialization interval and remain
unchanged. The first run is retained as historical output; only its
session-initialization value is superseded.
