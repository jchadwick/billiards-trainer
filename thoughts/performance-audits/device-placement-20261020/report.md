# PyTorch device-placement correction

This validation records the first reference-path repair for the Rust migration.
The regular PyTorch inference call previously omitted the configured device even
though the detector logged it as the active device. Both instrumented and normal
calls now pass the configured device to Ultralytics for `.pt` weights. Model
reload warm-up uses the same call path. Model diagnostics preserve the existing
`device` field as the request and expose `requested_device` and an observed
`effective_device` separately.

The real `billiards-yolov8n.pt` model (SHA-256
`f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`) was run
against `multiple_balls.png` (SHA-256
`e2cfcaef233cc819e1e679e28aafcbf46b88ce68da5c324e8d4b5ab6610151bd`) on an
Apple Silicon macOS host. The predictor reported `cpu` for the CPU run and
`mps` for the MPS run. Each produced five detections for five labels. The two
profiles had corresponding predicted centers within 0.000061 pixels. Both had
mean center error 2.453 pixels and maximum center error 4.475 pixels under a
one-to-one nearest-center match with a 160-pixel gate. Detailed machine and
input provenance is in
[`result.json`](result.json).

The recorded single-image elapsed times (767.94 ms CPU and 490.31 ms MPS) are
first-prediction measurements collected in sequence, so they are not a fair
performance comparison and are excluded from all throughput or latency claims.
This check demonstrates actual PyTorch predictor placement and a static-image
sanity result only. It does not establish temporal accuracy, live-camera
freshness, sustained performance, or Linux behavior.

The backend regression tests cover CPU, MPS and CUDA request propagation through
both profiled and unprofiled paths, unavailable runtime evidence, and the fact
that the PyTorch device option is not sent to the ONNX path. The focused suite
passed (14 tests including the existing inference-cache tests); Ruff, Black,
isort and mypy passed for the changed backend files. The full backend suite
reported 1,153 passed and 1 skipped. The configured total-coverage gate still
fails at 47.23% versus 80%, an existing repository-wide gap that remains open.

The currently constructed ONNX/CoreML session is still not used by the normal
inference path, native `.mlpackage` placement is not reported here, and detector
errors can still be represented as an empty detection list. These are not
validated by this repair. The next backend contract work must distinguish failed
observations from valid empty tables before the corrected Python path can serve
as a complete reference.
