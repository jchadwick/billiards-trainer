# Reusable frame-engine smoke, corrected timing

This run repeated the ten-image, unpaced recorded-input smoke after correcting
the audit's session-initialization boundary. It validates the shared inference
path and the measurement change; it is not a live-camera performance
qualification.

## Results

- Python PT matched all 73 Rust detections. There were zero unmatched candidate
  or reference detections, and the maximum center difference was 0.032512 px.
- ONNX Runtime reported `CPUExecutionProvider` for all 214 profiled nodes.
- Session initialization measured 11.591 ms for the measured engine and
  16.549 ms for the separately profiled provider probe. Both intervals include
  session creation and model-signature validation. Manifest and model/checkpoint
  hash verification happened once before those intervals.
- Across ten 3840x2160 stills, mean file decode was 40.571 ms, preprocessing
  9.700 ms, model inference 39.988 ms, decode/NMS 0.089 ms, and complete
  file-to-detection work 90.349 ms. These ten samples do not establish tail
  latency or production throughput.

The earlier smoke report's `measurement_session_init_ms` value of 91.625 ms
included provenance hashing because the audit called the public constructor,
which re-read the manifest and hashed the ONNX and checkpoint. That value is
superseded and must not be compared with `provider_probe_session_init_ms`.
The corrected audit reuses the already verified model contract and times only
ORT session creation plus signature validation. The reusable public
constructor still verifies provenance once when an application loads the
model.

## Scope and limits

The run uses a temporary ONNX export from the existing checkpoint and the
retained Python scorer. It processes still images without camera acquisition,
real source timestamps, temporal annotations, or calibrated physical positions.
The generic `cue` and `ball` classes cannot identify numbered balls. File
decode time is excluded from a future live RGB frame's inference path, while
camera conversion, queueing, state publication, and client receipt have not
been measured here. The run therefore provides no camera-to-client freshness
claim or production optimization recommendation.

The worktree was dirty during the run. Model, source-image and runtime
provenance are recorded in [run metadata](run-metadata.json); full output and
per-frame hashes are in [the audit JSON](audit-report.json), and the real
cross-runtime comparison is in [the equivalence report](equivalence/equivalence-20261020T173421.814570Z.md).
