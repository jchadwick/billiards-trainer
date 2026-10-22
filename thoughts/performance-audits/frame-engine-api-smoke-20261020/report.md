# Reusable frame-engine smoke

This was a code-path smoke for the new reusable `FrameInferenceEngine`, not a
performance audit or production qualification. It ran the existing real ONNX
export over the ten recorded 4K stills using unpaced replay and the retained
Python PT scorer.

## Result

- The shared Python scorer matched all 73 detections, with zero unmatched
  outputs and a maximum center difference of 0.032512 px.
- The observed ONNX Runtime provider was `CPUExecutionProvider` for all 214
  profiled nodes.
- Mean times over ten images were 39.650 ms file decode, 9.567 ms
  preprocessing, 41.014 ms model inference, 0.088 ms decode/NMS, and 90.320 ms
  complete image pipeline. Ten samples are too few for meaningful tail latency.
- The run requested no camera, delivered no real-time source timestamps, and
  measured no camera-to-client freshness. It does not establish detector
  accuracy: the input labels are sparse still-image annotations and the model
  still has only generic `cue` and `ball` classes.

## Reproduction details

- Source revision: `e748c0f99e8e8c089633aa282154a05697adb243`; the worktree was
  dirty during the run.
- Host: Apple M4 Max, macOS 26.6.2, arm64.
- Runtime: Rust 1.99.1; ONNX Runtime binding `ort` 2.0.0-rc.15; ONNX Runtime
  CPU provider.
- Input: `backend/vision/test_data`, ten 3840x2160 PNG/JPEG stills. Their
  encoded-image hashes are retained in `audit-report.json`.
- Model: `models/billiards-yolov8n.pt`, SHA-256
  `f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`; a
  temporary ONNX export, SHA-256
  `e4db94c3f1ecd9445134485e65b7536daf67904db6b87442a25641daa4dd161c`.
- Settings: fixed 1280-square input, confidence 0.40, IoU 0.45, maximum 300
  detections, CPU provider, unpaced processing.

The run artifacts are `audit-report.json`, `detection-batches.jsonl`, the
export manifest, and the Python equivalence JSON/Markdown. The ONNX binary is
not checked in; the retained exporter recreates it from the source checkpoint.

## Verification

The Rust replay now calls the same frame-level preprocessing, ONNX session, and
decode path exposed by `FrameInferenceEngine::infer_rgb`. `cargo test` passed
42 library tests, 7 binary tests, and 50 integration/contract tests. Strict
Clippy and formatting checks passed. The synthetic ONNX fixture test verifies
manifest hashes, input/output signatures, inference, and decoded locations;
the real-model replay and cross-runtime match above exercise the shared path
against the current export.

This smoke establishes that file replay and the reusable in-memory engine share
one detector path. It does not start a camera or connect the Rust provider to
FastAPI. Camera ownership, hardware timestamps, bounded live-frame scheduling,
and Mac/Linux device qualification remain open.

The original `measurement_session_init_ms` field was later found to include
provenance hashing and is not comparable with the provider-probe session time.
See the [measurement correction](measurement-correction.md) and
[corrected repeated smoke](../frame-engine-api-smoke-20261020-r2/report.md).
