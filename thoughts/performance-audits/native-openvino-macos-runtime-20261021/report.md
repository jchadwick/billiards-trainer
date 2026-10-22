# Native OpenVINO Runtime on Apple Silicon

- Date: 2026-10-21T20:15:24.914827+00:00
- Host: Apple M4 Max (arm64, macOS)
- OpenVINO: 2026.4.1-22982-e213a147257-releases/2026/4 from an isolated temporary PyPI install
- Runtime backend: `openvino-rs 0.11.1`
- Binary SHA-256: `089d2713222daaeeac4dd04898665e68279108bf12b13c29984f7c27b983b0c3`
- Workload: 30 recorded 3840×2160 still frames, unpaced, 1280-square model input
- Source revision: `127255103e7cd117f727a0842770ec079c15a87a` (worktree dirty)
- Corpus digest: `433a9871cef373f2aef35c83ee7f55fc85e13256dd4ca9f1e619c38d11306d9b`

The Rust direct OpenVINO backend initially failed to discover the official wheel's versioned `libopenvino_c` dylib. The application loader now searches versioned library names inside the configured `DYLD_LIBRARY_PATH`; the current release binary loaded OpenVINO CPU and produced a complete 30-frame report. A separate no-runtime invocation failed clearly without falling back. The runtime library stayed outside the project environment.

The same ONNX model, input frames, preprocessing, confidence threshold, and NMS threshold were run through ONNX Runtime CPU and native OpenVINO CPU. Detection counts matched on all 30 frames (416 detections each). Center associations matched 368 detections within 2 px and all 416 within 5 px. The 5 px association had mean/p50/p95/maximum center delta 0.917/0.812/2.461/2.815 px. Apple Silicon residuals are larger than the previous Linux comparison and require labeled-frame review before recommending either provider.

Median unpaced per-frame inference time was 40.92 ms for ONNX Runtime CPU and 45.03 ms for OpenVINO. This is one short development-Mac run, not a performance recommendation or real-time qualification. No temporal annotations were available, so this does not establish real-world accuracy, identity continuity, or freshness.

The raw runtime reports are retained as [`openvino.json`](openvino.json) and [`ort-cpu.json`](ort-cpu.json); the machine-readable summary is [`summary.json`](summary.json).
