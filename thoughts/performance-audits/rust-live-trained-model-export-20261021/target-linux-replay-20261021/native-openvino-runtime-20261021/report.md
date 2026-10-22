# Native OpenVINO target smoke and short A/B

This is a bounded implementation check on the user-provided Linux target. It verifies that the new direct OpenVINO C API backend loads the existing ONNX model, executes inference, and retains output parity with the ONNX Runtime CPU path on an identical short sample. It is not a production performance or accuracy qualification.

## Setup

- Target: Ubuntu 24.04.2, Intel Core i5-8600T at 2.30 GHz, 6 logical CPUs, 15 GiB RAM.
- Binary: built on the development Mac in an Ubuntu 24.04 x86_64 container with Rust 1.99.1; SHA-256 `01e22295cd157b572b96b2858e2f7f673e8dd4fe10df36ca1394558d9b9b0f0c`.
- Runtime: OpenVINO Runtime 2026.4.1 from the existing target test image. The host-level OpenVINO install was not present; the existing API and frontend containers were left untouched and healthy.
- Input: 30 consecutive 3840×2160 JPEG frames from the 120–180 second `full_table.mp4` segment. Both runs used identical frame hashes and the same verified model, confidence 0.4, IoU 0.45, 1280 input, and one warm-up inference. Runs were unpaced, sequential, CPU then OpenVINO.
- Worktree: revision `e748c0f99e8e8c089633aa282154a05697adb243`, dirty at run time; the implementation source set has SHA-256 `317a486537d3fac4a4025583c9915d59b2fd63b860fe7450d0cbc040a54fd3a4`. See `run-metadata.json` for file-level identity, model hashes, and run configuration.

## Results

| Measure | ONNX Runtime CPU | Native OpenVINO CPU | Change |
|---|---:|---:|---:|
| Images/s | 3.241 | 3.393 | +4.7% |
| Mean inference | 231.87 ms | 177.15 ms | -23.6% |
| p95 inference | 247.01 ms | 183.34 ms | -25.8% |
| Mean image-to-result | 308.53 ms | 289.15 ms | -6.3% |
| p95 image-to-result | 324.09 ms | 295.53 ms | -8.8% |

Both backends emitted 416 detections. Per-frame counts matched on all 30 frames. Matching same-class centers gives a mean difference of 0.00002053 px, p95 0.00016500 px, and maximum 0.00050000 px. This is numerical output parity on the selected frames, not evidence that either backend found every real ball.

OpenVINO improved the point estimate on all measured speed metrics. Since this is a single short run per backend on a Linux test machine that is not the final performance target, the result is directional and should not change the default provider. The production default remains `cpu`.

## Runtime behavior and limits

The report identifies `openvino-rs` and the native CPU backend. Per-node ONNX Runtime provider traces are unavailable by design; the native OpenVINO audit reports empty node counts rather than implying execution-provider graph partitioning. A direct invocation on the target host without OpenVINO libraries failed clearly with an OpenVINO load error and did not fall back to CPU.

The test runtime was inside a container because the OpenVINO installer is not yet visible to the host process. The deployed app containers still use their prior binary; container runtime packaging/mounting and app-level state delivery with the native backend remain to be integrated. Apple Silicon runtime execution, labeled accuracy, temporal tracking accuracy, numbered-ball identity, calibration, and camera-to-client freshness remain unqualified.

Machine-readable outputs are in `cpu-30.json`, `openvino-30.json`, and `run-metadata.json`. The earlier ONNX Runtime OpenVINO provider A/B is separate historical evidence and does not qualify this direct backend.

## Current-container loader confirmation

After the original A/B, the current Rust inference source was cross-built on the
development Mac with the locked Linux builder and run once inside the target's
existing API container, using that container's mounted OpenVINO 2026.4.1
libraries, the same model, and `frame-000001.jpg`. The binary reported
`openvino-rs` / `OpenVINO CPU (native)` and returned 14 detections; inference
took 180.54 ms and decode-to-result took 291.68 ms for this one cold-container
sample. Build identity was source revision
`127255103e7cd117f727a0842770ec079c15a87a`, with unrelated Python audit changes
in the worktree, and binary SHA-256
`9c5bf8960ab9491a157c567d840d61e3a94e01000bae7db6c6377fdd6e80c592`. The
runtime's versioned C API library was staged under `/tmp` with the
unversioned name expected by the binding. The test binary and copied runtime
files were removed afterward, and both API and frontend health checks remained
HTTP 200.

This initial loader smoke confirmed the native backend against the runtime
files mounted in the API container, but it did not route a replay through the
running API supervisor. At that time, inspection of the API image's inference
binary showed the older `ort` binding and `OpenVINOExecutionProvider`. The
follow-up deployment and supervisor-to-client smoke are recorded below. No
camera or frontend-render latency was measured.

## Follow-up API replay smoke

The API container was subsequently rebuilt and configured to launch the
runtime-selected native OpenVINO binary. A 30-frame corpus repeated ten times
at 5 fps completed with exit code 0. The API reported 221 accepted and
published state snapshots from 300 scheduled input slots. A five-second
WebSocket collection received 19 complete state messages, including 16 unique
source observations, with six source-sequence gaps. This verifies basic
supervisor-to-client delivery but shows the run did not sustain the requested
5 Hz. The collector ran on macOS, so no comparable cross-host latency is
reported; Rust-provider server-stage timing and resource samples were
unavailable. See the [target replay notes](../../../target-rust-replay-smoke-20261021/target-smoke-notes.md)
for the runtime-selected CPU/OpenVINO comparison and follow-up priorities.
