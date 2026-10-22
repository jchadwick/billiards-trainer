# Target OpenVINO runtime A/B

This short run checks whether the optional runtime provider loads on the Linux target and whether it changes inference cost or output on a matched sample. It is a relational check, not a production qualification.

## Setup

- Target: Ubuntu 24.04.2, Intel Core i5-8600T, 6 logical CPUs, 15 GiB RAM.
- The host-level OpenVINO install was not present. OpenVINO Runtime 2026.4.1 ran in a temporary Ubuntu 24.04 container on the target; the host was not changed.
- ONNX Runtime 1.31.0 (`ort` 2.0.0-rc.15), same OpenVINO-enabled x86_64 binary, model, thresholds, and 30 JPEGs for both runs.
- Input: first 30 consecutive 3840×2160 JPEG frames from the 120–180 second `full_table.mp4` segment. Replay was unpaced with 3 warm-up inferences per provider and synchronous JSONL export enabled.
- The OpenVINO per-node profile observed `{'OpenVINOExecutionProvider': 1}`. The CPU profile observed `{'CPUExecutionProvider': 219}`.

## Results

| Measure | CPU | OpenVINO CPU | Change |
|---|---:|---:|---:|
| Images per second | 3.004 | 3.418 | +13.8% |
| Mean inference | 235.6 ms | 175.1 ms | -25.6% |
| Inference p95 | 246.7 ms | 178.9 ms | -27.5% |
| Mean image-to-result | 330.1 ms | 287.0 ms | -13.1% |
| Image-to-result p95 | 353.7 ms | 292.3 ms | -17.4% |

OpenVINO lowered mean inference time by about 26% and mean image-to-result time by about 13% on this sample. Image decode and preprocessing were slightly slower in the OpenVINO run, so the end-to-end gain is smaller than the inference gain. With 30 frames and one run per provider, these are directional measurements.

## Output comparison

Both providers emitted 416 detections over the same 30 images, with matching per-frame detection counts on all 30 frames. All detections could be paired by class; the mean center delta was 0.000019 source pixels, the p95 was 0.000122 pixels, and the maximum was 0.000366 pixels. This is close numerical parity between providers, not evidence that the detections match the real balls: the clip has no ground-truth labels.

## Scope and next step

The provider was also exercised through the backend supervisor and retained frontend, with state received by an unauthenticated WebSocket subscriber; see the [application integration report](app-integration/report.md). That short smoke confirms startup and delivery, but its subscriber message counts are not a speed comparison and it does not measure source-to-client latency. The target host-level installer is still pending, so this experiment does not verify its `setupvars.sh` environment. CPU remains the configured default until labeled accuracy and a longer, repeated comparison support a change.

Detailed timing reports, detection batches, and the retained OpenVINO node profile sit beside this report. Machine-readable details are in [comparison.json](comparison.json) and [run-metadata.json](run-metadata.json).
