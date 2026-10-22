# Linux target recorded-video replay

**Run:** `rust-live-trained-model-export-20261021/target-linux-replay-20261021`

**Result:** CPU-only inference is substantially below the requested 30 fps replay cadence. This is a useful capacity signal for the detector on this host, not a qualification of camera-to-client latency or detection accuracy.

## What was run

The Linux x86_64 Rust inference executable was built on the development Mac in an Ubuntu 24.04 container, copied to `192.168.1.31`, and run directly on that Ubuntu host. A one-frame model-load/inference smoke test passed before the replay. This original CPU capacity run did not include the API or web frontend; a later 30-frame CPU/OpenVINO application integration smoke is documented in [the follow-up report](openvino-runtime-ab/app-integration/report.md).

The input was a 60-second segment, source-video offsets 120–180 seconds, from `full_table.mp4`. The 3840×2160, 30 fps source was extracted to 1,800 JPEG images. The Rust executable replayed those images at a requested 30 fps with latest-frame replacement enabled. It used 200 warm-up inferences and then wrote one JSON detection batch per processed frame. Input timestamps are synthetic replay-arrival timestamps; they are not camera exposure timestamps.

The host was Ubuntu 24.04.2 LTS, x86_64, Intel Core i5-8600T at 2.30 GHz (6 logical CPUs), and 15 GiB RAM. The run used ONNX Runtime 2.0.0-rc.15 with `CPUExecutionProvider` only. The model input was fixed at 1280×1280, with confidence 0.4 and IoU 0.45. No accelerator metrics were available.

## Findings

| Measurement | Result |
|---|---:|
| Replay cadence requested | 30 fps |
| Frames scheduled / decoded | 1,800 / 1,800 |
| Frames inferred | 274 (15.2% of scheduled) |
| Pending frames replaced | 1,526 (84.8%) |
| Observed source read/decode rate | 17.45 fps |
| Processed rate | 2.64 fps |
| Source-arrival to inference-complete p50 / p95 / p99 | 393.7 / 443.8 / 474.6 ms |
| Full image-decode-to-result p50 / p95 / p99 | 437.6 / 480.0 / 501.8 ms |

The p95 source-arrival-to-inference-complete latency is about 8.9 times the README’s 50 ms reference budget. That comparison is only a scale reference: this replay uses disk images and synthetic timestamps, not a live capture clock.

| Stage | Mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| JPEG decode | 59.8 ms | 63.7 ms | 88.4 ms | 99.3 ms |
| Resize / preprocessing | 56.4 ms | 53.8 ms | 68.4 ms | 68.9 ms |
| ONNX inference | 320.7 ms | 318.7 ms | 342.9 ms | 390.8 ms |
| Decode / NMS | 0.23 ms | 0.22 ms | 0.27 ms | 0.28 ms |

Model inference is the largest measured stage (about 73% of the mean image-to-result time). JPEG decode is specific to this image-replay harness and should not be counted as live camera cost. Preprocessing is material in this replay but needs measurement against the actual camera frame format before it is ranked as a live bottleneck.

The command took 152.45 seconds wall time, including 45.61 seconds for warm-up. The measured replay pipeline accumulated 103.71 seconds across its 274 processed images. Process CPU time was 584.86 seconds user and 51.35 seconds system, approximately 4.2 CPU cores averaged over command wall time. Sampled resident memory peaked at roughly 364,112 KiB (355 MiB); this was process sampling, not a kernel-recorded high-water mark.

## Detection and accuracy limits

The model emitted 3,291 generic `ball` detections across 274 processed images (mean 12.0, range 10–13) and at least one `cue` detection in 248 images. These are counts, not precision or recall. The run had no ground-truth labels, so it cannot establish missed-ball rate, false positives, center error, identity continuity, motion performance, or temporal freshness against real source timestamps.

The exported ONNX model has only two classes, `cue` and generic `ball`; it cannot distinguish the 6-, 7-, or 8-ball. Outputs are image-pixel coordinates. No table calibration or client receipt was exercised. The ONNX preprocessing path is documented as an OpenCV-compatible approximation, and tensor-level parity against the source model is unverified. The existing still-image labels do not provide the temporal annotations needed to qualify moving-ball accuracy.

The Coral appeared as a USB device on the target, but this run did not use it: there was no Edge TPU execution provider, no `/dev/apex_0`, and no deployed Edge TPU runtime or compiled TFLite model. Therefore this result measures CPU execution only.

## Ranked improvement work

1. **Qualify the Coral path with the actual model.** Build or select a fully integer-quantized TFLite detector that can compile for the connected Coral, install/use the supported runtime, and verify delegate/operator placement from runtime evidence. Compare CPU and Coral on identical labeled frames and temporal clips. Keep the Coral as an opportunity, not an assumed speedup: unsupported operators may fall back to CPU, and quantization may change accuracy.
2. **Sweep model size and CPU threading against accuracy labels.** Inference takes 320.7 ms on average at the current fixed 1280-square input, so test supported lower input sizes and explicit ONNX thread counts with identical input, warm-up, and replay policy. Record ball recall, center error, and freshness alongside throughput; do not recommend a smaller input until dense-table and motion accuracy remain acceptable.
3. **Measure the actual camera-format preprocessing path.** The 59.8 ms JPEG decode cost belongs to this harness. Once the webcam is connected, measure negotiated frame format, any color conversion/copy, resize, inference, and publication separately. The current 56.4 ms image preprocessing figure is a recorded-JPEG result, not an established live-camera cost.
4. **Extend the application-level transport audit.** A short 30-frame CPU/OpenVINO smoke now verifies API supervision, frontend proxying, unauthenticated WebSocket state subscription, and received scene messages ([report](openvino-runtime-ab/app-integration/report.md)). It does not measure full-run source-to-client latency, gaps against a temporal clock, queue growth, or slow-client isolation. Those remain the next validation.
5. **Add temporal ball identity after location quality is measured.** The current model has no numbered-ball classes. Collect reviewed temporal labels for occlusion, collisions, and dense tables, then evaluate a separate identity stage against the generic location detector. Keep identity inference from delaying measured centers.

The first priority is correct and fresh locations. No architecture change to scheduling, transport, or tracking is justified by this isolated CPU replay alone.

## Separate development-machine instrumentation comparison

The requested relative instrumentation A/B was measured on the development Mac and is not a target-host result. Across three 120-second runs per condition, OFF produced 19.413 states/s on average and ON produced 19.253 states/s, a point estimate of 0.82% lower throughput with instrumentation. Mean p95 receipt interval was 54.371 ms OFF and 54.886 ms ON, a 0.515 ms (0.95%) increase. No audit records were lost. With only three repetitions and visible run-to-run variation, these point estimates are inside the 2% budget but do not establish a statistically confident upper bound. No additional local performance runs were made.

## Reproducibility and artifacts

- Versioned frame-level result: [full-table-segment-paced-report.json](full-table-segment-paced-report.json)
- Frame detection batches: [full-table-segment-batches.jsonl](full-table-segment-batches.jsonl)
- Linux cross-build recipe: [Dockerfile](Dockerfile)
- One-frame load/inference smoke result: [one-frame-smoke.json](one-frame-smoke.json)
- Source video: `full_table.mp4`, SHA-256 `a1de2fe82fe5eb4c8722c4b64b7b5d416cead997c7cab491efa9975f8325c3a3`
- Rust source revision: `e748c0f99e8e8c089633aa282154a05697adb243`; deployed `perception/` source-tree SHA-256: `0d300b1d7163b12a4115adc9cc9e253711b7cb13e36cc38c8346e228521eda3e` (the same digest was calculated on the target copy).
- Deployed executable SHA-256: `71e867cac2d95dcb2f70a963acdeda429a5bfcfa77125140309f57fdb167d8d8`.
- Model hashes: ONNX `7beb3af7923290a5524fe1744de13bd66f4e43ea8a5048347bc8a12c51a1e4b5`; source checkpoint `f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`; export manifest `74c229ff058ec138b6194593921e5768dbeada092440ffd2c2a9f84e6b0dc707`.

The target run directory is `/home/jchadwick/billiards-rust-video-audit-20261021/`. The checked-in result files are kept under this audit folder; raw JPEG frames remain on the target. The short API/frontend state-delivery path is qualified for this recorded-image smoke; full-replay receipt timing, camera capture, Coral inference, and temporal accuracy remain unqualified.
