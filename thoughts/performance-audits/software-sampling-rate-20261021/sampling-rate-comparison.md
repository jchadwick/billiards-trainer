# Software camera-sampling rate comparison (2026-10-21)

This follow-up measures how a bounded latest-frame inference mailbox behaves at several software-paced input rates. It provides a relational CPU-capacity result for the target machine, not a qualified camera rate.

## Setup

- Target: Ubuntu 24.04.2, Intel Core i5-8600T (6 logical CPUs, 15.4 GiB RAM), with the backend and frontend app containers present. Each inference run used a one-off container capped at six CPUs. The webcam was disconnected; Coral was not used.
- Input: 60 seconds from a real 3840×2160, 30 FPS saved video (`full_table.mp4`, SHA-256 `a1de2fe82fe5eb4c8722c4b64b7b5d416cead997c7cab491efa9975f8325c3a3`, source interval 120–180 s). The stride sets retain original frame order and source filenames. See [sample manifest](raw/sampling-inputs.json).
- Detector: ONNX `billiards-yolov8n`, SHA-256 `7beb3af7923290a5524fe1744de13bd66f4e43ea8a5048347bc8a12c51a1e4b5`; 1280-square input, confidence 0.4, IoU 0.45, three warm-ups, OpenVINO native CPU. It recognizes cue/ball only, not numbered balls.
- Replay: each frame set was paced for 60 seconds through the production Rust latest-frame mailbox. Arrival timestamps were synthetic and assigned after image decode. Each rate ran once; four measurement windows were 60.0 seconds and the 4.286 FPS window was 60.2 seconds because its 258-frame schedule does not divide evenly into 60 seconds. Exact reports are retained under [raw results](raw/).

## Results

| Input FPS | Window | Input frames | Processed FPS in same window | Total processed | Pending frames replaced | Completion p95 | Completion p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.000 | 60.0 s | 180 | 2.983 | 180 | 0 | 321.8 ms | 455.3 ms |
| 3.333 | 60.0 s | 200 | 3.317 | 200 | 0 | 324.2 ms | 368.9 ms |
| 3.750 | 60.0 s | 225 | 3.717 | 224 | 1 | 448.3 ms | 477.2 ms |
| 4.286 | 60.2 s | 258 | 3.970 | 241 | 17 | 510.0 ms | 553.8 ms |
| 5.000 | 60.0 s | 300 | 3.683 | 222 | 78 | 530.4 ms | 548.6 ms |

Completion latency is synthetic frame-arrival-to-inference-complete, including any wait behind in-progress inference and image decode/preprocessing/inference. Same-window completion counts are used for `processed FPS`; the total column includes work that finished just after the 60-second window. The raw report also records the in-window received and processed counts, source-to-inference-start percentiles, and stage timing means.

## What the measurements support

- The two highest tested rates with no mailbox replacement were 3.0 and 3.333 FPS. All input frames completed after allowing the final in-flight work to drain; one completion per run fell just beyond the measurement window. Same-window processed rate was lower than input rate by 0.017 FPS in each case.
- At 3.75 FPS the mailbox replaced one pending frame and p95 completion latency rose to 448 ms. At 4.286 FPS it replaced 17/258 frames; at 5 FPS it replaced 78/300 frames (26%). Processing capacity flattened around 3.7–4.0 completed frames per second while completion p95 reached 510–530 ms.
- Thus, **3.333 FPS is the highest rate in this single sweep that showed no pending replacement**, while 3 FPS provides more operating margin. Neither is a production recommendation: p95 completion latency around 322–324 ms is already roughly one sample interval, and the experiment says nothing about whether a moving ball remains accurately localized at such sparse sampling.
- “Processed FPS no greater than received FPS” is necessary but not sufficient for current state. Use replacements, queue age, source-to-completion latency, and accuracy together. The latest-wins mailbox does keep overload bounded and exposes intentional losses; it cannot reduce the roughly 300 ms per-frame detector path measured here.

## Consequences and next experiments

1. Keep the latest-frame replacement policy. Do not allow FIFO replay to build a history backlog. Reject or visually suppress source states after the configured freshness limit; the UI now advances state age between messages and hides stale ball locations.
2. Do not set the webcam to 3 FPS based on this test. First test a real camera at multiple hardware modes, preserving exposure time and dropped-frame counters, then score detection recall and center error during fast motion and occlusion. A rate that loses moving balls is not a good balance even if queues stay empty.
3. Prioritize reducing the measured per-frame critical path. Decode averaged about 51 ms, preprocess about 68 ms, and inference about 190 ms near 3.33 FPS. Test efficient crop/ROI only with full-table coverage proof, model input-size or quantization variants against localization labels, and explicit OpenVINO thread settings. Keep the Coral A/B separate until its runtime is available and verified.
4. Measure camera-to-state and state-to-client receipt with the real camera, then compare state-only, trajectory, and viewer operation. These image replays do not measure transport, browser rendering, or end-to-end freshness.

## Limits

This is one machine, one 60-second video segment, one model, one run per rate, and no temporal labels. Synthetic replay timestamps start after decoding; the result does not prove the physical camera can deliver those rates or that a 3 FPS sample captures shots. The target camera was disconnected. The model’s two classes cannot identify a 6-ball, 7-ball, or 8-ball. CPU utilization, memory/allocation trends, queue receipt, and display latency were not measured here.

Provenance (revision, dirty-worktree fingerprint, target, model, and effective settings) is recorded in [summary.json](summary.json).
