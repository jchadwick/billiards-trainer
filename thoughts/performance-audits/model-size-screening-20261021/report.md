# Model input-size and runtime screening

**Run:** `model-size-screening-20261021`
**Purpose:** compare fixed model-input sizes and target CPU/OpenVINO behavior without changing production defaults. This is a screening run, not a release qualification.

## Result

The 960×960 export is the strongest next candidate to evaluate on a temporally labeled clip. It keeps all 52 labeled ball instances detected in the nine still-image set, while cutting mean ONNX CPU inference time by 44.5% against the 1280×1280 export on the development machine. Its mean full image-to-result time fell by 21.7%. The 640×640 export was faster again, but missed four of the 52 labeled balls and reduced ball F1 from 0.972 to 0.923, so it should not advance without retraining or a specific accuracy remedy.

The target-host verification rerun showed the same directional result: 960×960 reduced single-run p95 inference time by 42.9% on CPU and 41.8% on native OpenVINO relative to 1280×1280. The OpenVINO runtime was faster than ONNX Runtime CPU at both sizes in this bounded run. The target sample is only 30 sequential images per arm, so these figures show relative direction and do not qualify sustained rate, memory behavior, camera freshness, API delivery, or accuracy.

No production model size or provider default was changed.

## Development-machine repeated screen

Each size used the same nine annotated 3840×2160 still images, with two warm-up images and three unpaced repetitions. There were 27 measured image samples per size. Percentiles use nearest rank.

| Input | Mean inference | Inference p95 | Mean image-to-result | Image-to-result p95 | Ball precision / recall / F1 | Ball center error mean / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1280×1280 | 43.76 ms | 47.19 ms | 97.74 ms | 100.28 ms | 94.5% / 100% / 0.972 | 3.81 / 9.23 px |
| 960×960 | 24.28 ms | 25.85 ms | 76.54 ms | 81.32 ms | 94.5% / 100% / 0.972 | 3.99 / 9.78 px |
| 640×640 | 13.37 ms | 20.70 ms | 61.30 ms | 67.85 ms | 92.3% / 92.3% / 0.923 | 4.84 / 9.87 px |

All input sizes use exports from the same source checkpoint, with confidence 0.4, IoU 0.45, and at most 300 detections. The 960 export matched the reference model within the equivalence harness's 2 px tolerance on all 58 matched detections; its largest center difference was 0.75 px. At 640, one reference and one candidate detection were unmatched. This confirms runtime/export parity for this small sample, not the underlying model's physical accuracy.

The still labels contain 52 generic balls across nine images and three cue objects. The cue class was not successfully detected at 1280 or 960. At 640 there was one cue true positive, one false positive, and two misses. These labels do not test numbered-ball identity, motion, occlusion, settlement, or freshness over time.

Detailed runs, static scoring, and exported models are in [runs](runs/), [scores](scores/), and [exports](exports/).

## Target Linux relative A/B

The target is the Ubuntu 24.04.2 x86_64 HP ProDesk with an Intel i5-8600T at 2.30 GHz, six logical CPUs, and about 15.4 GiB RAM. Each arm processed the same 30 sequential 3840×2160 JPEGs, with two warm-ups. Replay was unpaced. These images came from a 30 fps video segment, but the harness did not pace them or use their timestamps for end-to-end camera latency.

| Provider | Input | Processed images/s | Inference mean / p95 | Image-to-result mean / p95 |
| --- | ---: | ---: | ---: | ---: |
| ONNX Runtime CPU | 1280×1280 | 3.25 | 231.62 / 250.93 ms | 307.81 / 326.96 ms |
| ONNX Runtime CPU | 960×960 | 4.72 | 124.25 / 143.29 ms | 208.58 / 226.79 ms |
| Native OpenVINO CPU | 1280×1280 | 3.37 | 176.74 / 184.55 ms | 290.19 / 301.76 ms |
| Native OpenVINO CPU | 960×960 | 5.63 | 104.39 / 107.47 ms | 177.65 / 188.07 ms |

At 960, the target CPU arm completed 45% more images per second than the CPU 1280 arm; the OpenVINO arm completed 67% more than OpenVINO at 1280. The p95 inference reductions were 42.9% for CPU and 41.8% for OpenVINO. At the same size, OpenVINO completed 19% more images per second than CPU at 960; this is one 30-image run per cell and should be treated as directional, not stable. A verification rerun recreated all four result files with the same target binary and the same frame hashes after an earlier local-copy checksum mismatch; final local file hashes match the target outputs exactly.

At the same input size, CPU and OpenVINO emitted the same class counts on all 30 frames. The 1280 outputs matched within 2 px for all 416 detections, with maximum center difference 0.0005 px; the 960 outputs matched for all 421 detections, with maximum difference 0.0017 px. This validates output parity between these two tested providers on this frame sample. It does not validate installation discovery on a clean host: the temporary OpenVINO test container exposed only a versioned `libopenvino_c.so.2641`, so an ephemeral unversioned symlink was needed for `openvino-rs` library discovery. Clean deployment-time library discovery remains to be tested.

Changing from 1280 to 960 changed the detection set: only 25 of 30 frames had equal class counts, and the same-class detections differed in location by up to 1.90 px after matching. These target frames have no labels, so this divergence does not establish which size is more correct.

The target comparison used the same native OpenVINO-enabled Rust executable for all arms. The executable SHA-256 is `01e22295cd157b572b96b2858e2f7f673e8dd4fe10df36ca1394558d9b9b0f0c`. Runtime version, model hashes, target/local result hashes, host data, and all 30 input-frame hashes are in [target-linux/host-metadata.json](target-linux/host-metadata.json); the raw outputs and inputs are retained under [target-linux](target-linux/).

The target process captured source revision `e748c0f99e8e8c089633aa282154a05697adb243` and a dirty-worktree count of 244 paths, but did not capture a content hash for that dirty tree. The executable hash is therefore the strongest exact identity for the tested binary; the missing dirty-tree hash limits source-level reproduction.

## Decision and next experiment

Advance 960×960 as an audit candidate only. The 640 result is currently unacceptable for a location-first detector because it loses labeled balls. Before changing the live default, run 1280 and 960 through the same paced temporal clips and calibration, then compare ball recall, center error in table coordinates, identity continuity, cue quality, source-to-client freshness, and queue/drop behavior. Include motion, dense layouts, occlusion, and settlement, and repeat the candidate on the target only for a bounded confirmation after the host has the camera connected.

The present model has only `cue` and generic `ball` classes. No input size can identify the 6-, 7-, or 8-ball from these outputs. Keep numbered-ball recognition as a separate later task so it cannot delay location publication.

## Reproduction files

- [run metadata](run-metadata.json)
- [target host and hash inventory](target-linux/host-metadata.json)
- [raw development-machine runs](runs/)
- [static CVAT/equivalence scores](scores/)
- [audit-only 960 and 640 exports](exports/)
- [30 target input frames](target-linux/input/frames/)
- [target raw outputs](target-linux/results/)
