# Rust real-model still-image localization audit

Run ID: `rust-cvat-ground-truth-20261020`

## Question and scope

This audit checks whether the current Rust ONNX inference path locates the
objects in the repository's nine CVAT-labeled still images, and whether Rust
and the retained Python checkpoint produce the same detections on those
images. It is a static, unpaced, CPU-only development-machine run. It does not
test a camera, temporal tracking, table calibration, physical position error,
numbered-ball identity, HTTP/WebSocket freshness, or Linux behavior.

All nine inputs are symlinks to the original files under
`backend/vision/test_data/`, so the sorted input contains exactly the images
listed in `annotations.xml` and no unannotated `grid.jpg`. The CVAT association
radii below are analysis settings, not accepted production error limits.

## Result

At a 20 px image-center association radius, Rust matched all 52 annotated balls:

| Class | TP | FP | FN | Precision | Recall | F1 | Matched center error, mean / p50 / p95 / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ball | 52 | 3 | 0 | 94.5% | 100% | 97.2% | 3.81 / 2.99 / 9.23 / 14.99 px |
| cue | 0 | 2 | 3 | 0% | 0% | 0% | unavailable |
| overall | 52 | 5 | 3 | 91.2% | 94.5% | 92.9% | 3.81 / 2.99 / 9.23 / 14.99 px |

At a stricter 10 px radius, ball results were 49 TP, 6 FP and 3 FN (89.1%
precision, 94.2% recall and 91.6% F1); matched center error was 3.22 px mean,
2.89 px median, 6.69 px p95 and 8.18 px maximum. The change in counts shows
why these association radii should not be mistaken for a production tolerance.

The retained Python checkpoint had the same per-class counts. Python and Rust
each emitted 57 detections, all paired with no unmatched outputs; the maximum
cross-runtime center difference was 0.000256 px. This is strong static output
equivalence for this checkpoint/export/configuration, not independent evidence
that the model is correct beyond these labels.

The ball false positives include one extra detection in `frame_with_cue.png`
near (1702, 326) and two detections in `motion_blur.png` near (436, 726) and
(526, 411), where CVAT labels only the cue. Cue localization is substantially
weaker: the detector missed all three cue labels within 20 px. The existing
model declares only generic `ball` and `cue` classes, so it cannot identify
8-ball versus 6-ball versus 7-ball.

## Runtime observation

Rust processed nine 3840×2160 PNGs serially using the ONNX Runtime CPU
provider, with one warm-up inference. The tool reported 97.56 ms mean
decode-to-detection time per image: 43.02 ms image decode, 9.65 ms
preprocessing, 44.80 ms inference, and 0.09 ms output decode/NMS. The aggregate
recorded-image time was 878.12 ms. This one-run file-replay figure includes PNG
decode and is not a measured live camera frame rate or end-to-end client
latency. The camera path may have a different decode/copy cost and must be
measured on the target device.

## Provenance and reproducibility

- Hardware: MacBook Pro, Apple M4 Max (14 CPU cores), 36 GB memory; macOS
  26.6.2, arm64.
- Rust runtime: `billiards-perception-infer`, release build,
  `CPUExecutionProvider` observed for every node.
- ONNX export SHA-256:
  `2444f8007934aa6a17584856b3049ebd1c1a6861c174408d7c3d549b41b45648`.
- Source PT checkpoint SHA-256:
  `f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`.
- Rust executable SHA-256:
  `d24ffa1d18e193cd17859772b7a800a29d7399a8d06fa4eb13fa68d21ee7a0c6`.
- CVAT XML SHA-256:
  `3960cbc90256c56b3378b3f31a381c7b09397ceaa2dfc4cf168b955647d4bd5d`.
- Settings: image size 1280×1280, confidence 0.4, NMS IoU 0.45, maximum 300
  detections, one warm-up, no calibration bundle, no frame pacing.
- Rust result and both scored reports are preserved beside this document:
  [`rust-result.json`](rust-result.json),
  [20 px score](scored/equivalence-20261020T192553.172605Z.md),
  [10 px score](scored-radius-10/equivalence-20261020T192733.614563Z.md).
  Machine-readable hardware, worktree, model, input and command provenance is
  recorded in [`run-metadata.json`](run-metadata.json).

The existing API integration path remains anonymous: this audit used no
credentials, and the separate real-model API smoke verifies observations over
anonymous HTTP and WebSocket connections. No authentication mechanism was
added.

## What to improve next

1. **Improve and expand the accuracy evidence before a production claim.** The
   nine stills contain 52 ball labels but no temporal annotations or verified
   physical center coordinates. Add held-out camera views, difficult occlusion
   and motion examples, and repeated placements measured in table coordinates.
   Report calibration error in millimeters separately from image-pixel center
   error.
2. **Reduce the three observed ball false positives.** Review the motion-blur
   examples and `frame_with_cue.png`; add representative hard negatives and
   label-review checks before changing confidence or NMS settings. Re-run the
   same scorer so any precision gain is checked against recall.
3. **Build numbered-ball recognition as a separate asynchronous capability.**
   The current output schema/model only distinguishes `ball` and `cue`. Use
   identity labels and held-out views to evaluate a crop classifier against a
   joint numbered detector, and do not delay fresh ball locations while
   identity remains unknown.
4. **Measure the live camera path on the target Mac and Linux setup.** This
   replay cannot choose or qualify the camera dependency. Record actual camera
   mode, frame age, dropped frames, read/copy/preprocess/inference timings,
   disconnect/restart behavior and client receipt. Preserve the one-camera-owner
   invariant and keep all connections anonymous.

These are evidence priorities, not a recommendation to change the model,
transport or architecture from a nine-image run. Prediction and trajectory
work remain secondary to fresh ball localization.
