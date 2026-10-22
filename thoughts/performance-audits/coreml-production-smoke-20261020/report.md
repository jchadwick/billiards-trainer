# Configured CoreML detector smoke

This run answers a narrow Stage 2 question: can the repository's configured
CoreML model load and return ball locations through the retained Python detector,
and what predictor settings does it actually use? It does not measure live
capture, gateway delivery, sustained rate, or physical-coordinate accuracy.

## Environment and model

The run used macOS arm64, Python 3.11.14, Ultralytics 8.4.182, CoreML Tools 9.0,
PyTorch 2.14.1, NumPy 2.3.5, and OpenCV 5.0.0.93 in an isolated temporary
environment. CoreML Tools warned that this PyTorch version is outside its
tested versions; loading and inference still completed for this pre-exported
CoreML package, but the production dependency set needs a deliberate
compatibility qualification. The
project's checked-in configuration selected
`models/billiards-yolov8n.mlpackage`, device `auto`, confidence 0.40, IoU 0.45,
and disabled automatic fallback. The exact model package tree hash and input
image hash are recorded in [result.json](result.json).

The first cold load exposed two code/dependency defects: the loader read
Ultralytics' lazy `YOLO` export from the module dictionary, and a missing
`coremltools` dependency was mislabeled as a missing Ultralytics package. Both
are fixed with focused regression tests. The CoreML runtime is now declared as
a macOS-only requirement in `backend/requirements.txt`, and `.mlpackage`
auto-selection reports the CoreML backend rather than MPS. The temporary test
environment did not alter the project's installed packages.

## Observed runtime behavior

The model loaded and returned five detections from the 3840×2160
`multiple_balls.png` fixture. Ultralytics resolved its input tensor to
`[1, 3, 1280, 1280]`, float32, normalized to `[0, 1]`. Other effective predictor
settings were `rect=true`, batch 1, max detections 300, confidence 0.40, and
IoU 0.45. A second inference returned the same detection count.

The model class map is only `cue` and `ball`, so it cannot distinguish the
8-ball from other numbered balls. CoreML reported a `CPU_AND_NE` compute-unit
policy. That policy does not reveal the actual per-operator device partition or
utilization; `effective_device` remains unavailable. The selected backend is
reported separately as CoreML.

## Direct Rust input and output comparison

For this same PNG, the CoreML predictor's input tensor and the previously
recorded Rust CPU preprocessor tensor were exactly equal across all 4,915,200
float values. The encoded-image hashes also match. Comparing the saved detector
outputs through the existing equivalence tool paired all five detections within
2 px; the largest center difference was 1.15 px and the largest confidence
difference was 0.02354. The [tensor measurements](tensor-comparison.json) and
[output comparison](coreml-vs-rust-static-comparison.json) preserve the details.

This narrows the one-PNG discrepancy to model/export/runtime output behavior,
rather than image decoding or preprocessing. The CoreML package has no verified
source checkpoint manifest, so the comparison cannot distinguish weight
lineage from exporter or runtime effects. It does not establish artifact
equivalence; the one JPEG decoder mismatch and broader image/temporal coverage
remain open.

## Fixture comparison

All five predicted centers were paired with the five image-plane labels. Their
center errors were 5.33, 1.77, 2.90, 4.16, and 0.40 pixels: mean 2.91 px,
maximum 5.33 px. Two of five were within 2 px, and four of five within 5 px.
This confirms that the configured path produces locations and also shows that
one successful image is not enough to establish accuracy. The labels do not
measure physical table coordinates, temporal continuity, occlusion recovery,
or exposure-to-client freshness.

## Next evidence needed

- Compare the captured Python tensor with the Rust tensor for this and several
  JPEG fixtures; `rect=true` alone is not proof of equivalent preprocessing.
- Run the identical recorded and paced sequences through both providers and the
  retained headless receiver.
- Build held-out, complete annotations with physical table coordinates and
  temporal identities before accepting a localization tolerance.
- Qualify a portable model/profile on Linux; this checked-in `.mlpackage` is a
  macOS CoreML artifact and does not by itself establish a Linux deployment.
- Measure sustained end-to-end freshness and resource use separately from the
  model-only smoke.
