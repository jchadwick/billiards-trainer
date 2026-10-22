# Static Python PT-to-ONNX comparison

This run checks that exporting the current checkpoint to ONNX preserves the
same static detections under matched Ultralytics inference settings. It is a
descriptive prototype, not a release gate or an accuracy/performance claim.

The source checkpoint is
`billiards-yolov8n.pt` (SHA-256
`f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`). A
temporary fixed-shape ONNX export was produced with Ultralytics 8.4.182,
PyTorch 2.14.1, ONNX Runtime 1.31.0, input size 1280, CPU, batch 1, FP32, and
NMS disabled. Both models received the same decoded OpenCV BGR frames through
Ultralytics with `rect=False`, confidence 0.4 and IoU 0.45. The export used
opset 18 and ran with `CPUExecutionProvider`.

All 28 detections had a class-preserving one-to-one match. The maximum center
difference was 0.000245 pixels, and the maximum confidence difference was
0.00000030. Both models returned no detections on the empty-table image. The
four cases were `multiple_balls.png` (5), `cue_aiming.png` (7), `empty_table.png`
(0), and `clustered_balls.png` (16). Their annotation ball counts were 5, 6, 0,
and 16 respectively; the aiming image also has one cue annotation while this
model emits cue as class 0. The machine-readable record includes per-image
hashes and per-detection errors.

This shows numerical agreement between the two Python inference paths for these
still images. It does not show that their common predictions are physically
accurate. The model has only generic `cue`/`ball` labels, so this check says
nothing about identifying 8-ball versus 6-ball, and the still images cannot
establish temporal continuity, occlusion recovery, motion accuracy, or live
freshness. Input tensors were not captured independently, and no throughput or
latency comparison was performed. The temporary ONNX file was removed after the
run.

See [result.json](result.json) for source, image and export hashes and exact
per-image comparison data. The original export probe is documented in the
[model export characterization](../model-export-characterization-20261020/report.md).
