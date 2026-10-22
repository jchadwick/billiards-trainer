# Model export characterization

This is an artifact and compatibility check for the existing model assets. It
does not measure detector accuracy, temporal tracking, end-to-end latency, or a
Rust implementation.

## Findings

The `models` link resolves to the sibling checkout
`/Users/jchadwick/repos/jchadwick/billiards-model/v2`, so model contents are not
versioned with this repository. It contains `billiards-yolov8n.pt` and
`billiards-yolov8n.mlpackage`, but no ONNX export or training-run directory.
The PT checkpoint SHA-256 is
`f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`.

Loading the PT checkpoint reports Ultralytics 8.4.182, PyTorch 2.14.1, task
`detect`, stride `[8, 16, 32]`, and exactly two classes: `cue` and `ball`. The
CoreML package metadata also names those two classes, records a 1280 by 1280
input, Ultralytics 8.3.185, CoreML Tools 8.3.0, and PyTorch 2.8.0. Its package
SHA-256 is
`a18334e92e24e6c9e4e59a8c4977e9aa3f33d6ec141834f8ac3c860f57261d92`. No
manifest connects that package to the current PT checkpoint, so they must not
yet be treated as equivalent derivatives.

Repository metadata conflicts with the deployed checkpoint: `models/config.json`
lists two class names but says `num_classes: 3`; the legacy
`backend/vision/models/yolov8n-pool.yaml` describes a different 17-class, 640
by 640 model. The checked-in runtime default points to the external CoreML
package. These metadata files are not reliable evidence for class count or
model lineage. The current deployed detector distinguishes cue from generic
object ball; it cannot support numbered identity such as 6 versus 7 versus 8.

## Export probe

The existing exporter selected the lexically latest training run when invoked
directly and ignored its `output_dir` argument. No such training-run directory
exists beside the available PT weights. The exporter now requires an explicit
source checkpoint, writes to its requested output directory, refuses accidental
overwrite, and emits a versioned manifest with source/artifact hashes, class
names, export options, and tool versions. ONNX simplification is opt-in because
`onnxslim` is absent from the current environment; the exporter does not
silently install it.

An ONNX derivative was exported from the explicit PT checkpoint into a
temporary directory with Ultralytics 8.4.182 and PyTorch 2.14.1, using a static
1280 by 1280 input, batch 1, FP32, CPU export, raw output (NMS disabled), and no
simplification. The temporary output passed `onnx.checker` and opened in ONNX
Runtime 1.31.0 with `CPUExecutionProvider`. It was 12,743,602 bytes with SHA-256
`5bb6bc0b8ada6790544a4db30f88699be6a89d77cb10e5ab84eadd4261f91909`. Its input
was `images: float32[1,3,1280,1280]`; its raw output was
`output0: float32[1,6,33600]`. The temporary export and generated manifest were
removed after inspection; they did not modify the sibling model checkout.

This probe establishes that the current PT checkpoint exports and loads as a
static CPU ONNX graph. It does not prove Python PT/ONNX preprocessing and
detection equivalence. No inference comparison, class-aware accuracy check,
CoreML export, MPS placement for the CoreML package, or Linux run was part of
this initial export probe. A follow-up PT/ONNX comparison was then performed;
see the
[static equivalence report](../pt-onnx-static-equivalence-20261020/report.md).

## Next evidence required

1. Capture golden preprocessing tensors and add a repeatable comparison command
   that records the exact input tensors as well as detection-level differences.
2. Reconcile the three class-count metadata sources and establish the actual
   export lineage for the current CoreML package.
3. Add temporally labeled motion, collision, occlusion, interruption, and
   restart sequences before making identity continuity or live-accuracy claims.
4. Keep export equivalence separate from sustained performance and
   camera-to-client freshness qualification.

Machine and artifact details are in [result.json](result.json). The exporter
regression tests are in
[`backend/tests/unit/test_model_export_tool.py`](../../../backend/tests/unit/test_model_export_tool.py).
