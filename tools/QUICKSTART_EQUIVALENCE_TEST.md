# Quick start: model-output comparison

This utility compares two YOLO model providers on identical image frames. It
does not compare the full backend or measure real-time performance. Optional
CVAT scoring reports image-space accuracy on labeled stills; it does not
qualify a model. Read [the full guide](README_EQUIVALENCE_TEST.md) before using
results to qualify a model.

## Compare a checkpoint with an export

Export a selected checkpoint to a temporary or dedicated directory:

```bash
python tools/export_high_res_model.py \
  --model models/billiards-yolov8n.pt \
  --output-dir /tmp/billiards-model-export \
  --format onnx \
  --imgsz 1280
```

Then compare both providers on the same video, image, or image directory:

```bash
tools/run_equivalence_test.sh \
  --video backend/vision/test_data/multiple_balls.png \
  --reference-model models/billiards-yolov8n.pt \
  --candidate-model /tmp/billiards-model-export/billiards-yolov8n.onnx \
  --frames 100
```

For a single still image, the input path itself is sufficient. The shell option
remains named `--video` for compatibility, but accepts a video, image, or
directory. The Python command accepts the input as its positional argument.

To score a saved Rust recorded-image result with the retained Python reference,
use the direct Python command and the same image corpus used for Rust inference:

```bash
python tools/test_backend_equivalence.py backend/vision/test_data \
  --reference-model models/billiards-yolov8n.pt \
  --candidate-result thoughts/performance-audits/rust-inference-prototype-20261020/evidence/decoder-parity-followup-cpu.json \
  --frames 10 --imgsz 1280 --confidence 0.4 --iou 0.45 \
  --output-dir thoughts/performance-audits/rust-inference-prototype-20261020/tool-run
```

This mode accepts saved PNG/JPEG still-image results only, validates source
hashes and inference settings. It does not measure timing; accuracy is scored
only when the optional CVAT annotation arguments are supplied. The shell
wrapper currently handles model-to-model comparisons; see the [full
guide](README_EQUIVALENCE_TEST.md) for the saved-result contract and limits.

To score image-space accuracy against the existing CVAT boxes, pass the
annotation XML and choose an association radius. The radius controls one-to-one
center matching; it is an audit setting, not a production acceptance target.

```bash
python tools/test_backend_equivalence.py thoughts/performance-audits/rust-cvat-ground-truth-20261020/input \
  --reference-model models/billiards-yolov8n.pt \
  --candidate-result thoughts/performance-audits/rust-cvat-ground-truth-20261020/rust-result.json \
  --ground-truth-cvat backend/vision/test_data/annotations.xml \
  --ground-truth-radius-px 20 \
  --frames 9 --imgsz 1280 --confidence 0.4 --iou 0.45 \
  --output-dir thoughts/performance-audits/equivalence-tests
```

CVAT images must map unambiguously to all processed stills and use the model's
class names. The report separates per-class and overall detections and includes
matched center-error statistics. These labels do not cover temporal behavior,
calibrated table coordinates, or numbered-ball identity.

## Read the reports

The command prints a short summary and writes a JSON evidence record plus a
Markdown summary under `thoughts/performance-audits/equivalence-tests/` by
default. Specify `--output-dir` to use another location. Review unmatched
detections and the center/confidence differences in JSON. A successful exit
only means the comparison completed and reports were saved; it is not a model
acceptance verdict.

Use `--help` on either command for supported options. The Python comparison
requires an explicit reference checkpoint and exactly one candidate model or
saved Rust result. Model mode fails if either model cannot load or class maps
differ. Its default 2-pixel tolerance only associates corresponding detections
and does not define acceptable localization error.
