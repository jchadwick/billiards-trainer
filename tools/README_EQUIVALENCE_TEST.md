# YOLO model-output equivalence comparison

`test_backend_equivalence.py` compares the retained Python PT model against a
second YOLO model or a saved Rust `billiards-perception-infer` JSON result. Use
it to characterize detector output changes on recorded still images. It does
not compare `VisionModule`, physics, tracking, cue geometry, or the full
camera-to-client path.

This is a descriptive comparison. A close match shows that two providers
produce similar outputs; it does not establish that either output is physically
correct. Ground-truth scoring is optional and limited to annotated still-image
box centers. Real-time freshness requires a separate test.

## Run a comparison

Provide a video, a single image, or a directory of images, along with both model
paths:

```bash
tools/run_equivalence_test.sh \
  --video backend/vision/test_data/multiple_balls.png \
  --reference-model models/billiards-yolov8n.pt \
  --candidate-model /path/to/model.onnx
```

The direct Python command accepts the same options:

```bash
python tools/test_backend_equivalence.py INPUT \
  --reference-model MODEL \
  --candidate-model MODEL \
  --frames 100 \
  --tolerance 2.0 \
  --imgsz 1280 \
  --confidence 0.4 \
  --iou 0.45 \
  --device cpu \
  --output-dir thoughts/performance-audits/equivalence-tests
```

To compare the retained Python checkpoint with a saved Rust recorded-image
result, select `--candidate-result` instead of `--candidate-model` and pass the
same image or image directory used for the Rust run. To score the new
detection-batch v1 artifact, also pass its JSONL file:

```bash
python tools/test_backend_equivalence.py backend/vision/test_data \
  --reference-model models/billiards-yolov8n.pt \
  --candidate-result thoughts/performance-audits/detection-batch-jsonl-smoke-20261020-fixed/audit-report.json \
  --candidate-batches-jsonl thoughts/performance-audits/detection-batch-jsonl-smoke-20261020-fixed/detection-batches.jsonl \
  --frames 100 \
  --tolerance 2.0 \
  --imgsz 1280 \
  --confidence 0.4 \
  --iou 0.45 \
  --output-dir thoughts/performance-audits/equivalence-tests
```

`--candidate-model` and `--candidate-result` are mutually exclusive and one is
required. `--candidate-batches-jsonl` requires `--candidate-result`, because the
aggregate report supplies source paths, encoded-image hashes, model lineage and
settings that do not belong in the frame-scoped batch contract. The JSONL may
come directly from the Rust file sink or from the Rust
`billiards-perception-receive-detection-batches` command after framed loopback
TCP delivery. The tool checks every JSONL line against the shared detection-batch
v1 JSON Schema, then checks sequence ordering, run/session identity, model
hashes, class map, detector settings, timestamp order, source dimensions, and
detection geometry before joining detections to the aggregate report. The
receiver records separate local monotonic receipt times and ordering; this
scorer continues to evaluate detection output and does not subtract timestamps
from different process clock epochs.

The saved-result path accepts only the Rust tool's
`recorded_static_images_unpaced` schema. It rejects video inputs, unsupported
schemas/workloads, different class maps or detector settings, and reference
checkpoints whose SHA-256 does not match the Rust result's recorded source
checkpoint. Every input image must have exactly one Rust result, and every
Rust image entry must map one-to-one to an input with a matching encoded-file
SHA-256. If a path cannot be mapped uniquely, the tool stops with an error
rather than guessing. The shell wrapper currently exposes the model-candidate
mode; use the direct Python command for saved Rust artifacts.

### Optional CVAT still-image scoring

When an image corpus has CVAT image annotations, add `--ground-truth-cvat` to
score both the retained Python reference and the candidate against those labels.
For a saved Rust result, use the exact still-image corpus supplied to Rust:

```bash
python tools/test_backend_equivalence.py thoughts/performance-audits/rust-cvat-ground-truth-20261020/input \
  --reference-model models/billiards-yolov8n.pt \
  --candidate-result thoughts/performance-audits/rust-cvat-ground-truth-20261020/rust-result.json \
  --ground-truth-cvat backend/vision/test_data/annotations.xml \
  --ground-truth-radius-px 20 \
  --frames 9 --imgsz 1280 --confidence 0.4 --iou 0.45 \
  --output-dir thoughts/performance-audits/equivalence-tests
```

The XML must contain image annotations made of boxes whose labels match the
model class names. Every processed image must map unambiguously to one XML
image by exact relative path, unique path suffix, or unique basename, and image
dimensions must agree. Missing or ambiguous matches fail instead of being
silently omitted. CVAT track/video annotations and other shapes are not
scored. The association radius is configurable; it is a pixel-distance rule
for one-to-one class-aware center matching, not a production tolerance.
Reports include overall and per-class TP/FP/FN, precision, recall, F1, and the
error distribution for matched centers. Unmatched labels and detections remain
false negatives and false positives, respectively. Empty annotated images are
included as negative examples.

Scores are image-space measurements against this annotation set. They do not
establish calibrated table-coordinate accuracy, temporal continuity, ball
number identity, or live-camera performance. Review per-frame unmatched
examples and annotation quality before drawing conclusions.

An encoded-file hash proves that both reports refer to the same file bytes; it
does not prove that Rust and Python decoders produce identical pixels. The Rust
prototype uses a different image decoder and its preprocessing parity remains
under evaluation. For pixel-level parity, compare saved preprocessed tensors
separately. This equivalence report does not make a timing, real-time freshness,
ground-truth accuracy, or production qualification claim; Rust timings are
carried only as candidate provenance and are not used in the score.

The defaults use the current detector confidence and NMS IoU values, a square
1280 input, CPU, and the repository's stated 2-pixel center association
tolerance. These settings are recorded with each run. Change them explicitly
when evaluating another model input shape or provider. The `--tolerance` value
only controls which reference/candidate detections count as corresponding; it
does not qualify model accuracy.

The comparison stops with an error if either model cannot load or the class
maps differ. It never falls back to an unrelated detector. With two model
paths, both receive the same OpenCV BGR frame and identical Ultralytics
prediction arguments. With a saved Rust result, the Python reference model is
run on the mapped OpenCV-decoded frame and compared to the detections stored in
that result.

## Reports

Each run writes a versioned JSON record and a concise Markdown summary under
the selected output directory. The JSON records model hashes and package
component hashes, class maps, observed backend/device/provider information,
input hash, adjacent export-manifest content/hash when present, effective
comparison settings, aggregate mismatch counts, center/confidence differences,
and up to `--max-trace-frames` detailed frame records. Frame records identify
the source image by stable relative path (or video frame number). By default,
at most 100 frames are processed and detailed.
Saved Rust-result comparisons also record the candidate JSON's SHA-256, ONNX
and source-checkpoint hashes, requested and observed execution providers, and
Rust workload/schema provenance. The effective image size, confidence, IoU,
and detection cap are retained; the result adapter requires an integer schema
version and verifies the reference checkpoint, class map, settings, complete
source-image coverage, and encoded source hashes before scoring.

Exit status zero means the comparison completed and reports were written. It
does not mean the models passed a production accuracy or performance gate.

## Scope and limits

- The tool compares class-aware detections after Ultralytics decoding and NMS.
- Without `--ground-truth-cvat`, it does not compute precision/recall against
  annotations. With it, the optional score covers CVAT image boxes only, as
  described above.
- It does not evaluate class identity beyond the model's exposed class names.
  A generic `ball` class cannot distinguish numbered balls.
- It does not report performance. Use the bounded production audit for warm-up,
  throughput, latency percentiles, freshness, queueing, loss, and interference.
- The default output limit bounds work and trace size; `--frames` and
  `--max-trace-frames` can be adjusted for a larger investigation.

The machine-readable result is intended as evidence for the retained Python
reference and future Rust/provider adapters, not as a competing benchmark
format.
