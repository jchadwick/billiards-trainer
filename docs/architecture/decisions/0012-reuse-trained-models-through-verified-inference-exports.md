# ADR-0012: Reuse trained models through verified inference exports

- Status: proposed
- Date: 2026-10-19
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The project already has trained PT weights, CoreML artifacts, export/training
tools and labeled inputs. The audited model is a generic ball/cue detector.
Changing weights, preprocessing, backend and language together would obscure
accuracy regressions. The recent MPS label/CPU execution mismatch demonstrates
that configured provider names are insufficient evidence.

## Decision

Preserve source PT weights and existing assets. Extend the existing export
workflow to create explicit versioned derivatives and manifests instead of
creating a second training/export pipeline or retraining for the Rust migration.

Each manifest records source/export hashes, model class map, exporter/runtime
versions, input layout/dtype/shape, RGB/BGR handling, normalization, resizing and
letterbox geometry, output format, NMS location/thresholds and precision.
Existing CoreML artifacts require validated source lineage before treating them
as equivalent to another PT file.

Prove Python PT against the export first, then the identical export in Rust.
Use identical decoded frames and deliberately equivalent preprocessing before
moving to video/camera comparisons. Validate tensor and detection differences
within declared tolerances; do not demand bit equality across precision changes.

Use an inference adapter with ONNX Runtime through Rust bindings as the first
portable candidate. The recorded-image prototype currently pins `ort`
2.0.0-rc.15 and ONNX Runtime 1.31.x; these are prototype dependencies, not an
accepted production binding or packaging strategy. Qualify macOS CoreML and
Linux CPU; Linux acceleration is a separate hardware profile. Requested
provider, actual execution, partition/fallback and effective input properties
must be reported honestly. No speedup is assumed from the language or provider
name.

## Consequences

### Positive

- Reuses model investment and Python experimentation/training.
- Separates export correctness from Rust integration and backend performance.
- Supports platform-specific acceleration behind one observation contract.
- Model/backend failures become reproducible rather than hidden defaults.

### Negative

- Export validation and manifests add maintenance work.
- Supported operators and numerical behavior can differ by backend.
- Native runtime libraries still require packaging; Rust is not automatically
  a dependency-free single-file deployment.
- Initial localization migration does not add reliable numbered-ball recognition.

## Alternatives considered

### Train a new detector during the rewrite

This adds an independent accuracy variable and data cost. Consider new models
only as separately measured improvements after the reference path is understood.

### Load source PT directly in every Rust deployment

A LibTorch adapter may be evaluated if justified, but it is not necessary to
preserve weights and can retain substantial deployment dependencies. Compare
it behind the same interface if exported backends fail qualification.

### Select a newer model, smaller resolution or quantization by default

These may help but change accuracy/precision independently. They need separate
candidate evidence rather than becoming undocumented migration side effects.

## Verification

- Export hashes and lineage reproduce from an explicit source model.
- Golden preprocessing tensors and decoded-frame detections pass equivalence
  tolerances across Python reference, Python export and Rust export paths.
- Both target platforms load the real model and report actual execution placement.
- Held-out labels pass accuracy/completeness checks before performance ranking.
- Warm and sustained runs distinguish initialization from steady-state execution.

## Revisit conditions and related records

Select another adapter/provider if this candidate fails compatibility, accuracy,
packaging or sustained freshness gates. Changing weights, precision or supported
production backend requires an ADR update or successor with new evidence.
ADR-0017 governs external evaluation; ADR-0019 governs numbered identity.


## Evidence and implementation tracking

Status is proposed; Rust migration implementation and qualification are planned,
not completed. This record governs migration stages 2.

- [Migration plan](../../../thoughts/rust-perception-migration-plan.md).
- [Architectural review and evidence limitations](../../../thoughts/architectural-review-2026-10-19.md).
- [Existing backend/vision/detection/yolo_detector.py](../../../backend/vision/detection/yolo_detector.py).
- [Existing tools/export_high_res_model.py](../../../tools/export_high_res_model.py).
- [PyTorch device-placement correction evidence](../../../thoughts/performance-audits/device-placement-20261020/report.md).
- [Model export and asset characterization report](../../../thoughts/performance-audits/model-export-characterization-20261020/report.md) and [machine-readable result](../../../thoughts/performance-audits/model-export-characterization-20261020/result.json).
- [Static PT/ONNX comparison on four labeled stills](../../../thoughts/performance-audits/pt-onnx-static-equivalence-20261020/report.md) and [machine-readable result](../../../thoughts/performance-audits/pt-onnx-static-equivalence-20261020/result.json).
- [Explicit-source exporter tests](../../../backend/tests/unit/test_model_export_tool.py).
- The temporary Python still-image comparison utility and focused tests were
  retired under [ADR-0085](0085-retire-legacy-audit-tooling-in-favor-of-rust-owned-evidence-paths.md).
  Its historical comparison artifacts remain linked above; there is no current
  standalone differential evaluator.
- [Ten-still PT/ONNX comparison JSON](../../../thoughts/performance-audits/pt-onnx-static-equivalence-20261020/tool-run-corpus/equivalence-20261020T042945.248691Z.json) and [summary](../../../thoughts/performance-audits/pt-onnx-static-equivalence-20261020/tool-run-corpus/equivalence-20261020T042945.248691Z.md).

The Stage 0 Python prerequisite now passes the requested device to Ultralytics
for PyTorch inference, reports its observed predictor device separately from the
configured request, and applies the same selection during model reload warm-up.
CPU and MPS placement and static-image detections were checked with the original
PT weights. This does not select or qualify a Rust binding, prove ONNX/CoreML
execution, or complete the export-equivalence decision; this ADR remains proposed
until those stage-specific choices are resolved.

The current explicit-source exporter records lineage and toolchain data and was
used to create a temporary ONNX derivative from the actual PT checkpoint. That
graph passed `onnx.checker` and loaded in ONNX Runtime CPU with a fixed
`[1, 3, 1280, 1280]` input and raw `[1, 6, 33600]` output. On four existing
labeled stills, Python PT and ONNX had equal detection counts and matching class
IDs; maximum center deviation was 0.00025 px. This is a static same-preprocessor
comparison, not ground-truth accuracy or performance evidence. Golden tensor
capture, temporal evaluation, CoreML lineage/runtime, sustained performance, and
Linux qualification remain open; this ADR remains proposed.

The Rust inference prototype now verifies both the ONNX file and the explicitly
provided source checkpoint against the export manifest before loading. Its
recorded-image path reproduced Python PT outputs on ten stills: CPU had 73 of
73 detections paired within 2 px with a maximum center difference of 0.033 px;
macOS auto policy had 73 of 73 paired with a maximum difference of 0.453 px.
Nine PNG preprocessing tensors matched the Ultralytics/OpenCV tensors exactly;
the one JPEG fixture differed because the image decoders differ. A 30-sample
per-provider timing comparison and actual ORT node placement are preserved in
the [Rust inference prototype report](../../../thoughts/performance-audits/rust-inference-prototype-20261020/report.md).
On the tested M4 Max, auto policy partitioned nodes between CoreML and CPU and
reduced the model-only preprocessing/inference/decode p95 to 27.48 ms versus
55.86 ms for CPU-only, while auto initialization averaged 973 ms versus 11 ms.
This is not a CoreML-only profile or an end-to-end freshness measurement.
The decoder was subsequently aligned with Ultralytics' strict confidence test
and pre-clipping NMS order. Adversarial unit cases cover those semantics; a
post-change CPU replay retained the same 73 detections and identical centers
on this corpus. The follow-up output and trace are in the report's evidence
directory.

The accuracy comparison remains inadequate for production: on eight stills,
18 of 52 annotated centers matched within 2 px. Labels are image-plane and may
be incomplete; no temporal or physical-coordinate accuracy was measured.
Correction: the configured `models/billiards-yolov8n.mlpackage` declares
`imgsz: [1280, 1280]` in its Ultralytics export metadata, and the retained
Ultralytics 8.4.182 predictor uses that exported size. The earlier statement
that the effective input size was unknown because the detector call omits
`imgsz` was too broad. The 1280-square comparison therefore matches the
configured package's declared input dimensions. This still does not prove
CoreML-versus-ONNX output parity or exact package lineage: the CoreML package
has no verified checkpoint manifest, and provider-specific preprocessing and
decoding remain distinct. The Rust/provider choices remain proposed. Python's
retained binary labels cannot establish numbered-ball identity.

The retained comparison command was also exercised using a fresh ONNX export
from the same checkpoint. Its versioned JSON captures model/package hashes,
the adjacent exporter manifest and hash, observed `CPUExecutionProvider`,
effective inference settings, stable frame source names, and per-frame output
differences. This is repeatable static provider-equivalence evidence, not an
accuracy or production-performance qualification. See the linked ten-still
run evidence above.

The comparison adapter now also accepts the Rust recorded-static-image result
schema. On all ten repository images it verified the source checkpoint and
class map, inference settings, one-to-one input coverage and encoded-image
hashes, then matched 73 CPU Rust detections against Python PT with no unmatched
outputs. Maximum center difference was 0.0325 px and maximum confidence
difference was 0.000402. The comparison JSON records the ONNX/result/checkpoint
hashes, actual CPU provider, fixed-square settings and detection cap. This
proves reproducible saved-result scoring for this static workload only; it
does not resolve the separate JPEG decoder pixel difference, live-detector
preprocessing mismatch, label accuracy shortfall, temporal freshness or Linux
qualification. See the [Rust-result comparison](../../../thoughts/performance-audits/rust-inference-prototype-20261020/tool-run/equivalence-20261020T123923.419656Z.json)
and [summary](../../../thoughts/performance-audits/rust-inference-prototype-20261020/tool-run/equivalence-20261020T123923.419656Z.md).

A later Linux ARM64 smoke built the same Rust executable in Debian 13 with
Rust 1.92 and ran the hash-verified ONNX export through ONNX Runtime CPU over
the same ten repository stills. All 73 detections matched the retained Python
PT reference with no unmatched outputs and 0.0325 px maximum center
difference. This verifies Linux ARM64 container execution and repeatable
cross-runtime scoring for this static corpus; the container shares the Mac
development machine's hardware and does not qualify physical Linux deployment
or performance. The model's still-label accuracy, live freshness, temporal
behavior, and CoreML package lineage remain unresolved. The recorded
[Linux smoke report](../../../thoughts/performance-audits/linux-perception-smoke-20261020/report.md),
[run metadata](../../../thoughts/performance-audits/linux-perception-smoke-20261020/run-metadata.json),
and [equivalence result](../../../thoughts/performance-audits/linux-perception-smoke-20261020/equivalence/equivalence-20261020T152704.641143Z.md)
retain the environment and output hashes.

A direct decoded-pixel probe then confirmed the JPEG difference originates
before resize: for `grid.jpg`, 695,791 of 8,294,400 pixels differ in at least
one RGB channel between Rust `image-rs` and Python OpenCV, with mean absolute
channel difference 0.06365/255 and maximum 3/255. The [captured hashes and
measurements](../../../thoughts/performance-audits/rust-inference-prototype-20261020/evidence/raw-jpeg-decoder-comparison.json)
are reproducible with the diagnostic example in `perception/README.md`. This
is one JPEG only and does not establish a localization impact; leave decoder
selection open pending a broader image set and Linux qualification.

The reusable Rust `FrameInferenceEngine` now accepts an in-memory RGB image;
recorded replay calls the same preprocessing, ONNX execution and decoding path.
A deterministic ONNX fixture exercises the frame API, while a real-model
ten-still run matched all 73 Python PT detections with no unmatched outputs
and 0.032512 px maximum center difference. Review found that the first run's
measurement-session field included repeated provenance hashing; the corrected
run separates this work and reports 11.591 ms session initialization, compared
with 16.549 ms for the provider probe. This is integration evidence only: it
does not qualify a live source, camera freshness, temporal accuracy, physical
positions, numbered identity or production performance. See the
[corrected frame-engine smoke](../../../thoughts/performance-audits/frame-engine-api-smoke-20261020-r2/report.md)
and its [machine-readable result](../../../thoughts/performance-audits/frame-engine-api-smoke-20261020-r2/run-metadata.json).

The checked-in CoreML package was subsequently loaded and executed through the
retained Python detector in an isolated macOS arm64 environment with Ultralytics
8.4.182 and CoreML Tools 9.0. Its predictor resolved to a 1280-square
float32 `[1, 3, 1280, 1280]` tensor, confidence 0.40, IoU 0.45, batch 1, and
maximum 300 detections. The selected backend is CoreML; its configured compute
unit policy is `CPU_AND_NE`, but runtime partition and actual unit utilization
are not exposed by this measurement. The auto-device selector previously
reported MPS for this `.mlpackage`; it now selects CoreML on macOS, and model
metadata separates backend family from physical device. `backend/requirements.txt`
now installs CoreML Tools only on macOS, matching the platform-specific package
format. On one five-ball annotated image, all five objects were detected, with
mean center error 2.91 px and maximum 5.33 px; only two centers were within
2 px. This still-image sample is not production accuracy evidence or a
performance result. The package exposes only generic `cue` and `ball` labels
and cannot establish numbered-ball identity. See the
[smoke report](../../../thoughts/performance-audits/coreml-production-smoke-20261020/report.md)
and [machine-readable result](../../../thoughts/performance-audits/coreml-production-smoke-20261020/result.json).
On the same PNG, Python CoreML and Rust CPU preprocessing tensors were exactly
equal across all 4,915,200 values. Their five detections paired within 2 px,
with maximum center difference 1.15 px and maximum confidence difference
0.02354. The CoreML package still has no verified source manifest, so this is
not proof of shared weights or export equivalence. See the
[tensor comparison](../../../thoughts/performance-audits/coreml-production-smoke-20261020/tensor-comparison.json)
and [output comparison](../../../thoughts/performance-audits/coreml-production-smoke-20261020/coreml-vs-rust-static-comparison.json).

Before accepting this decision for implementation, resolve its relevant prototype
and compatibility questions and record the evidence. Acceptance records commitment;
it does not claim all verification has passed. Link implementing changes and actual
verification artifacts as they exist. Follow the [decision register](../README.md)
for status changes and supersession.

The Rust library now exposes a reusable `FrameInferenceEngine` that loads the
same hash-verified model and class map as recorded replay, accepts one immutable
RGB frame, and returns detections plus separately measured preprocessing,
inference, and decode/NMS durations. The replay path delegates its prepared
frames to this engine, keeping the harness and future in-memory frame sources
on one implementation. A deterministic ONNX fixture exercises session and
decode behavior. A ten-still real-model smoke matched all 73 detections against
Python PT with zero unmatched and 0.032512 px maximum center difference; mean
CPU inference was 41.014 ms over ten unpaced samples. This is integration
evidence only. It does not qualify a live provider, production accuracy,
camera freshness, sustained performance, or physical Linux hardware. See the
[frame-engine smoke report](../../../thoughts/performance-audits/frame-engine-api-smoke-20261020/report.md)
and [run metadata](../../../thoughts/performance-audits/frame-engine-api-smoke-20261020/run-metadata.json).
