# Model artifacts

The Rust perception runtime is the supported inference path. It loads a fixed
ONNX export whose adjacent manifest binds the artifact hash, input shape, and
source checkpoint hash. The `.pt` checkpoint is provenance for that export; it
is not a runtime inference format.

The active detector has two generic classes: `cue` and `ball`. It does not
identify numbered balls. Do not infer 6/7/8-ball identity from the current
model.

## Current model and export workflow

The current trained checkpoint and exported ONNX artifact are maintained in
the sibling `billiards-model` repository and in recorded audit evidence under
`thoughts/performance-audits/`. Use `tools/export_high_res_model.py` only when
creating a new lineage-bound export; see [`README_MODEL_EXPORT.md`](../../../tools/README_MODEL_EXPORT.md)
for its optional dependencies and checks.

A successful export or a still-image match does not establish localization
accuracy, temporal identity continuity, or live capacity. Use the Rust recorded-
video replay command for same-machine capacity/freshness comparisons, and use
reviewed temporal annotations with the Rust temporal evaluator for accuracy.
See [`docs/PERFORMANCE_AUDIT.md`](../../../docs/PERFORMANCE_AUDIT.md).

## Performance and accuracy evidence

There is no universal FPS or latency promise for a model artifact. Results
depend on the host, provider, input resolution, effective settings, and input
workload. Do not use old Python benchmark instructions or unverified targets as
acceptance criteria. The canonical tools are:

- `scripts/run-rust-video-audit.sh` for bounded PTS-paced freshness or unpaced
  processing-capacity replay.
- `scripts/import-cvat-temporal-annotations.sh` and
  `scripts/evaluate-temporal-localization.sh` for reviewed temporal labels and
  localization/freshness evaluation.
- `scripts/performance-audit-live.sh` for end-to-end live state delivery on a
  configured runtime.

A replay without representative reviewed labels is performance evidence only.
A candidate configuration cannot be recommended for production from throughput
or model-to-model similarity alone.
