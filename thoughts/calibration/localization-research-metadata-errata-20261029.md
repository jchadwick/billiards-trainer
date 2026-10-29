# Localization research metadata corrections

The authoritative final revision-9 saved-proposal comparisons are:

- `actual-gpu-r9-canonical-20261029/summary.json`: actual target Intel GPU OpenVINO proposals; 15/15 accepted, including all nine manually referenced balls.
- `actual-onnx-r9-canonical-20261029/summary.json`: actual production Rust CPU ONNX proposals; 15/15 accepted, including all nine manually referenced balls.

Both reports bind the exact source image, corpus, production refiner source, and
frozen Rust harness by SHA-256. Their Markdown counts are generated from the
machine-readable report. Their proposal origin is read from the selected source
record, rather than copied from an earlier backend's narrative. The source
refiner is revision 9, committed as `b8168238`; audit schema is version 6.

Earlier exploratory runners were copied between PyTorch, CPU ONNX, and target
GPU corpora. Some retained a hardcoded Markdown accepted count or a PyTorch-only
provenance limitation. Those inherited descriptions are metadata errors; they do
not change the source-pixel traces, rejection reasons, or JSON result counts.
In particular, the actual-GPU revision-8 comparison accepted **13/15**, not the
15/15 stated by any inherited narrative. The revision-8 CPU ONNX full pipeline
accepted 14/15. The initial PyTorch revision-8 comparison accepted 15/15. These
are different proposal sets and must not be substituted for one another.

Use each earlier report's `summary` and `results` for its numerical outcome and
the explicitly selected corpus/backend evidence for its detector provenance.
Retain earlier artifacts as historical experiments, not final qualification.
The canonical revision-9 reports replace the inherited narrative; they do not
rewrite a revision-8 outcome as a revision-9 success.

The initial `actual-onnx-proposals-20261029/corpus.json` was copied from the GPU
corpus and retained its top-level `proposal_origin` string, even though its
measured-scene boxes and nested status were replaced with the actual CPU ONNX
batch. `corpus-canonical.json` corrects that selected-scene provenance, and
`batch-v1.json` preserves the actual raw production batch used to create it.
The original input is retained so its historical hash remains reproducible.

These are image-localization availability comparisons on frozen saved pixels.
They do not establish physical center truth, independent placements, or
millimeter accuracy. Nine physical references have unknown measurement
uncertainty. Relaxed per-proposal/batch timing budgets are used to complete the
spatial audit; its wall time is not a target-device latency benchmark or the
selected deployment's resource budget.

## Reproduce the canonical comparisons

From the repository root, with no other Cargo task discovering examples:

```sh
mkdir -p perception/examples
cp thoughts/calibration/offline-actual-gpu-r9-canonical-20261029.rs perception/examples/research_gpu_r9.rs
cargo run --release --manifest-path perception/Cargo.toml --example research_gpu_r9
cp thoughts/calibration/offline-actual-onnx-r9-canonical-20261029.rs perception/examples/research_onnx_r9.rs
cargo run --release --manifest-path perception/Cargo.toml --example research_onnx_r9
rm perception/examples/research_gpu_r9.rs perception/examples/research_onnx_r9.rs
```

These commands use frozen proposals and saved pixels. They perform no inference,
network access, or camera capture. They overwrite their corresponding canonical
output folders; preserve a copy before comparing a changed estimator revision.
