# ADR-0085: Retire legacy audit tooling in favor of Rust-owned evidence paths

- Status: accepted
- Date: 2026-10-23
- Deciders: Project maintainers
- Supersedes: ADR-0082
- Superseded by: None

## Context

ADR-0082 selected shell for thin glue and Rust for substantial audit tools, but
retained the Python collector and detector-profile benchmark as temporary
references. The Rust live runner, timing receiver, resource sampler, temporal
evaluator, and reviewed CVAT importer now own the supported measurement and
label-scoring paths. The Python collector and profile benchmark remain coupled
to the retired Python `VisionModule` service; their documented viewer, event
reducer, and workload commands no longer match supported executables. Several
small Rust reducers and one-shot wrappers only existed to feed that collector.

The same audit found abandoned training helpers that silently encoded a
19-class dataset, clamped labels to 16 classes, or ignored the dataset manifest.
Those behaviors conflict with the current generic `cue`/`ball` runtime model
contract. Preserving stale tools to avoid deleting code would leave misleading
commands and tests that claim coverage of unsupported workflows.

The ball-grid distortion experiment was also not suitable for position-critical
calibration: its own documented sample reports roughly 360 px RMS error and
coefficients clamped to their configured limits. The root and vision-specific
pip requirement lists are unreferenced and overlap the maintained Python
project groups or compatibility requirements.

## Decision

Use the Rust live performance runner as the only supported live state-delivery
audit and use the Rust temporal evaluator/importer for labeled temporal
localization evidence. Retire the Python collector and detector-profile
benchmark, their collector-only tests, and Rust adapter binaries/modules whose
only caller was the retired collector. Preserve the shared Rust receiver,
timing accumulator, resource sampler, trajectory accumulator, current audit
runner, temporal evaluator, temporal annotation importer, and detection-batch
receiver where each still serves a live workflow or test contract.

Remove obsolete profiling-only Python dependencies, duplicated command
wrappers, stale training setup/repair/class-validation tools, the Python
model-equivalence evaluator, its test suite, example-only coordinate demos
whose behavior is covered by direct unit tests, and deployment aliases that
have no supported caller. Retire the single-image ball-grid distortion
experiment because its documented error is unusable for precise localization.
Delete unreferenced root and vision-specific pip requirement lists; retain the
referenced compatibility requirements and Python project groups. Keep offline
dataset frame extraction, dataset merging, the tested chessboard/table
calibration authoring path, and ONNX/CoreML model export as optional Python
tools because they depend on existing OpenCV/Ultralytics artifacts and do not
run in the application server. CoreML output remains only for the Python
compatibility detector; ONNX is the Rust runtime artifact. Keep the Python
compatibility service's optional in-process profiler/status route while that
service remains, but do not treat it as an evaluator for the Rust runtime. Do
not retain standalone Python performance or accuracy evaluators.

Document missing capabilities as gaps rather than implying parity: the current
Rust live runner does not measure browser/video-viewer interference, and there
is no supported Rust recorded-video capacity/freshness replay tool yet. Use
reviewed temporal labels and the Rust evaluator for accuracy when scene
snapshots are available. Evaluate a recorded-video replay as a separate
follow-up under ADR-0084. Do not change live scheduling, latest-frame, or
slow-client contracts as part of this tooling cleanup.

## Consequences

### Positive

- One live audit entry point and one labeled temporal evaluator define current
  evidence semantics; removed commands cannot silently report on the retired
  Python runtime.
- The server runtime remains Rust, while optional model/data authoring tools
  retain the Python libraries required by their upstream formats.
- Test effort follows retained workflows instead of preserving CLI contracts
  for dead tools.
- The unsupported ball-grid estimator can no longer produce high-error lens
  correction files that are mistaken for measured calibration.

### Negative

- Viewer interference and recorded-video capacity/freshness experiments are
  explicitly unavailable until separate Rust tooling is implemented.
- Cross-runtime still-image parity reports remain historical; there is no
  current differential evaluator. The remaining standalone Python tools are
  for dataset/model/calibration authoring, not evaluation. The compatibility
  service's profiler is service instrumentation only and will be retired with
  that service.
- Historical reports and ADRs mention removed tools; those references are
  evidence of past work, not runnable instructions.

## Alternatives considered

### Keep the Python collector as a compatibility reference

Rejected because it measured a retired Python service, its commands referenced
missing modules, and maintaining its tests after the Rust cutover would make
unsupported workflows appear current. Historical reports remain available.

### Port every Python utility to Rust

Rejected because dataset preparation and model export use the existing
Ultralytics/PyTorch authoring stack off-device; duplicating those upstream
formats in Rust does not improve the Rust server or live evidence path. The
still-image equivalence evaluator was also rejected as a maintained tool:
similarity to a source checkpoint is not correctness, and the Rust temporal
evaluator is the accuracy authority. Prior parity reports remain historical.

## Verification

- `cargo test --locked --manifest-path perception/Cargo.toml` passes the Rust
  library, binary, and integration suites; strict Clippy and format checks pass.
- Focused tests pass for maintained model export, dataset merge, and local
  deployment contracts; the retired equivalence suite is no longer collected.
- The performance-audit guide names only current executables and accurately
  states viewer, recorded-video, and accuracy limits.
- Repository searches find no current workflow instructions for the retired
  Python collector, profile benchmark, reducer adapters, dataset repair wizard,
  ball-grid distortion estimator, unreferenced pip lists, example-only demos,
  or removed deployment aliases. Historical evidence and ADR rationale remain
  intact.
- The architecture-decision validator accepts this record and its register
  entry.
