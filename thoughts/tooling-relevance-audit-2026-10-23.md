# Tooling relevance audit — 2026-10-23

This review checked current executable entry points, build and setup targets,
tests, documentation references, dependency manifests, and Rust audit tools.
The same caller, distinct-capability, and meaningful-validation tests were
applied to already-ported Rust tools; implementation language alone did not
count as a reason to keep a utility. Historical reports and ADR rationale
remain available as evidence; they are not treated as current commands or
baselines.

## Current tools and disposition

| Area | Current disposition | Reason and limits |
| --- | --- | --- |
| `billiards-server`, `billiards-perception-live`, capture helper, configuration authority, and game-session authority | Keep | These are supported Rust runtime processes or subprocess contracts. |
| Rust camera discovery | Keep | Read-only inventory helps select a camera without opening it; useful operational diagnostic. |
| Rust still-image inference | Keep as diagnostic | Useful to inspect a model and decoder; still images do not establish live capacity, freshness, or accuracy. |
| Rust live performance audit | Keep as canonical | Measures state delivery and optional detector timings with bounded artifacts. |
| Rust temporal label importer/evaluator | Keep as canonical | Imports reviewed video labels and scores temporally aligned state; accuracy evidence still depends on adequate labels. |
| Rust detection-batch receiver | Keep | Records validated replay batches and local receipt evidence; not an asynchronous production transport. |
| Rust configuration/session one-shot authorities | Keep | They are called by the gateway and have bounded protocols; they are runtime components rather than experiments. |
| `scripts/performance-audit-*.sh`, temporal wrappers, deploy build/run/update, MJPEG fixture, readiness watcher, development setup verifier | Keep | Thin shell entry points call supported binaries or perform bounded local development/deployment duties. |
| `tools/export_high_res_model.py` | Keep | Its lineage-bound ONNX/CoreML export workflow has a separate opt-in dependency target. ONNX is used by Rust; CoreML only by the Python compatibility detector. Export alone does not establish parity or accuracy. |
| `tools/merge_yolo_datasets.py` | Keep | Class-name remapping and safe output handling are directly tested; it prepares reviewed data and does not create labels. |
| `tools/dataset_creator.py` | Keep as optional frame extractor | Its quality/edge-case values are heuristics. Timestamps are approximate, and output is candidate imagery only. CLI help was exercised; extraction itself has no focused automated test. |
| Python chessboard/table calibration path | Keep temporarily | It uses the maintained calibration implementation and has direct behavior tests. It remains off-device compatibility tooling; its OpenCV YAML is not a Rust live calibration bundle. |
| Python compatibility service's profiler and audit recorder | Retain only with that service | This in-process instrumentation measures the Python compatibility app, not the Rust runtime. It is historical diagnostic surface, not an approved Rust performance evaluator, and can be deleted when the compatibility service is removed. |
| `frontend/projector` and its UDP/chessboard senders | Keep as a separate client | It is distinct from the `/live` browser application; its senders exercise the projector's local protocol. They do not validate live server-to-browser delivery. |
| `frontend/visualizer` LÖVE trajectory client | Retire | README marked its WebSocket dependency as future work, the required `lib.websocket` module is absent, the initialization test fails, and camera/ball/table rendering are placeholders. It does not serve the primary `/live` workflow. |

## Retired in this pass

- The Python performance collector, standalone detector profiler benchmark,
  still-image checkpoint-equivalence evaluator, their focused CLI/equivalence
  tests, and the Rust adapters whose only caller was the Python collector. This
  includes the timing/resource/trajectory summary adapters, server/delivery/
  overhead reducers, and synthetic capture-transport benchmark; shared Rust
  accumulators, samplers, and receiver code still used by the canonical runner
  remain.
- The duplicate one-window Rust performance-audit CLI was removed because the
  canonical runner already supports a single zero-warm-up measurement window.
- The one-off RGB decoder probe was removed after its diagnostic result was
  recorded; its input path is not used by live GStreamer capture.
- Old training setup, label-fixing, label-validation, and YOLO training tools
  that encoded a 19-class workflow, clamped labels to 16 classes, or ignored the
  dataset manifest. Ball-number classification remains explicitly outside the
  migration.
- The one-image ball-grid distortion experiment. Its guide reported roughly
  360 px RMS and coefficients at clamp limits, which is not acceptable evidence
  for camera correction or precise ball positions.
- Three executable coordinate/cue/refinement examples and the example-runner
  test file. Their behavior is covered more directly by core geometry,
  coordinate, resolution, and hybrid-refinement tests.
- The incomplete parallel LÖVE trajectory visualizer, its failing initialization
  runner, focused implementation notes, and its Make targets. Keep the distinct
  projector client.
- Root `PROMPT.md`, `PROMPT - visualizer.md`, and `PLAN.md`, whose instructions
  described the retired Python/LOVE2D architecture and contradicted the current
  Rust runtime and `/live` browser workflow.
- Dedicated coordinate-scaling implementation notes for the retired visualizer;
  current coordinate contracts stay in the runtime docs and the separate
  projector's maintained protocol tests.
- The unused root and vision-specific pip requirement manifests, an unused
  deployment backup utility, a TPU diagnostic for the old Python detector, and
  redundant deployment/setup aliases.
- Unused graphics, hot-reload, scikit-image, Numba, SQLAlchemy, email-validation,
  and pytest-mock dependencies found in manifests without corresponding imports
  or tests. Scikit-learn was added to the active Python tooling group because
  compatibility color calibration imports it.
- The project now declares Python 3.10 as its minimum because the retained
  ONNX Runtime dependency no longer resolves for Python 3.9. The heavy model
  exporter stack is an explicit extra and install target; dataset helpers keep
  using the smaller base tooling set.
- Obsolete profiler-only dependency groups and Sphinx build dependencies;
  project documentation is Markdown and generated Rust API docs use `cargo doc`.
- The redundant Python test extra; the development extra already includes the
  test runner. Heavy Ultralytics/ONNX/CoreML export packages are isolated behind
  `make install-model-tools` so frame extraction does not install that stack.

## Gaps and follow-up priorities

1. Implement the proposed Rust recorded-video replay path in ADR-0084 so paced
   freshness and unpaced processing capacity can be compared using the same
   saved input and Rust inference pipeline.
2. Measure browser/video-viewer interference separately. Current live audit
   receipts do not represent browser rendering or physical display latency.
3. Decide whether the Python compatibility service remains. If it does, its
   profiler/status route is explicitly compatibility-only; retire it together
   with that service rather than investing in parity with the Rust runner.
4. Qualify or migrate camera intrinsic/distortion calibration before removing
   Python compatibility code. The retained Python calibration files produce
   OpenCV YAML for the old service, and do not directly configure Rust live.
5. Add focused deterministic extraction tests only if `dataset_creator.py`
   remains an actively used annotation workflow. Do not treat its heuristic
   scores or frame timestamps as ground truth.
6. Keep the generic `cue`/`ball` detector separate from numbered-ball identity;
   no current model/tool proves 6/7/8-ball classification.

## Verification performed

- ADR validation: 86 records accepted.
- Python backend tests: 2,952 passed, 2 skipped (the two skips require external
  recorded model inputs).
- Python backend coverage: 81.83%, above the configured 80% gate.
- Rust suite: passed, including unit, binary, and integration targets.
- Browser tests: 70 passed.
- Production build: `make build` passed for the browser client and Rust runtime.
- Model-export utility tests: 7 passed, including lazy optional dependency
  loading and the actionable missing-Ultralytics path.
- Public-state fixture generator: 35 differential cases generated.
- `uv lock --check`, Docker Compose configuration, Make help output, supported
  offline-tool `--help` commands, and shell syntax checks passed.
- Strict Rust Clippy, Make's isolated-target Rust test target, Python lint for
  backend/tools/projector helpers and backend type checks, browser typecheck/tests,
  ADR validation, lockfile check, public fixture generation, Docker Compose
  config, shell syntax, retired-path search, and diff whitespace checks passed
  after pruning. The removed visualizer's
  pre-removal smoke test failed 7/8 at the missing WebSocket dependency, which
  was the evidence for retiring it.
