# Tooling relevance audit — 2026-10-23

This review checked current executable entry points, build and setup targets,
tests, documentation references, dependency manifests, and Rust audit tools.
The same caller, distinct-capability, and meaningful-validation tests were
applied to already-ported Rust tools; implementation language alone did not
count as a reason to keep a utility. Historical performance reports and ADR
rationale remain available as evidence; they are not treated as current
commands or baselines.

## Current tools and disposition

| Area | Current disposition | Reason and limits |
| --- | --- | --- |
| `billiards-server`, `billiards-perception-live`, capture helper, configuration authority, and game-session authority | Keep | These are supported Rust runtime processes or subprocess contracts. |
| Rust camera discovery | Keep | Read-only inventory helps select a camera without opening it; useful operational diagnostic. |
| Rust still-image inference | Keep as diagnostic | Useful to inspect a model and decoder; still images do not establish live capacity, freshness, or accuracy. |
| Rust live performance audit | Keep as canonical | Measures state delivery and optional detector timings with bounded artifacts. |
| Rust recorded-video replay (`billiards-perception-video-replay`) | Keep as canonical, narrowly scoped | PTS-paced latest-frame mode measures detector-side freshness; unpaced mode measures processing capacity. It reuses current Rust inference and optional temporal scoring, but predecodes video and does not represent live camera/decoder contention or browser rendering. |
| Rust temporal label importer/evaluator | Keep as canonical | Imports reviewed video labels and scores temporally aligned state; accuracy evidence still depends on adequate labels. |
| Rust detection-batch receiver and synchronous still-image TCP sender | Retire | The only retained workflow was the retired still-image equivalence evaluator. Current video scoring uses bounded JSONL artifacts; live state receipt uses the scene transport and audit subscriber. The old receiver's tests no longer protect a supported caller. |
| Rust configuration/session one-shot authorities | Keep | They are called by the gateway and have bounded protocols; they are runtime components rather than experiments. |
| `perception/contracts/public-state/generate_fixtures.py` | Keep as migration-only test oracle | Captures the actual Python adapter output used for Rust differential fixtures. It is not a server/runtime tool and should be removed when the compatibility oracle and its fixtures are re-baselined. |
| `camera-nokhwa-build-spike` Cargo feature and CI check | Retire | It compiled a native dependency but had no camera source factory, runtime caller, or capture evidence. The supported capture work proceeds through the supervised GStreamer adapter; the compile-only experiment and CI leg were removed. |
| Python FastAPI profiler/status endpoints | Test-only legacy reference; not operational tools | The FastAPI package is explicitly offline differential-test code and is not installed or runnable as the application server. These endpoints and the Python profiler do not measure the Rust runtime; keep only while Python compatibility tests require them, then remove with that test surface. |
| `backend/vision/calibrate_from_grid.py` and its YAML guide | Retire | Automatic corner detection is a removed stub, manual four-corner output targets the retired Python video module, and the YAML does not configure Rust live perception. The command and workflow guides are removed; Rust live calibration remains a separately versioned bundle. |
| `scripts/performance-audit-*.sh`, `scripts/run-rust-video-audit.sh`, temporal wrappers, deploy build/run/update, MJPEG fixture, readiness watcher, development setup verifier | Keep | Thin shell entry points call supported binaries or perform bounded local development/deployment duties. |
| `tools/export_high_res_model.py` | Keep | Its lineage-bound ONNX/CoreML export workflow has a separate opt-in dependency target. ONNX is used by Rust; CoreML only by the Python compatibility detector. Export alone does not establish parity or accuracy. |
| `tools/merge_yolo_datasets.py` | Keep | Class-name remapping and safe output handling are directly tested; it prepares reviewed data and does not create labels. |
| `tools/dataset_creator.py` | Retired under ADR-0087 | No active caller or focused extraction test existed. Random sampling errors when the requested count exceeds the frame count, hybrid sampling divides by zero for count 1, and timestamp/quality fields were heuristic. The source video and reviewed Rust temporal-label workflow remain authoritative. |
| Python chessboard/table calibration implementation | Keep only as legacy test/reference code | Its behavior has unit coverage, but the retired grid-calibration CLI and its YAML guide were removed. OpenCV YAML does not configure the Rust live calibration bundle. |
| Projector client and UDP test senders | Keep as isolated optional client; exclude from `/live` evidence | This separate LÖVE application receives a custom loopback UDP protocol with no current Rust publisher or WebSocket client. Its tests validate local payload/coordinate behavior only; they do not exercise capture, detection, or browser delivery. ADR-0086 preserves it as a distinct projection use case. |
| `frontend/visualizer` LÖVE trajectory client | Retire | README marked its WebSocket dependency as future work, the required `lib.websocket` module is absent, the initialization test fails, and camera/ball/table rendering are placeholders. It does not serve the primary `/live` workflow. |

## Already-ported Rust executables, evaluated individually

| Executable | Disposition | Distinct role and evidence |
| --- | --- | --- |
| `billiards-server` | Keep: product runtime | Serves the HTML/JS/CSS application, API, and current scene read model; Rust HTTP/WebSocket integration tests exercise the real gateway contracts. |
| `billiards-perception-live` | Keep: product runtime | Owns the live perception process contract and composes capture, inference, calibration, tracking, and audit boundaries; lifecycle and live integration tests cover current behavior. |
| `billiards-perception-gstreamer-probe` | Keep: product capture helper | Despite its name, this is the supervised GStreamer capture helper selected by deployment and the live supervisor. Its helper/source protocols and lifecycle tests cover real subprocess behavior; physical-camera qualification remains a separate gap. |
| `billiards-perception-config` | Keep: runtime authority | The gateway launches it for configuration validation and updates; bounded file/lock behavior and command integration have dedicated tests. |
| `billiards-perception-game-session` | Keep: runtime authority | The gateway launches it for session lifecycle operations; process and lifecycle contracts have dedicated tests. |
| `billiards-perception-camera-discovery` | Keep: read-only diagnostic | Reports enumerated video sources without opening a device. It has bounded output and command tests, and answers a different operational question from capture qualification. |
| `billiards-perception-infer` | Keep: narrow model diagnostic | Runs current Rust inference on PNG/JPEG inputs and can produce raw detection/scene JSONL. It is useful for model-load and pre/post-processing smoke checks, but is not the canonical video freshness or temporal accuracy tool. Its unsupported batch-TCP mode was removed. |
| `billiards-perception-video-replay` | Keep: canonical saved-video detector audit | Reuses production Rust inference with PTS-paced latest-frame and unpaced capacity modes. It reports decoder/model provenance, bounded traces and accuracy when reviewed labels are supplied; it does not measure live camera or browser delay. |
| `billiards-performance-audit-run` | Keep: canonical live delivery audit | Measures live state freshness, throughput, gaps, client receipt and optional trajectory/resource interference using bounded artifacts. Browser rendering remains outside its evidence boundary. |
| `billiards-temporal-evaluate` | Keep: accuracy authority | Scores scene observations against video-bound reviewed annotations, including localization, tracking, cue, and freshness metrics. Unlabeled runs cannot establish accuracy. |
| `billiards-temporal-import-cvat` | Keep: annotation adapter | Imports reviewed CVAT Video 1.1 keyframes into the bounded, video-bound annotation contract; it converts labels and does not author or validate their truth. |

## Remaining shell and data tools, evaluated by caller and output

| Tool | Disposition | Evidence and boundary |
| --- | --- | --- |
| `scripts/deploy/build-dist.sh`, `run.sh`, `update.sh` | Keep | The Make/deployment path builds and launches the current Rust runtime; the local deployment tests reject legacy executables. |
| `scripts/performance-audit-live.sh`, `performance-audit-run.sh`, `run-rust-video-audit.sh` | Keep | Thin wrappers for the canonical Rust live runner, one-window measurement, and recorded-video CLI; no separate measurement logic is duplicated in shell. |
| `scripts/evaluate-temporal-localization.sh`, `import-cvat-temporal-annotations.sh` | Keep | Thin entry points for the Rust evaluator and reviewed-annotation importer. |
| `scripts/serve-test-video-mjpeg.sh` | Keep: test fixture server | Provides an isolated local MJPEG source for capture/replay tests; it does not measure camera exposure or physical capture. |
| `scripts/verify-development-setup.sh`, `watch-rust-readiness.sh` | Keep: development operations | Check required local tooling and wait for explicit Rust readiness respectively; they do not perform accuracy or performance analysis. |
| `tools/merge_yolo_datasets.py` | Keep | The generic class-map merge behavior and safe output handling are directly unit-tested; it does not make or repair labels. |
| `tools/export_high_res_model.py` | Keep | Creates lineage-bound ONNX/CoreML artifacts with focused tests; ONNX is consumed by Rust, while CoreML only serves the Python compatibility detector. |

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
- The unowned video frame dataset creator. Its random and hybrid strategies
  have invalid count edge cases, its frame timing was approximate, its
  heuristic metadata was not ground truth, and no focused extraction test or
  active caller supported retaining it. Keep the tested dataset merger and
  lineage-bound model exporter instead.
- The one-image ball-grid distortion experiment. Its guide reported roughly
  360 px RMS and coefficients at clamp limits, which is not acceptable evidence
  for camera correction or precise ball positions.
- The `nokhwa` compile-only camera spike and CI leg. It tested dependency
  compilation only and was never a capture adapter.
- The Rust detection-batch loopback receiver, still-image TCP producer option,
  and receiver tests. Their only maintained consumer was the retired model
  equivalence workflow; the current video path writes bounded JSONL, and live
  state receipt is covered by the scene transport and performance-audit
  subscriber.
- The Python grid-calibration command and its stale guides. The purported
  automatic detector was a stub, and manual-corner YAML output did not configure
  the Rust live calibration bundle.
- Three executable coordinate/cue/refinement examples and the example-runner
  test file. Their behavior is covered more directly by core geometry,
  coordinate, resolution, and hybrid-refinement tests.
- The incomplete parallel LÖVE trajectory visualizer, its failing initialization
  runner, focused implementation notes, and its Make targets. Keep the distinct
  projector client.
- Obsolete 2024/2025 ball-event, Hough/YOLO tuning, missing-test, table-detector,
  cleanup-plan, and cleanup-verification notes that pointed to removed tools,
  nonexistent tests, the retired Python runtime, or unsupported accuracy claims.
  Current decisions live in ADRs and the maintained performance/tool audit.
- Root `PROMPT.md`, `PROMPT - visualizer.md`, and `PLAN.md`, whose instructions
  described the retired Python/LOVE2D architecture and contradicted the current
  Rust runtime and `/live` browser workflow.
- Dedicated coordinate-scaling implementation notes for the retired visualizer;
  current coordinate contracts stay in the runtime docs and the separate
  projector's maintained protocol tests.
- Python profiler/status endpoints are not active product diagnostics: the
  Python FastAPI app exists only for offline compatibility tests. This pass
  classifies them accurately but leaves them attached to that compatibility
  test surface; they are not evidence for Rust behavior.
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

1. Measure browser/video-viewer interference separately. Current live audit
   receipts do not represent browser rendering or physical display latency.
2. Decide when to remove the remaining Python FastAPI compatibility test
   surface, including its profiler and diagnostic routes; it is not an
   operational backend service or Rust performance tool.
3. Keep the generic `cue`/`ball` detector separate from numbered-ball identity;
   no current model/tool proves 6/7/8-ball classification.

The recorded-video command's real-detector runs are functionality and
development-machine comparison checks, not production baselines: they used one
machine and no reviewed temporal labels. The [paced run](performance-audits/video-replay-smoke-20261023/real-recording/video-replay-report.md)
processed 863 of 1,177 source frames (314 latest-frame replacements), with
21.95 processed observations/s against 29.97 source frames/s and 77.46 ms p95
source-to-inference completion. The [capacity run](performance-audits/video-replay-smoke-20261023/capacity/video-replay-report.md)
processed all frames at 22.22 observations/s. Neither supports an accuracy
claim or a production recommendation; ADR-0084 records scope and the missing
permanent ffmpeg/model end-to-end test.

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
- After removing the orphaned detection-batch TCP path, the Rust all-target
  suite passed again (306 library tests plus all binary and integration targets),
  strict Clippy and formatting passed, and a CLI help test confirms that JSONL
  batch export remains while the retired TCP option is absent. The two focused
  Python camera/color calibration modules also passed all 58 tests with
  coverage disabled for that focused run.
- The dataset frame extractor and its stale guide were retired after confirming
  there were no callers or extraction tests and inspecting its invalid count
  edge cases. Dataset-merge tests remain the verification for the supported
  dataset preparation path.
