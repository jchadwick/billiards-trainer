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
| `billiards-server`, `billiards-perception-live`, and supervised capture helper | Keep | These are supported Rust runtime processes with lifecycle or integration coverage. |
| Rust configuration-authority CLI | Keep as developer utility | `make config-check` uses the bounded one-shot validator; the running Rust gateway uses `ConfigurationAuthority` in-process, and the CLI has process-level tests. |
| Rust game-session-authority CLI | Keep only as a Python compatibility subprocess | Its only caller is the retained Python compatibility API. The supported Rust gateway uses `GameSessionStore` in-process; the CLI is not part of the supported Rust runtime and lacks a dedicated binary process test. |
| Rust camera discovery | Keep | Read-only inventory helps select a camera without opening it; useful operational diagnostic. |
| Rust still-image inference | Keep as diagnostic | Useful to inspect a model and decoder; still images do not establish live capacity, freshness, or accuracy. |
| Rust live performance audit | Keep as canonical | Measures state delivery and optional detector timings with bounded artifacts. |
| Rust recorded-video replay (`billiards-perception-video-replay`) | Keep as canonical, narrowly scoped | PTS-paced latest-frame mode measures detector-side freshness; unpaced mode measures processing capacity. It reuses current Rust inference and optional temporal scoring, but predecodes video and does not represent live camera/decoder contention or browser rendering. |
| Rust temporal label importer/evaluator | Keep as canonical | Imports reviewed video labels and scores temporally aligned state; accuracy evidence still depends on adequate labels. |
| Rust detection-batch receiver and synchronous still-image TCP sender | Retire | The only retained workflow was the retired still-image equivalence evaluator. Current video scoring uses bounded JSONL artifacts; live state receipt uses the scene transport and audit subscriber. The old receiver's tests no longer protect a supported caller. |
| `perception/contracts/public-state/generate_fixtures.py` | Keep as migration-only test oracle | Captures the actual Python adapter output used for Rust differential fixtures. It is not a server/runtime tool and should be removed when the compatibility oracle and its fixtures are re-baselined. |
| `camera-nokhwa-build-spike` Cargo feature and CI check | Retire | It compiled a native dependency but had no camera source factory, runtime caller, or capture evidence. The supported capture work proceeds through the supervised GStreamer adapter; the compile-only experiment and CI leg were removed. |
| Python FastAPI profiler/status endpoints and fake system diagnostics | Retired | They measured the retired Python vision pipeline or returned fixed diagnostic values, had no supported product caller, and could not characterize the Rust runtime. ADR-0088 removes the routes, hooks, configuration, and instrumentation-only tests. |
| `backend/vision/calibrate_from_grid.py` and its YAML guide | Retire | Automatic corner detection is a removed stub, manual four-corner output targets the retired Python video module, and the YAML does not configure Rust live perception. The command and workflow guides are removed; Rust live calibration remains a separately versioned bundle. |
| `scripts/performance-audit-*.sh`, `scripts/run-rust-video-audit.sh`, temporal wrappers, deploy build/run/update, MJPEG fixture, readiness watcher, development setup verifier | Keep | Thin shell entry points call supported binaries or perform bounded local development/deployment duties. |
| `tools/export_high_res_model.py` | Keep | Its lineage-bound ONNX/CoreML export workflow has a separate opt-in dependency target. Rust needs the ONNX size to match `perception.rust_live.image_size` and verifies model/checkpoint hashes. ONNX is used by Rust; CoreML only by the Python compatibility detector. Export alone does not establish parity or accuracy. |
| `tools/merge_yolo_datasets.py` | Retired under ADR-0089 | It had no consumer outside its tests and guide, sorted class IDs in a way that could invert Rust cue/ball semantics, and allowed class sets rejected by live inference. The repository has no maintained training pipeline to justify preserving it. |
| `tools/dataset_creator.py` | Retired under ADR-0087 | No active caller or focused extraction test existed. Random sampling errors when the requested count exceeds the frame count, hybrid sampling divides by zero for count 1, and timestamp/quality fields were heuristic. The source video and reviewed Rust temporal-label workflow remain authoritative. |
| Python chessboard/table calibration implementation | Keep only as legacy test/reference code | Its behavior has unit coverage, but the retired grid-calibration CLI and its YAML guide were removed. OpenCV YAML does not configure the Rust live calibration bundle. |
| `frontend/web` | Keep as the minimal operator client | ADR-0090 keeps `/live` with direct MJPEG and a state-only reconnecting WebSocket. Legacy configuration, diagnostics, monitoring, navigation, generic video controls, persistence, and their tests were unreachable from the route graph and were removed. |
| Native projector client | Keep as isolated optional client; exclude from `/live` evidence | This separate LÖVE application receives a custom loopback UDP protocol with no current Rust publisher or WebSocket client. Keep only its meaningful Lua coordinate-contract tests; fake Python senders, an unused Docker dev image, unsupported fused-binary instructions, and the unrelated love.js browser package are retired under ADR-0091. ADR-0086 still preserves the native projection use case. |
| `frontend/visualizer` LÖVE trajectory client | Retire | README marked its WebSocket dependency as future work, the required `lib.websocket` module is absent, the initialization test fails, and camera/ball/table rendering are placeholders. It does not serve the primary `/live` workflow. |

## Already-ported Rust executables, evaluated individually

| Executable | Disposition | Distinct role and evidence |
| --- | --- | --- |
| `billiards-server` | Keep: product runtime | Serves the HTML/JS/CSS application, API, and current scene read model; Rust HTTP/WebSocket integration tests exercise the real gateway contracts. |
| `billiards-perception-live` | Keep: product runtime | Owns the live perception process contract and composes capture, inference, calibration, tracking, and audit boundaries; lifecycle and live integration tests cover current behavior. |
| `billiards-perception-gstreamer-probe` | Keep: product capture helper | Despite its name, this is the supervised GStreamer capture helper selected by deployment and the live supervisor. Its helper/source protocols and lifecycle tests cover real subprocess behavior; physical-camera qualification remains a separate gap. |
| `billiards-perception-config` | Keep: developer validation command | `make config-check` invokes it for explicit configuration validation. Runtime configuration reads and updates use the library in-process through the Rust gateway; binary process contracts cover its one-shot protocol. |
| `billiards-perception-game-session` | Keep: Python compatibility adapter only | The Python compatibility route invokes it through `backend/perception/rust_game_session.py`. The Rust gateway uses the session store directly in-process; this binary is not part of the supported runtime and its lifecycle is not separately tested as a process. |
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
| `tools/export_high_res_model.py` | Keep | Creates lineage-bound ONNX/CoreML artifacts with focused tests; ONNX is consumed by Rust, while CoreML only serves the Python compatibility detector. |

## Browser client tools

| Tool or package | Disposition | Evidence and boundary |
| --- | --- | --- |
| Vitest 5, Testing Library, jsdom, TypeScript typecheck, Vite build/serve | Keep | The focused suite covers `/live` freshness, 16-ball slots, direct stream URL handling, coordinate mapping, and state-WebSocket behavior; the built assets are served by Rust. |
| Orphan Playwright diagnostic spec and `@playwright/test` | Retired | It targeted a nonexistent route, had no runner/configuration, and made no assertions about the live client. It did not provide browser E2E evidence. |
| React Query, TanStack devtools, Chart.js, `web-vitals`, Tailwind/PostCSS, and duplicate scripts | Retired | No imports or application consumers remained after the client was reduced to `/live`; explicit CSS replaces the utility framework. The earlier forms-plugin note was stale: no plugin remained in the active config or dependency manifest. |
| Browser E2E against real Rust state plus MJPEG | Gap; not represented by unit tests | The retained tests use controlled stream/socket inputs. Add a focused browser smoke test when an owned local integration fixture exists. |

The locked browser dependency tree reports **zero npm audit vulnerabilities** in
both production and development dependencies. Retiring the vulnerable Tailwind
3 build chain avoided carrying seven development-only advisories through a
major CSS-framework migration.

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
- The unowned video frame dataset creator and generic dataset merger. The
  creator had invalid count edges and heuristic timing; the merger had no
  workflow caller and could silently assign class IDs that conflict with Rust
  runtime semantics. Keep the lineage-bound model exporter and reviewed
  temporal-label workflow instead.
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
- The Python projector UDP message generators and their self-referential tests,
  the combined local launcher with broad process killing, an unreferenced
  Docker dev image, unsupported Linux-only fused-binary instructions, and the
  optional love.js browser packager. The fake sender did not exercise product
  capture, detection, or delivery, and the generated browser artifact was not
  the supported HTML/JavaScript/CSS client. Keep the native projector and its
  Lua coordinate-contract check under ADR-0091.
- Obsolete 2024/2025 ball-event, Hough/YOLO tuning, missing-test, table-detector,
  cleanup-plan, and cleanup-verification notes that pointed to removed tools,
  nonexistent tests, the retired Python runtime, or unsupported accuracy claims.
  Current decisions live in ADRs and the maintained performance/tool audit.
- Root `PROMPT.md`, `PROMPT - visualizer.md`, and `PLAN.md`, whose instructions
  described the retired Python/LOVE2D architecture and contradicted the current
  Rust runtime and `/live` browser workflow.
- Dedicated coordinate-scaling implementation notes for the retired visualizer;
  current coordinate contracts stay in runtime docs and the separate
  projector's Lua coordinate-contract check.
- Python profiler/status endpoints, fake system diagnostics, and the Python
  event collector are removed under ADR-0088. The compatibility API remains an
  offline test reference, but no longer preserves measurement-only routes or
  hooks for the retired Python perception pipeline. Orphan diagnostics models
  and the unproduced performance/metrics WebSocket message types are removed;
  actual Core/WebSocket delivery counters and REST health data remain.
- The unused root and vision-specific pip requirement manifests, an unused
  deployment backup utility, a TPU diagnostic for the old Python detector, and
  redundant deployment/setup aliases.
- Unused graphics, hot-reload, scikit-image, Numba, SQLAlchemy, email-validation,
  and pytest-mock dependencies found in manifests without corresponding imports
  or tests. Scikit-learn was added to the active Python tooling group because
  compatibility color calibration imports it.
- The project now declares Python 3.10 as its minimum because the retained
  ONNX Runtime dependency no longer resolves for Python 3.9. The heavy model
  exporter stack is an explicit extra and install target; the retired dataset
  merger no longer motivates a Python tooling install path.
- Obsolete profiler-only dependency groups and Sphinx build dependencies;
  project documentation is Markdown and generated Rust API docs use `cargo doc`.
- The redundant Python test extra; the development extra already includes the
  test runner. Heavy Ultralytics/ONNX/CoreML export packages are isolated behind
  `make install-model-tools` for the retained model-export workflow.

## Gaps and follow-up priorities

1. Measure browser/video-viewer interference separately. Current live audit
   receipts do not represent browser rendering or physical display latency.
2. Keep the generic `cue`/`ball` detector separate from numbered-ball identity;
   no current model/tool proves 6/7/8-ball classification.
3. The checked-in `config.json` selects the legacy `.mlpackage` under Python
   vision settings, has no explicit `perception.rust_live` ONNX/checkpoint
   paths, and points at a missing `assets/demo3.mp4`; its integer camera ID is
   not a qualified GStreamer source. A valid 1280 ONNX export and matching
   manifest now exist in the external model repository, with two classes
   (`cue=0`, `ball=1`). The local Rust model smoke test loads that export, but
   the product launcher still needs an effective capture/model configuration
   before live startup is testable.
4. The Python calibration-bundle helper has test callers but no live Rust
   product caller. Retire the helper and migrate its tests when the Python
   compatibility capture-settings path is removed; preserve the shared
   calibration schema and Rust-consumed fixture.
5. CoreML export creates an artifact the Rust runtime cannot consume. It is
   retained only for the offline Python detector reference; remove its format
   branch and export dependency when that reference no longer needs CoreML.

The recorded-video command's real-detector runs are functionality and
development-machine comparison checks, not production baselines: they used one
machine and no reviewed temporal labels. The earlier [paced run](performance-audits/video-replay-smoke-20261023/real-recording/video-replay-report.md)
processed 863 of 1,177 source frames (314 latest-frame replacements), with
21.95 processed observations/s against 29.97 source frames/s and 77.46 ms p95
source-to-inference completion. The later [model smoke run](performance-audits/dev-model-smoke-2026-10-23/video-replay-report.md)
processed 853 frames (324 replacements) at 21.70 observations/s against
29.97/s, with 78.35 ms p95 completion. Neither run supports an accuracy claim
or production recommendation. The capacity mode and qualified live capture
path remain untested.

## Verification performed

- ADR validation before ADR-0091: 90 records accepted.
- Python backend tests after retiring the isolated dataset merger: 2,840 passed,
  2 skipped (the skips require external
  recorded model inputs).
- Python backend coverage: 82.55%, above the configured 80% gate.
- Rust `cargo test --locked --all-targets --quiet`: passed, including 306
  library tests and the binary/integration targets.
- Browser tests after the `/live`-only trim: 16 passed across 7 focused files.
- Browser TypeScript typecheck and production Vite build passed; the app now
  retains direct MJPEG, reconnecting state WebSocket, current-state freshness,
  16 fixed ball slots, and coordinate mapping. There is no real-browser E2E
  test against a running Rust process and MJPEG stream.
- Frontend npm audit: zero vulnerabilities for both the full locked dependency
  tree and the production-only dependency tree.
- Production build: `make build` passed for the browser client and Rust runtime.
- Model-export utility tests: 7 passed, including lazy optional dependency
  loading and the actionable missing-Ultralytics path.
- Public-state fixture generator: 35 differential cases generated.
- `uv lock --check`, Docker Compose configuration, Make help output, supported
  offline-tool `--help` commands, and shell syntax checks passed.
- Strict Rust Clippy, Make's isolated-target Rust test target, Python lint for
  backend/tools and backend type checks, browser typecheck/tests,
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
  edge cases. Dataset authoring remains external; reviewed temporal annotations
  are the supported path for accuracy evaluation.
- This pass's backend verification: `make test-python` passed all 2,840 tests
  with the configured coverage gate; Ruff, strict mypy, Black, isort, and ADR
  validation passed. The dedicated shell workflow syntax-checked all 11
  tracked scripts and its YAML parsed successfully.
- The live-only browser client and disconnected dataset-merger retirement
  passed the combined Rust/browser/Python test run. The frontend passed its
  TypeScript check and production build. ADR-0091 adds the projector tooling
  follow-up and must pass the current structure and cross-reference validator.
- The current projector-tooling pass retains only the native LÖVE client and
  Lua coordinate tests. Its Python fake message senders, shell launcher with
  broad process killing, and love.js browser package were removed. The latest
  `make test-all` passed 306 Rust library tests, all Rust binary/integration
  targets, 16 browser tests, and 2,837 Python compatibility tests with two
  model-dependent skips; backend coverage remained 82.55%. The isolated local
  deployment-policy check passed all 10 tests. Lua coordinate checks, Lua
  syntax checks, shell syntax checks, whitespace validation, and the 91-record
  ADR validation passed. The first Rust suite attempt had one intermittent
  ffprobe output-bound assertion failure; the isolated test and full rerun both
  passed.
