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
| Rust game-session-authority CLI | Keep only while the offline Python compatibility fixture is retained | The Python route/reference uses it in compatibility tests; the supported Rust gateway uses `GameSessionStore` in-process. It is not a product runtime command and should be retired with the Python compatibility reference. |
| Rust camera discovery | Keep | Read-only inventory helps select a camera without opening it; useful operational diagnostic. |
| Rust still-image inference and Python API replay mode | Retire | The image CLI had no supported caller or deployed role. The Python replay supervisor launched that removed binary with an incompatible interface, and its fake-child tests concealed the break. Replay-v2 gateway ingress and repeat-count config had no current producer after retirement. Shared engine tests, live tests, and offline saved-video evaluation preserve supported evidence workflows. |
| Rust live performance audit | Keep as canonical | Measures state delivery and optional detector timings with bounded artifacts. |
| Rust recorded-video replay (`billiards-perception-video-replay`) | Keep as canonical, narrowly scoped | PTS-paced latest-frame mode measures detector-side freshness; unpaced mode measures processing capacity. It reuses current Rust inference and optional temporal scoring, but predecodes video and does not represent live camera/decoder contention or browser rendering. |
| Rust temporal label importer/evaluator | Keep as canonical | Imports reviewed video labels and scores temporally aligned state. V1 remains pixel-only; v2 adds independently measured table coordinates with explicit uncertainty and a required millimeter gate. Recorded-video calibration feeds this scorer, but held-out physical accuracy evidence still needs to be collected. |
| Rust detection-batch receiver and synchronous still-image TCP sender | Retire | The only retained workflow was the retired still-image equivalence evaluator. Current video scoring uses bounded JSONL artifacts; live state receipt uses the scene transport and audit subscriber. The old receiver's tests no longer protect a supported caller. |
| `perception/contracts/public-state/generate_fixtures.py` | Keep as migration-only test oracle | Captures the actual Python adapter output used for Rust differential fixtures. It is not a server/runtime tool and should be removed when the compatibility oracle and its fixtures are re-baselined. |
| `camera-nokhwa-build-spike` Cargo feature and CI check | Retire | It compiled a native dependency but had no camera source factory, runtime caller, or capture evidence. The supported capture work proceeds through the supervised GStreamer adapter; the compile-only experiment and CI leg were removed. |
| Python FastAPI profiler/status endpoints and fake system diagnostics | Retired | They measured the retired Python vision pipeline or returned fixed diagnostic values, had no supported product caller, and could not characterize the Rust runtime. ADR-0088 removes the routes, hooks, configuration, and instrumentation-only tests. |
| `backend/vision/calibrate_from_grid.py` and its YAML guide | Retire | Automatic corner detection is a removed stub, manual four-corner output targets the retired Python video module, and the YAML does not configure Rust live perception. The command and workflow guides are removed; Rust live calibration remains a separately versioned bundle. |
| `scripts/performance-audit-*.sh`, `scripts/run-rust-video-audit.sh`, temporal wrappers, deploy build/run/update, MJPEG fixture, development setup verifier | Keep, with distinct boundaries | Thin shell entry points call supported binaries or perform bounded local development/deployment duties. `performance-audit-live.sh` builds release binaries. `serve-test-video-mjpeg.sh` is a manual, one-client network-capture fixture that paces a saved video; it tests the HTTP-MJPEG ingest path but does not simulate camera exposure, driver buffering, or a physical camera. |
| `tools/export_high_res_model.py` | Keep: ONNX only | The lineage-bound ONNX artifact feeds the supported Rust CPU, auto, and OpenVINO paths. CoreML output served only the offline Python compatibility detector and is retired under ADR-0092. Overwrite restores the previous model/manifest pair if either staged promotion fails. The exporter and Rust loader consume the same v1 manifest fixture, and Rust rejects unknown schema versions. Export alone does not establish runtime parity or accuracy. |
| `tools/merge_yolo_datasets.py` | Retired under ADR-0089 | It had no consumer outside its tests and guide, sorted class IDs in a way that could invert Rust cue/ball semantics, and allowed class sets rejected by live inference. The repository has no maintained training pipeline to justify preserving it. |
| `tools/dataset_creator.py` | Retired under ADR-0087 | No active caller or focused extraction test existed. Random sampling errors when the requested count exceeds the frame count, hybrid sampling divides by zero for count 1, and timestamp/quality fields were heuristic. The source video and reviewed Rust temporal-label workflow remain authoritative. |
| Python chessboard/table calibration implementation | Keep within the Python compatibility service for now | It still has callers in the compatibility API, streaming, and video modules. It is not part of the Rust live path; the unused bundle-conversion bridge and tests have been retired. The grid-calibration CLI and YAML guide remain removed. |
| `frontend/web` | Keep as the minimal operator client | ADR-0090 keeps `/live` with direct MJPEG and a state-only reconnecting WebSocket. Legacy configuration, diagnostics, monitoring, navigation, generic video controls, persistence, and their tests were unreachable from the route graph and were removed. |
| Native projector client | Keep as isolated optional client; exclude from `/live` evidence | This separate LÖVE application receives a custom loopback UDP protocol with no current Rust publisher or WebSocket client. Keep only its meaningful Lua coordinate-contract tests; fake Python senders, an unused Docker dev image, unsupported fused-binary instructions, and the unrelated love.js browser package are retired under ADR-0091. ADR-0086 still preserves the native projection use case. |
| `frontend/visualizer` LÖVE trajectory client | Retire | README marked its WebSocket dependency as future work, the required `lib.websocket` module is absent, the initialization test fails, and camera/ball/table rendering are placeholders. It does not serve the primary `/live` workflow. |

## Already-ported Rust executables, evaluated individually

| Executable | Disposition | Distinct role and evidence |
| --- | --- | --- |
| `billiards-server` | Keep: product runtime | Serves the HTML/JS/CSS application, API, and current scene read model; Rust HTTP/WebSocket integration tests exercise the real gateway contracts. |
| `billiards-perception-live` | Keep: product runtime | Owns the live perception process contract and composes capture, inference, calibration, tracking, and audit boundaries; lifecycle and live integration tests cover current behavior. |
| `billiards-perception-gstreamer-probe` | Keep: product capture helper | Despite its name, this is the supervised GStreamer capture helper selected by deployment and the live supervisor. Its helper/source protocols and lifecycle tests now run with the exact `camera-gstreamer-source` feature used by production; physical-camera qualification remains a separate gap. |
| `billiards-perception-config` | Keep: developer validation command | `make config-check` invokes it for explicit configuration validation. Runtime configuration reads and updates use the library in-process through the Rust gateway; binary process contracts cover its one-shot protocol. |
| `billiards-perception-game-session` | Keep only for offline compatibility tests | The old Python route reaches it through `backend/perception/rust_game_session.py`; that route is assembled only by the test fixture. The Rust gateway uses the session store directly in-process, so this binary is not a product runtime dependency. |
| `billiards-perception-camera-discovery` | Keep: read-only diagnostic | Reports enumerated video sources without opening a device. It has bounded output and command tests, and answers a different operational question from capture qualification. |
| `billiards-perception-infer` | Retired in this pass | Its only callers were its source, tests, and documentation. The shared inference engine remains covered by live and video-replay tests. The CLI-specific image directory replay, synthetic pacing, raw tensor dumps, retained provider profile, and replay-v2 TCP bridge did not add a validated workflow. |
| `billiards-perception-video-replay` | Keep: canonical saved-video detector audit | Reuses production Rust inference with PTS-paced latest-frame and unpaced capacity modes. It reports decoder/model provenance, bounded traces and accuracy when reviewed labels are supplied; it does not measure live camera or browser delay. |
| `billiards-performance-audit-run` | Keep: canonical live delivery audit | Measures live state freshness, throughput, gaps, client receipt, optional trajectory receipt collection, and managed detector-process resources using bounded artifacts. Trajectory receipt collection does not enable/disable prediction computation, and browser rendering remains outside its evidence boundary. |
| `billiards-temporal-evaluate` | Keep: accuracy authority | Scores scene observations against video-bound reviewed annotations, including localization, tracking, cue, and freshness metrics. Unlabeled runs cannot establish accuracy. |
| `billiards-temporal-import-cvat` | Keep: annotation adapter | Imports reviewed CVAT Video 1.1 keyframes into the bounded, video-bound annotation contract; it converts labels and does not author or validate their truth. |

## Remaining shell and data tools, evaluated by caller and output

| Tool | Disposition | Evidence and boundary |
| --- | --- | --- |
| `scripts/deploy/build-dist.sh`, `clean-dist.sh`, `run.sh`, `update.sh` | Keep | The Make/deployment path builds and launches the current Rust runtime. Vite now builds inside package staging, and a partial frontend-build failure test verifies the served `backend/static` tree and current package remain intact. Isolated shell tests also cover bundle layout, ownership checks, prior-bundle retention, rollback, cleanup, PID ownership, graceful/forced stop, and systemd restart behavior; the separate Rust process test exercises the real staged server. |
| Docker/Compose image and Make targets | Keep as a separate optional Linux gateway/network-source lane | The image builds the Rust runtime and Compose defaults to gateway-only mode. The service has no physical camera device mapping, and CI builds but does not start the image or exercise its health check. It is not physical-camera qualification or a substitute for native macOS packaging. |
| `scripts/performance-audit-live.sh`, `performance-audit-run.sh`, `run-rust-video-audit.sh` | Keep | Thin wrappers for the canonical Rust live runner, external subscriber, and recorded-video CLI; no measurement logic is duplicated in shell. Both performance-audit entry points and video replay build/run release binaries so the observer does not add debug-build overhead. |
| `scripts/evaluate-temporal-localization.sh`, `import-cvat-temporal-annotations.sh` | Keep | Thin entry points for the Rust evaluator and reviewed-annotation importer. |
| `scripts/serve-test-video-mjpeg.sh` | Keep: manual network-capture fixture | Streams a saved video at its source frame rate over one-client HTTP multipart MJPEG and restarts for a reconnecting client. It can exercise network-source capture and the end-to-end software path, but cannot stand in for physical camera timing or qualify live camera freshness. |
| `scripts/verify-development-setup.sh` | Keep: development setup | Checks required local tooling; it does not duplicate application readiness or detector accuracy checks. |
| `make security-check` | Keep, fail closed | It composes independent Rust and browser dependency checks; `cargo-audit` is required, and a missing Rust audit tool fails instead of reporting a misleading successful partial check. `make test-tooling` covers both its failure and dispatch paths. |
| `backend/README.md` and `backend/requirements.txt` | Retire | The 936-line README described retired Python launchers and workflows; the unbounded requirements file duplicated and diverged from `pyproject.toml`. Current setup instructions use the root runtime guide and the checked-in Python lock. |
| Python module `SPECS.md` files and streaming implementation guide | Retire | Their 2,014 lines prescribed an unimplemented multi-stream Python service, described retired vision commands, and included production claims for numbered-ball recognition. Current API, coordinate, runtime, and ADR documents are authoritative. |
| Root `Makefile` | Keep the focused build/test/run/deploy interface; remove the false `install` alias and broad process kill | `install` only ran `build`, and `kill-processes` used machine-wide `pkill -x`, which could stop another checkout. Local builds now reuse installed browser packages; `make dev` and release packaging still perform locked installs. |
| `scripts/deploy/build-dist.sh` | Keep; lock Rust dependency resolution | It is the package builder. Release builds pass `cargo build --locked`; `make test-deploy` tests package assembly, safe replacement and rollback, frontend failure isolation, cleanup, and updater lifecycle. Tests run through the system Bash path to catch macOS Bash compatibility issues. |
| `.github/workflows/shell-scripts.yml` | Keep and broaden | It syntax-checks every tracked `*.sh` and the active `.githooks/pre-commit`, not only `scripts/**`; deployment and audit smoke tests cover key behavior, but syntax coverage is not full behavioral coverage or ShellCheck. |
| Make targets for Python compatibility tests/lint | Keep as compatibility checks | The user has allowed the Python service/reference to remain for now. Do not mistake its tests for Rust runtime or live-camera qualification. |
| `scripts/watch-rust-readiness.sh` | Retired | It had no callers, while the managed audit runner has a stronger first-fresh-state readiness gate and lifecycle handling. |
| `make test-rust-camera` | Keep: production feature test | Runs the all-target Rust suite with the exact physical GStreamer source feature enabled and uses a separate Cargo target directory. `make test-all` includes it; the ordinary feature-free `make test-rust` remains usable without native GStreamer development libraries. |
| Python dependency installs and `uv.lock` | Keep one lock-backed source | CI and the optional Make install targets sync from `pyproject.toml` using `uv.lock`; compatibility CI omits the heavyweight optional model-export extra, whose tests use fakes and whose real checkpoint workflow remains a separate opt-in. The duplicate, unbounded `backend/requirements.txt` was removed. |

## Browser client tools

| Tool or package | Disposition | Evidence and boundary |
| --- | --- | --- |
| Vitest 5, Testing Library, jsdom, TypeScript typecheck, Vite build/serve | Keep | The focused suite covers `/live` freshness, 16-ball slots, direct stream URL handling, coordinate mapping, and state-WebSocket behavior; the built assets are served by Rust. |
| Orphan Playwright diagnostic spec and `@playwright/test` | Retired | It targeted a nonexistent route, had no runner/configuration, and made no assertions about the live client. It did not provide browser E2E evidence. |
| React Query, TanStack devtools, Chart.js, `web-vitals`, Tailwind/PostCSS, and duplicate scripts | Retired | No imports or application consumers remained after the client was reduced to `/live`; explicit CSS replaces the utility framework. The earlier forms-plugin note was stale: no plugin remained in the active config or dependency manifest. |
| Browser E2E against real Rust state plus MJPEG | Gap; not represented by unit tests | The retained tests use controlled stream/socket inputs. Add a focused browser smoke test when an owned local integration fixture exists. |
| Browser CI | Keep: focused workflow | `.github/workflows/web.yml` triggers on `frontend/web/**` and runs locked dependency install, Vitest, TypeScript typecheck, and production build. It does not claim real-browser or live-camera E2E coverage. |
| `npm run preview` | Retired | No repository workflow called it, and Vite preview served only the static bundle without the Rust API needed by `/live`; keep `npm run dev` for frontend work and `make run` for the integrated application. |

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
- Removed the no-caller Rust-readiness watcher, duplicate geometric-collision
  shell wrapper, and duplicate backend-local Makefile. The root Makefile and
  canonical managed audit now own those entry points.
- Corrected root Python test targets to select their named test directories,
  and removed zero-test pytest shortcuts. The CI workflow no longer points
  Python linters at nonexistent projector Python files or filters out
  unmarked integration/system cases. It runs the retained Python suite once
  with coverage and runs the kept projector Lua coordinate contract directly.
- Added a frontend-only CI workflow so `/live` tests, TypeScript validation, and
  the production build run when browser sources change.
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
- The unreferenced Python video distortion-correction guide. It documented
  inactive settings and unqualified timing values for a retired launcher; the
  Python implementation remains only as offline compatibility code, and the
  supported Rust capture path has its own explicit calibration contract.
- The `nokhwa` compile-only camera spike and CI leg. It tested dependency
  compilation only and was never a capture adapter.
- The Rust detection-batch loopback receiver, still-image TCP producer option,
  and receiver tests. Their only maintained consumer was the retired model
  equivalence workflow; the current video path writes bounded JSONL, and live
  state receipt is covered by the scene transport and performance-audit
  subscriber.
- The Rust `billiards-perception-infer` still-image command and its unique
  synthetic-paced replay path. It duplicated the shared inference engine and
  lacked a current workflow caller; video replay now owns recorded input, while
  the live audit owns state delivery. Still-image tensor dumps, provider-profile
  retention, and replay-v2 TCP scene output were not retained because no
  supported end-to-end workflow depended on them. The shared engine, provider
  selection, calibration, tracking, and JSONL artifact code used by video
  replay remain.
- The Python API's `recorded_replay` mode, its supervisor, fake-child and
  integration tests, repeat-count settings, and stale executable-path
  protection. Those tests never launched the current video-replay CLI, whose
  artifact workflow cannot act as an API publisher. The Rust gateway's unused
  replay-v2 TCP ingress was also removed; live-v3 is its only scene producer
  protocol. ADR-0094 records the decision and retained boundaries.
- The Python compatibility API's `/debug/ball-detection` static-page route.
  Its referenced `backend/static/debug/ball-detection.html` did not exist, the
  supported Rust server never registered the route, and no test asserted it.
  The route and test-fixture registration were removed.
- The Python grid-calibration command and its stale guides. The purported
  automatic detector was a stub, and manual-corner YAML output did not configure
  the Rust live calibration bundle.
- The Python calibration-bundle conversion bridge and its direct tests under
  ADR-0096. Rust owns production solving, persistence, and projection; the
  shared skewed-perspective JSON remains only as a historical geometry fixture
  for Rust and compatibility request-contract tests.
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
- Unlinked Python shared-memory capture and streaming plans, setup guides, and
  distortion experiments. Their `python -m backend.video` instructions point
  to a retired launcher; current capture and direct MJPEG delivery are owned by
  the Rust server. The migration ADRs and this audit retain the useful history.
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
  ONNX Runtime dependency no longer resolves for Python 3.9. The heavy ONNX
  exporter stack is an explicit extra and install target; the retired dataset
  merger no longer motivates a Python tooling install path.
- The model exporter now emits ONNX only. The CoreML package branch had no
  consumer in the supported Rust runtime; its compatibility detector and old
  CoreML evidence remain historical, while `coremltools` was removed from the
  optional model-export dependency group under ADR-0092.
- Obsolete profiler-only dependency groups and Sphinx build dependencies;
  project documentation is Markdown and generated Rust API docs use `cargo doc`.
- The redundant Python test extra; the development extra already includes the
  test runner. Heavy Ultralytics/ONNX/CoreML export packages are isolated behind
  `make install-model-tools` for the retained model-export workflow.
- The unreferenced `frontend/web/Dockerfile.dev`, which had no Make, Compose,
  CI, or documented workflow caller and used an unlocked dependency install.
- The unused `frontend/web` `preview` script, which served built static files
  without the Rust API required by `/live` and had no documented or automated
  caller.
- The orphaned `backend/HEALTH_MONITORING.md`, whose test command referenced a
  deleted file and described the retired Python deployment. Current runtime
  health evidence belongs to the Rust gateway contracts.

## Gaps and follow-up priorities

1. Collect reviewed temporal labels with independently surveyed table-space
   positions, then run the Rust evaluator. The current tests prove evaluator
   semantics but there is no representative truth set and no production
   localization-accuracy result.
2. Add a Docker/Compose runtime smoke if this optional deployment lane is
   retained. CI currently verifies image construction only; there is no
   container startup/API health check, and Compose does not pass through a
   physical camera.
3. Measure browser/video-viewer interference separately. Current live audit
   receipts do not represent browser rendering or physical display latency.
4. Keep the generic `cue`/`ball` detector separate from numbered-ball identity;
   no current model/tool proves 6/7/8-ball classification.
5. The checked-in `config.json` still selects the legacy `.mlpackage` under
   Python vision settings, points at missing `assets/demo3.mp4`, and uses an
   integer camera ID that is not a qualified GStreamer source. It has not been
   qualified as the product launcher config. The Rust live path is now
   end-to-end testable with an explicitly validated Rust-default-derived
   config and local MJPEG input; see the matched live audit reports below. A
   valid 1280 ONNX export and matching manifest exist in the external model
   repository, with two classes (`cue=0`, `ball=1`). Migrating the checked-in
   default config and qualifying its physical source remain open.
6. The offline Python compatibility detector still contains CoreML loading
   behavior, but no maintained exporter produces new CoreML packages. Remove
   that detector branch with the compatibility reference when the Python
   reference is retired; preserve the historical model and results as evidence.
The retired `frontend/projector/love.js` packager is absent from the tracked
repository, but this developer checkout still has an ignored nested Git
repository at that path with local changes. It was not altered during this
review because it is independent checkout data, not a root-repository tool.

## Follow-up validation (2026-10-23)

The retained Rust live audit is no longer merely CLI-tested: the supported
`scripts/performance-audit-live.sh` workflow completed matched three-by-120
second state-delivery runs, with and without nested pipeline timing, on the
development Mac. Unique source-frame update throughput changed from 21.980/s to
22.108/s (+0.58%), while mean p95 read-completion-to-publication age changed
from 78.33 ms to 77.05 ms. The audit-only capture-mailbox metric showed 32.57 ms
p95 queue residence and paired frame data showed only 2.4 microseconds p95
between dequeue and inference start. See
[the matched audit](performance-audits/rust-mailbox-overhead-20261023/comparison.md).

A same-source 20 fps cap reduced unique source-frame skips from 7.86/s to
0.39/s and read-completion-to-publication p95 from 77.05 ms to 67.72 ms, with
19.62 fresh updates/s at the 20 fps cap. This uses a looped HTTP-MJPEG fixture,
not a physical webcam, and supplies no temporal ground truth. See
[the camera-rate comparison](performance-audits/camera-rate-20261023/comparison.md).
The fixture exercises the Rust HTTP-MJPEG helper; it does not establish camera
exposure age, webcam support for an exact 20 fps mode, or Linux target behavior.

The separate [20-minute bounded-memory soak](performance-audits/live-bounded-soak-20261023/comparison.md)
completed 1,200 measured seconds with all state receipts retained and 1,189
available one-second detector-process resource samples. Detector RSS stayed
between 253.4 and 299.5 MiB and ended below its initial sample. The detector
used about 4.6 logical CPU cores at p95 on the development Mac. Ball state was
marked cached in 8,293 of 26,374 messages; this is not an accuracy result and
requires temporal labels. The optional timing trace hit its configured 64 MiB
retention bound and evicted 14,270 records, with no producer or queue drops, so
the long-run stage-distribution section is partial. This run does not qualify
the physical Linux camera, browser interference, or detection accuracy.

The repository acceptance run passed the Rust tests, GStreamer camera-feature
tests, browser tests and typecheck, and retained Python compatibility tests:
2,827 passed, one model-dependent skip, and 82.48% Python coverage against an
80% gate. Strict Rust Clippy, browser typecheck, formatting, and validation of
94 ADRs also passed. The Python API remains as the explicitly retained
compatibility service; retired inference, replay, profiler, training, and
dataset experiments are not required to run or verify the Rust live path.

These results do not close the remaining tool and system gaps: temporal
accuracy labels; a physical Linux webcam and hardware-provider qualification;
viewer interference; and the offline compatibility detector's remaining
CoreML branch. The 20-minute state/memory soak is complete, while its bounded
timing trace is intentionally partial after reaching retention limits. The
shared live-audit and replay CLIs still need hardware-free command-to-artifact
tests; the importer-to-evaluator process workflow now has that coverage.

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
- Model-export utility tests: 6 passed, including lazy optional dependency
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
  deployment-policy check passed all 10 tests. A parallel-load-sensitive Rust
  ffprobe fixture timeout was raised from one to five seconds; its separate
  explicit timeout assertion remains, and `make test-rust` passes all targets.
  Lua coordinate checks, Lua
  syntax checks, shell syntax checks, whitespace validation, and the 91-record
  ADR validation passed. The isolated ffprobe-bound test and subsequent full
  Rust and repository suites passed.
- The follow-up tool sweep removed the duplicate backend `Makefile`, collision
  test wrapper, and unreferenced Rust-readiness watcher. The live audit wrapper
  now builds release executables. CI now covers the retained Lua coordinate
  test, lints only existing Python sources, and avoids rerunning integration
  and system suites before the same tests run again under full coverage. The
  production GStreamer feature is now exercised by both CI platform targets
  and `make test-rust-camera`. Its first run exposed a helper-readiness wait
  shorter than the configured source deadline; the test now derives its wait
  from that configured deadline and passes.
- Verification after that change: the retained Python suite passed 2,837 tests
  with one model-dependent skip and one deselected model-assets case at 82.55%
  coverage; Ruff, Black, isort, and strict mypy passed. The production-feature
  Rust suite passed all targets, the projector Lua contract and syntax checks
  passed, all workflow YAML files parsed, and the live audit wrapper built its
  release tools and returned CLI help successfully. The browser workflow also
  passed locally: `npm ci`, all 16 Vitest tests, TypeScript typecheck, and the
  production build; npm reported zero vulnerabilities.
- The consolidated `make test-all` entry point passed after the cleanup: 306
  default-feature Rust tests, 318 production-camera-feature Rust tests, all 16
  browser tests, and 2,837 Python tests with two model-dependent skips at
  82.55% coverage.
- The expanded tool review retired the exporter’s unused CoreML output path and
  removed `coremltools` from the optional model-export dependency set; the
  offline Python compatibility detector and historical CoreML artifacts remain
  outside the maintained export workflow. The external WebSocket audit wrapper
  now builds and runs its subscriber in release mode to reduce observer
  overhead. The exporter’s six focused tests, Ruff/Black, `uv lock --check`,
  all 92 ADR checks, shell syntax/target guards, and the release subscriber
  `--help` smoke passed. `make test-all` passed 306 Rust tests, 318
  camera-feature Rust tests, 16 browser tests, and 2,836 Python tests with two
  model-dependent skips at 82.55% coverage; `make lint format-check` and
  `make build` also passed. The release CLI smoke does not represent a live
  receipt-latency measurement.
- The second pass removed the no-op `make install` alias and the global process
  killer from `make run`, kept local browser dependency installation out of
  each build, and made release packaging use `cargo build --locked`. Exporter
  examples no longer pass an unsupported `--format` flag or advertise CoreML.
  Shell CI now checks all 13 tracked shell scripts. The focused deployment
  contract passed 10 tests with coverage disabled for that isolated run, the
  browser production build passed, and the corrected exporter `--help` output
  matches its documented options. A focused pytest invocation with the default
  full-suite coverage gate first failed at 0% coverage; rerunning the same tests
  without that unrelated gate passed all 10.
- The retained Python install/test descriptions now match the actual surface:
  compatibility/reference tests and ONNX export, with no bundled dataset
  preparation or supported legacy calibration CLI. Development setup now checks
  for Vite itself, the dependency required by `make build-frontend`, instead of
  accepting any `node_modules` directory. The setup verifier passed on this Mac.
- The physical-accuracy extension keeps the Rust temporal evaluator as the
  single scored-label path. V1 pixel labels and importer output remain
  compatible; v2 requires independently measured physical positions, declared
  table axes/dimensions, a stated 95% uncertainty, and an explicit millimeter
  match gate. Recorded-video replay now applies the existing calibration bundle
  and reports its provenance and calibrated coverage; capacity mode does not
  claim source-to-calibration latency. Schema semantics and gate errors are
  checked before detector startup. Validation passed: 312 Rust library tests,
  all default-feature Rust binary/integration targets, 7 shared Python schema
  tests, strict Clippy, formatting, `git diff --check`, and all 93 ADR checks.
  No held-out independent physical label set exists yet, so no physical
  accuracy or production-readiness claim is supported.

### Final cross-project verification (2026-10-23)

- `make test-all` passed the Rust default and production GStreamer-feature
  suites, the browser suite (16 tests), and the Python compatibility suite
  (2,827 passed, 1 skipped) at 82.48% coverage, above its configured 80% gate.
- `make format-check` and `make lint` passed, including strict Rust Clippy;
  the 94 Architecture Decision Records validate, shell syntax checks pass, and
  `git diff --check` is clean.
- `make build` passed the browser production bundle and Rust server, live
  perception, and supervised GStreamer capture binaries. These checks establish
  build and contract integrity, not physical-camera qualification, detector
  accuracy, or browser-rendering latency.

### Tool inventory follow-up (2026-10-23)

- The actual temporal-import CLI now feeds its generated annotation file into
  the actual temporal-evaluate CLI in a hardware-free process test. It checks
  annotation identity/contents, stdout and report-file agreement, and exact
  matched/false-positive/false-negative counts. Added equivalent process-level
  command-to-artifact tests for the real live-performance-audit and
  recorded-video replay CLIs. The live test checks fresh unique observations,
  report agreement, bounded samples, and fixture shutdown. The replay test runs
  the real tiny ONNX model behind deterministic fake media tools and checks
  input/model hashes, processed-frame counts, output artifacts, and explicit
  accuracy-unavailable labeling. All three targeted CLI workflow tests pass.
- Removed the unused Vite `preview` package script after confirming it served
  only static assets and had no repository caller. Frontend unit tests (16),
  TypeScript typecheck, and the production Vite build passed after removal.
- After the tooling changes, `make test-all` passed 312 default-feature Rust
  tests, 324 production-camera-feature Rust tests, 16 browser tests, and 2,827
  Python compatibility tests with one model-dependent skip at 82.48% coverage.
  `make lint format-check` and `make build` also passed.
- The inventory now includes Docker/Compose as an optional Linux gateway and
  network-source lane. CI builds the image but does not start it, and Compose
  does not map through a physical camera; do not use it as camera qualification.
- The Python model exporter and Rust loader now consume the same small v1
  lineage fixture. Rust validates the manifest schema version, hashes, and
  ordered class map; the fixture is not an inference model. Exporter tests,
  loader tests, the live-application test, and the real-inference replay CLI
  process test passed.
- A separate inventory check confirmed the retired `love.js` packager is not
  tracked or registered by the root repository. An ignored nested checkout
  with local modifications remains on this developer machine and was left
  untouched.
- Tracked the 2,395-byte ONNX fixture already embedded by live Rust tests; it
  was ignored and untracked, making clean-checkout test builds non-reproducible.
  The real-detector live-application integration test passes using the tracked
  fixture.
- Added a staged-bundle `run.sh` process test. It launches the actual compiled
  Rust server with capture disabled, verifies health and the copied web root,
  then checks SIGTERM shutdown and PID-file cleanup. This does not exercise
  package construction, update/rollback, systemd, camera capture, or hardware.
- ADR-0095 supersedes the unimplemented Python trajectory-workload switch in
  ADR-0048. The Rust server has no trajectory producer to isolate, and
  `--include-trajectory` is receiver-only; current performance documentation
  and findings now state that boundary explicitly.
- Final verification including the new deployment process test: `make test-all`
  passed 312 default Rust library tests, 324 GStreamer-feature Rust library
  tests, 16 browser tests, and 2,827 Python compatibility tests with one
  model-dependent skip at 82.48% coverage. The deployment test passed in both
  Rust configurations. `make lint format-check`, `make build`, `git diff
  --check`, and validation of all 95 ADRs passed.

### Final tooling relevance and reproducibility pass (2026-10-23)

- Split dependency audits into independently runnable Rust and browser checks.
  The combined `make security-check` now fails if `cargo-audit` is missing,
  rather than reporting success after only the browser audit. `make test-tooling`
  verifies both missing-auditor failure and Rust audit command dispatch. The
  browser audit reports zero vulnerabilities; the Rust audit could not be run
  on this development machine because `cargo-audit` is not installed.
- Shell CI now checks the active `.githooks/pre-commit` as well as all tracked
  shell scripts. The deployment package and restart smoke tests pass.
- Python installs now use the committed `uv.lock` in Make and both Python CI
  jobs. `backend/requirements.txt` was removed as a duplicate unbounded source;
  the model-export extra remains separate from the lean compatibility/test
  environment. A fresh locked compatibility/test environment passed 2,822 tests
  with one model-dependent skip at 82.48% coverage, Ruff, strict mypy, Black,
  and isort. Model export itself remains a separately installed opt-in path.
- Removed the 936-line backend README and 2,014 lines of stale Python module
  specifications and streaming proposals. They prescribed retired launchers,
  an unimplemented multi-stream server, and numbered-ball behavior not present
  in the product. Current API, coordinate, runtime, and ADR documentation is
  retained as the authority.
- Corrected the recorded-video guide to describe the existing raw-batch and
  scene JSONL options and same-run reviewed-label scoring, instead of proposing
  flags already implemented by the Rust replay tool. Removed a tracked test
  file's formatting drift exposed by the pinned formatter.
- Validation after this pass: Rust all-target and browser tests passed;
  deployment and tooling smoke tests passed; Python compatibility tests and
  focused exporter/schema tests passed; `make lint format-check`, Python lint,
  type checks and formatting checks passed; all five workflow YAML files
  parsed; `uv lock --check`, whitespace checks, and validation of all 97 ADRs
  passed.
