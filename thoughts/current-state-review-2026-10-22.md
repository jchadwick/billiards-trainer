# Current state and recommended next milestone

Reviewed 2026-10-22 against the current dirty worktree, source, architecture
records, and available Linux test-machine artifacts. This is a review and
recommended sequence, not a claim that migration or production qualification
is complete. No production defaults were changed for this review.

## Assessment

Keep the existing direction: Rust owns all application-domain behavior, the
Python backend remains the service gateway, and the existing frontend, trained
weights, calibration authoring and evaluation tools are retained. Python game,
physics, prediction, configuration and calibration logic are migration debt,
not permanent owners. The main remaining problem is delivering
fresh, physically accurate locations through that complete application path.
Another language, transport or frontend rewrite is not the next useful step.

The Rust implementation is substantial: verified model inference, runtime
CPU/native OpenVINO selection, calibrated snapshot contracts, association,
bounded latest-value transport, supervised physical capture and diagnostic
export exist. Rust replay reaches the retained API and clients. The API now
allows explicit Rust/live selection behind the Python service. The bundled
Python provider is still the temporary rollback/default during qualification,
not the destination architecture. Rust mode does not start Python Core, Vision, or
integration services. Legacy preview, frame-capture, and direct camera
diagnostic routes now fail closed rather than opening a second Python camera;
a Rust-fed preview and camera-backed calibration workflow still need to be
implemented. This API wiring passed local contract tests and was started against
the Linux target. Its first container-side camera inventory saw no devices;
read-only `/run/udev` and `/sys` mounts made discovery return the same stable
selector as the host. With that corrected, the API still failed before
inference because the capture helper timed out waiting for its first sample.
This is a startup failure, not end-to-end state, accuracy, or performance
qualification.

The latest Linux tests used the available i5-8600T machine. Its results support
relational experiments, not a performance guarantee for the final deployment
hardware. At inspection, no application Docker containers were running on it;
the camera audit containers have finished. This is not a continuously running,
integrated Rust live deployment.

## Evidence that changes the priorities

### Inference is the immediate latency blocker

The illuminated-camera run used 1920×1080 MJPEG at 10 FPS, emitted 1920×1080
RGB8, and ran the real model with 1280-square input through ONNX Runtime CPU.
All 62 completed inference attempts exceeded the configured 100 ms parent-read
freshness limit. No states were published. Audit export was complete, with no
record losses.

| Startup diagnostic | p50 | p95 |
|---|---:|---:|
| Model inference | 267.1 ms | 292.3 ms |
| Preprocessing | 37.6 ms | 38.3 ms |
| Complete inference engine | 304.8 ms | 329.9 ms |
| Parent read completion to inference completion | 317.3 ms | 363.4 ms |

These are nested stage measurements from a failed startup, not an ordinary
warm-up/measured baseline. Do not add their percentiles. Camera exposure time
is unavailable, so even the final row is not sensor-to-client latency.

The interrupted 5 FPS experiment completed. It still ran 62 inferences, all
rejected afterward, plus 24 frames rejected before inference. Completion-age
p95 was 372.5 ms; inference p95 was 284.1 ms. Both 10 and 5 FPS produced zero
states. One short run per mode cannot establish a small speed difference, but
lower FPS clearly did not restore fresh delivery. The advertised 3 FPS mode
timed out before its first sample and produced no inference data. Its cause is
unresolved; do not conclude that the hardware cannot support 3 FPS.

The earlier software-paced replay also found no replacements at 3–3.33 FPS,
but still had completion p95 around 322–324 ms. Latest-only scheduling bounds
backlog; it cannot shorten an inference already in progress. Capture rate,
inference admission rate and preview rate should remain separate controls.

Native OpenVINO already works as a runtime-selected backend on Mac and Linux.
A prior identical-image Linux smoke reduced inference p95 from about 247 to
183 ms, but complete image-to-result p95 improved only from 324 to 296 ms.
That is a useful experimental lead, not a solution to a 100 ms deadline. The
latest physical-camera audit used CPU and did not expose the OpenVINO runtime
inside its container. Reproducible runtime packaging must replace temporary
library staging before a live provider comparison.

### Accuracy remains a first-class missing gate

The nine-still labeled evaluation matched 52/52 balls at a 20 px association
radius, with three extra ball detections and 9.23 px p95 matched-center error.
At a 10 px radius it matched 49 balls with six false positives and three misses.
Cue localization matched zero of three labels at 20 px. This is a small static
corpus, not evidence of motion accuracy or physical precision on this table.

The current model recognizes generic ball and cue classes. It cannot identify
the 6, 7 or 8 ball. Numerical equivalence between Python, Rust and OpenVINO
does not establish that any of them correctly locates the actual balls.

Calibrated millimeter-coordinate machinery exists, but physical error across
the table has not been qualified. Lens distortion, the exact frame geometry,
calibration revision and the distinction between a detected image center and
the physical ball location all matter. The Rust projection is a homography;
it has no explicit ball-height/parallax correction. Measure any position
dependent bias before deciding whether a correction or center-refinement
stage is necessary.

### Preview and calibration still need the Rust-owned source

The Rust provider can now start through `RustLiveSupervisor`, and the service
blocks legacy routes that would create a second Python camera. The application
currently has no Rust-fed preview bridge; these endpoints return 503 in Rust
mode. Calibration capture still depends on the Python camera path and cannot
be used in Rust mode yet.

Preview, reference stills, raw/rectified geometry, calibration/configuration
application and release/restart must all use the same Rust-owned source in
Rust mode. The scene compatibility adapter currently accepts raw image
coordinates because the retained preview requires them. Physical coordinates
are carried separately. Preserve that distinction rather than silently
mixing coordinate spaces to get an overlay working.

## Recommended sequence

1. **Checkpoint the current implementation.** Preserve all uncommitted work;
   review and group it into module-sized changes before another large feature.
   The worktree currently has roughly 147 modified/untracked entries, including
   many tests and audit artifacts. Keep a concise current capability matrix
   alongside the append-only migration history. Coverage is no longer the
   blocker; do not add broad tests merely to raise the percentage.

2. **Define the localization acceptance envelope and collect its labels.**
   Record required position error in millimeters, maximum observation age,
   useful update coverage during motion, false/missing-ball tolerance and
   recovery behavior. The current 100 ms guard is a configured limit, while
   50 ms remains a reference goal. Neither is achieved. Label a small fixed
   corpus from this camera covering the full table, motion, dense layouts,
   occlusion and settled balls; measure held-out physical ball placements.

3. **Find a viable inference profile with a compact accuracy/speed matrix.**
   Package native OpenVINO into the same reproducible runtime as capture.
   Compare CPU/OpenVINO and 1280/960/640 input variants, exporting derivatives
   from the existing weights where needed. Screen cheaply, then interleave
   repeated runs of the best candidates on identical frames and settings.
   Rank ball recall and location error before fresh-result age and throughput.
   Report complete preprocessing-to-result cost, not inference alone. Test
   thread settings and measured preprocessing changes only where justified.
   Do not raise the freshness limit and call the result a speed improvement.

4. **Complete one opt-in camera-to-browser vertical slice.** Add a Rust-fed
   cached newest preview/reference-frame bridge and finish camera-backed
   calibration without creating a second owner. Retain the frontend and
   endpoints. Apply
   calibration/configuration by immutable revision with acknowledgment. Report
   camera/process health separately from fresh-localization readiness, so an
   overloaded detector is diagnosable. Add one real-camera acceptance test
   through the existing browser/receiver, then interruption/restart,
   calibration changes and slow-viewer cases.

5. **Qualify the complete path before changing the default.** Use the planned
   repeated runs, bounded-memory soak, audit overhead comparison and held-out
   physical/temporal accuracy checks. Measure same-machine client receipt on
   comparable clocks and disclose exposure timing limits. Validate Mac/Linux
   compatibility separately from final-hardware performance. Keep latest-only
   replacement, stale rejection, no-auth operation and consumer isolation.

6. **Add secondary capability afterward.** Ball numbering should consume
   bounded crops asynchronously, preserve unknown identity and never delay
   locations. Predictions and training overlays consume the latest scene on
   separate budgets. Evaluate Coral only as a separately versioned quantized
   model/runtime experiment if the simpler runtime/profile work still misses
   the localization envelope.

The next milestone should be: **the existing browser receives fresh, measured
ball locations from one Rust-owned camera, with verified table-position error
and honest stale/unknown status.** It does not depend on ball numbers,
predictions, a Python removal, or replacing TCP. Once bytes have entered TCP,
they cannot be recalled; bounded queues, deadlines, replacement and consumer
expiry provide the required best-effort application semantics.

## Verification status

- Latest backend run after Rust live API wiring: 2,981 passed, two
  model-dependent skips, 80.08% coverage against the unchanged 80% gate. The
  skipped real-model integration checks remain a qualification gap.
- Latest default Rust suite: 136 library tests, seven inference CLI tests,
  six live CLI tests and applicable integration tests passed. The Linux
  physical-source release build succeeded. This is not a claim of current
  full feature-enabled platform qualification.
- Frontend checked during this review: TypeScript passes; 58 tests in 13 files
  pass. These do not establish physical-camera browser latency or accuracy.
- ADR validator: 57 records pass. Scoped Python Ruff, Black, and mypy pass for
  the changed API/provider/settings modules. Full backend tests pass with the
  configured coverage gate after the latest wiring changes.
- The global `git diff --check` still flags extensive whitespace/CRLF issues
  in existing modified source files. Resolve line-ending policy as part of the
  checkpoint; the worktree is not yet a clean, reviewable change set despite
  the passing functional checks. No bulk source formatting was performed by
  this review.

## Evidence

- [10 FPS camera diagnostic](performance-audits/target-physical-camera-qualification-20261022/20261022-040555-efa22995/report.md)
- [5 FPS failure and diagnostic](performance-audits/target-physical-camera-qualification-20261022/20261022-041134-4a04a28e/failure.json)
- [3 FPS source-start failure](performance-audits/target-physical-camera-qualification-20261022/20261022-041120-212da569/failure.json)
- [Software sampling sweep](performance-audits/software-sampling-rate-20261021/sampling-rate-comparison.md)
- [Native OpenVINO comparison](performance-audits/rust-live-trained-model-export-20261021/target-linux-replay-20261021/native-openvino-runtime-20261021/report.md)
- [Migration history](rust-perception-migration-plan.md)
- [Architecture decision register](../docs/architecture/README.md)
