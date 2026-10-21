# Architecture decision register

Architecture decisions are written before dependent implementation, starting with
the migration foundation. This register covers the [Rust perception migration
plan](../../thoughts/rust-perception-migration-plan.md) and the existing accepted
architecture. It is the entry point for finding the rationale, alternatives,
tradeoffs and validation obligations behind consequential choices.

## Status and scope

ADRs 0001–0008 and 0020–0021 are accepted decisions. ADR-0010 is accepted for the
v1 scene-snapshot semantics; ADR-0013 is accepted for the v1 calibrated table
coordinate contract; ADR-0023 is superseded by ADR-0024, which accepts the
shared scene read model with Core-independent localization and calibration-gated
table metadata. ADR-0011 is superseded by ADR-0025, which rejects authentication
for the local perception stream and application APIs while retaining bounded
loopback transport. ADR-0028 supersedes only ADR-0025's handshake epoch
behavior and extends its no-authentication decision across repository-controlled
build and deployment workflows. ADRs 0009, 0012 and 0014–0019
remain **proposals** written before the corresponding Rust migration code.
ADR-0027 is accepted for
post-readiness source-health reporting. ADR-0029 is accepted for source-reported
camera-control readbacks. ADR-0030 is accepted for request-bound live startup
acknowledgements; the contract and opt-in sender/receiver handshake are
implemented, while live-provider and application wiring remain work.
ADR-0031 is accepted for a Rust-owned live capture runtime that composes the
validated request, source preflight, live acknowledgement, frame pipeline, and
bounded shutdown while keeping concrete camera selection separate.
ADR-0032 is accepted for Python-owned Rust live process supervision: Python
passes one exact capture-settings request to the Rust child and validates its
source acknowledgement through the existing ingress; provider selection stays
closed until a real adapter and platform qualification exist. ADR-0033 accepts
a dedicated Rust live executable and injected-source runner, separate from the
recorded-image evaluator. The runner is tested with the real ONNX model and a
deterministic source; the standalone binary still fails closed because no
qualified camera adapter is registered, and production API wiring remains open.
ADR-0036 is accepted for evaluating native camera backends behind a supervised,
killable helper process, with synthetic and opt-in physical GStreamer paths.
The physical path remains a prototype only; it does not register a production
backend. Device cancellation/release, sustained freshness, accuracy, and
macOS/Linux hardware qualification remain open; exact camera-control readbacks
are required only for explicit control requests.
ADR-0037's non-capturing GStreamer DeviceMonitor inventory remains in force;
ADR-0040 supersedes its response identity fields with a versioned stable capture
selector. The size-bounded inventory still does not qualify capture or change
live-provider readiness.
ADR-0038 proposes native camera-control readbacks behind ADR-0036's helper when
explicit controls are requested; generic GStreamer capture may be evaluated
for driver defaults only after identity and negotiated mode are verified.
ADR-0039 accepts a v3 `source_defaults` policy that performs no camera-control
writes, while preserving strict readbacks for explicit `require` requests and
leaving v1/v2 unchanged. The user specified a regular webcam without
per-camera configuration; exact model and Linux host remain unspecified.
ADR-0040 versions stable camera selectors in the discovery response so Linux
provider paths can be passed through the unchanged capture-settings identifier
contract. It retains ADR-0037's non-capturing inventory decision and does not
enable arbitrary/default camera selection.
ADR-0041 accepts a default-off, bounded Rust timing side channel for correlating
live pipeline stage costs with scene receipt. Rust spans/export and the Python
collector are implemented and tested without changing the public scene schema
or adding authentication. A three-run-per-condition development-Mac A/B had
point estimates within the 2% overhead budget and no audit loss, but variation
left the budget qualification inconclusive; target live overhead remains open.
ADR-0042 accepts deterministic eviction of old unmatched tracks under valid
capacity pressure so current measured locations continue to publish. Tracker
and capture-pipeline tests cover the behavior; temporal identity accuracy and
physical-camera pressure remain unqualified.
ADR-0043 records the earlier ONNX Runtime OpenVINO provider experiment and is
superseded by ADR-0045. The `openvino` runtime selection now uses the native
OpenVINO C API through `openvino-rs`, while `cpu` and `auto` retain ONNX Runtime.
The earlier target A/B remains historical provider evidence and does not
qualify the direct backend. Target-native runtime/model parity and Apple
Silicon runtime qualification remain open; labeled accuracy and full-replay
client latency also remain open.
ADR-0044 accepts one detector source setting whose integer or HTTP(S) URI
selects local-device or MJPEG reading, plus a distinct runtime viewer URL for
browser-reachable streaming. Python capture and frontend selection are wired;
the Rust live MJPEG adapter and target camera qualification remain open.
ADR-0022 is proposed for the first
source-time tracking prototype. No new record claims that the prototype,
performance improvement or platform qualification already exists.
The proposal set captures a concrete starting design; unresolved choices must be
settled with the named evidence before the dependent implementation commits to them.

Acceptance means the project commits to a decision, not that implementation or
verification is complete. Record implementation progress and actual evidence
separately. Do not require an additional user confirmation for routine acceptance
within already authorized work; apply the user's direction and the recorded
evidence, and surface material changes in scope or tradeoffs.

## Migration decisions

| ADR | Decision | Status | Required before migration stage | Main acceptance evidence |
|---|---|---|---|---|
| [0009](decisions/0009-rust-perception-engine-behind-the-retained-python-gateway.md) | Rust perception with retained Python gateway, frontend and physics | Proposed | 0–1 | Boundary/API inventory and a real-model vertical slice; independent primary publisher |
| [0010](decisions/0010-immutable-source-timed-scene-snapshots.md) | Immutable complete snapshots, source time, failure and confidence semantics | Accepted | 1 | Shared fixtures round-trip through Rust and validate in Python; clock, receiver-age and transport qualification remain |
| [0011](decisions/0011-bounded-local-transport-for-perception-snapshots.md) | Historical bounded local transport proposal with per-launch auth | Superseded by 0025 | 1 | Keep framing, bounded queue, reconnect, and freshness evidence; no credential contract |
| [0012](decisions/0012-reuse-trained-models-through-verified-inference-exports.md) | Preserve models; explicit export/backend equivalence and actual placement | Proposed | 2 | Manifest lineage and equivalent real-model inference on target platforms |
| [0013](decisions/0013-calibrated-table-positions-with-image-coordinate-compatibility.md) | Physical table coordinates alongside unchanged legacy image semantics | Accepted | 3 qualification | Python/Rust perspective fixture agreement; held-out physical measurements remain required before accuracy claims |
| [0014](decisions/0014-python-owned-revisioned-configuration-and-calibration.md) | Python persistence/authoring with acknowledged Rust applied revisions | Proposed | 1/4/5 | Versioned Python/Rust capture-request contract and persisted config revisions pass; runtime application acknowledgements and raw/rectified calibration reference access remain |
| [0015](decisions/0015-exclusive-camera-ownership-and-owned-frame-buffers.md) | One acquisition owner and owned in-process frames | Proposed | 5 | Device-independent envelope/mailbox semantics are tested; physical camera modes, release/restart/handover and target-platform behavior remain open |
| [0016](decisions/0016-prioritize-localization-and-isolate-optional-consumers.md) | Continuous localization with independently bounded optional work | Proposed | 3 | Dense-motion accuracy, blocked-consumer/interference tests and preserved prediction lifecycle |
| [0017](decisions/0017-retain-python-tools-as-cross-runtime-acceptance-tests.md) | Reuse Python scoring/tools as external runtime acceptance harness | Proposed | 0/2 | Shared real-adapter/black-box tests and versioned accuracy/performance results |
| [0018](decisions/0018-capability-gated-reversible-rollout-on-macos-and-linux.md) | Qualified platform/provider capabilities and reversible cutover | Proposed | 4 onward | Explicit replay/live readiness, target-hardware qualification and exercised rollback |
| [0019](decisions/0019-asynchronous-ball-identity-separate-from-location-tracks.md) | Asynchronous number evidence separate from location and track identity | Proposed | 7 | Held-out identity accuracy/unknown rate, provenance and localization interference checks |
| [0020](decisions/0020-deduplicate-polled-source-observations-before-state-updates.md) | Process each source observation once; preserve freshness across repeated polls | Accepted | 0 | Same-frame replay leaves Core and receipt freshness unchanged; new frame and restarted writer are accepted |
| [0021](decisions/0021-frame-scoped-detection-batches-before-tracked-scene-snapshots.md) | Publish honest per-frame detections before tracked scenes | Accepted | 2–3 | Shared fixtures round-trip; real-model TCP replay reaches the Python receiver and passes PT comparison; paced freshness and live receiver semantics remain open |
| [0022](decisions/0022-immediate-measured-localization-with-gated-track-association.md) | Publish measured centers immediately; use bounded global assignment for track continuity | Proposed | 3 | Deterministic source-time/calibration tests and opt-in paced scene JSONL; model-backed replay, held-out physical/temporal accuracy, switches, reacquisition and freshness remain required |
| [0023](decisions/0023-single-scene-read-model-for-http-and-websocket.md) | Use one current Rust-scene read model for HTTP bootstrap and WebSocket state | Superseded | 4 | Historical decision: existing Python HTTP fixture unchanged and one publisher read model |
| [0024](decisions/0024-core-independent-scene-http-and-calibration-gated-metadata.md) | Serve scene localization without Core and gate table metadata on matching calibration revisions | Accepted | 4 | Scene HTTP works without Core; table alignment is explicit; UTC receipt and monotonic freshness clocks remain distinct |
| [0025](decisions/0025-unauthenticated-loopback-perception-stream.md) | No application or internal-stream authentication; retain bounded loopback IPC | Superseded by 0028 for handshake; no-auth retained | 1/4 | No-authentication choice retained by 0028; original handshake epoch details superseded |
| [0026](decisions/0026-explicit-scene-evidence-in-websocket-state.md) | Preserve anonymous Rust localization and source evidence in the existing state stream schema | Accepted | 4/6 | Scene and Python state fixtures validate; real-socket delivery preserves scene provenance |
| [0027](decisions/0027-explicit-perception-source-health-after-readiness.md) | Report source freshness and process outcome; keep stale HTTP state explicit | Accepted | 4 | Anonymous lifecycle test covers child failure, source health, source silence, and nullable stale frame number |
| [0028](decisions/0028-epoch-scoped-unauthenticated-scene-transport.md) | Scope reconnect watermarks to producer epochs; keep app, IPC, and project automation credential-free | Accepted | 4/5 | Rust/Python tests cover same/new epochs; API, workflow, and local-only deployment tests cover no-auth behavior |
| [0029](decisions/0029-source-reported-camera-control-readback.md) | Separate camera-control readback from requested values and capability declarations | Accepted | 5 | Rust reports preserve applied/unsupported/unknown states; live-readiness preflight rejects mismatched required controls |
| [0030](decisions/0030-request-bound-capture-acknowledgement-for-live-startup.md) | Bind live startup readiness to the exact request and source-session report | Accepted | 4/5 | Rust and Python v3 handshake implemented and tested; live provider, Python process wiring, and hardware qualification remain |
| [0031](decisions/0031-rust-live-capture-runtime-composition.md) | Compose request validation, source ownership, v3 readiness and perception in one Rust runtime | Accepted | 5 | Injected-source lifecycle and handshake tests pass; concrete camera adapter, application wiring, and platform qualification remain |
| [0032](decisions/0032-python-supervisor-validates-rust-live-startup.md) | Python supervises Rust live startup with one request and validates source readiness | Accepted | 5 | Exact request delivery, bounded settings, model/tracking arguments, request-bound v3 acknowledgement, first usable scene and ordered shutdown deadlines; camera adapter, API wiring and platform qualification remain |
| [0033](decisions/0033-separate-rust-live-capture-process-from-replay-evaluator.md) | Keep persistent live capture in a separate Rust executable with an injected source factory | Accepted | 5 | Real-model deterministic-source process test verifies v3 readiness, scene identity and release; a qualified camera adapter and target-device tests remain |
| [0034](decisions/0034-separate-camera-input-mode-from-emitted-rgb-frame-geometry.md) | Keep raw camera mode separate from emitted RGB8 frame geometry in capture-settings v2 | Accepted | 1/5 | Python/Rust v1 compatibility and v2 default/transformed fixtures validate; acknowledgement and per-frame geometry are checked; hardware adapter qualification remains |
| [0035](decisions/0035-version-temporal-localization-evaluation-data.md) | Score versioned temporal labels through one engine-neutral localization evaluator | Accepted | 3/8 | Strict CVAT Video 1.1 importer records explicit review and source-video provenance; Python and Rust adapters share scoring; held-out temporal/physical labels remain required for accuracy claims |
| [0036](decisions/0036-contain-native-camera-capture-in-a-supervised-helper-process.md) | Isolate native capture and use feature-gated GStreamer for physical qualification | Accepted | 5 | Five-second Mac run: 150 unique states at 29.99/s, zero identity/order/gap errors, receipt-interval p95 37.228 ms, no audit loss; test-model accuracy is unmeasured; API selection, Linux hardware, sustained freshness, and production accuracy remain open ([report](../../thoughts/performance-audits/20261021-103042-aeb9f8d8/report.md)) |
| [0037](decisions/0037-non-capturing-cross-platform-camera-discovery-probe.md) | Enumerate camera metadata without opening a device | Superseded by 0040 for response identity fields | 5 | Bounded inventory and no-open semantics remain; selector shape is defined by ADR-0040 |
| [0038](decisions/0038-use-platform-native-camera-control-readbacks-behind-the-capture-helper.md) | Use native control readbacks for explicitly configured controls inside the supervised helper | Proposed | 5 | Generic default-control behavior is handled by ADR-0039; explicit control adapters, value units, and target hardware still need qualification |
| [0039](decisions/0039-represent-driver-default-camera-controls-explicitly-in-capture-settings-v3.md) | Represent no-write OS/driver defaults explicitly while retaining strict optional controls | Accepted | 5 | Cross-runtime v3 request/ack implementation and source-default hardware qualification remain |
| [0040](decisions/0040-version-stable-camera-capture-selectors.md) | Give stable provider identities a bounded selector without changing capture-settings v1/v2/v3 | Accepted | 5 | Schema-v2, exact selector resolution, and Python v3 config bridge pass; discovery UI and physical Mac/Linux qualification remain |
| [0041](decisions/0041-bound-opt-in-live-pipeline-timing-export.md) | Export correlated Rust live-stage timings over a bounded opt-in side channel | Accepted | 5 | Rust spans/export and Python collector tests pass; development-Mac A/B point estimates are within budget but statistically inconclusive; target live overhead remains open ([target replay limits](../../thoughts/performance-audits/rust-live-trained-model-export-20261021/target-linux-replay-20261021/report.md)) |
| [0042](decisions/0042-preserve-live-localization-under-bounded-tracker-pressure.md) | Evict unmatched tracking history under valid capacity pressure and preserve current measured locations | Accepted | 3/5 | Tracker and capture-pipeline tests cover deterministic eviction, degraded state, and continued localization; temporal accuracy and physical-camera pressure remain open |
| [0043](decisions/0043-runtime-selectable-openvino-for-rust-onnx.md) | Earlier ONNX Runtime OpenVINO execution-provider experiment | Superseded by 0045 | Historical | Target provider run and A/B are preserved as historical evidence; they do not qualify the native backend |
| [0044](decisions/0044-infer-camera-source-from-source-value-and-separate-viewer-url.md) | Infer detector input type from `vision.camera.device_id`; keep browser URL independently configurable | Accepted | 5 | Python URI/device routing, runtime viewer selection, config validation, and freshness-bounded queue tests pass; Rust stream adapter and target hardware remain open |
| [0045](decisions/0045-use-the-native-openvino-runtime-backend-for-openvino-selection.md) | Use the native OpenVINO Runtime backend for explicit OpenVINO selection | Accepted | 3/3 | Apple Silicon binary compiles; bounded Linux test-container smoke and 30-frame CPU/native output comparison pass. Host installation, API-container runtime packaging/state delivery, Apple Silicon runtime execution, labeled accuracy, and end-to-end client latency remain open ([native backend report](../../thoughts/performance-audits/rust-live-trained-model-export-20261021/target-linux-replay-20261021/native-openvino-runtime-20261021/report.md)) |

These stage references are dependencies, not evidence that the stages have run.
Backend bindings, camera-specific adapters, precision/resolution profiles and
numeric product tolerances are deliberately conditional on their respective
prototype/measurement gates. When selected, document the choice and rejected
alternatives in the owning ADR or a focused additional record before its use.

ADR-0024 supersedes ADR-0023 only for Core availability, calibration alignment,
and receipt-time semantics. ADR-0023 remains the record of why HTTP and
WebSocket use the same scene read model.

## Existing decisions and migration relationships

| Existing ADR | Relationship |
|---|---|
| [0001](decisions/0001-phase-aware-vision-scheduling-and-prediction-lifecycle.md) | ADR-0016 proposes replacing the motion-time perception policy. Preserve trust/recovery and prediction freeze protections. Do not mark 0001 superseded until replacement is accepted and its retained scope is explicit. |
| [0002](decisions/0002-background-trajectory-prediction.md) | Preserve detached inputs, bounded pending work, lifecycle and shutdown; strengthen absolute freshness through 0010. |
| [0003](decisions/0003-bounded-websocket-delivery.md) | Retain bounded public client delivery. Internal transport 0011 must preserve the same freshness intent. |
| [0004](decisions/0004-validated-config-updates.md) | Retain validation and atomic persistence; 0014 adds the process application/acknowledgement boundary. |
| [0005](decisions/0005-distributed-rate-limiter.md) | Retain admission rules and one table owner. Provider selection is not multi-worker camera ownership. |
| [0006](decisions/0006-api-model-and-error-contracts.md) | Retain transport/domain separation; 0010/0013 add observation evidence and compatibility mapping. |
| [0007](decisions/0007-explicit-coordinate-scaling-contract.md) | Retain legacy canonical-pixel/scale contracts. ADR-0013 introduces physical authority for the new perception model; record any actual scope supersession explicitly. |
| [0008](decisions/0008-opt-in-performance-audit.md) | Retain default-off bounded instrumentation and accuracy-first verification; 0017/0018 require actual backend provenance and credible overhead/cutover evidence. |

ADR-0021 clarifies ADR-0010's Stage 2 dependency: a complete detection batch is
the first Rust-to-receiver location message; the tracked scene snapshot remains
the later boundary after real association is available. Neither changes the
existing public API or frontend contract.

## Required depth for each new decision

Each consequential decision records:

1. Concrete context, user priorities, constraints and evidence, including uncertainty.
2. One precise decision, ownership and the contracts it changes.
3. Credible alternatives and why they were rejected or remain conditional.
4. Positive and negative consequences, costs, failure modes and compatibility.
5. Executable verification requirements and links to actual results when available.
6. Related/superseded records, implementation stage and conditions for revisiting it.

Use separate records for separate consequential choices. Routine local refactors
do not need a record for each function or crate. If implementation requires a
material decision missing here—new camera API, inference provider, process
boundary, transport format, persistence owner, coordinate convention, scheduling
policy or removal of an old capability—write it before implementing that choice.

## Workflow throughout implementation

- At the beginning of each module task, identify its ADRs, evidence and unresolved
  choices. A plan entry alone is not an accepted decision.
- Write/update the proposed record before dependent code. Resolve needed prototype
  questions and commit the chosen decision as accepted when justified. Experiments
  may run under proposed records and must be labeled as experiments.
- Link each implementing change to its ADRs; link the ADR back to tests/results and
  record unverified obligations honestly. Prototype success is not production readiness.
- Preserve historical rationale. When an accepted choice changes materially, create
  a successor; update both records' Supersedes/Superseded by metadata and explain
  retained scope. Proposals do not silently invalidate existing accepted contracts.
- Keep this register's statuses/stages synchronized with records. Record rejection
  rationale when a proposal is abandoned; use a clearly explained deprecated status
  if retaining the record without a selected replacement.
- Validate records and local links in each documentation/architecture change.
  The existing manage-architecture-decisions validator checks the numbered records;
  verify references/status consistency as well, since syntax alone is insufficient.

Keep run reports and exploratory analysis in thoughts/. ADRs in this directory
contain durable decisions with links to that evidence. Do not copy large raw traces
into the decision history or treat hypothetical benefits as measured results.
