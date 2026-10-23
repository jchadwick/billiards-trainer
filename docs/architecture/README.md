# Architecture decision register

Architecture decisions are written before dependent implementation, starting with
the migration foundation. This register covers the [Rust perception migration
plan](../../thoughts/rust-perception-migration-plan.md) and the existing accepted
architecture. It is the entry point for finding the rationale, alternatives,
tradeoffs and validation obligations behind consequential choices.

## Status and scope

ADRs 0001–0004, 0006–0008 and 0020–0021 are accepted decisions. ADR-0005 is
superseded for the supported Rust runtime by ADR-0081; its offline Python
reference behavior remains. ADR-0010 is accepted for the
v1 scene-snapshot semantics; ADR-0013 is accepted for the v1 calibrated table
coordinate contract; ADR-0023 is superseded by ADR-0024, which accepts the
shared scene read model with Core-independent localization and calibration-gated
table metadata. ADR-0011 is superseded by ADR-0025, which rejects authentication
for the local perception stream and application APIs while retaining bounded
loopback transport. ADR-0028 supersedes only ADR-0025's handshake epoch
behavior and extends its no-authentication decision across repository-controlled
build and deployment workflows. ADR-0066 sets the destination: all deployed
server-side components run in Rust, while the browser application remains
HTML, JavaScript, and CSS. The Python service is a migration bridge. Model
training, dataset preparation, and other annotation workflows remain
off-device. The retained Python FastAPI package is an offline compatibility
test reference, not a running migration bridge. The reviewed temporal-label
importer is Rust under ADR-0083. ADR-0085
supersedes ADR-0082's temporary Python collector-retention requirement while
preserving its shell-for-glue and Rust-for-complex-tooling decision. The Rust
live runner owns state and optional trajectory receipt audits, detector timing,
and optional process-resource sampling; the Rust temporal evaluator owns
reviewed-label scoring. The retired Python collector/profile benchmark,
still-image equivalence evaluator, and collector-only Rust adapters are
historical. ADR-0084 implements bounded recorded-video capacity/freshness
replay; browser/video-viewer interference and representative temporal accuracy
evidence remain open.
Optional dataset preparation, camera calibration authoring, and model export
remain off-device. Standalone Python accuracy/performance evaluators are
retired. Python profiler/status endpoints remain only in the offline test
reference and do not audit the Rust runtime. ADR-0086 also retires the
disconnected LÖVE trajectory visualizer;
the supported browser client remains the `/live` application and the separate
projector client remains available.
ADR-0064 is superseded because it selected a Rust/WASM browser
client and permanent Python gateway. ADR-0065 assigns unified configuration
authority to Rust and preserves current route contracts during the gateway
transition. ADR-0067 assigns game-session metadata and reset persistence to an
isolated Rust store while preserving scene locations. ADR-0070 moves
configuration reads directly onto the Rust gateway with bounded authority
work and the existing browser response contract. ADR-0071 moves configuration
updates to Rust while preserving validation, revision and executable-path
protection semantics. ADR-0072 moves reset to the same gateway while keeping
the legacy all/user/section behaviors. ADR-0073 moves configuration import,
export and download to the Rust gateway with bounded multipart and output
handling. The browser remains HTML/JavaScript/CSS; no Rust/WASM client is
planned. ADR-0074 serves game-session metadata and reset directly from the Rust
gateway while keeping metadata failures separate from ball localization.
Ball-number classification remains excluded because it is nonfunctional.
ADR-0080 makes the Rust gateway the only supported application server entry
point across development and deployment; Python API modules remain available
only as offline compatibility references and test fixtures.
ADR-0081 removes the obsolete Python/Nginx deployment path and makes direct
Rust delivery the only supported HTTP, WebSocket, and static-client topology.
The Rust gateway retains global bounded WebSocket admission and delivery but
does not currently enforce the legacy per-IP attempt quota; revisit that bound
before expanding beyond the trusted local-network deployment assumption.
ADR-0005's Python rate-limiter design remains historical for the offline
reference suite and is superseded for supported-runtime enforcement by ADR-0081.
ADR-0076 extends replay-v2 to accept the versioned Rust public-state envelope
while preserving raw snapshot compatibility and receiver-local freshness.
ADRs 0009 and 0014
are superseded for permanent domain ownership;
ADR-0012 and 0015–0019 remain **proposals** written before the corresponding
Rust migration code.
ADR-0027 is accepted for
post-readiness source-health reporting. ADR-0029 is accepted for source-reported
camera-control readbacks. ADR-0030 is accepted for request-bound live startup
acknowledgements; the contract and opt-in sender/receiver handshake are
implemented, while live-provider and application wiring remain work.
ADR-0031 is accepted for a Rust-owned live capture runtime that composes the
validated request, source preflight, live acknowledgement, frame pipeline, and
bounded shutdown while keeping concrete camera selection separate.
ADR-0032's process-contract decisions remain in force under its partial
supersession by ADR-0057: the Rust configuration authority builds one exact
capture-settings request, which the Python supervisor passes to the Rust child
and binds to its source acknowledgement. ADR-0057 allows explicit Rust live
selection for qualification behind the service, with Python still the bundled
default and platform/accuracy qualification still required. ADR-0058 moves
live public-state conversion and freshness lifecycle into Rust while
preserving the raw scene contract.
ADR-0061 extends that lifecycle to paced Rust replay while preserving raw JSONL;
the configured Python provider remains present as the reference/default path.
ADR-0059 assigns table-plane solving to Rust with exact capture provenance,
and ADR-0060 assigns persistence to a revisioned Rust store. ADR-0062 defines a
bounded Rust-owned current-frame preview for provenance-bearing calibration;
ADR-0063 supersedes its latest-frame submission rule and validates stable active
capture context so frozen images remain usable as capture advances. Preview UI
integration now uses the Rust frame and maps source pixels; Rust calibration
submission, store-to-live activation and physical accuracy qualification remain
open. ADR-0033 accepts
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
contract. ADR-0054 defined Linux resolution through stable V4L symlinks, and
ADR-0075 adds the exact-node sysfs/udev fallback for containers that do not
mount those links. The selector schema, hash, and exact-match rules remain from
ADR-0040. Neither decision enables arbitrary/default camera selection.
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
qualify the direct backend. A bounded Apple Silicon runtime/model smoke now
passes and its static CPU comparison has matching detection counts; labeled
accuracy and full-replay client latency remain open.
ADR-0044 accepts one detector source setting whose integer or HTTP(S) URI
selects local-device or MJPEG reading, plus a distinct runtime viewer URL for
browser-reachable streaming. Python capture and frontend selection are wired;
the Rust live MJPEG adapter and target camera qualification remain open.
ADR-0045 accepts the direct OpenVINO Runtime backend for explicit selection;
bounded Apple Silicon and Linux runtime/model comparisons now pass, while
API-container state delivery is smoke-tested but below the requested 5 Hz;
server-stage telemetry, labeled temporal accuracy, and sustained client
freshness remain open. ADR-0046 accepts process-local capture timing
segments for the supervised camera helper without claiming exposure timing.
ADRs 0047 and 0048 accept receiver-side scene-transport measurement and an
audit-only trajectory-workload control. ADR-0049 is superseded by ADR-0050,
which accepts normalized Python/Rust audit evidence and fresh-observation
counting. ADR-0051 accepts bounded repeat counts for paced replay; Rust/Python
tests and a short repeated-corpus API replay smoke pass, and repeated imagery
remains load evidence rather than accuracy evidence. ADR-0052 accepts
latest-value, best-effort semantics for perception state across the pipeline.
Browser WebSocket and local TCP remain transport choices, not a promise to
deliver every frame; single-slot live-frame admission, stale-before-Core
rejection, state-event coalescing, client expiry, and source-order guards are
implemented and regression-tested. Target freshness, physical-camera age, and
location accuracy remain to be qualified.
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
| [0009](decisions/0009-rust-perception-engine-behind-the-retained-python-gateway.md) | Historical retained Python gateway proposal | Superseded by 0056 | 0–1 | Retain snapshot/publisher/compatibility invariants; permanent deployed ownership moves to Rust |
| [0010](decisions/0010-immutable-source-timed-scene-snapshots.md) | Immutable complete snapshots, source time, failure and confidence semantics | Accepted | 1 | Shared fixtures round-trip through Rust and validate in Python; clock, receiver-age and transport qualification remain |
| [0011](decisions/0011-bounded-local-transport-for-perception-snapshots.md) | Historical bounded local transport proposal with per-launch auth | Superseded by 0025 | 1 | Keep framing, bounded queue, reconnect, and freshness evidence; no credential contract |
| [0012](decisions/0012-reuse-trained-models-through-verified-inference-exports.md) | Preserve models; explicit export/backend equivalence and actual placement | Proposed | 2 | Manifest lineage and equivalent real-model inference on target platforms |
| [0013](decisions/0013-calibrated-table-positions-with-image-coordinate-compatibility.md) | Physical table coordinates alongside unchanged legacy image semantics | Accepted | 3 qualification | Python/Rust perspective fixture agreement; held-out physical measurements remain required before accuracy claims |
| [0014](decisions/0014-python-owned-revisioned-configuration-and-calibration.md) | Historical Python-owned configuration proposal | Superseded by 0056 | 1/4/5 | Retain versioned requests and revision acknowledgments; native configuration/calibration replacement remains required |
| [0015](decisions/0015-exclusive-camera-ownership-and-owned-frame-buffers.md) | One acquisition owner and owned in-process frames | Proposed | 5 | Device-independent envelope/mailbox semantics are tested; physical camera modes, release/restart/handover and target-platform behavior remain open |
| [0016](decisions/0016-prioritize-localization-and-isolate-optional-consumers.md) | Continuous localization with independently bounded optional work | Proposed | 3 | Dense-motion accuracy, blocked-consumer/interference tests and preserved prediction lifecycle |
| [0017](decisions/0017-retain-python-tools-as-cross-runtime-acceptance-tests.md) | Retain Python scoring/tools as temporary differential references | Superseded through 0082 by 0085 | 0/2 | Differential Python evaluators are retired; current evidence is Rust-owned under ADR-0085 |
| [0018](decisions/0018-capability-gated-reversible-rollout-on-macos-and-linux.md) | Qualified platform/provider capabilities and reversible cutover | Proposed | 4 onward | Explicit replay/live readiness, target-hardware qualification and exercised rollback |
| [0019](decisions/0019-asynchronous-ball-identity-separate-from-location-tracks.md) | Asynchronous number evidence separate from location and track identity | Proposed | 7 | Held-out identity accuracy/unknown rate, provenance and localization interference checks |
| [0020](decisions/0020-deduplicate-polled-source-observations-before-state-updates.md) | Process each source observation once; preserve freshness across repeated polls | Accepted | 0 | Same-frame replay leaves Core and receipt freshness unchanged; new frame and restarted writer are accepted |
| [0021](decisions/0021-frame-scoped-detection-batches-before-tracked-scene-snapshots.md) | Publish honest per-frame detections before tracked scenes | Accepted | 2–3 | Shared detection-batch contract remains used by live capture and replay JSONL; the retired standalone TCP receiver and its tests remain historical; paced freshness and temporal/physical accuracy evidence remain open |
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
| [0032](decisions/0032-python-supervisor-validates-rust-live-startup.md) | Python supervises Rust live startup with one request and validates source readiness | Superseded by 0057 (selection gate only) | 5 | Exact request delivery, bounded settings, model/tracking arguments, request-bound v3 acknowledgement, first usable scene and ordered shutdown deadlines remain active |
| [0033](decisions/0033-separate-rust-live-capture-process-from-replay-evaluator.md) | Keep persistent live capture in a separate Rust executable with an injected source factory | Accepted | 5 | Real-model deterministic-source process test verifies v3 readiness, scene identity and release; a qualified camera adapter and target-device tests remain |
| [0034](decisions/0034-separate-camera-input-mode-from-emitted-rgb-frame-geometry.md) | Keep raw camera mode separate from emitted RGB8 frame geometry in capture-settings v2 | Accepted | 1/5 | Python/Rust v1 compatibility and v2 default/transformed fixtures validate; acknowledgement and per-frame geometry are checked; hardware adapter qualification remains |
| [0035](decisions/0035-version-temporal-localization-evaluation-data.md) | Score versioned temporal labels through one engine-neutral localization evaluator | Accepted | 3/8 | Strict CVAT Video 1.1 importer records explicit review and source-video provenance; Python and Rust adapters share scoring; held-out temporal/physical labels remain required for accuracy claims |
| [0036](decisions/0036-contain-native-camera-capture-in-a-supervised-helper-process.md) | Isolate native capture and use feature-gated GStreamer for physical qualification | Superseded by 0057 (selection gate only) | 5 | Helper isolation remains; explicit API selection is now allowed for qualification; Linux hardware, sustained freshness, and production accuracy remain open ([report](../../thoughts/performance-audits/20261021-103042-aeb9f8d8/report.md)) |
| [0037](decisions/0037-non-capturing-cross-platform-camera-discovery-probe.md) | Enumerate camera metadata without opening a device | Superseded by 0040 for response identity fields | 5 | Bounded inventory and no-open semantics remain; selector shape is defined by ADR-0040 |
| [0038](decisions/0038-use-platform-native-camera-control-readbacks-behind-the-capture-helper.md) | Use native control readbacks for explicitly configured controls inside the supervised helper | Proposed | 5 | Generic default-control behavior is handled by ADR-0039; explicit control adapters, value units, and target hardware still need qualification |
| [0039](decisions/0039-represent-driver-default-camera-controls-explicitly-in-capture-settings-v3.md) | Represent no-write OS/driver defaults explicitly while retaining strict optional controls | Accepted | 5 | Cross-runtime v3 request/ack implementation and source-default hardware qualification remain |
| [0040](decisions/0040-version-stable-camera-capture-selectors.md) | Give stable provider identities a bounded selector without changing capture-settings v1/v2/v3 | Superseded by 0054 for Linux identity resolution | 5 | Selector schema, hash, exact selection, and Python v3 bridge remain active |
| [0041](decisions/0041-bound-opt-in-live-pipeline-timing-export.md) | Export correlated Rust live-stage timings over a bounded opt-in side channel | Accepted | 5 | Rust spans/export and Python collector tests pass; development-Mac A/B point estimates are within budget but statistically inconclusive; target live overhead remains open ([target replay limits](../../thoughts/performance-audits/rust-live-trained-model-export-20261021/target-linux-replay-20261021/report.md)) |
| [0042](decisions/0042-preserve-live-localization-under-bounded-tracker-pressure.md) | Evict unmatched tracking history under valid capacity pressure and preserve current measured locations | Accepted | 3/5 | Tracker and capture-pipeline tests cover deterministic eviction, degraded state, and continued localization; temporal accuracy and physical-camera pressure remain open |
| [0043](decisions/0043-runtime-selectable-openvino-for-rust-onnx.md) | Earlier ONNX Runtime OpenVINO execution-provider experiment | Superseded by 0045 | Historical | Target provider run and A/B are preserved as historical evidence; they do not qualify the native backend |
| [0044](decisions/0044-infer-camera-source-from-source-value-and-separate-viewer-url.md) | Infer detector input type from `vision.camera.device_id`; keep browser URL independently configurable | Accepted | 5 | Python URI/device routing, runtime viewer selection, config validation, and freshness-bounded queue tests pass; Rust stream adapter and target hardware remain open |
| [0045](decisions/0045-use-the-native-openvino-runtime-backend-for-openvino-selection.md) | Use the native OpenVINO Runtime backend for explicit OpenVINO selection | Accepted | 3/3 | Bounded Linux and Apple Silicon runtime/model smokes, 30-frame CPU/native output comparisons, and target API state delivery pass; 5 Hz freshness, labeled temporal accuracy, and Rust server-stage telemetry remain open ([Linux report](../../thoughts/performance-audits/rust-live-trained-model-export-20261021/target-linux-replay-20261021/native-openvino-runtime-20261021/report.md), [Apple Silicon report](../../thoughts/performance-audits/native-openvino-macos-runtime-20261021/report.md), [target replay notes](../../thoughts/performance-audits/target-rust-replay-smoke-20261021/target-smoke-notes.md)) |
| [0046](decisions/0046-segment-supervised-camera-capture-timing.md) | Segment supervised camera capture timing across process boundaries | Accepted | 5 | Timed helper stages and parent framed-read completion use separate process-local durations; camera exposure remains unavailable; verify timed/legacy protocol and audit-disabled paths |
| [0047](decisions/0047-measure-rust-scene-transport-at-the-python-receiver.md) | Measure Rust scene transport at the Python receiver | Accepted | Audit | Opt-in bounded counters report accepted snapshots, payload sizes, gaps and latest-only queue pressure without changing the Rust wire protocol |
| [0048](decisions/0048-isolate-trajectory-workload-during-state-path-audits.md) | Isolate trajectory workload during state-path audits | Accepted | Audit | State-only runs suppress prediction admission only while audit control is active, preserving Core updates and production prediction behavior |
| [0049](decisions/0049-explicit-performance-audit-count-semantics.md) | Make performance audit count semantics explicit | Superseded by 0050 | Audit | Historical sampling and retained-event counting policy; use ADR-0050's normalized fresh-observation and source-update metrics |
| [0050](decisions/0050-normalize-audit-evidence-across-python-and-rust-runtimes.md) | Normalize audit evidence across Python and Rust runtimes | Accepted | Audit | Collector preserves each runtime's source identity, separates fresh observations from cached/failed updates, and reports unavailable clock or repository identity explicitly |
| [0051](decisions/0051-repeat-recorded-input-for-bounded-paced-audits.md) | Repeat recorded input for bounded paced audits | Accepted | Audit | Configurable repeats assign monotonically increasing synthetic frame indexes and require a maximum frame cap; Rust/Python tests and a bounded repeated-corpus API smoke pass; longer audit windows remain |
| [0052](decisions/0052-latest-state-best-effort-freshness.md) | Prefer the latest state throughout the perception pipeline | Accepted | 3–6 | Backend flood/slow-consumer and stale-before-Core tests plus frontend freshness/order tests pass; target freshness, physical-camera age, and location accuracy remain to be qualified |
| [0053](decisions/0053-use-five-percent-default-detail-sampling-for-performance-audit.md) | Use five-percent default detail sampling for performance audit | Accepted | Audit | Same-input three-run instrumentation comparison stays within the overhead budget on the development Mac; other providers and physical cameras remain unqualified |
| [0054](decisions/0054-resolve-linux-camera-selectors-from-stable-os-links.md) | Resolve Linux camera selectors from stable OS links when GStreamer reports a transient V4L node | Superseded by 0075 | 5 | Direct stable-link matching remains active; container fallback is now defined by ADR-0075 |
| [0055](decisions/0055-export-timing-evidence-for-stale-live-observations.md) | Export stage timing evidence when live inference is rejected as stale | Accepted | Audit | V3 Rust/Python contract reports capture, inference and freshness durations without publishing stale states; target timing attribution and live readiness remain open |
| [0056](decisions/0056-retain-the-python-api-gateway-while-migrating-domain-logic-to-rust.md) | Retain Python HTTP/WebSocket gateway; migrate all application-domain logic to Rust | Superseded by 0064 | Runtime ownership | Historical migration boundary; Python gateway was a staged destination before ADR-0066 moved the full server to Rust |
| [0057](decisions/0057-allow-explicit-rust-live-perception-behind-the-python-service.md) | Allow explicit Rust live perception behind the Python service | Accepted | Runtime selection | Live path is reachable through the API without Python Core/Vision/integration ownership; stable selector, model/helper validation and first-localization readiness remain mandatory; physical accuracy, freshness, and sustained qualification remain open |
| [0058](decisions/0058-use-rust-lifecycle-for-live-public-state.md) | Use the Rust lifecycle for live public state | Accepted | Live state ownership | Rust emits versioned public-state delivery with unchanged raw scene; Python validates/caches/relays; target live qualification remains open |
| [0059](decisions/0059-solve-table-calibration-in-rust-with-capture-provenance.md) | Solve table calibration in Rust with capture provenance | Accepted | Calibration authority | Rust solves provenance-bound v1 homographies; atomic persistence, route delegation, and physical calibration accuracy remain open |
| [0060](decisions/0060-store-table-calibration-as-rust-owned-revisioned-state.md) | Store table calibration as Rust-owned revisioned state | Accepted | Calibration authority | Rust store and capture-control protocol are wired; bounded FastAPI snapshot/save proxy passes focused tests; browser authoring integration and macOS/Linux filesystem qualification remain open |
| [0061](decisions/0061-use-rust-public-state-lifecycle-for-paced-replay.md) | Use Rust public-state lifecycle for paced replay | Accepted | Replay state ownership | Rust paced TCP replay emits native lifecycle delivery while raw JSONL remains unchanged; Python validates and relays through HTTP/WebSocket; real-model replay freshness evidence remains open |
| [0062](decisions/0062-serve-a-bounded-rust-owned-current-frame-for-calibration.md) | Serve a bounded Rust-owned current frame for calibration | Accepted | Calibration provenance | One-slot Rust preview, bounded on-demand encoding and descriptor-bound source-pixel calibration; gateway/UI integration and camera qualification remain open |
| [0063](decisions/0063-validate-calibration-against-stable-capture-context.md) | Validate calibration against stable capture context | Accepted | Calibration authority | Frozen-frame selections are checked against the active stable session context and store CAS; gateway/UI integration and physical accuracy remain open |
| [0064](decisions/0064-defer-rust-web-client-and-exclude-ball-number-classification.md) | Defer Rust/WASM browser client to the final stage and exclude ball-number classification | Superseded by 0066 | Migration scope | Historical scope; classifier remains excluded, but the browser stays HTML/JavaScript/CSS and all server services move to Rust |
| [0065](decisions/0065-move-unified-runtime-configuration-authority-to-rust.md) | Move unified runtime configuration authority to Rust | Accepted | Configuration authority | One revisioned Rust owner for defaults, validation, atomic persistence, and production v3 capture-request construction; all configuration file routes now use the Rust gateway, while desired/applied revision tracking remains |
| [0066](decisions/0066-run-server-services-in-rust-and-retain-the-standard-web-client.md) | Run server services in Rust and retain the standard web client | Accepted | Deployment/runtime boundary | Replace the Python service after Rust routes, lifecycle, and operations pass compatibility and platform gates; keep the browser HTML/JavaScript/CSS and defer integration work until server qualification |
| [0067](decisions/0067-own-game-session-metadata-in-rust.md) | Own game-session metadata in Rust | Accepted | Game state | Bounded revisioned Rust metadata store and reset adapter leave scene locations/calibration untouched; history/events/export and final in-process Rust gateway integration remain open |
| [0068](decisions/0068-introduce-a-rust-latest-state-http-and-websocket-gateway.md) | Introduce a Rust latest-state HTTP and WebSocket gateway | Accepted | Gateway foundation | 186 Rust library tests plus all targets, strict Clippy/fmt, three real slow-reader stress passes, and a standalone replay-v2-to-HTTP process smoke pass; Python remains active and live camera, full-route, deployment, and platform gates remain open |
| [0069](decisions/0069-rust-gateway-validates-live-capture-readiness.md) | Validate live capture readiness in the Rust gateway | Accepted | Gateway live ingress | Require revision-bound v3 acknowledgement from Rust configuration authority; retain browser contract and rebase freshness to receiver-local monotonic time |
| [0070](decisions/0070-serve-configuration-reads-from-rust.md) | Serve configuration reads from the Rust gateway | Accepted | Gateway configuration reads | Preserve current config/schema response shapes, run file authority on bounded blocking workers, and retain anonymous access; reset is in 0072 and import/export are in 0073 |
| [0071](decisions/0071-serve-configuration-updates-from-rust.md) | Serve configuration updates from the Rust gateway | Accepted | Gateway configuration update route | Preserve browser update/validation contracts, guard deployment executable paths, serialize bounded mutations and commit against the original revision; reset is in 0072 and import/export are in 0073 |
| [0072](decisions/0072-serve-configuration-reset-from-rust.md) | Serve configuration reset from the Rust gateway | Accepted | Gateway configuration reset route | Preserve anonymous all/user/section semantics and response envelopes using authority CAS; import/export are in 0073, while remaining route parity and cutover stay open |
| [0073](decisions/0073-serve-configuration-import-and-export-from-the-rust-gateway.md) | Serve configuration import and export from the Rust gateway | Accepted | Gateway configuration import/export | Preserve multipart, JSON/YAML, checksum and download contracts with bounded parsing/output, validation, mutation CAS and executable-path protection; existing browser remains HTML/JavaScript/CSS |
| [0074](decisions/0074-serve-game-session-state-directly-from-the-rust-gateway.md) | Serve game-session state directly from the Rust gateway | Accepted | Gateway game-state metadata and reset | Read the revisioned store on bounded blocking work using a securely opened root and root-anchored, no-follow file access; keep optional failures from suppressing localization and preserve the current scene on reset |
| [0075](decisions/0075-resolve-containerized-linux-camera-identities-from-udev-records.md) | Resolve containerized Linux camera identities from udev records | Accepted | Linux camera identity | Preserve exact stable selector by binding the sysfs device number to its udev V4L alias when `/dev/v4l` is not mounted; target helper must still capture a valid first frame |
| [0076](decisions/0076-accept-rust-public-state-deliveries-over-replay-v2.md) | Accept Rust public-state deliveries over replay-v2 | Accepted | Gateway replay ingress | Preserve raw v1 payloads and accept strict lifecycle envelopes from paced Rust replay; rebase elapsed ages locally and leave live-v3 readiness unchanged |
| [0077](decisions/0077-use-a-bounded-remote-mjpeg-source-for-local-rust-qualification.md) | Use a bounded remote MJPEG source for local Rust qualification | Accepted | Mac-first capture qualification | Five 1280×720 RGB8 frames and a live Rust detector→gateway smoke pass; sampled state stayed fresh at roughly 100–180 ms, while table calibration and accuracy ground truth remain open |
| [0078](decisions/0078-serve-the-prebuilt-browser-client-from-the-rust-gateway.md) | Serve the prebuilt browser client from the Rust gateway | Accepted | Rust static delivery | Safe asset delivery and SPA routing pass; the live-only client and configured remote MJPEG viewer work on Mac, while Rust-owned physical-camera video remains open |
| [0079](decisions/0079-reduce-the-first-rust-hosted-client-to-live-view.md) | Reduce the first Rust-hosted client to the live view | Accepted | Mac-first browser qualification | Only `/live` remains in the route tree; root redirects and unknown routes show a live-only notice; local Rust state and remote 1280×720 MJPEG smoke pass, while calibration and production video delivery remain open |
| [0080](decisions/0080-use-rust-as-the-only-supported-application-server.md) | Use Rust as the only supported application server | Accepted | Rust server cutover | Python launch commands fail with migration guidance; Rust, browser, compatibility, Docker and distribution checks pass |
| [0081](decisions/0081-use-direct-rust-http-delivery-without-nginx.md) | Use direct Rust HTTP delivery without Nginx | Accepted | Direct Rust deployment | Retire the Python/Nginx installer and standalone Nginx image; keep bounded Rust WebSocket concurrency, payload, and write behavior |
| [0082](decisions/0082-use-shell-for-glue-and-rust-for-complex-audit-tools.md) | Use shell for glue and Rust for complex audit tools | Superseded by 0085 | Audit tooling migration | Shell/Rust ownership split remains; temporary Python collector retention ended when the Rust live audit and temporal-label path became canonical |
| [0083](decisions/0083-import-reviewed-temporal-annotations-with-rust.md) | Import reviewed temporal annotations with Rust | Accepted | Temporal label authoring | Rust CLI validates reviewed CVAT keyframes, video identity and timing with bounded inputs and atomic output; generated-video Python/Rust parity passed and the Python importer was retired |
| [0084](decisions/0084-replay-recorded-video-with-rust-perception.md) | Replay recorded video with Rust perception | Accepted | Saved-video evaluation | PTS-paced latest-frame and unpaced capacity modes run through current Rust inference; predecode excludes live codec contention and labels are still required for accuracy claims |
| [0085](decisions/0085-retire-legacy-audit-tooling-in-favor-of-rust-owned-evidence-paths.md) | Retire legacy audit tooling in favor of Rust-owned evidence paths | Accepted | Tooling scope | Rust owns live, saved-video, and labeled temporal evidence; offline dataset/model-authoring tools remain Python, while viewer interference and representative accuracy evidence remain open |
| [0086](decisions/0086-retire-the-incomplete-native-visualizer.md) | Retire the incomplete native visualizer | Accepted | Client scope | Remove the disconnected trajectory-only LÖVE client; keep the `/live` browser application and distinct projector client |

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
| [0003](decisions/0003-bounded-websocket-delivery.md) | Retain bounded client and ordered control delivery; ADR-0052 tightens replaceable perception messages to latest-value best effort. |
| [0004](decisions/0004-validated-config-updates.md) | Retain validation and atomic persistence; 0014 adds the process application/acknowledgement boundary. |
| [0005](decisions/0005-distributed-rate-limiter.md) | Historical offline Python reference policy; ADR-0081 supersedes its per-IP enforcement for the supported Rust runtime, which retains global bounded WebSocket admission. |
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
