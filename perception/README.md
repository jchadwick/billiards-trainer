# Rust runtime and perception tools

This crate contains the supported server-side runtime and the maintained
perception tools. `billiards-server` serves the browser application and owns
the application lifecycle; `billiards-perception-live` runs live capture and
detection through the supervised GStreamer helper. The physical-camera source
is feature-gated and still requires hardware qualification on each target.

Use `billiards-performance-audit-run` for live state delivery and freshness,
`billiards-perception-video-replay` for recorded-video detector freshness or
capacity, and `billiards-temporal-evaluate` for accuracy against reviewed
labels. Their distinct evidence boundaries and commands are documented in
[`docs/PERFORMANCE_AUDIT.md`](../docs/PERFORMANCE_AUDIT.md). There is no
maintained still-image inference CLI: a still image cannot establish live
freshness or temporal accuracy, and the old command duplicated shared inference
paths without a distinct supported workflow.

The current model reports generic `cue` and `ball` classes only. Numbered-ball
identity such as 6/7/8-ball classification remains outside the supported model
contract. Historical prototype notes may describe the former Python
compatibility supervisor; they are not the supported server entry point.

## Reusable RGB-frame inference

`inference::FrameInferenceEngine` loads the same hash-verified ONNX export and
class map as the replay tool, then accepts one immutable `RgbImage` or packed
borrowed RGB8 bytes through `infer_rgb8`. The byte-slice path validates geometry
and reads a matching-size capture buffer directly, avoiding an intermediate
full-frame `RgbImage` copy. Resizing and model-tensor allocations remain, and
are outside `CaptureFrameBudget` accounting. It returns a complete detection
list plus separate preprocessing, model inference, and decode/NMS durations.
The input owner supplies frame identity, source timestamps, and queue timing;
engine timings do not include camera capture, encoded-image decoding, tracking,
or publication. The engine reports the requested provider policy only; actual
per-node provider placement is unavailable and must not be inferred.

The engine is mutable and intended to remain on one inference worker. It does
not schedule, retain, or drop frames. `capture_pipeline::CaptureScenePipeline`
provides a synchronous frame consumer around this API, while the capture
receiver's capacity-one mailbox handles latest-frame replacement. The Rust
gateway owns the supported application lifecycle and supervises live capture
when selected by its validated configuration. Physical camera accuracy and
sustained freshness still require qualification on the intended hardware.

The integration test loads a small deterministic ONNX fixture to exercise
manifest validation, tensor creation, session execution, output validation,
and decoding without network access or a large model artifact. Recorded-video
replay uses this same frame engine. A historical ten-still comparison with
the Python scorer remains in the audit archive; that scorer has been retired.
The small unpaced run is integration evidence, not a live freshness or
performance qualification.

## Device-independent capture boundary

`capture::RgbFrameEnvelope` owns tightly packed RGB8 pixels and the matching
source-frame identity. It preserves an optional source-clock timestamp and its
event label; the host monotonic read-completion point is stored separately.
Capture-settings v2 keeps raw requested/negotiated camera modes separate from
the output RGB8 dimensions. V2 preflight requires the source report to confirm
those dimensions and checks every emitted frame against them; v1 remains
byte-compatible and assumes output dimensions equal the negotiated profile.
The internal `capture::LatestFrameMailbox` is a thread-safe capacity-one handoff
scoped to one source session. A new offer replaces only the pending frame and
increments the replacement counter. Graceful close drains a final frame,
cancellation discards pending work, and producer failure remains visible after
any pending frame is received. The runtime grants exactly one
`CaptureFrameReceiver` per session; it is not cloneable, and delivers frames
through a synchronous callback that borrows the envelope only for the
callback's duration. This prevents a consumer from retaining a received frame
through the API while requesting another. The callback may still allocate
derived data, and the capture source must not retain extra application-owned
RGB buffers.

`capture::CaptureRuntimeCoordinator` owns the active worker and configured
frame-size limit. Its start method accepts only an unopened
`CaptureSourceFactory`; callers cannot inject a gate. All coordinators share
one process-wide gate because this runtime supports one physical source. The
source is opened, read, released, and dropped on one worker thread. A timed-out
stop keeps the worker and gate alive, and blocks a replacement start until the
old source exits. Unconfirmed release, destructor panic, or unexpected worker
panic quarantines the gate for the process lifetime. Multi-camera support needs
a configured source-keyed owner registry before it is added.

Each source read receives both a per-frame RGB8 limit and the process-shared
`CaptureFrameBudget`. After learning negotiated dimensions, an adapter should
use `CaptureFrameBudget::allocate_rgb8_buffer` before copying or converting
pixels into application-owned memory. The budget reserves each allocation's
actual `Vec` capacity and remains charged while its buffer is retained.
Envelope construction and mailbox publication recheck frame length, capacity,
and budget ownership. The per-frame default is 24 MiB, enough for one
3840×2160 RGB8 frame; configuration accepts up to 1 GiB. The aggregate limit
is a separate required runtime input with no Rust default. All public runtime
coordinators share one process-wide aggregate budget and reject conflicting
limits. Native camera-driver buffers and temporary source-side conversions
remain outside application-buffer accounting.

The Rust configuration contract exposes
`perception.capture.maximum_aggregate_rgb_frame_bytes` with a 72 MiB default
(three 24 MiB application buffers) and validates that it is at least the
configured per-frame limit. The schema also bounds it to 1 GiB. The Rust
gateway constructs and passes the exact v3 capture request to the supervised
live detector. This budget contract does not qualify camera accuracy or
sustained freshness.

The source gate coordinates lifecycle ownership; it does not identify or
authorize a client. The application, HTTP/WebSocket APIs, and local perception
stream remain anonymous and use no credentials. The coordinator's gate is
process-wide and cannot coordinate another process that independently opens a
camera. The Rust live executable constructs the coordinator and pipeline
around the supervised GStreamer helper. The helper receives the exact capture
request, uses an
enumerated stable selector, validates negotiated input/output geometry, and
does not write camera controls. A short macOS smoke verified frame delivery
and child shutdown with the built-in camera, but did not verify table accuracy,
driver-level frame age, sustained operation, or Linux behavior. The live path
is available through the Rust gateway when validated live configuration and
the required GStreamer feature are selected. Preview-to-calibration-to-live
activation and physical accuracy/freshness qualification remain open.

## Opt-in camera discovery

`billiards-perception-camera-discovery` reports GStreamer `Video/Source`
inventory without opening a camera. It uses `DeviceMonitor` to read display
names, device classes, provider properties, and advertised caps; it never
creates a source element, starts a capture pipeline, negotiates a mode, or
reads frames. The command is independent of the live executable and does not
change the capture-settings contract.

Build and run it only when explicitly requested:

```bash
cargo run --release --manifest-path perception/Cargo.toml \
  --no-default-features --features camera-gstreamer-discovery \
  --bin billiards-perception-camera-discovery -- --max-output-bytes 1048576
```

The optional feature requires the GStreamer 1.x development files discoverable
by `pkg-config` at build time, and the GStreamer runtime plus platform device
providers at execution time. A missing provider can produce a valid empty
inventory. The default output limit is 1 MiB; the CLI accepts 512 bytes through
16 MiB. If metadata does not fit, the versioned JSON marks `truncated` and
counts omitted fields/devices while staying within the byte limit. Missing
provider values and advertised caps are represented as `null` rather than
guessed.

Identifier classification is deliberately narrow. `avf.unique_id` is reported
as platform-persistent on macOS. Linux `/dev/v4l/by-id/...` and
`/dev/v4l/by-path/...` symlink paths are reported as platform-persistent;
`/dev/videoN` is included only as an enumerated/transient identifier. Other
provider properties are emitted as metadata, but are not promoted to stable
identity. The top-level provider_status distinguishes a started device
provider from no installed provider matching Video/Source. The JSON contains
the host OS/architecture/version when available and the initialized GStreamer
version. This command is an inventory aid only;
it does not establish that any camera can be opened, that advertised caps can
be negotiated, or that capture works on the host.

## Capture-to-scene library pipeline

`capture_pipeline::CaptureScenePipeline` accepts each
`RgbFrameEnvelope` synchronously, invokes the inference engine on its borrowed
RGB8 bytes, validates a complete detection batch, builds a tracked scene, and
submits that scene to the existing newest-only loopback writer. Construct it
before starting capture so its process-local monotonic origin predates the
first frame. A provided source timestamp keeps its clock domain and event
stage; if none is provided, the pipeline uses host read-completion time labeled
`camera_read_complete`, never exposure. Duplicate frames, non-increasing
same-domain source times, and observations beyond the configured freshness
limit are rejected before publication. Inference failure and a successful
empty observation remain distinct.

The runtime may provide an `AppliedCaptureCalibration` containing the
validated calibration bundle and the profile and revisions acknowledged by
the control boundary. Startup rejects a bundle that disagrees with the
tracking profile, rectification revision, or configured image space. Every
frame is checked again for matching image space and dimensions. Only a
complete valid projection publishes table millimeters; absent calibration is
reported as unavailable, and incompatible calibration is reported invalid
without stale table coordinates. The pipeline still does not assign ball
numbers or run predictions. Each accepted result reports host read-completion
to inference-start and scene-queue latency, plus a separate geometry duration
when calibration is configured. These use the local monotonic clock and do not
represent camera exposure or socket receipt.

A consumer loop can call it inside the receiver callback:

```rust,ignore
let outcome = pipeline.receive_one(&mut receiver, timeout);
```

`receive_one` processes the frame inside the receiver callback and returns only
derived owned results; it cannot retain the frame borrow. The writer keeps a
single pending scene, replaces older pending state, and performs socket I/O on
its own worker. Its protocol remains unauthenticated:
run and source-session identifiers are provenance only, and no credentials are
sent or required. The caller must handle receiver terminal outcomes and call
`pipeline.shutdown()` to collect final transport counters. This library seam
has a hardware-independent integration test that composes a deterministic
capture source, the public receiver and pipeline APIs, and a schema-validating
anonymous loopback receiver. That proves software wiring and lifecycle only;
it does not qualify a selected camera adapter, provider launch, live freshness,
or hardware. Detector resize/tensor work is not covered by the capture RGB
allocation budget.

## Recorded-video replay and accuracy evaluation

`billiards-perception-video-replay` reuses the inference engine used by live
capture. Select `realtime-latest-frame` to replay source presentation timing
with latest-frame replacement, or `capacity` to decode and process every source
frame as quickly as possible. The two modes answer different questions. Replay
predecodes into a bounded RGB spool, so it does not measure camera acquisition
or live decoder contention; use the live performance audit for end-to-end state
delivery. See [`docs/PERFORMANCE_AUDIT.md`](../docs/PERFORMANCE_AUDIT.md) for
the maintained commands, defaults, report fields, and interpretation limits.

The video-replay command records model and video hashes, selected stream,
effective inference settings, provider, revision/worktree identity, replacement
counts, and timing distributions. Optional bounded traces and JSONL detection
or scene artifacts support deeper analysis. Calibration can add table-space
coordinates only when the bundle, source image space, camera profile, and
applied revisions agree. Scene artifacts can be scored with reviewed temporal
annotations; without those labels, localization accuracy is unavailable.

Use `billiards-temporal-import-cvat` to import explicitly reviewed CVAT video
annotations and `billiards-temporal-evaluate` to score scene snapshots against
them. The annotation must refer to the exact video and stream geometry. V2
physical-coordinate scoring requires independently measured table coordinates,
stated uncertainty, and an explicit millimeter match gate. The repository does
not currently contain representative held-out temporal and physical labels, so
no production accuracy claim is supported.

The model exporter at `tools/export_high_res_model.py` produces the verified
ONNX artifact consumed by Rust. The current model detects generic `cue` and
`ball` classes only; numbered-ball classification is outside its contract.
OpenVINO is an explicit CPU inference backend on macOS and Linux, while CPU and
automatic-provider policies use ONNX Runtime. Provider selection and model
loadability do not establish equivalent accuracy or target-machine speed; see
[ADR-0045](../docs/architecture/decisions/0045-use-the-native-openvino-runtime-backend-for-openvino-selection.md)
and the specific run reports for the evidence and limits.

## Capture implementation limits

The strict capture-settings parser accepts v1, v2, and v3. V3 retains the
explicit RGB8 output geometry from v2 and tags camera-control behavior with
`requested.camera_controls`. `{"mode":"source_defaults"}` makes no control
request and does not require capabilities or readbacks; source-reported
supported, unsupported, and unknown states remain unchanged. The `require`
mode preserves strict capability and exact-readback checks. The synthetic
GStreamer helper exercises both modes, but does not establish physical camera
control behavior. V1 and v2 wire requests retain their existing strict
semantics. See the [v3 contract](contracts/capture-settings/v3/README.md).

The frame envelope, mailbox, and fake-source worker described above preserve
source timestamp semantics and exercise software ownership sequencing. They do
not qualify physical capture-buffer reuse. The live executable and supervised
GStreamer helper are implemented; enabling the physical source requires the
`camera-gstreamer-source` feature. Fake and network-stream tests prove software
wiring only, not driver timing or physical table accuracy.

The mailbox bounds pending frame count, and the per-frame maximum bounds each
application-owned RGB8 vector. The one-session receiver API allows one active
consumer and lends a frame only during its synchronous callback. A shared
aggregate budget now accounts for the capacity of budgeted application RGB8
buffers across source allocation, the mailbox, and the borrowed consumer
callback. It does not cover native driver buffers or adapter-side temporary
conversions. The earlier worker prototype exposed independent owner gates;
`CaptureOwnerGate` is now private and all runtime coordinators share one
process-wide gate for the current single-source product. That gate cannot
coordinate another process opening the camera. The gate enforces process-local
capture ownership; source/session IDs are provenance labels and do not restrict
which local client may connect.
See
[`ADR-0015`](../docs/architecture/decisions/0015-exclusive-camera-ownership-and-owned-frame-buffers.md)
for the open hardware and lifecycle qualification requirements.

`CaptureRuntimeCoordinator::start_preflighted(expected, factory)` connects the
revisioned `live_config` expectation to the capture worker. It first checks the
coordinator's per-frame and process-shared aggregate RGB8 limits, then opens the
source on its owner thread and validates its `CaptureSessionReport` before the
first read. A bounded readiness wait exposes session-bound pending, compatible,
rejected, failed, or stopped status. A wait timeout does not release the worker
or source-owner gate. On rejection, the mailbox is failed and waiters are woken
before the owner thread releases the source; rejected sessions perform no reads.
Release and quarantine behavior remains on that owner thread.

Preflight verifies device identity, requested and negotiated video profiles,
required reported capabilities, timestamp semantics, and actual RGB8 dimensions
against the negotiated profile before a frame can be published. The report
cannot prove that a camera profile ID, image space, calibration, or exposure/gain
values were applied when the adapter does not report those values. In
particular, a compatible preflight is not a camera-control acknowledgement.

Construct `CaptureScenePipeline` before starting capture so its monotonic clock
origin precedes the first frame. After receiving the compatible session result,
call `bind_preflight(compatibility)` once and before consuming any frame. Binding
requires the same explicit capture-configuration revision, image space, and
maximum frame age, and cross-checks the tracker camera profile and rectification
revision against the complete validated calibration bundle. The capture
configuration revision is distinct from the tracker's local configuration
revision. A bound pipeline rejects a wrong source session, dimensions, or
timestamp semantics before inference. Without applied calibration, table
positions remain unavailable. Existing unbound replay-pipeline use remains
compatible.

The live executable composes this preflight and pipeline with an opt-in
feature-gated GStreamer physical source. The Rust gateway forwards the same
immutable capture request and validates request-bound readiness. A
macOS built-in-camera smoke passed through the complete local loopback path with
the checked-in test detector and then shut down cleanly; it is integration
evidence only. No calibrated table positions, production-model accuracy,
reliable exposure timestamp, sustained freshness, preview handover, or Linux
physical behavior has been qualified. The feature-enabled all-target suite and
optimized live/helper builds also pass in an Ubuntu 24.04 x86_64 container with
GStreamer 1.24, but
that container had no V4L2 device.
