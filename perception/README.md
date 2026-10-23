# Rust perception prototype

This directory contains the Rust side of the staged perception migration. It
currently provides the versioned scene contract, length-prefixed framing, and
an opt-in recorded-image ONNX inference prototype with optional detection-batch
JSONL or loopback TCP output. A separate live executable can also run an
explicitly selected, feature-gated GStreamer camera through a supervised helper
process. Its default source remains unavailable, and the production API still
uses its existing provider selection. The library exposes a reusable one-frame
RGB inference API, which recorded replay and the live runner share.

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
or publication. The engine reports the requested provider policy only. Use a
profiled audit session to establish which ONNX Runtime providers actually
executed model nodes.

The engine is mutable and intended to remain on one inference worker. It does
not schedule, retain, or drop frames. `capture_pipeline::CaptureScenePipeline`
provides a synchronous frame consumer around this API, while the capture
receiver's capacity-one mailbox handles latest-frame replacement.
The physical GStreamer path is an opt-in qualification candidate, not a
qualified production provider. FastAPI's production lifecycle still selects
Rust only for recorded-input replay; the standalone supervisor can exercise the
live camera path without changing that API default.

The integration test loads a small deterministic ONNX fixture to exercise
manifest validation, tensor creation, session execution, output validation,
and decoding without network access or a large model artifact. The recorded
replay uses this same frame engine. A real-model ten-still smoke exercised that
shared path and matched the existing Python scorer; its small unpaced run is
integration evidence, not a live freshness or performance qualification.

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

The Python configuration schema exposes
`perception.capture.maximum_aggregate_rgb_frame_bytes` with a 72 MiB default
(three 24 MiB application buffers) and validates that it is at least the
configured per-frame limit. The schema also bounds it to 1 GiB. The Rust live
runner derives its aggregate budget from the exact v3 request supplied by the
Python supervisor. Production API lifecycle wiring remains disabled, so this
does not mean the application has selected or qualified Rust live capture.

The source gate coordinates lifecycle ownership; it does not identify or
authorize a client. The application, HTTP/WebSocket APIs, and local perception
stream remain anonymous and use no credentials. The coordinator's gate is
process-wide and cannot coordinate another process that independently opens a
camera. The opt-in Rust live executable now constructs the coordinator and
pipeline around the supervised GStreamer helper; FastAPI still does not start
that live path. The helper receives the exact capture request, uses an
enumerated stable selector, validates negotiated input/output geometry, and
does not write camera controls. A short macOS smoke verified frame delivery
and child shutdown with the built-in camera, but did not verify table accuracy,
driver-level frame age, sustained operation, or Linux behavior. Preview and
calibration handover remain unimplemented, so Rust live stays unavailable to
normal provider selection.

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

## Recorded-image inference

The `billiards-perception-infer` binary runs one PNG/JPEG image or every PNG/JPEG
under a directory. It requires explicit model paths, provider policy,
image-size, confidence, and IoU arguments, and emits a versioned JSON report to
stdout. It can also retain the JSON report and the ONNX Runtime profile used to
identify the providers that actually executed nodes.

Unpaced replay measures capacity by processing every image in order. Add
`--replay-rate-fps <rate>` for synthetic real-time pacing with a one-frame
latest-work mailbox. While inference is busy, a newer decoded input replaces a
pending one; source-frame IDs and replacement counts expose those skips. The
reported source timestamp is synthetic monotonic `frame_arrival` after image
decoding, never camera exposure time. The requested cadence, source-read
lateness, source-to-inference-start and source-to-inference-complete timings
are reported separately. Detection-batch TCP and JSONL output remain
synchronous replay tools and can delay the measured consumer.

Optional detection-batch v1 output publishes one complete record per image.
Use `--detection-batches-jsonl PATH` for an artifact or
`--detection-batches-tcp 127.0.0.1:PORT` for a synchronous framed connection to
the standalone Rust receiver binary. Both require `--engine-instance-id ID`
and `--source-session-id ID`. The TCP mode also requires explicit
maximum-message, connect-timeout, and write-timeout arguments; the receiver
requires the expected count, frame limit, deadlines, and a new JSONL output
path.

For an experimental tracked-scene replay, add `--scene-snapshots-jsonl PATH`
or the `--scene-snapshots-tcp 127.0.0.1:PORT` destination and
provide every `--tracking-*` policy setting shown below. This mode requires
paced replay and detection-batch output, preserves raw detector batches, and
builds one validated scene-snapshot v1 record for each processed frame. The
source time remains synthetic post-decode `frame_arrival`; `snapshot_publication`
is captured after tracking, immediately before encoding. Without a calibration
bundle, table coordinates remain unavailable. A configured but incompatible
calibration produces a degraded scene with image measurements retained and
table measurements empty.

TCP scene output serializes on the inference thread and submits to a bounded
latest-only slot serviced by a dedicated writer. At most one frame is pending
while one can be in flight; a newer pending frame replaces the older one. The
writer reconnects with capped exponential backoff (defaults: 100 ms initial,
2,000 ms maximum). The protocol-version 2 hello carries the engine-instance
and source-session IDs already present in each scene, so the ready response
includes a watermark only for the matching producer epoch. These IDs are data
provenance, not authentication credentials. A changed engine or source session
starts a fresh receiver sequence epoch and clears stale queued scene state.
After reconnect, only a newer latest snapshot is sent.
The report distinguishes queued records and successful local socket writes from
receiver acceptance; the receiver's own counters are authoritative for accepted
snapshots. The handshake carries no credential. The companion Python ingress
binds to loopback and accepts one active producer connection.
The scene TCP option remains recorded-replay infrastructure. In explicit
`rust` + `recorded_replay` mode, FastAPI supervises the Rust process and
publishes received scenes to the HTTP state read model and existing WebSocket
stream. Those scenes do not enter Core physics, and this path does not own a
live camera or provide preview/predictions. Scene JSONL remains synchronously
flushed per frame.

```bash
cargo run --release --manifest-path perception/Cargo.toml --bin billiards-perception-infer -- \
  --onnx /tmp/billiards-onnx/model.onnx \
  --source-checkpoint models/billiards-yolov8n.pt \
  --input backend/vision/test_data \
  --provider cpu \
  --image-size 1280 \
  --confidence 0.4 \
  --iou 0.45 \
  --replay-rate-fps 30 \
  --detection-batches-jsonl /tmp/billiards-batches.jsonl \
  --engine-instance-id scene-audit-01 \
  --source-session-id recording-01 \
  --scene-snapshots-jsonl /tmp/billiards-scenes.jsonl \
  --tracking-ball-class-id 1 \
  --tracking-cue-class-id 0 \
  --tracking-camera-profile-id camera-profile-a \
  --tracking-association-gate-px 18.5 \
  --tracking-association-prediction-horizon-ms 50 \
  --tracking-lost-after-ms 200 \
  --tracking-maximum-track-age-ms 1000 \
  --tracking-maximum-active-tracks 24 \
  --tracking-recent-observation-capacity 32 \
  --tracking-configuration-revision 1 \
  --tracking-rectification-revision 1
```

The explicit tracking numbers above are an example audit configuration, not
recommended production thresholds. Add `--calibration-bundle`,
`--source-image-space`, and all three applied camera/revision arguments to test
physical coordinates. The bundle must match source image space, dimensions,
camera profile, and applied revisions exactly. The report records effective
tracking settings, per-frame tracking and encode/enqueue time, scene count and skips through
the associated source-frame IDs. Temporal tracking accuracy, held-out physical
position error, slow-client isolation and platform qualification remain open.

For TCP scene output, replace the JSONL option with
`--scene-snapshots-tcp 127.0.0.1:PORT` and provide
`--scene-snapshots-max-message-bytes`, `--scene-snapshots-connect-timeout-ms`,
and `--scene-snapshots-write-timeout-ms`. The reconnect settings default to
100 ms and 2,000 ms and can be overridden with
`--scene-snapshots-reconnect-initial-delay-ms` and
`--scene-snapshots-reconnect-maximum-delay-ms`. The receiver must be an
application-owned `SceneSnapshotIngress` using protocol version 2; a standalone
operator-facing receiver command has not been added yet.
`mean_encode_enqueue_ms` measures serialization plus bounded queue submission,
not socket delivery.

Build the Rust receiver once, then run it before the replay producer:

```bash
cargo build --release --manifest-path perception/Cargo.toml \\
  --bin billiards-perception-receive-detection-batches

perception/target/release/billiards-perception-receive-detection-batches \\
  --bind-address 127.0.0.1:9000 \\
  --expected-batches 10 \\
  --maximum-batches 10000 \\
  --maximum-message-bytes 16777216 \\
  --accept-timeout-ms 30000 \\
  --session-timeout-ms 120000 \\
  --read-timeout-ms 5000 \\
  --maximum-detail-records 10000 \\
  --batches-jsonl /tmp/received-batches.jsonl \\
  --output-json /tmp/receiver-report.json

perception/target/release/billiards-perception-infer \\
  --onnx /tmp/billiards-onnx/model.onnx \\
  --source-checkpoint models/billiards-yolov8n.pt \\
  --input backend/vision/test_data \\
  --provider cpu \\
  --image-size 1280 \\
  --confidence 0.4 \\
  --iou 0.45 \\
  --detection-batches-tcp 127.0.0.1:9000 \\
  --detection-batches-max-message-bytes 16777216 \\
  --detection-batches-connect-timeout-ms 3000 \\
  --detection-batches-write-timeout-ms 5000 \\
  --engine-instance-id stage2-run-01 \\
  --source-session-id recorded-images-01 \\
  --output-json /tmp/rust-replay-report.json
```

Both output paths are created without overwriting existing data. If replay
fails after output creation, the receiver JSONL may contain a valid prefix.
Treat receipt as complete only when both processes succeed and the receiver
reports the expected ordered batch count. The receiver records local monotonic
receipt times, ordering, gaps, duplicates, and bounded per-batch details; it
does not subtract Rust timestamps from the Python clock or claim one-way
latency. The detection-batch TCP sender performs synchronous writes during replay, so
the reported `timings.batch_export_ms` includes serialization and socket
delivery, while `summed_image_pipeline_ms` covers image decode, preprocessing,
inference, and decode/NMS only. This is a recorded-replay check, not a live
camera pipeline.

The Rust detection-batch receiver binds to loopback and accepts one producer
connection. Its message, expected-batch, and receipt-detail limits are
configurable and bounded. It does not implement producer authentication,
asynchronous queues, or reconnection.
Unpaced still images preserve
`source.timestamp: null`; paced replay publishes synthetic `frame_arrival`
timestamps from the Rust monotonic domain. Frame IDs follow original input
order and are scoped by the supplied source session and engine instance.

Build and test from the repository root:

```bash
cargo fmt --manifest-path perception/Cargo.toml -- --check
cargo test --manifest-path perception/Cargo.toml
cargo clippy --manifest-path perception/Cargo.toml --all-targets -- -D warnings
```

Export the current trained checkpoint to a temporary directory with the
retained exporter. The model remains outside this repository. The runtime
requires the explicit source checkpoint to verify its hash, the exported ONNX
hash, and the manifest class map:

```bash
uv run --project . python tools/export_high_res_model.py \
  --model models/billiards-yolov8n.pt \
  --output-dir /tmp/billiards-onnx \
  --imgsz 1280

cargo run --release --manifest-path perception/Cargo.toml --bin billiards-perception-infer -- \
  --onnx /tmp/billiards-onnx/billiards-yolov8n.onnx \
  --source-checkpoint models/billiards-yolov8n.pt \
  --input backend/vision/test_data \
  --provider cpu \
  --image-size 1280 \
  --confidence 0.4 \
  --iou 0.45 \
  --max-detections 300 \
  --dump-preprocessed-directory /tmp/billiards-rust-inputs \
  --detection-batches-jsonl /tmp/billiards-detection-batches.jsonl \
  --engine-instance-id rust-audit-20261020-01 \
  --source-session-id recording-20261020-01 \
  --output-json /tmp/billiards-rust-inference.json \
  --retain-provider-profile /tmp/billiards-ort-profile.json
```

Confidence, IoU, model image size, model paths, and provider policy are
required; the `0.4`, `0.45`, and `1280` values above are for the existing
equivalence harness, not recommended production thresholds. `--max-detections`
defaults to 300 to match
the common Ultralytics cap; the chosen value is recorded. `--warmup-runs`
defaults to one and can be set to zero. `--intra-op-threads` is optional; when
omitted, the selected backend chooses its own thread settings and the report
records that it was omitted. OpenVINO applies an explicit value to its CPU
inference-thread property; ONNX Runtime applies it as its intra-op thread count.

The runtime accepts only the exported single-image signature (`images` float32
NCHW input and `output0` float32 `[1, 6, N]` output), and verifies it against
the fixed input size and the two contiguous classes in the export manifest.
Unexpected names, shapes, dtypes, and non-finite outputs fail the run. The
current checkpoint provides generic `cue` and `ball` classes only. It does not
identify numbered balls such as the 6, 7, or 8 ball.

The Rust preprocessor performs aspect-preserving, centered letterboxing,
OpenCV-style half-pixel bilinear sampling, RGB channel ordering, NCHW layout,
and `[0,1]` normalization. It decodes files with `image-rs`. In the current
10-image comparison, all nine PNG tensors matched the Ultralytics/OpenCV
reference exactly, while the JPEG tensor differed due to decoder behavior.
See the [recorded-image prototype report](../thoughts/performance-audits/rust-inference-prototype-20261020/report.md)
before treating this as preprocessing parity. The currently configured CoreML
package declares a 1280-square input, though its exact checkpoint lineage and
CoreML-versus-ONNX outputs remain unverified.

`--dump-preprocessed-directory` optionally writes raw little-endian float32
NCHW tensors, with each path recorded in the JSON result. Load them with NumPy
as `np.fromfile(path, dtype="<f4").reshape(1, 3, image_size, image_size)` and
compare directly with the Python/OpenCV input tensor.

One isolated `grid.jpg` comparison found 695,791 of 8,294,400 decoded pixels
with at least one channel differing between `image-rs` and OpenCV; mean
absolute channel difference was 0.06365/255 and maximum was 3/255. This is
single-image decoder evidence, not detection-accuracy evidence, and does not
justify adding another decoder. The one-off raw-byte probe was retired; the
input and result hashes are preserved in the
[raw decoder comparison](../thoughts/performance-audits/rust-inference-prototype-20261020/evidence/raw-jpeg-decoder-comparison.json).

The current comparison settings use a 1280 square input. The configured
`models/billiards-yolov8n.mlpackage` declares `imgsz: [1280, 1280]` in its
Ultralytics export metadata, and the retained runtime uses that exported size
when preparing inputs. The live detector does not pass `imgsz` explicitly, so
this behavior depends on the selected artifact metadata. The checkpoint used
for the ONNX comparison has the same SHA-256 as the ONNX export manifest's
source checkpoint, but the configured CoreML package has no lineage manifest;
its exact relationship to that checkpoint and CoreML-versus-ONNX output parity
remain unverified. Thus 1280 is a supported effective-size observation for the
current configured package, not proof of full production-path parity. The
results are static, unpaced image processing, not camera-to-client latency,
temporal accuracy, or real-time capacity evidence.

`--provider cpu` disables automatic accelerators and requests the CPU provider.
`--provider auto` explicitly selects ONNX Runtime's max-efficiency device
policy. On macOS this can partition the graph between CoreML and CPU; node
provider evidence is reported rather than assuming the policy's preferred
provider ran every node. Linux behavior and a directly requested CoreML-only
profile still require qualification.

OpenVINO is an explicit runtime choice in normal macOS and Linux builds. Build
the inference executable with
`cargo build --release --manifest-path perception/Cargo.toml --bin billiards-perception-infer`
and select it with `--provider openvino`; API replay continues to default to
`cpu`. This selection uses the direct OpenVINO Runtime C API through
`openvino-rs`, not ONNX Runtime's OpenVINO execution provider. The model remains
the same verified ONNX export, and preprocessing, decoding, calibration, and
scene publication are shared with the ONNX Runtime path. The backend targets
the OpenVINO CPU device and maps an explicit inference-thread setting to
OpenVINO's `INFERENCE_NUM_THREADS` CPU property.

The binary is built with runtime linking, so OpenVINO need not be installed on
the build host. At launch, OpenVINO Runtime 2025.1 or newer must be discoverable
through the platform environment (`LD_LIBRARY_PATH` on Linux or
`DYLD_LIBRARY_PATH` on macOS), an OpenVINO install directory, or a supported
standard installation path. The Rust supervisor forwards the platform library
path only for an explicit OpenVINO selection. In container deployments, the
OpenVINO runtime must be installed in or mounted into the API container itself;
a host installation alone is invisible to the isolated child process. The
current API image does not package OpenVINO by default. Missing or incompatible
runtime libraries fail the request; it never silently falls back to CPU.
Actual Apple-Silicon runtime execution still needs a host smoke test. See
[ADR-0045](../docs/architecture/decisions/0045-use-the-native-openvino-runtime-backend-for-openvino-selection.md)
for the decision and qualification requirements. ADR-0043's target A/B
measured the earlier ONNX Runtime provider implementation and does not qualify
this native backend.

Runtime provenance identifies `openvino-rs` and the native CPU backend. This
API does not expose ONNX Runtime per-node provider assignment, so OpenVINO node
counts and retained ORT provider traces are unavailable; the audit says so
explicitly rather than implying graph partition or fallback evidence. The safe
OpenVINO tensor API copies prepared input into the persistent native input
tensor, and that copy is included in inference timing. CPU and auto continue to
use ONNX Runtime `2.0.0-rc.15` with the bundled ONNX Runtime 1.31 distribution.
Their short profiling session and separate measurement session remain
unchanged.
This is a prototype choice, not an approved dependency or deployment strategy.
The observed macOS auto-policy profile is evidence for this one model/runtime
combination; it does not qualify CoreML or Linux deployment.

The existing scene snapshot contract and four-byte framing tests remain
available through `cargo test`. Socket ownership, camera acquisition,
transport backpressure, production accuracy acceptance, and macOS/Linux
qualification are subsequent migration stages governed by the architecture
records in [`../docs/architecture/`](../docs/architecture/README.md).

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
not acquire a camera, connect to inference or the API, or qualify physical
capture-buffer reuse. The executable remains recorded-image-only; no camera
library or platform adapter has been selected.

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

The standalone live executable now composes this preflight and pipeline with an
opt-in feature-gated GStreamer physical source. The Python supervisor forwards
the same immutable capture request and validates request-bound readiness. A
macOS built-in-camera smoke passed through the complete local loopback path with
the checked-in test detector and then shut down cleanly; it is integration
evidence only. The API continues to reject Rust live selection. No calibrated
table positions, production-model accuracy, reliable exposure timestamp,
sustained freshness, preview handover, or Linux physical behavior has been
qualified. The feature-enabled all-target suite and optimized live/helper
builds also pass in an Ubuntu 24.04 x86_64 container with GStreamer 1.24, but
that container had no V4L2 device.
