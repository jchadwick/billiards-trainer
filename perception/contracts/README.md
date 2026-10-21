# Perception contracts

Contracts in this directory define the process-neutral boundary between the
primary perception engine and the retained Python gateway. Versioned schemas
are immutable once published; incompatible semantics require a new version.

## Scene snapshot v1

`v1/scene-snapshot.schema.json` defines a complete, replaceable ball scene.
`source.session_id` plus `source.frame_id` identify the input observation;
`engine_instance_id` separates engine restarts; `snapshot_revision` tracks
semantic updates; and `transport_sequence` orders sent messages. These values
are distinct because an annotation or health update can create a new snapshot
without creating a new camera observation.

Every timestamp carries its clock type, opaque clock-domain identity, unit,
and semantic stage. Nanosecond ticks are decimal strings so Unix-epoch values
remain exact in JSON clients that use IEEE-754 numbers. The
`shared_memory_publication` stage means the time bytes became available to a
consumer; it never implies camera exposure. Consumers may subtract timestamps
only when a platform test has proved that their clock domains are comparable.
Otherwise each process reports its own segment and does not invent end-to-end
age.

Each ball carries a stable track ID independently of its nullable numbered-ball
identity, its own last observed source/time, a raw or rectified image-space
measurement with its image-space dimensions, an optional calibrated millimeter
position, and a separately tagged filtered or extrapolated estimate. An empty
list with `success` means a successful empty observation. `failed`, `degraded`,
and `skipped` are explicit outcomes and cannot be mistaken for a newly observed
empty table.

Calibration has an explicit valid/invalid/unavailable state. Table coordinates
are forbidden while calibration is invalid or unavailable. Position uncertainty
is separately available or explicitly unavailable; detector confidence is not
used as a substitute for spatial uncertainty.

The schema intentionally permits unavailable table positions and unknown
identity. The current generic `cue`/`ball` model cannot supply numbers 1–15;
identity evidence can arrive in a later annotation snapshot without refreshing
the underlying location observation. Cue evidence has an independent
observed/not-observed/unavailable status, so missing cue work does not invalidate
ball locations. The API adapter remains responsible for mapping this richer
internal contract into unchanged public v1 payloads.

`tracking::SceneTrackerV1` is the first source-time association prototype. It
emits the detector's measured center immediately, uses bounded global
association only to maintain IDs, and preserves the last observed measurement
and timestamp through occluded/lost states. It requires a caller-supplied actual
snapshot-publication timestamp; it never relabels the detector batch's earlier
publication time. Recorded replay can optionally attach table coordinates when
the entire frame passes the immutable calibration bundle's applied context.
Missing calibration marks table coordinates unavailable; a configured but
incompatible calibration degrades the scene while retaining image measurements.
The replay prototype is not connected to live capture, bounded scene transport,
or the Python gateway.
Its emitted example is pinned in
[`scene-rust-tracked-success.json`](v1/scene-rust-tracked-success.json), which
both the Rust round-trip tests and Python JSON Schema tests validate.

## Table calibration v1

`calibration/v1/table-calibration.schema.json` defines the immutable geometry
bundle Rust applies to an image measurement. Its homography maps source-image
pixels to millimeters on the playing-surface plane. The coordinate frame is
center-origin: corner 0 is the image-space top-left calibration corner, and
corners 1, 2, and 3 continue clockwise; positive X points from corner 0 to 1,
and positive Y from corner 0 to 3. This matches the retained Python
`CameraCalibrator` table transform after converting its meter output to
millimeters.

The bundle is valid only for its exact camera profile, image space, source
resolution, calibration revision, and rectification revision. The consumer
must map detector coordinates back to source-image pixels first. It never
silently rescales or adjusts for a crop. A raw-image homography does not remove
lens distortion; an unrectified frame requires a calibration authored for that
same raw profile or its physical coordinates remain unavailable. The transform
does not imply a position-accuracy guarantee: table uncertainty remains
unavailable until held-out physical measurements justify it. The shared
[center-origin fixture](calibration/v1/table-calibration-center-origin.json)
pins the serialized frame and units.
The Python-generated perspective fixture
([bundle](calibration/v1/table-calibration-python-export.json)) is shared by
the Python exporter and Rust projection tests to cover meter-to-millimeter
conversion and non-affine transforms end to end.

The Rust `geometry::calibrate_detection_batch` adapter applies this bundle to
every source-image detection in a successful frame batch only when the caller
provides the exact applied camera profile and revisions. It preserves the
original frame/detection identity and rejects the entire transformed result if
any point is incompatible. It is an in-process geometry layer, not a new wire
schema. The recorded-image audit can opt in with a bundle and explicit applied
profile/revisions; it retains the raw detection batch and reports unavailable
coordinates for incompatible frames. Its only public options constructor loads
and validates the bundle while hashing those exact bytes, so reported bundle
provenance cannot drift from the geometry used in the replay. Live capture still
lacks an applied calibration control path.

## Detection batch v1

`detection-batch/v1/detection-batch.schema.json` is the Stage 2 inference
boundary. Each immutable batch contains the full model output for one input
frame, including source and inference timing, model hashes and class map,
effective decoding settings, and source-image coordinates. A source timestamp
may be null when its clock is unavailable; replay must not substitute a later
inference or publication timestamp. Detection indices identify entries only
inside the same engine/session/frame batch and are not tracking IDs.

The batch contract distinguishes successful empty inference from failed or
skipped work. It keeps the raw detection boundary honest until Stage 3 adds
cross-frame association, when the existing tracked-scene v1 contract can be
emitted without changing its semantics. See the [detection batch contract
notes](detection-batch/v1/README.md), [schema](detection-batch/v1/detection-batch.schema.json),
and [shared fixtures](detection-batch/v1/).

## Capture settings request v1

`capture-settings/v1/capture-settings.schema.json` defines the Python-owned
request sent to a future Rust capture provider. Shared default and
calibration-bearing fixtures are validated by both runtimes. The request keeps
device controls and reconnect policy separate from source-reported settings;
matching preflight metadata is not an acknowledgment of every requested value.
Explicit backend requests must match the opened report's canonical identifier;
`auto` accepts the provider's selected backend. See the
[capture settings notes](capture-settings/v1/README.md).

## Capture settings request v2

`capture-settings/v2/capture-settings.schema.json` keeps the requested and
negotiated profiles as raw camera input modes and requires
`requested.output_frame_geometry` for the emitted RGB8 image. V2 frame limits
and calibration dimensions are checked against this output geometry. Rust
accepts v1, v2, and v3 through an explicit version dispatcher; direct v1 parsing
remains strict and unchanged. The v2 default fixture mirrors the effective
application defaults at 1920×1080 RGB8 input/output. A separate transformed
example requests MJPG input at 1920×1080 and emits 1280×720 RGB8.

## Capture settings request v3

`capture-settings/v3/capture-settings.schema.json` retains v2's required
emitted RGB8 geometry and adds a tagged `requested.camera_controls` policy.
`{"mode":"source_defaults"}` sends no control request and imposes no
camera-control capability or readback requirements; the opened source report
still carries its honest supported, unsupported, or unknown statuses.
`{"mode":"require","requirements":...,"requested":...}` keeps the
strict v1/v2 capability and exact-readback contract. Unknown policy tags and
unexpected policy fields are rejected. The source-default fixture does not
invent control values, while a separate require fixture pins strict behavior.
V1/v2 parsing and their wire shapes remain unchanged. Source-report and
acknowledgement v2 already carry all observed control statuses, so v3 does not
introduce another acknowledgement schema.

## Capture session acknowledgement v2

`capture-session-ack/v2/capture-session-ack.schema.json` requires the source
report to include the explicitly observed emitted RGB8 dimensions. The Rust
live sender selects this acknowledgement only after v2 or v3 output geometry
has matched during preflight. The existing v1 source report omits this field
and retains its byte/schema shape; v1 remains limited to output dimensions
equal to the negotiated profile.

## Capture session acknowledgement v1

`capture-session-ack/v1/capture-session-ack.schema.json` defines the source
report returned for one opened capture session. It binds to the exact request
revision and canonical JSON digest, then keeps the source session, effective
frame geometry, and per-control readbacks distinct from requested values. A
compatible acknowledgement must match the supplied request before live scenes
can be admitted. The schema, Python validator, opt-in receiver-side v3 handshake,
and Rust live sender are implemented. Rust waits for a matching v3 `ready`
before processing captured frames, while replay remains on v2. The live camera
provider and non-compatible Rust outcome reporting remain incomplete. See the
[acknowledgement notes](capture-session-ack/v1/README.md) and
[ADR-0030](../../docs/architecture/decisions/0030-request-bound-capture-acknowledgement-for-live-startup.md).
