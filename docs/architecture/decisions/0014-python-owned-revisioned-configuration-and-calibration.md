# ADR-0014: Python-owned revisioned configuration and calibration

- Status: superseded
- Date: 2026-10-19
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: ADR-0056 (permanent Python configuration/calibration authority)

## Context

Historical proposal: ADR-0056 moves configuration and calibration authority to
Rust while allowing the Python HTTP service to retain route adapters. The
revision and acknowledgment rationale below remains relevant to that gateway
contract.

The project has validated atomic configuration persistence, existing calibration
routes/screens, and stored assets worth retaining. Splitting perception into a
new process creates a risk of independently mutable settings, unacknowledged
changes, or observations produced with a different calibration than the gateway
assumes. Some current routes access concrete Vision objects directly.

## Decision

Retain Python as the configuration persistence and calibration-authoring
authority. Rust receives an immutable validated effective configuration and
calibration bundle with a content identity/revision. It reports whether the
bundle is accepted, applied, rejected or requires restart, plus actual negotiated
camera/model/backend properties.

Distinguish desired persisted state from applied runtime state. A persisted
change cannot be represented as applied before acknowledgement. Unsupported
settings fail explicitly; supported keys must not become silent no-ops.
Define capabilities and restart requirements at the control boundary.

Each observation reports the revisions actually used. Applying new geometry
invalidates affected tracks/overlays/predictions according to their contracts;
in-flight older work remains labeled with its old revision and cannot overwrite
a newer spatial state. Only compatible calibration can support readiness for
physical-position operations.

Keep calibration computation and UI initially. Obtain source-identified raw and
rectified full-resolution stills from Rust for appropriate workflows, separately
from low-resolution preview. Camera intrinsic calibration cannot use rectified
preview as a substitute for the required original evidence.

## Consequences

### Positive

- Preserves validated persistence and existing user workflows.
- Gives every measured scene reproducible effective settings.
- Separates configuration success from runtime application failure.
- Makes cross-process calibration changes and restart behavior inspectable.

### Negative

- Commands need correlation, deadlines, acknowledgements and idempotent retry rules.
- Persisted and applied state can temporarily differ and must be surfaced.
- Existing routes need explicit control adapters.
- Calibration compatibility checks may expose previously tolerated bad assets.

## Alternatives considered

### Let both processes read and mutate config.json independently

This preserves file format but not one authority or atomic application semantics.
It creates revision ambiguity and can expose partially coordinated changes.

### Move calibration UI, solvers and persistence into Rust immediately

Possible later, but unnecessary to improve primary localization and expensive
relative to reusing established workflows.

### Assume a file write means the engine applied the settings

This hides rejection, camera negotiation failure, restart requirements and
in-flight results from older geometry.

## Verification

- Commands exercise accepted/applied/rejected/restart-required responses,
  timeout, retry, duplicate request IDs and worker restart.
- Existing atomic persistence/validation tests remain applicable.
- Calibration replacement during processing never mixes pixels and revisions.
- Missing/incompatible calibration blocks only operations requiring it and
  reports degraded capabilities accurately.
- Raw/rectified reference images preserve source/mode metadata through UI routes.

## Revisit conditions and related records

Move authority only through a successor ADR that defines persistence and
migration of existing assets. Preserve ADR-0004 validation/atomic persistence
and ADR-0002 spatial invalidation protections.


## Evidence and implementation tracking

Status is proposed; Rust migration implementation and qualification are planned,
not completed. This record governs migration stages 1, 4 and 5.

- [Migration plan](../../../thoughts/rust-perception-migration-plan.md).
- [Architectural review and evidence limitations](../../../thoughts/architectural-review-2026-10-19.md).
- [Existing backend/config.py](../../../backend/config.py).
- [Existing backend/api/routes/calibration.py](../../../backend/api/routes/calibration.py).

Before accepting this decision for implementation, resolve its relevant prototype
and compatibility questions and record the evidence. Acceptance records commitment;
it does not claim all verification has passed. Link implementing changes and actual
verification artifacts as they exist. Follow the [decision register](../README.md)
for status changes and supersession.

### Expected capture preflight prototype (2026-10-20)

The Rust `live_config` module validates an explicitly supplied expected capture
configuration against metadata available from one opened source session. It
binds a safe configuration revision, exact device and requested/negotiated
video profiles, timestamp semantics, required control capabilities, freshness
and RGB8 memory limits, and any validated calibration bundle. It rejects
metadata mismatches and unsupported or unknown required capabilities. The
`live_config_preflight` integration tests exercise the real capture coordinator
with a fake source and verify profile, device, timestamp, capability, revision,
calibration, and limit behavior. `cargo fmt --check`, strict Clippy, and all
170 perception tests pass.

This is an expectation/compatibility preflight, not a runtime acknowledgement.
`CaptureSessionReport` does not report applied camera-profile IDs, image-space
or calibration revisions, or exposure/gain values, so a matching report cannot
prove those settings were applied. The initial prototype did not connect the
expectation to a capture owner; the later integration below adds that software
gate while leaving the Python bridge and actual-application acknowledgement
open. ADR-0014 remains proposed pending the control protocol,
application acknowledgements, replacement/in-flight revision semantics, and
raw/rectified reference-image compatibility evidence. No credentials or
authentication fields are present.

### Preflight-gated capture and pipeline binding (2026-10-20)

`CaptureRuntimeCoordinator::start_preflighted(expected, factory)` now checks
the coordinator's per-frame and aggregate RGB8 limits before worker startup,
opens the source on its owner thread, and compares its session report with the
expected device, requested/negotiated profiles, required reported capabilities,
timestamp semantics, and frame geometry before the first read. It reports
session-bound pending/compatible/rejected/failed/stopped readiness through a
bounded wait. A readiness timeout does not release worker ownership. A rejected
report fails and wakes the mailbox before source release and skips all reads;
release and quarantine remain owned by the capture thread.

The caller constructs `CaptureScenePipeline` before capture starts to establish
the monotonic clock origin, then binds the returned compatibility exactly once
before receiving frames. Binding verifies the explicit capture-configuration
revision, image space, maximum frame age, tracker camera profile and
rectification revision, and complete validated calibration bundle. This
capture revision is separate from the tracker's local revision. Once bound, the
pipeline rejects a mismatched source session, dimensions, or timestamp
semantics before inference; absent applied calibration leaves table positions
unavailable. The pre-existing unbound replay path remains available.

Rust formatting, strict Clippy, and all 182 perception tests pass. The focused
anonymous API, workflow, and local-deployment Python selection passes 44 tests
with coverage collection disabled; this run does not measure the repository
coverage gate. The Python effective-settings/revision/ack bridge is still
missing, the current report cannot acknowledge settings it does not contain,
and no physical camera adapter or accuracy qualification is enabled. This work
does not add authentication; no-auth remains an invariant across APIs, clients,
IPC, and project-controlled workflows.

### Python-owned capture-settings request bridge (2026-10-20)

The configuration store persists `metadata.configuration_revision`, starting
at the schema revision and incrementing atomically for each changed commit.
Partial updates cannot set it, and full replacements cannot choose its value.
Legacy files without the field receive the schema revision in memory before
their next commit. Malformed revisions fail closed, and a reload of the same
file cannot lower the currently loaded revision. The resolver reads one
deep-copied `get_all()` snapshot so its settings and revision always come from
the same committed configuration state.

`backend/perception/capture_configuration.py` resolves the current effective
camera settings and serializes the strict `CaptureSettingsSnapshotV1` request.
It carries device/backend requests, requested and expected negotiated modes,
an effective camera-profile fingerprint, image space and revisions, optional
validated calibration, read-completion timestamp semantics, requested camera
controls, reconnect/stop policy, frame-freshness and memory limits. The
fingerprint covers device selection, backend, video mode, pixel format, image
space, and rectification revision. Photometric controls and calibration
revision are excluded because they do not define the geometric mapping. The
fingerprint is a configuration identity, not hardware serial evidence. The
initial 100 ms maximum frame age is an adjustable configuration default; it is
not a measured latency result. The resolver rejects file paths and unsupported
timestamp assumptions instead of treating them as camera settings.

`perception/src/capture_settings_wire.rs` rejects unknown fields and unsupported
schema versions, enforces identifier and resource bounds, checks cross-field
rules, and routes the request through the existing Rust constructors for
profiles, calibration, limits and preflight settings. Both default and
calibration-bearing JSON fixtures pass through Python's resolver, JSON Schema
validation, and Rust's parser. Explicit backend requests are compared with the
opened report's canonical `backend_id`; `auto` accepts any supported reported
backend. Requested controls and lifecycle policy remain separately available
to a future adapter; no request field is represented as applied or verified.
Eighty-two focused Python capture/configuration tests, seven Rust
wire-contract tests, and seventeen Rust preflight tests pass, along with Python
formatting, linting, and production-module typing, Rust formatting, and strict
Clippy. The full backend suite passes 1,491 tests with two model-dependent
skips; configured coverage is 52.14%, below the 80% gate.

This bridge is a contract and resolver, not a live-provider connection. The
API continues to reject Rust live mode. The source report now carries actual
exposure mode/value, gain, brightness, contrast, saturation, autofocus and
buffer readbacks independently from capability declarations. Preflight checks
required exposure and gain readbacks against the corresponding requested values
and rejects unknown, unsupported, or mismatched required controls before the
first source read. Optional control readbacks remain explicitly visible. At
this checkpoint, frame geometry was not yet included in the source report; the
following progress entry records that addition. The report is still not
transported as a Python-visible runtime acknowledgement. Adapter selection,
target-hardware qualification, and the repository's 80% coverage gate remain
open before enabling live capture, so ADR-0014 remains proposed.

### Source-reported camera-control readbacks (2026-10-20)

ADR-0029 accepts a typed `applied(value)` / `unsupported` / `unknown` readback
for each requested camera control. Rust stores those readbacks in the opened
`CaptureSessionReport`, which binds them to the capture session. The strict
Python request parser carries exposure mode, optional manual exposure value,
and gain into expected settings. Required controls need concrete request values
and matching source readbacks; otherwise the coordinator rejects the source
before its first read. A capability marked `supported` alone no longer suffices
for the required value check. Optional controls can remain unknown or
unsupported without blocking localization.

Formatting, strict Clippy and all 195 Rust tests pass. These fake-source tests
verify the state model and ordering of the pre-read gate, not physical device
behavior. Full application acknowledgement still needs the effective profile,
image-space/rectification and calibration revisions to be confirmed by the
components that apply them, then conveyed to Python with the source-session
identity. No camera adapter is selected, and Rust live mode remains unavailable.

### Source-reported frame geometry (2026-10-20)

The `CaptureSessionReport` now includes source-reported frame geometry bound to
the opened session: capture configuration revision, camera-profile ID, output
image space, and rectification revision. `ExpectedCaptureSettings` compares all
four values with the Python request before the first frame read, and the
resulting compatibility object carries the validated geometry into the scene
pipeline's one-time session binding. The Rust request parser maps the Python
snapshot's profile and geometry fields into those expectations.

All-target Rust tests, including mismatch, serialization, and reject-before-read
cases, pass (196 tests); formatting and ADR validation pass. These tests prove
the software gate only. No live adapter is selected, no hardware negotiation
has supplied these values, and the report is not yet a Python-visible runtime
acknowledgement. Calibration compatibility continues to be validated and bound
at the scene-pipeline boundary. Keep ADR-0014 proposed until the runtime
acknowledgement, replacement/in-flight semantics, raw/rectified reference-image
workflow, and hardware qualification are verified.
