# ADR-0030: Request-bound capture acknowledgement for live startup

- Status: accepted
- Date: 2026-10-20
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

> Current status: the live-v3 acknowledgement remains active. Its historical
> recorded-replay v2 compatibility note was retired by ADR-0094.

## Context

At the time of this decision, Python resolved one immutable, revisioned
capture-settings request, and Rust could compare that request with an opened
source report before its first read. ADR-0065 has since moved request
construction to the Rust configuration authority. The report contains
requested-versus-negotiated modes, source capability declarations, timestamp
semantics, source-reported required control readbacks, and source-reported
output geometry. Those values are available before processing starts, while
source properties that cannot be observed remain explicitly unknown.

The existing loopback scene handshake binds engine and source-session identity
and preserves the recorded-replay path. Scene data must not be admitted as
ready for live capture until Python has checked the Rust source report against
the exact request revision it supplied. Configuration can change while a
process is starting, so revision alone is not enough to correlate the response;
the acknowledgement also needs a digest of the exact canonical request bytes.
The digest is provenance/correlation data, not a secret or proof of which
process produced the message. No camera adapter or live Rust executable is
currently qualified, so this decision defines the contract and startup rule;
it does not enable live capture.

## Decision

Define a strict `capture-session-ack` v1 contract and use it for live-mode
startup on the existing local scene-transport control path. The acknowledgement
contains schema version, configuration revision, SHA-256 of the exact canonical
capture-request JSON bytes, outcome (`compatible`, `rejected`, `failed`, or
`stopped`), and the opened `CaptureSessionReport` when one exists. The source
report retains source-session ID and source-reported values; the acknowledgement
must never copy requested values into applied fields.

The Rust owner continues to reject required profile, geometry, capability,
timestamp, exposure, and gain mismatches before its first capture read. It sends
the acknowledgement with the producer startup handshake. Python independently
checks schema, request revision and digest, source-session identity, requested
versus negotiated profiles, required capabilities, timestamp semantics,
required control readbacks, and reported frame geometry against the exact
snapshot it supplied. Python sends `ready` and admits scene snapshots only
after a compatible acknowledgement passes those checks. Rejected, failed,
stopped, stale, malformed, or mismatched acknowledgements do not establish
live readiness.

Use scene-transport protocol v3 for this live-mode startup handshake. Preserve
protocol v2 and its existing behavior for recorded replay. Keep the request
digest and producer/session IDs as non-secret correlation and provenance
values; the message carries no credential, login, permission, or authentication
field. This acknowledgement confirms cross-process agreement over reported
software state. It does not prove that a physical camera honored the adapter's
claims. Physical adapter readback, calibration application, and platform
qualification remain separate evidence gates.

## Consequences

### Positive

- Python can report whether the runtime opened the exact settings revision it
  resolved, rather than treating process launch or a matching first scene as
  sufficient readiness.
- Rust and Python independently enforce the pre-read source gate and the
  request/report comparison, while reusing the existing bounded loopback
  handshake and session identity.
- Recorded replay remains compatible and does not pretend to have camera
  settings it never applied.

### Negative

- Live and replay startup use different protocol versions; both peers must
  support v3 before live capture can be enabled.
- The report is duplicated across the local boundary and must remain bounded,
  versioned, and consistent with the capture-settings schema.
- Agreement between two processes does not establish hardware truth. A faulty
  adapter can still report values it did not apply.

## Alternatives considered

### Treat process launch or the first scene as runtime acknowledgement

Rejected because a live scene does not identify whether the current Python
configuration revision and its required source readbacks were checked. It can
also arrive after an outdated or incompatible configuration has begun use.

### Put applied fields into the Python request

Rejected because requested settings cannot serve as evidence that the source
applied them. It would collapse configuration intent and runtime observation
into one unauditable value.

### Replace the scene channel with a new control service

Deferred because the existing loopback transport already owns bounded local
startup and producer identity. A second process channel would add lifecycle,
ordering and cleanup work before a camera provider exists. Revisit only if the
handshake cannot meet measured deadlines or ownership requirements.

### Use protocol v3 for all recorded replay producers

Rejected for this stage because replay has no camera settings to acknowledge;
keeping v2 avoids an unnecessary compatibility break. A later migration may
unify versions when a concrete replay requirement justifies it.

## Verification

- Contract-schema tests reject unknown fields, invalid outcomes, malformed
  hashes, invalid numeric values, unbounded reports, and missing required
  fields.
- Python validator tests cover matching request/report pairs, stale revisions,
  wrong request digests, wrong session IDs, profile and geometry mismatches,
  missing or unknown required controls, optional unsupported controls, and
  rejected/failed/stopped outcomes.
- Rust serialization tests prove that the acknowledgement includes its opened
  source report and does not substitute requested values for readbacks.
- Protocol tests prove live v3 cannot send `ready` or admit a scene before a
  compatible acknowledgement, while recorded replay v2 retains current
  behavior. A capture mismatch remains a zero-read failure.
- Keep live provider selection disabled until a real adapter and target
  macOS/Linux hardware pass control, geometry, freshness, accuracy and lifecycle
  qualification.

## Revisit conditions and related records

ADR-0014 remains the authority for Python-owned persisted configuration,
calibration, revision semantics, and the broader application acknowledgement.
ADR-0028 continues to define producer-epoch watermarks and protocol-v2 replay
behavior; this record adds the request-bound acknowledgement required for
future live startup.
ADR-0029 defines source-reported control readbacks, while ADR-0015 governs
exclusive camera ownership. Revisit v3 only with measured transport limitations
or a change in runtime ownership; any wire change must keep strict schema,
bounded messages, request correlation, and recorded-replay compatibility
explicit.

## Evidence and implementation tracking

The current source report and pre-read gate live in
[`perception/src/capture.rs`](../../../perception/src/capture.rs) and
[`perception/src/live_config.rs`](../../../perception/src/live_config.rs).
Python creates the request in
[`backend/perception/capture_configuration.py`](../../../backend/perception/capture_configuration.py).
The existing framed transport is in
[`perception/src/scene_transport.rs`](../../../perception/src/scene_transport.rs)
and [`backend/perception/scene_ingress.py`](../../../backend/perception/scene_ingress.py).
The versioned schema and Python validator now live in
[`perception/contracts/capture-session-ack/v1/`](../../../perception/contracts/capture-session-ack/v1/)
and [`backend/perception/capture_acknowledgement.py`](../../../backend/perception/capture_acknowledgement.py).
Their shared default/rejected fixtures and 29 validator tests pass. The Python
scene ingress accepts a configured live-v3 acknowledgement before returning
`ready`, preserves recorded replay on v2, rejects changed reports under a
reused source identity, and includes the larger acknowledgement hello in its
message-bound check. Rust hashes the exact bytes from Python's deterministic
request encoder, carries the digest and full report through preflight, and
performs the live v3 handshake on its dedicated bounded writer before allowing
the pipeline to process frames. The v3 response must echo the exact revision
and digest. Live transport tests cover the serialized acknowledgement, the
matching echo, mismatched echo rejection, identity binding, and v2 preservation.
Rust acknowledgement construction, serialization, and sender-side tests pass in
the 203-test all-target suite; the focused Python ingress and acknowledgement
selection passes 70 tests, with Ruff, Black, isort and mypy passing. Rejected,
failed and stopped outcomes remain schema-only, and no real source adapter
currently supplies qualified hardware evidence or connects the live path to the app.
