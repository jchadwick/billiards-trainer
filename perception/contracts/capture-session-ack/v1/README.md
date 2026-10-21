# Capture session acknowledgement v1

This contract reports the outcome of one Rust capture-session preflight back to
the Python process that supplied the settings request. The acknowledgement is
bound to the request's positive configuration revision and SHA-256 of its
canonical compact JSON bytes. `source_report.session_id` must match the source
session identity declared in the producer handshake.

The `compatible` outcome is allowed only with a complete source report and no
failure. `rejected`, `failed`, and `stopped` never establish readiness. A report
contains source-reported values, including explicit `applied`, `unsupported`,
and `unknown` control readbacks. It does not repeat request values as applied
values. Python compares required controls, source-reported output geometry,
profiles, capabilities, timestamp semantics, request revision and digest before
returning `ready` or admitting live scene snapshots.

The request digest is computed over the exact bytes emitted by Python's
deterministic `CaptureSettingsSnapshotV1.to_json()` encoder. Rust retains that
digest from parsing through the validated preflight binding; callers must pass
those request bytes through unchanged. The request digest and source/engine
session identifiers are non-secret correlation and provenance data. They do
not identify or authenticate a process. Successful validation proves that both
runtimes agree on the request and source report; it does not prove that a
physical camera honored the adapter's report. Hardware qualification and
calibration application remain separate gates.

The default fixture matches the shared default request and demonstrates a
compatible report with an unsupported optional control. The rejected fixture
demonstrates a required gain readback that differs from the requested value.
The Python scene ingress supports an opt-in protocol-v3 mode that validates the
acknowledgement before returning `ready`; recorded replay keeps the existing v2
handshake. Rust now serializes an acknowledgement only from a compatible
preflight report, sends it over the existing bounded writer, and waits for
Python's `ready` response to echo the exact revision and digest before the
capture pipeline can process frames. The receiver and sender both account for
the larger live hello when enforcing message limits. Rejected/failed/stopped
outcomes are represented by the schema but are not yet emitted by the Rust
runtime. No platform camera provider is connected or enabled.
