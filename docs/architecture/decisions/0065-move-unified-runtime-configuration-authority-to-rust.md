# ADR-0065: Move unified runtime configuration authority to Rust

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The application currently keeps its full configuration document in
`config.json`, but Python owns its defaults, validation, revision counter,
atomic persistence, and HTTP route decisions. Rust live capture receives a
Python-computed capture-settings request and validates that request again
before opening a source. This duplicates policy and leaves Python as the
effective authority for camera and perception settings, contrary to
ADR-0064's migration boundary.

The existing configuration document also contains gateway and UI settings.
Public routes support partial updates, section reset, validation-only checks,
warnings with an explicit force option, import/replace/merge, schema export,
and revisioned persistence. A migration must preserve those contracts while
making a single owner responsible for cross-field validation and atomic
updates. Rust runtime configuration and camera settings have a second state:
the persisted desired revision may differ from the revision actually applied
to a running source until startup or a device readback acknowledges it.

Calibration has its own provenance-bound Rust store and revision under
ADR-0060. Its physical calibration record remains separate from the general
configuration document; legacy table-editor corners remain compatibility
data and cannot grant physical-coordinate readiness.

## Decision

Make a Rust configuration authority the only owner of the existing unified
configuration document, including defaults, schema, type/range/cross-field
validation, warning and force decisions, revision assignment, reset/import
semantics, and persistence. Keep the existing JSON document shape and public
HTTP route contracts so the current HTML/JavaScript/CSS browser client can
remain in place through the server migration.

Use one versioned, language-neutral configuration contract as the source for
the defaults and schema. Rust embeds and serves this contract. Python may keep
compatibility projections for typing and route adaptation, but it must not
maintain a second mutable copy of validation rules or defaults.

Expose the authority through a bounded one-shot local command interface using
versioned JSON on standard input/output. The command handles snapshot, schema,
validate-only, partial update, full replace/import, section reset, and export
operations. Python remains the HTTP/WebSocket and process-supervision gateway:
it parses accepted import file formats, invokes the Rust command, maps its
explicit result to existing HTTP responses, and refreshes any in-process
read-through snapshot. It must fail closed when the authority is unavailable;
it must not write `config.json`, locally decide whether a requested update is
valid, or silently fall back to Python persistence.

This Python gateway is transitional. ADR-0066 sets the final deployed server
language to Rust and retains the standard HTML/JavaScript/CSS browser client.
The configuration contract, store ownership, revision behavior, and validation
requirements in this ADR remain active when the Rust gateway replaces the
Python adapter.

Serialize mutations under one cross-process lock. A mutation supplies the
revision it read; Rust rejects stale writes, applies updates to the current
document, validates the complete merged result, and atomically persists before
publishing the new snapshot. Rust assigns revisions. A no-op does not advance
the revision; failed validation or persistence leaves both memory and disk
unchanged. Imported data cannot choose or decrease the store-managed revision.
Legacy documents missing a revision are upgraded through a documented,
tested migration path.

Keep desired configuration distinct from runtime-applied configuration. A
successful store update reports the desired revision and any restart
requirement; it does not imply that the camera or inference process has applied
it. Runtime state becomes applied only after the Rust capture/runtime
acknowledgement identifies the matching revision and reports actual negotiated
properties. A newer update invalidates that applied match until a later
acknowledgement.

Resolve production v3 capture settings in Rust from the revisioned effective
configuration and embedded defaults. The bounded authority exposes a
read-only `capture_settings` operation that generates the camera request,
profile fingerprint, controls policy, memory limits, and configuration revision
in Rust, then validates the generated request with the strict Rust wire parser.
The live supervisor must obtain this request through the Rust authority and
fail closed if it is unavailable; it must not reconstruct a request from
Python defaults. Python may preserve the resulting JSON contract for process
handoff, but the child still hashes those exact bytes and checks the request
against the opened source before reading frames. The legacy Python resolver
remains only for offline differential tooling and acceptance tests. The
production request currently carries `calibration: null`; moving calibration
selection into this request requires a separate provenance-bound decision and
must not infer calibration from its configured revision alone.

## Consequences

### Positive

- One implementation owns defaults, validation, revisioning and persistence
  across all config sections, including cross-section updates.
- Existing HTTP routes and JSON shapes can survive while the browser remains
  frozen during runtime migration.
- Concurrent gateway requests and direct local operations cannot silently
  overwrite a newer document or publish an unpersisted snapshot.
- Persisted desired settings cannot be mistaken for settings actually applied
  by a camera or live runtime.
- Python remains useful for the service boundary, file-format adaptation,
  differential evaluation and training tools without owning runtime policy.

### Negative

- Python startup and configuration routes depend on a Rust command binary and
  must surface a clear unavailable error rather than falling back.
- A one-shot subprocess adds bounded process startup cost to mutations and
  requires stable stdin/stdout framing, deadlines, and capped diagnostics.
- The Rust authority must temporarily preserve some Python compatibility and
  route semantics that are not perception-specific.
- A global revision advances for gateway-only and runtime settings alike, so
  applied camera/runtime revision must remain a separate acknowledged field.
- Existing test suites and offline tools require a deliberate Rust oracle and
  differential fixture path instead of mutating a Python singleton directly.

## Alternatives considered

### Keep Python as the general configuration owner

Rejected because the user directed all application runtime behavior outside
the retained backend service to move to Rust, and Python currently constructs
effective capture settings before Rust receives them. Keeping Python as the
authority would preserve split policy and revision semantics.

### Let Python and Rust write the same configuration file

Rejected because independent validators and revision counters cannot provide
atomic merged validation or reliable cross-process compare-and-swap behavior.

### Maintain separate configuration stores for gateway and runtime sections

Rejected for this migration because existing imports and partial updates span
sections. Splitting the file would require a multi-store transaction and would
change established export and route behavior. Revisit only with an explicit
versioned product contract and demonstrated need.

### Add a persistent Rust network control server

Rejected for the initial authority because configuration writes are
infrequent and the existing service already supervises the native process.
A bounded one-shot local command avoids another long-lived listener and
preserves the no-auth local deployment model. Revisit if measured subprocess
overhead or concurrent-client requirements justify a persistent control
channel.

### Keep only Rust's capture-settings request and leave config in Python

Rejected because strict parsing of a Python-resolved request protects the
capture process but does not move configuration decisions, persistence, or
revisions out of Python.

## Implementation status (2026-10-22)

The Rust authority foundation is implemented in
`perception/src/configuration_authority.rs` and the
`billiards-perception-config` one-shot command. It embeds the v1 defaults and
schema, requires an explicit configuration path, bounds request/configuration/
response sizes, validates merged documents, and handles snapshot, schema,
defaults, validation, revision-checked update/replace/section reset, and export.
Writes use a cross-process lock, compare-and-swap revisions, and atomic
replacement. A mutation whose success payload cannot fit is rejected before
the write. Targeted tests caught and corrected that ordering defect, missed
trusted-proxy CIDR validation, and URI control-character acceptance.

The Python bounded client and `Config` read-through facade are implemented.
Startup loads a Rust snapshot before configuration-dependent imports when Rust
authority mode is explicitly selected. Configuration routes then send
validation, schema/defaults, exports, reset, and mutations to Rust and map
revision conflicts, validation failures, warning rejections, and authority
failures to the existing HTTP boundary. Route tests assert that Rust mode does
not call Python validators/defaults/schema builders or write through Python.
The facade serializes local CAS mutations and cache publication under one lock.
The backend config route suite (168 focused config/startup tests) and strict
type/lint/format checks pass after this adapter slice.

This is still a qualification path, not a default production cutover:
`BILLIARDS_CONFIG_AUTHORITY=rust` is explicit, and Python mode remains available
while compatibility is established. Production Rust live capture now requires
that Rust authority mode and obtains its v3 request from the `capture_settings`
operation. The Python resolver remains for offline differential tests and the
performance-audit tooling only. The generated request is compared against the
existing v3 golden fixture, the service-side adapter verifies the returned
revision, and an isolated temporary-config command test proves full-object
parity for selector, strict-control, output-geometry and frame-budget cases.
Desired-versus-applied revision reporting and using the authority directly in
the final Rust HTTP service remain subsequent migration work.

The Rust authority's formatting and strict Clippy checks pass. The Rust library
suite passes 160 tests, and its isolated configuration command integration
suite passes 20 tests, including full-object v3 capture-request parity and
read-only behavior. An actual one-shot command invocation against a temporary
configuration file produced a v3 request matching the Python differential
reference for a stable GStreamer selector, required controls, scaled output
geometry, and exact frame limits. The full backend suite passes 3,140 tests
with two model-dependent skips and 80.06% coverage; the small margin above the
80% gate means future coverage-affecting changes must retain the full-suite
check.

## Verification

The following gates are still required for completing this decision:

- Rust unit and integration tests cover defaults/schema agreement, strict
  types, unknown fields, ranges, cross-field validation, warnings/force,
  partial-merge validation, replace/import behavior, reset-section behavior,
  revision ownership, no-op updates, legacy missing-revision migration,
  compare-and-swap conflicts, concurrent processes, restart reads, and
  persistence-failure preservation.
- Cross-runtime fixtures compare the existing public defaults, schema,
  revisioned snapshots, capture-settings v3 request and operation responses
  against Rust output.
- Existing backend config and route contract tests must remain green against
  the Rust-backed gateway adapter, including validate-only, import/export,
  reset, warning/force, and error status behavior.
- Rust live tests prove that the resolved capture request identifies the
  desired configuration revision; a running runtime must report an applied
  revision only after its exact request/source acknowledgement.
- Supervisor tests must cover a missing binary, timeout, malformed/oversized
  protocol data, nonzero exit, bounded diagnostics, restart, and no Python
  persistence fallback.
- CI validates Rust formatting, Clippy, all perception targets, Python format
  and lint checks, backend unit/integration suites, and the repository coverage
  gate. macOS/Linux package and live-camera qualification remain separate
  release gates.

## Related records

- [ADR-0066](0066-run-server-services-in-rust-and-retain-the-standard-web-client.md)
  supersedes the Python gateway and Rust/WASM client destination. The browser
  remains HTML/JavaScript/CSS, and the Python configuration adapter is
  transitional.
- [ADR-0060](0060-store-table-calibration-as-rust-owned-revisioned-state.md)
  defines the separate provenance-bound Rust calibration store.
- [ADR-0030](0030-request-bound-capture-acknowledgement-for-live-startup.md)
  defines exact capture-request and applied-source acknowledgement semantics.
- [Native domain migration plan](../../../thoughts/rust-domain-migration-plan.md)
  tracks staged implementation and qualification.
