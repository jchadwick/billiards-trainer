# ADR-0057: Allow explicit Rust live perception behind the Python service

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: ADR-0032 provider-selection gate; ADR-0036 provider-selection gate
- Superseded by: None

## Context

The accepted migration target moves capture, image processing, inference,
tracking and localization to Rust while retaining the Python API service as a
client-facing gateway. Rust has a live executable, bounded capture runtime,
tracking, and a supervised scene protocol, but the API provider selector still
rejects `rust` with `live` mode. The API can already supervise Rust recorded
replay. Physical webcam capture requires an explicit stable GStreamer selector,
helper binary, and exact request-bound startup handshake. The default provider
must remain reversible until live accuracy and freshness are qualified.

## Decision

Allow `perception.provider=rust` with `perception.mode=live` as an explicit
configuration choice. The Python service launches one `RustLiveSupervisor` and
keeps ownership of HTTP/WebSocket delivery, public state compatibility, health,
and child supervision. Rust is the sole camera, inference, and tracking owner
for that run. Live mode requires a validated v3 capture request with a stable
physical source selector and the matching feature-enabled helper. It fails
closed on unavailable hardware, settings mismatch, stale startup, or child
failure; it never starts or falls back to Python vision. Keep the default
provider as Python only as a temporary rollback path until the platform and
accuracy gates support the Rust cutover. The accepted destination is Rust for
all application-domain behavior; retaining Python is limited to the service
gateway and approved development/training/evaluation tools.
The Rust live provider also requires `BILLIARDS_CONFIG_AUTHORITY=rust`; its v3
capture request must come from the Rust configuration authority and cannot be
reconstructed from Python defaults. Recorded replay remains a separate mode
and uses the existing replay supervisor.

## Consequences

### Positive

- The live Rust engine can be qualified through the real API and existing
  WebSocket contracts without rewriting the backend service.
- One camera and one localization publisher own each selected live run.
- Failed Rust startup is visible and cannot silently create competing Python
  capture.

### Negative

- Python and Rust paths coexist during migration and require matched testing.
- Live Rust configuration must resolve the exact model, camera selector,
  calibration, helper, and process limits before startup.
- The existing Python public-state publisher remains a gateway contract adapter
  until its responsibilities are deliberately migrated or reclassified.

## Alternatives considered

### Keep Rust live unavailable until every platform gate passes

Rejected because it prevents the real API lifecycle from being used for the
qualification needed to complete those gates. Availability remains explicit and
non-default; this does not authorize a production default change.

### Fall back to Python if Rust fails

Rejected because it could open a second camera, conceal a Rust failure, and
publish mixed or stale state from two owners.

### Remove the Python service in the same change

Rejected because the user permits the backend service to remain Python and the
service can remain the stable client contract while domain behavior migrates.

## Verification

- Provider tests accept only supported provider/mode pairs and retain Python as
  the bundled default.
- Rust live settings tests require Rust configuration authority and exact
  stable-selector configuration, validate model lineage, and reject missing
  binaries or incompatible camera settings before process creation.
- API lifecycle tests prove Rust live startup/stop never initializes Python
  vision or integration services, and no fallback occurs on failure.
- Live supervisor tests prove request-bound handshake, first-localization
  readiness, source silence, child failure and bounded shutdown.
- Target API startup currently fails before inference: container-side discovery
  needs read-only `/run/udev` and `/sys` mounts to resolve the stable selector,
  then the capture helper times out waiting for its first frame. No target
  accuracy, freshness, or performance qualification is claimed yet.

## Scope clarification

This decision supersedes only the requirement in ADR-0032 and ADR-0036 that
Rust live remain unavailable through API provider selection. Their exact-request
handshake, supervised helper, bounded process lifecycle, and physical hardware
qualification requirements remain in force. Explicit Rust live selection is a
qualification path; it does not qualify accuracy, freshness, sustained
performance, or make Rust the bundled default.

The Python service starts no Python Core, Vision, or integration runtime when
Rust is selected. It remains responsible for service routes, state contract
translation, WebSocket publication, health, and supervised process lifecycle.
Legacy preview, frame-capture, and camera-diagnostic routes return 503 while
Rust owns live capture, rather than opening a second Python camera. A Rust-fed
preview and camera-backed calibration workflow remain follow-up work.
