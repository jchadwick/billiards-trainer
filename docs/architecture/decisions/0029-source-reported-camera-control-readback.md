# ADR-0029: Source-reported camera-control readback

- Status: accepted
- Date: 2026-10-20
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The Python capture-settings contract carries requested exposure mode, exposure,
gain, brightness, contrast, saturation, focus and buffer settings. Rust capture
metadata currently reports whether selected controls are supported, but a
capability declaration does not say that a requested value was applied. Treating
the request or capability as an applied value could mark a live source ready
with different camera behavior than the configuration describes.

The device-independent source boundary can be extended and tested with fake
adapters without selecting a physical camera library. A real adapter and target
hardware are still required to establish that its readbacks reflect device
behavior.

## Decision

Keep requested settings, capability declarations and source readbacks as
separate facts. The opened source report carries a typed readback for each
supported control in the v1 request: exposure mode, exposure value, gain,
brightness, contrast, saturation, autofocus and buffer size. Each value is
reported as `applied(value)`, `unsupported`, or `unknown`; only `applied` carries
a value. Do not substitute a requested value when the source cannot report one.

The source produces these readbacks only after opening and attempting to apply
the requested settings. The report binds them to the opened capture session.
Capabilities remain separate: a supported control does not prove an applied
value, and an unknown readback is not treated as an unsupported control.

For settings whose capability is required by the current request contract, a
live preflight succeeds only when the source reports the requested exposure
mode and requested gain as applied with exactly matching values. A requested
manual exposure value is also required to match when present. Optional control
readbacks remain visible even when unknown or unsupported, but do not block
localization readiness. No tolerance or normalization policy is introduced
without measurements from the selected adapter and camera; an adjusted required
value therefore fails closed until a follow-up decision defines how to handle
that device behavior.

The source report now also carries the applied frame-geometry fields
(configuration revision, camera-profile ID, image space, and rectification
revision), and preflight compares them with the request before the first read.
That extends the source-report contract but does not prove a physical device
honored the values. Table calibration remains a pipeline-owned fact and is
validated separately. ADR-0030 defines the future cross-process acknowledgement
of the opened report. This ADR still covers camera-control readback only; it
does not select a camera library or enable the live provider.

## Consequences

### Positive

- Requested values cannot be mistaken for actual device state.
- Unknown and unsupported controls remain explicit rather than being filled
  from configuration defaults.
- Required exposure and gain values can be checked before the first frame is
  read or published.
- Fake-source tests can verify the protocol before camera hardware is selected.

### Negative

- Source adapters must query or otherwise reliably observe applied controls
  after setting them.
- Some devices quantize, clamp or asynchronously update control values; those
  devices remain incompatible with exact matching until evidence supports a
  bounded normalization policy.
- The full runtime-application acknowledgement still needs to combine source
  readbacks with application/pipeline-owned image-space and calibration state.

## Alternatives considered

### Treat capability support or a successful setter call as proof of application

Rejected because capability is not a value readback, and some camera APIs
silently clamp, defer or ignore control writes.

### Copy requested values into the source report

Rejected because this would fabricate evidence when the source cannot query or
confirm its effective state.

### Add a numeric tolerance before selecting a target camera

Deferred because device ranges and quantization are not known. Any comparison
policy must be based on readbacks from the selected hardware and remain
configuration-driven rather than embedding an unmeasured tolerance.

## Verification

- Rust serialization tests cover each readback state and confirm that only the
  applied state has a value.
- Metadata construction rejects non-finite numeric readbacks and invalid
  applied gain values.
- Preflight tests accept matching required values and reject mismatched,
  unsupported or unknown required values before the first source read.
- Optional unknown/unsupported readbacks remain visible without blocking
  preflight.
- Rust wire parsing preserves the request values in `ExpectedCaptureSettings`,
  while opened-session serialization emits only source-reported readbacks.
- The full Rust suite, formatting and strict Clippy pass. Python persistence and
  the 80% repository coverage gate are unchanged by this Rust-only slice.

## Revisit conditions and related records

ADR-0014 governs persisted configuration revisions, calibration and the
cross-process application acknowledgement. ADR-0015 governs camera ownership
and adapter lifecycle. Add a successor decision before accepting quantized
control matching, camera-specific behavior, or a new readback source.

## Evidence and implementation tracking

- [Migration plan](../../../thoughts/rust-perception-migration-plan.md).
- [Rust source metadata](../../../perception/src/capture.rs).
- [Rust live preflight](../../../perception/src/live_config.rs).
- [Rust request parser](../../../perception/src/capture_settings_wire.rs).
- [Rust preflight readback tests](../../../perception/tests/live_config_preflight.rs).
- [Capture settings request contract](../../../perception/contracts/capture-settings/v1/README.md).

This decision defines a software contract, not hardware qualification or a
claim that a camera adapter has been implemented.
