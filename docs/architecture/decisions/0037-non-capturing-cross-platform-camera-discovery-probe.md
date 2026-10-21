# ADR-0037: Non-capturing cross-platform camera discovery probe

- Status: superseded
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: ADR-0040, only for discovery-report identity and selector fields

ADR-0040 replaces only the discovery response's identity/selector shape. This
record's decision to use bounded, non-capturing DeviceMonitor inventory remains
in force.

ADR-0039 later added a v3 `source_defaults` policy for generic webcams; explicit
exposure/gain readbacks are required only when a caller requests controls with
the strict `require` policy.

## Context

Rust live perception is already composed through model inference, tracking,
request-bound readiness, scene transport, and shutdown, but its executable has
no camera source. The synthetic GStreamer helper validates framing and
supervision without enumerating or opening a camera. A physical adapter cannot
be selected responsibly until the intended table camera, its advertised
modes, and platform identity are known. At the time this record was authored,
the capture request used a numeric device index and required exposure-mode and
gain readbacks. ADR-0039 later introduced v3 `source_defaults` for webcams
without configured controls; a display name or enumerated `/dev/videoN` path
still cannot safely identify a Rust capture source.

The development Mac exposes AVFoundation camera devices through GStreamer
DeviceMonitor, including a built-in camera and a Continuity Camera. The user
clarified the intended hardware is a regular webcam with no per-camera
configuration; the exact model and Linux host are unspecified. Discovery can
collect platform-reported identifiers, element/provider properties, and
advertised caps without opening a camera stream. Advertised caps do not prove
that a mode will negotiate or that any camera controls can be applied.

## Decision

Add a separate, opt-in `billiards-perception-camera-discovery` diagnostic
executable behind a GStreamer feature. It uses `GstDeviceMonitor` to enumerate
`Video/Source` devices, then emits a versioned JSON inventory containing
GStreamer/platform versions, display name, device class, provider properties,
and advertised caps. Preserve property names and values as reported; expose
only platform-specific identifiers that the probe can identify explicitly,
and label their stability instead of converting a transient enumeration index
into a persistent ID. Report whether a matching Video/Source provider started
or no provider matched, so an empty device list does not hide a missing plugin.
Bound serialized output with a caller-configurable maximum.

The probe does not call `create_element`, construct a capture pipeline, read
frames, set controls, or enable a live provider. Empty device lists and
unsupported properties are represented explicitly. The inventory is
diagnostic evidence for a later device-ID/control-contract decision, not a
source-readiness report. Production selection remains gated by a physical
camera adapter that verifies negotiated mode, no-write `source_defaults`
behavior (or exact control readbacks when explicit controls are requested),
bounded interruption/release, and freshness on the chosen macOS and Linux
devices. GStreamer's add-filter result is used to distinguish no matching
provider from a started provider with no currently visible devices, as
documented by the [GstDeviceMonitor API](https://gstreamer.freedesktop.org/documentation/gstreamer/gstdevicemonitor.html).

## Consequences

### Positive

- Establishes which GStreamer source providers, device identities, and modes
  are visible on each target without letting exploratory work take ownership
  of the camera.
- Supplies concrete evidence for a stable device-ID contract and adapter
  feasibility before changing the accepted capture-settings wire format.
- Keeps the synthetic helper and production live provider semantics separate.

### Negative

- Adds a GStreamer development/runtime dependency for this opt-in executable.
- Enumeration is platform/plugin dependent and may return incomplete
  properties; advertised caps are not a successful-open or control-readback
  guarantee.
- The probe does not resolve camera buffering, exposure/gain normalization,
  timestamps, frame freshness, or release/reopen behavior.

## Alternatives considered

### Open a camera and report a negotiated stream

Deferred because opening a physical camera can interrupt an active application
and still would not prove applied exposure/gain or source-timestamp semantics.
The intended table camera and Linux target are not known yet.

### Use numeric device indexes as the portable camera identity

Rejected because enumeration order is not a stable cross-platform identity and
does not distinguish an intended table camera from built-in or virtual
devices.

## Verification

- Unit tests validate stable-identifier classification, caps/property
  serialization, empty inventories, and the output-size limit using fixtures.
- A feature-gated process integration test executes the command, validates
  the JSON shape, and enforces the caller-supplied output bound.
- The macOS/Linux Rust workflow runs the discovery process test together with
  the synthetic helper tests; a no-device Linux runner may legitimately emit
  an empty inventory.
- Capture and review actual device inventory on the development Mac, without
  opening any device, and record only the non-sensitive result summary in the
  migration plan.
- ADR-0036's physical qualification gates remain required before live provider
  selection.

## Implementation progress (2026-10-21)

The optional camera-gstreamer-discovery feature now builds
billiards-perception-camera-discovery; the command reads only DeviceMonitor
metadata, explicitly reports no matching provider, and applies the configured
output bound. Unit tests cover known and
unknown identifier forms, missing values, empty results, and metadata
truncation. A process integration test executes the binary and validates the
bounded JSON report without assuming any physical device is attached. The
macOS/Linux workflow runs that test beside the synthetic helper checks.

On the development Mac, the command reported the built-in MacBook Pro Camera
through AVFoundation and identified its provider ID as avf.unique_id. It did
not create a source element, negotiate a stream, or read a frame. The combined
feature-enabled Rust target suite and strict Clippy pass locally, as do Rust
formatting and the default-feature binary check; GStreamer remains absent from
the default feature graph. Hosted Linux/macOS workflow results and inventory
from the intended table camera remain pending. This verifies discovery and
wire-output behavior only, not live capture, control application, or freshness.
