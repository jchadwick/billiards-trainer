# ADR-0038: Use platform-native camera control readbacks behind the capture helper

- Status: proposed
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

Live capture always requires verified source identity and negotiated mode.
Explicitly configured exposure or gain additionally requires capability and
applied-value readbacks. The opt-in camera inventory found the development
Mac's AVFoundation provider, but inventory does not show that a source can
apply the requested controls.

Inspection of the installed GStreamer 1.28.3 `avfvideosrc` confirms that it
exposes device index/name and video properties, but no exposure or gain
controls. Its `device-index` is an enumeration index; upstream documents
`unique-id` as a new property in GStreamer 1.30. Its advertised caps do not
prove negotiated mode or stable source selection. The official
`avfvideosrc` property list documents this element surface:
[GStreamer AVFoundation source](https://gstreamer.freedesktop.org/documentation/applemedia/avfvideosrc.html).

Platform APIs expose different control models. Apple AVFoundation requires a
device configuration lock before setting hardware properties such as exposure
mode and supports stable device identification; Linux V4L2 defines camera
control IDs and device-reported values. These capabilities still require
verification on the selected physical camera. See [Apple's AVCaptureDevice
documentation](https://developer.apple.com/documentation/avfoundation/avcapturedevice)
and the [Linux V4L2 camera-control reference](https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html).
The user clarified that the intended device is a regular webcam with no
per-camera configuration; its exact model and the Linux host are unspecified.

## Decision

Keep GStreamer DeviceMonitor for opt-in inventory and GStreamer
`videotestsrc` for synthetic pipeline and helper-process tests. A generic
GStreamer physical source may be evaluated for the v3 `source_defaults`
control policy, provided it selects the requested device without relying on a
transient index and verifies the negotiated mode before readiness. Its source
element's presence, advertised caps, requested properties, or device index are
not evidence that a camera mode or explicit control was applied.

When v3 `require` requests explicit controls, evaluate platform-native control
and identity APIs inside the supervised helper from ADR-0036. On macOS the
helper must bind the selected AVFoundation device by its persistent identifier
and verify every requested exposure and ISO/gain value. On Linux it must bind a
stable V4L2 device path, query the selected driver's capabilities and values,
and report exact control units. A helper may use GStreamer for pixels only if
conversion does not hide source identity, negotiated mode, timestamps, or
native-control readbacks.

Keep live unavailable until a source verifies identity and negotiated mode,
and interruption, freshness, and release/reopen pass on both target systems.
Enable explicit control requirements only after their cross-platform values
are defined and exact readbacks qualify. This ADR proposes the adapter
boundary; it does not select a Rust binding, helper language, or production
backend.

## Consequences

### Positive

- Keeps requested controls separate from source-reported values and prevents a
  generic plugin from falsely satisfying strict readiness.
- Lets the Rust parent retain one versioned helper protocol while each
  platform handles its actual device controls.

### Negative

- Strict custom controls require platform-specific helper code, packaging, and
  a precise cross-platform definition for exposure and gain values.
- Driver defaults can vary by webcam, operating system, and lighting; image
  quality still needs validation on the table.

## Alternatives considered

### Use GStreamer as the complete cross-platform camera adapter

Not selected as a universal strict-control implementation. The installed
macOS element exposes neither exposure nor gain controls, and its direct
selector is an enumerated index; the stable `unique-id` property is documented
beginning with GStreamer 1.30. Linux `v4l2src` exposes V4L2 controls, but those
controls and units remain driver-specific. A GStreamer path may still satisfy
v3 `source_defaults` after exact source identity and mode are verified.

### Use a single generic Rust camera crate for both operating systems

Deferred until the selected devices pass exact-mode, control-readback,
freshness, interruption, release, and reopen tests through that crate. A
shared API surface alone does not prove matching semantics or safe lifecycle
behavior.

## Verification

- The discovery command must continue to avoid opening a device and report
  stable versus transient identity explicitly.
- A source-default request must not perform control writes and must preserve
  actual Supported/Unsupported/Unknown readbacks.
- Each explicit-control helper must test requested-versus-applied mismatches,
  missing controls, mode mismatch, and identity mismatch.
- Physical qualification must verify source identity, negotiated mode, frame
  freshness, timestamp semantics, interruption, release, and reopen on macOS
  and Linux; verify exposure/gain readbacks when strict controls are requested.
- Validate the helper against ADR-0015, ADR-0029, ADR-0031, ADR-0034, and ADR-0036.
