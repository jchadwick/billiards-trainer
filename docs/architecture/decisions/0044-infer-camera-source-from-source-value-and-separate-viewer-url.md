# ADR-0044: Infer detector source from one value and configure viewers separately

- Status: accepted
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None
- Implementation stage: Python capture accepts local device indexes and HTTP(S)
  MJPEG URIs; the frontend reads a separate runtime viewer URL. The Rust live
  stream adapter and target camera qualification remain open.

## Context

The Linux target has a systemd-managed uStreamer service that owns `/dev/video0`
and exposes an MJPEG stream on port 8080. During inspection the service was
running, but the camera was unplugged, the stream reported `online: false`, and
it had no clients. The target backend container had no camera device mapping
and was configured for recorded replay. In the repository, `LiveView` used the
API's video route, while the Python `CameraCapture` accepted only a numeric
device for live capture. Neither path represented the intended local detector
subscription and direct browser subscription.

The detector and browsers can reach the same physical stream through different
network names. The detector should use a loopback URI where possible. A browser
needs an address reachable from its own machine, such as the target's LAN
hostname. One URL therefore cannot serve both roles reliably.

## Decision

Use `vision.camera.device_id` as the detector's single source setting. Infer the
reader from its value: a nonnegative integer selects a directly opened local
camera; an anonymous HTTP(S) URI selects an MJPEG network source such as
uStreamer's `/stream` endpoint. Reject embedded credentials, malformed ports,
fragments, and unsupported schemes. Do not add a second detector source-mode
flag. The existing stable GStreamer selector remains a direct-device identity
for its Rust-specific capture adapter.

Use a separate optional `api.stream.viewer_url` for the browser image source.
When set to a valid anonymous HTTP(S) URI, Live View subscribes directly to
that stream. When unset or invalid, it uses the retained API video route. The
API and WebSocket continue to carry detections, positions, freshness, and other
application state; image bytes do not need to pass through them when direct
viewer delivery is configured.

For an uStreamer deployment, configure the detector with a loopback source URI
and configure `api.stream.viewer_url` with a browser-reachable address. Keep
camera mode, resolution, frame rate, and camera controls owned by uStreamer in
that mode. The detector URI takes precedence over the legacy recorded-video
path so it can be selected by changing the detector source value alone. For a
numeric local camera, the legacy video path remains a fallback and must be
cleared when live device capture is intended. The Python capture reader keeps
a one-frame application queue and uses OpenCV's available network-video
backend. The Rust live provider must add an equivalent supervised source
adapter before it can consume this URI.

## Consequences

### Positive

- A single detector setting selects either the direct camera path or a network
  MJPEG source from its value, without a parallel mode flag.
- Detector and browser can use host-appropriate addresses for the same stream.
- Configured browser video can bypass API image fan-out while scene and state
  delivery retain their existing contracts.
- The normal numeric camera-index default and API video fallback remain
  available for existing deployments.

### Negative

- HTTP/MJPEG decoding support depends on the OpenCV backend available in the
  installed Python build; runtime open failures must remain visible in camera
  health rather than falling back to a different source silently.
- Direct browser delivery requires the configured address to be reachable by
  every viewer. An HTTP stream may be blocked as mixed content if the viewer
  page itself is served over HTTPS.
- Rust live capture currently rejects HTTP MJPEG URIs. The accepted source
  contract is not complete across providers until the supervised Rust stream
  adapter and lifecycle tests are implemented.
- uStreamer is Linux-target infrastructure; macOS direct-camera capture remains
  an independent platform path.

## Alternatives considered

### Add a separate `capture_mode` enum

Rejected because it duplicates information already present in the source
value. It also creates invalid combinations, such as selecting `ustreamer`
while supplying a device index or selecting direct capture with a URL.

### Force the detector and browser to use one URL

Rejected because loopback is best for the local detector, while remote browsers
need a LAN-reachable address. The host part of those URLs is intentionally
different even when both address the same uStreamer service.

### Route every video viewer through the API

Retained only as the fallback for configurations without `api.stream.viewer_url`.
For the uStreamer target, direct subscriptions avoid using the API as an image
fan-out point and allow the detector and viewers to consume independently.

## Verification

- Backend configuration tests verify that device indexes and anonymous HTTP(S)
  source URIs validate, and that unsafe or malformed URIs are rejected.
- Camera-capture tests verify URL inference, use of OpenCV's automatic backend,
  skipped physical-camera controls, and a one-frame application queue.
- Frontend tests verify that runtime `api.stream.viewer_url` selects the direct
  image feed, an unset URL uses the API fallback, and unsafe URLs are rejected.
- Target inspection confirmed the systemd service and endpoint but found the
  camera offline and no subscribers. No live detector or browser fan-out has
  been qualified on that target yet.

## Revisit when

- The camera is connected and the Linux detector and browser can be tested
  concurrently against the same uStreamer source.
- The Rust live provider is enabled for physical capture; its URL adapter must
  preserve bounded freshness, request-bound source identity, restart behavior,
  and helper-process cancellation.
- Measured MJPEG decode cost or image quality shows that the source should
  publish another camera-native format or resolution.
