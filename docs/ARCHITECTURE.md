# Runtime architecture

## Application boundary

The supported application has one server-side runtime: Rust. The browser
client remains HTML, CSS, and JavaScript and is served as static files by the
Rust gateway. The browser route `/` redirects to `/live`; the live page is the
only product view.

`billiards-server` owns configuration, HTTP and WebSocket routes, static-file
serving, and supervision of live capture. The live detector reads frames from
the selected source, performs inference and tracking, and sends scene updates
to the gateway over bounded loopback TCP. The gateway converts the latest
scene into the public state contract and publishes it to connected clients.
The capture path uses the configured source and the GStreamer helper where
applicable. Docker and the local deployment bundle contain Rust executables,
model/runtime files, configuration, and the built browser assets; they do not
install or launch Python.

## State and freshness

Live updates are best-effort and freshness-first. Queues are bounded, newer
state may replace older pending state, and slow consumers may miss intermediate
updates. The system does not promise delivery of every camera frame. A detected
state is only current to its source observation; the server preserves source
identity, frame identity, timestamps, and observation status for consumers.

The browser loads `api.stream.viewer_url`; ball state is received separately
over WebSocket. Production can point the viewer URL at
`/api/v1/video/identified.mjpg`, where the gateway bridges the detector's
loopback-only frame preview and attaches the exact capture identity to each
JPEG part. The browser buffers a bounded set of decoded stills and draws spatial
guidance only over the exact JPEG whose identity matches the perception
decision. If the pair is missing or any producer/session/geometry revision
differs, the view withholds the spatial overlay. Legacy MJPEG remains visible
as video without spatial overlays. The gateway bridge is limited to configured
preview clients and frame sizes.

## Deployment boundary

The current supported deployment is colocated: the capture/inference producer
connects to the Rust gateway over a loopback-only TCP scene protocol, while the
browser loads video from its configured URL. `billiards-mjpeg-fanout` is a
saved-frame qualification fixture, not a production camera-streamer service.
It emits per-part frame identity headers for viewer tests, but the production
capture stream does not yet provide that identity contract. The HTTP and
WebSocket API also has no authentication. Keep the service on loopback until
the planned remote roles, authenticated transport, and production frame
identity path are implemented and qualified; a trusted LAN or VPN alone does
not make the current unauthenticated API safe to expose.

## Current scope

The detector reports generic ball locations and cue status. It does not
reliably identify numbered balls, and the running server has no trajectory
prediction producer. Those features are outside the current live path. The
current `/live` UI and the Rust state contracts are the reference for supported
behavior.

## Runtime entry points

- `perception/src/bin/server.rs`: HTTP/WebSocket gateway and process owner.
- `perception/src/bin/live.rs`: live capture, inference, and scene producer.
- `perception/src/bin/gstreamer_probe.rs`: GStreamer capture helper.
- `perception/src`: configuration, source capture, inference, tracking,
  localization, public state, and bounded delivery.
- `frontend/web`: browser application and client tests.
- `backend/Dockerfile` and `scripts/deploy`: production packaging and service
  deployment.

See the [API reference](API_REFERENCE.md) and [configuration guide](CONFIG.md)
for their current contracts. The [phase-aware perception guide](PHASE_AWARE_PERCEPTION.md)
describes scheduling, state trust, and rate qualification.
