# Runtime architecture

## Application boundary

The supported application has one server-side runtime: Rust. The browser
client remains HTML, CSS, and JavaScript and is served as static files by the
Rust gateway. The browser includes setup, an editable `/practice` sandbox, and
the camera based `/live` coach.

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

The browser loads `api.stream.viewer_url` directly; ball state is received
separately over WebSocket. The newest video continues independently of state
delivery; the live view does not freeze or rewind it to find a matching frame.
Perception owns the whole-table layout and reports trusted or stale last-known
centers. The browser renders compatible centers as solid or dim dashed markers
and does not retain a second ball layout. Frame IDs and optional fitted outlines
remain provenance for diagnostics, not prerequisites for normal markers.

Live advice uses the active table projection when its source and calibration
context match the video and state. That projection maps source-image pixels to
table-plane millimeters with the origin at the playing-surface center. The
browser inverts the same homography to draw suggested paths. If usable table
coordinates or a trusted layout are unavailable, it pauses advice; compatible
image-space center markers may still be shown. The cue-ball choice, target, and
pocket are manual UI selections. Producer marker labels are track IDs, not
recognized ball numbers. Direct-pot suggestions check cue-to-contact and
object-to-pocket clearance, then rank by configured policy. They show a
ghost-ball/contact aim and estimated initial object path, not legal-game rules,
spin, banks, combinations, or complete post-contact physics.

`/practice` uses editable simulated positions and shared direct-pot geometry.
It is self-contained and does not read camera or producer state.

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

## Current scope and validation

The detector reports generic ball centers and cue status; it does not identify
numbered balls. Direct-pot guidance is computed by the browser from producer
centers and physical table geometry, not by a trajectory-prediction producer.
It estimates initial paths only. Automated software checks cover the
implemented UI and producer lifecycle. Acceptance on a physical billiards
table has not been performed, and no physical accuracy, end-to-end latency, or
deployment performance qualification is claimed.

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
describes scheduling, state trust, and configurable work rates.
