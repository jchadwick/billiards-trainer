# API reference

The supported API is implemented by `billiards-server`. The default local
Makefile port is `8001`; the Docker image listens on port `8000`. The HTTP and
WebSocket API is unauthenticated and intended for loopback/local development.
Do not expose it to a LAN or the public Internet. The scene producer listener
is also loopback-only. Remote authenticated service roles are not implemented
yet; see [Runtime architecture](ARCHITECTURE.md#deployment-boundary).

## Health

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health/live` | Confirm the gateway is responding |
| `GET` | `/api/v1/health/ready` | Report readiness of the state source |

## Configuration

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/config` | Read active configuration |
| `PUT` | `/api/v1/config` or `/api/v1/config/` | Validate and apply a partial update |
| `GET` | `/api/v1/config/schema` | Read the Rust-owned configuration schema |
| `POST` | `/api/v1/config/reset` | Reset supported configuration values |
| `POST` | `/api/v1/config/import` | Validate and import configuration data |
| `GET` | `/api/v1/config/export` | Export configuration data |
| `GET` | `/api/v1/config/export/download` | Download an exported configuration file |

Updates use the versioned Rust schema and revisioned persistence described in
[Configuration](CONFIG.md).

## Game state

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/game/state` | Read the latest localized state and optional session metadata |
| `POST` | `/api/v1/game/reset` | Reset optional session metadata |

Session metadata is separate from detected ball locations.

## WebSocket state stream

Connect to `/api/v1/websocket/ws`. The gateway sends the latest available state
after the WebSocket handshake; no subscription message is required. Delivery
is latest-value and best-effort, so intermediate states can be coalesced or
dropped when clients are slow.

```javascript
const socket = new WebSocket('ws://127.0.0.1:8001/api/v1/websocket/ws');
socket.onmessage = ({ data }) => {
  const state = JSON.parse(data);
  console.log(state);
};
```

The versioned scene and public-state schemas and fixtures live under
[`perception/contracts`](../perception/contracts/). The detector does not
identify numbered balls or produce shot trajectories. The browser's live coach
uses producer centers and a compatible table projection to calculate direct-pot
guidance; this is a UI estimate, not a server prediction or complete physics
model.

Public state exposes `observation_status.perception`, containing `layout_trusted`,
`trust_reason`, whole-table motion evidence ages, and the current cue result.
Inside a WebSocket message, the full prefix is `data.observation_status.perception`.
A clean settled scan
replaces the producer layout wholesale, including a valid empty layout. Failed
or occluded scans do not clear it. Motion or expired motion evidence makes the
layout stale while retaining last-known centers. Consumers must not treat stale
centers as proof of a clear path or cache them as a second layout. A slow scan
can be accepted when source context, ordering, motion generation, and quiet
continuity still match; capture timestamps remain unchanged.

`observation_status.perception.table_projection` describes the active source-pixel to
center-origin table-millimeter homography, physical table dimensions, and
camera/calibration context. The compatibility HTTP root also exposes it as
`table`, with its matching `table_calibration_revision`. Clients should reject
spatial output when known source, configuration, calibration, rectification,
or geometry context is incompatible. The source-selection configuration
revision is a minimum bound: a newer configuration on the same selected source
may remain compatible, but a producer older than the selected source revision
cannot. A video source generation change invalidates prior display context.
Frame-ID differences alone are not a spatial mismatch.

Cue results are ephemeral. A negative evaluation clears the cue immediately;
capture-age expiry and motion/context changes prevent old results from
reappearing. The default cue evidence bound is configured as 250 ms in the
current defaults. A cue line may be visible without signed direction; in that
case its direction is unknown and it does not drive a live path prediction.
The manual desired-aim suggestion is independent of cue detection.

## Browser assets and camera video

The Rust server serves the built HTML, JavaScript, and CSS bundle. The browser
loads the configured `api.stream.viewer_url` directly and receives detection
state through the WebSocket. It does not proxy video through the Rust API. The
live view always uses the newest video and renders compatible producer centers
independently of video frame IDs. When physical table mapping is unavailable,
compatible image-space centers can still be displayed, while table-space advice
is disabled. Production ball markers use fixed physical-radius circles
projected through the active mapping when available; a compatible detector
center remains usable if optional precision refinement is absent or invalid.
Marker labels are stable producer track IDs for manual selection, not ball
numbers.

### Identified MJPEG metadata for compatible viewers

The live product view does not require an identified MJPEG stream or an exact
JPEG/perception frame join. A separate viewer that chooses to correlate a
particular JPEG with state may use the following identity metadata; context
must match, while frame identity is provenance rather than a production marker
gate. Optional fitted outlines remain tied to their exact source frame.

Cue geometry carries its own `source_frame_id` as provenance. The product
checks current source/calibration context and capture age; it does not require
the video viewer to display that exact frame. Ball measurements retain their
original observation age, and phase-only updates do not make them newer.

Each `multipart/x-mixed-replace` part must include a valid `Content-Length`,
`Content-Type: image/jpeg`, and these part headers:

| Header | Meaning |
| --- | --- |
| `X-Frame-Id` | Unique frame identifier within the source session |
| `X-Engine-Instance-Id` | Perception engine identity |
| `X-Source-Session-Id` | Capture session identity |
| `X-Configuration-Revision` | Applied capture/perception configuration revision |
| `X-Calibration-Revision` | Calibration revision used to localize state |
| `X-Rectification-Revision` | Image rectification revision |

The configured stream origin must permit browser `fetch` access with an
appropriate CORS policy when it is cross-origin. Frame bytes, part headers,
and retained decoded frames are bounded by `api.stream.frame_pairing`; see
[Configuration](CONFIG.md#identified-video-and-exact-frame-overlays). The Rust
gateway route `/api/v1/video/identified.mjpg` emits the identity headers from
the capture preview descriptor; the preview TCP listener stays loopback-only.
