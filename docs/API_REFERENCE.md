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
[`perception/contracts`](../perception/contracts/). Numbered-ball recognition
and trajectory predictions are not produced by the current detector.

## Browser assets and camera video

The Rust server serves the built HTML, JavaScript, and CSS bundle. `/` redirects
to `/live`. The live view fetches the configured `api.stream.viewer_url`
directly in the browser and receives detection state through the WebSocket. It
does not proxy video through the Rust API.

### Identified MJPEG parts

Spatial overlays are drawn only when a decoded JPEG part and the perception
decision have the same frame, capture session, engine, configuration,
calibration, and rectification identities. The browser keeps a bounded ring of
decoded stills; it never substitutes a nearby frame. An unmatched identified
frame can be shown without a spatial overlay. A legacy MJPEG stream without
the identity headers remains viewable through the browser's native image
stream, but spatial overlays are disabled.

Cue geometry carries its own `source_frame_id`. The viewer shows cue geometry
only when that source frame is the exact displayed frame. Ball positions may
remain available from an older measurement with their original observation
age, while the decision frame identifies the current image used for aiming;
those identities must not be conflated.

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
