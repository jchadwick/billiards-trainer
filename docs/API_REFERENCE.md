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

## Camera and table setup

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/setup` | Read capture status, selected source, configuration revision, and calibration status |
| `GET` | `/api/v1/setup/cameras` | Discover local camera selectors without opening a source |
| `PUT` | `/api/v1/setup/source` | Save a stable selector or HTTP(S) MJPEG URL and restart capture |
| `GET` | `/api/v1/setup/calibration/frame` | Freeze an identified JPEG and return its capture token and calibration store revision |
| `POST` | `/api/v1/setup/calibration` | Solve and save table calibration from the frozen frame |
| `GET` | `/api/v1/setup/calibration/checkerboard.pdf` | Download the four-page, actual-size checkerboard print |
| `GET` | `/api/v1/setup/calibration/captures/{mode}` | List compatible saved slots and their thumbnails |
| `PUT` | `/api/v1/setup/calibration/captures/{mode}/{slot}` | Save or replace a captured still, with measurements for a ball sample |
| `DELETE` | `/api/v1/setup/calibration/captures/{mode}/{slot}` | Remove one saved capture |
| `DELETE` | `/api/v1/setup/calibration/captures/{mode}` | Clear the compatible saved captures for one wizard |
| `GET` | `/api/v1/setup/calibration/ball-placement` | Read the projected target and current ball placement feedback |
| `POST` | `/api/v1/setup/calibration/ball-candidate` | Propose a ball outline on a single frozen still for operator review |
| `POST` | `/api/v1/setup/calibration/optics` | Fit and activate optics from checkerboard stills |
| `POST` | `/api/v1/setup/calibration/ball-centers` | Fit and activate measured ball-center calibration |

Setup status includes `precision_calibration.optics` and
`precision_calibration.ball_centers`. Each reports a compatibility state and
may include a validation detail. The separate `calibrated` flag represents
active table geometry.

Source updates accept `source` and `expected_revision`. Calibration accepts
`schema_version: 1`, the unchanged frame `descriptor`,
`expected_store_revision`, `table_width_mm`, `table_height_mm`, and four
`corners_px` entries with `x_px` and `y_px` in clockwise order from the
top left. Stale revision or frame conflicts return HTTP 409. Successful
calibration restarts the producer to load the saved result.

The capture modes are `optics` and `ball-centers`. To save a still, send its
unchanged `descriptor`, `jpeg_base64`, and `capture_token`. For ball centers,
include `image_center_px`, `radius_px`, and the independently measured
`table_center_mm`. The wizard fits the selected `slots` directly from disk.
Deleting the input photos does not delete the previously fitted profile.
Saved thumbnails and the single-still endpoints are separate from live video,
which the browser receives directly from the camera service.

The placement query accepts `x_mm` and `y_mm` relative to the playing
surface center. It returns projected target and detected ball geometry in
raw camera pixels. The placement check helps align the ball with the
prescribed spot. The physical mark still supplies the reference used for fitting.
When `observation_available` is false, the projected target can still be drawn,
but there is no current ball observation to confirm placement.

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

## Runtime diagnostics WebSocket

Connect to `/api/v1/diagnostics/ws` for complete, versioned runtime-health
snapshots. The gateway sends an initial snapshot even when no producer or scene
is available, then pushes updates on evidence changes and the configured
diagnostics heartbeat. This is a separate stream from the state WebSocket; it
does not republish game state or change the state stream's latest-value
semantics.

Snapshots use `schema_version: 1` and include gateway identity/revision/build,
producer reporter state and active or last-known model/build metadata, stream
freshness settings, and seven named stage assessments: `gateway_service`,
`source_lifecycle`, `capture_progress`, `model_initialization`,
`scheduled_perception_work`, `scene_delivery`, and `localization`. Stage
details include stable status/reason/source fields and, when applicable,
progress ages, in-flight job age/deadline, job counters, scheduler state, and
failure evidence. Last-known producer details are explicitly separate from
active producer metadata.

The channel is push-only; clients should reconnect to resume updates and should
mark the latest snapshot stale after its `stream.stale_after_ms` interval. A
browser connection failure is not proof that the gateway or producer failed.
The HTTP health endpoints remain available for explicit liveness/readiness
checks. The contract and fixtures are in
[`perception/contracts/browser-diagnostics/v1`](../perception/contracts/browser-diagnostics/v1/).

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

The API does not offer an MJPEG endpoint. Live Diagnostics loads its image
directly from the camera service. The frame identity contract above describes
the requirements for a viewer that correlates a separate identified stream
with perception state. Setup exposes a single bounded calibration still.
