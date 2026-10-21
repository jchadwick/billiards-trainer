# Billiards Trainer API Reference

The running API's OpenAPI document at `/openapi.json` is the authoritative schema for request and response models. The routes below are registered by `backend/api/main.py`; values in these examples follow the current Pydantic transport models.

## Base URLs

- HTTP: `http://localhost:8000`
- WebSocket: `ws://localhost:8000/api/v1/websocket/ws`
- HTTP API prefix: `/api/v1`

## Coordinate contract

The canonical image plane is 3840 × 2160 pixels. Every image-space point or vector is encoded as `{ "x": number, "y": number, "scale": [sx, sy] }`; compute canonical x/y by multiplying the corresponding value by its scale factor. `[1, 1]` is required for values already in canonical space. Omitted scale is invalid. Coordinate magnitude and camera resolution are never used to guess scale.

```json
{
  "x": 100,
  "y": 50,
  "scale": [6, 6]
}
```

This represents `(600, 300)` canonical pixels. The scale factors map source values into canonical space; when source resolution is needed for another conversion, retain it explicitly. To map canonical coordinates into a frame of size `W × H`, multiply x by `W / 3840` and y by `H / 2160`.

Scale every spatial value, including points on boundaries and bounds. Independent geometry uses independent scale fields:

- Bounding boxes carry scale for the origin and width/height.
- Vision detections use `radius_scale`, which may be anisotropic and then represents an ellipse. Core/API game-state ball and pocket radii use `radius_scale` and `pocket_radius_scale` respectively and require equal x/y factors because they are circular values.
- Table width and height use `dimensions_scale` and `dimensions_unit: "canonical_pixels"`.
- Physical table dimensions and calibration correspondences are separate values marked `unit: "meters"`; they are connected to image points through calibration.
- Normalized table and projector coordinates are distinct declared spaces and are never interpreted as image pixels.

The default configured `table.playing_area_corners` and `table.marker_dots` are recorded at 640 × 360 and each point carries `[6, 6]`. The config also records that reference resolution. Do not remove per-point scale and reconstruct it from the resolution setting. See [Coordinate Systems](COORDINATE_SYSTEMS.md) for conversion examples and repository-specific fixture conventions.

## Health and readiness

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Health details |
| `GET` | `/api/v1/health/ready` | Readiness, including selected rate-limit store availability |
| `GET` | `/api/v1/health/live` | Liveness |
| `GET` | `/api/v1/health/version` | Version information |
| `GET` | `/api/v1/health/metrics` | System and application metrics |

## Configuration

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/config` | Read current configuration |
| `PUT` | `/api/v1/config/` | Validate/apply a partial configuration update |
| `POST` | `/api/v1/config/reset` | Restore supported defaults |
| `POST` | `/api/v1/config/import` | Validate/import configuration data |
| `GET` | `/api/v1/config/export` | Export current configuration |
| `GET` | `/api/v1/config/schema` | Read configuration schema and supported defaults |
| `GET` / `POST` | `/api/v1/config/table/playing-area` | Read/update calibrated playing-area geometry |

Configuration writes validate the merged configuration. `validate_only=true` checks without writing. `force_update=true` may bypass warnings, not schema or validation errors. Playing-area corners and optional marker dots require `x`, `y`, and `scale` for every point.

Example corner update:

```http
POST /api/v1/config/table/playing-area
Content-Type: application/json
```

```json
{
  "corners": [
    {"x": 38, "y": 43, "scale": [6, 6]},
    {"x": 606, "y": 39, "scale": [6, 6]},
    {"x": 603, "y": 324, "scale": [6, 6]},
    {"x": 39, "y": 324, "scale": [6, 6]}
  ],
  "marker_dots": [
    {"x": 182, "y": 182, "scale": [6, 6]},
    {"x": 459, "y": 181, "scale": [6, 6]}
  ],
  "calibration_resolution_width": 640,
  "calibration_resolution_height": 360
}
```

## Game state

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/game/state` | Read current game state; accepts `include_events` and `include_trajectories` query flags |
| `GET` | `/api/v1/game/history` | Read retained state history |
| `POST` | `/api/v1/game/reset` | Reset game state; game type and table-preservation options are query parameters |
| `GET` | `/api/v1/game/stats` | Read game statistics |
| `POST` | `/api/v1/game/export` | Start an export |
| `GET` | `/api/v1/game/export/{export_id}/download` | Download an export |

The state response uses separate API models for balls, cue, and table. Its spatial fields are scale-aware. `GameStateResponse` has no trajectory field; use the WebSocket trajectory stream for trajectory updates. Example excerpt:

```json
{
  "frame_number": 42,
  "balls": [
    {
      "id": "cue",
      "number": 0,
      "position": {"x": 1920, "y": 1080, "scale": [1, 1]},
      "velocity": {"x": 0, "y": 0, "scale": [1, 1]},
      "is_cue_ball": true,
      "is_pocketed": false,
      "confidence": 0.99,
      "last_update": "2026-10-19T12:00:00Z"
    }
  ],
  "table": {
    "width": 3200,
    "height": 1600,
    "dimensions_scale": [1, 1],
    "dimensions_unit": "canonical_pixels",
    "pocket_positions": [
      {"x": 320, "y": 280, "scale": [1, 1]}
    ],
    "playing_area_corners": [
      {"x": 320, "y": 280, "scale": [1, 1]},
      {"x": 3520, "y": 280, "scale": [1, 1]},
      {"x": 3520, "y": 1880, "scale": [1, 1]},
      {"x": 320, "y": 1880, "scale": [1, 1]}
    ],
    "pocket_radius": 72,
    "pocket_radius_scale": [1, 1],
    "pocket_radius_unit": "canonical_pixels",
    "real_world_dimensions": {"width": 2.54, "height": 1.27, "unit": "meters"},
    "surface_friction": 0.2
  }
}
```

Ball response position and velocity use `{x, y, scale}`. WebSocket ball-state payloads also include `radius`, `radius_scale`, and `radius_unit`. Table image extents and physical dimensions must remain separate.

## Calibration

The calibration router is mounted under `/api/v1/vision/calibration`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/vision/calibration/start` | Start calibration; type, force-restart, and timeout are query parameters |
| `POST` | `/api/v1/vision/calibration/{session_id}/points` | Add a screen/world correspondence |
| `POST` | `/api/v1/vision/calibration/{session_id}/apply` | Apply completed calibration |
| `POST` | `/api/v1/vision/calibration/{session_id}/validate` | Validate calibration |
| `GET` | `/api/v1/vision/calibration/{session_id}` | Read a session |
| `GET` | `/api/v1/vision/calibration/` | Read current calibration and table corners |

Calibration point requests pair a scaled image point with an explicitly meter-valued physical point:

```json
{
  "session_id": "cal_20261019_example",
  "point_id": "table_top_left",
  "screen_position": {"x": 38, "y": 43, "scale": [6, 6]},
  "world_position": {"x": -1.27, "y": -0.635, "unit": "meters"},
  "confidence": 1.0
}
```

## Vision and video

Vision operations are mounted under `/api/v1/vision`; video streaming and frame retrieval are under `/api/v1/stream/video`. The exact operation list and schemas are available from the running OpenAPI document. Spatial values returned by vision endpoints must follow the same explicit scale contract above. Video frames additionally include their actual pixel width and height; those frame dimensions describe the image, not a coordinate scale to infer for unrelated payloads.

## WebSocket

The application provides `/api/v1/websocket/ws`. Connect and send a subscription message using the protocol schemas in `backend/api/websocket/schemas.py`:

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/websocket/ws');
ws.onopen = () => ws.send(JSON.stringify({
  type: 'subscribe',
  timestamp: new Date().toISOString(),
  data: {streams: ['state', 'trajectory', 'alert', 'config']}
}));
```

Messages use an envelope containing `type`, `timestamp`, and `data`. State and trajectory coordinates follow the same scale rules as HTTP. Control notifications and errors retain message ordering. WebSocket video/frame broadcast is disabled; use the HTTP streaming/frame endpoints for video.

## Errors and request correlation

HTTP responses use the registered domain error contracts and normalized framework status handling. Responses include an `X-Request-ID` correlation header; WebSocket connection errors carry the connection request ID. Server failures expose a public error message and correlation ID while diagnostic exception details remain in server logs.
