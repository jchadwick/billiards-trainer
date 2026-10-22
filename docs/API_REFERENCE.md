# Rust Gateway API Reference

The supported application API is implemented by the Rust `billiards-server`.
The Python FastAPI package is retained for offline compatibility tests and is
not the active application contract. Only routes listed here are currently
registered by the Rust gateway; historical Python-only routes are not
available in the supported application.

## Base URLs

- HTTP: `http://localhost:8000`
- State WebSocket: `ws://localhost:8000/api/v1/websocket/ws`
- API prefix: `/api/v1`

## Health and readiness

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health/live` | Confirm the Rust gateway process is responding |
| `GET` | `/api/v1/health/ready` | Report whether the current state source is ready |

The container health check uses `/api/v1/health/live`. Readiness is separate:
the gateway may be alive while no localized state has arrived.

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

Configuration requests use the Rust-owned versioned schema, bounded request
handling, and revisioned persistence described in
[Configuration](CONFIG.md). The API is unauthenticated by design.

## Current game state

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/game/state` | Read current localized state with optional session metadata |
| `POST` | `/api/v1/game/reset` | Reset optional game-session metadata |

The game-state route keeps session metadata separate from ball localization; a
metadata-store failure does not suppress a valid current scene. The browser
application currently exposes only the `/live` view.

## WebSocket state stream

Connect to `/api/v1/websocket/ws` to receive the current state stream. The
gateway sends the latest available state after the WebSocket handshake; no
separate subscription message is required. Delivery is best-effort and
latest-value: intermediate states may be coalesced or dropped so a slow client
does not force delivery of stale frames.

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/websocket/ws');
ws.onmessage = (event) => {
  console.log('Latest state:', JSON.parse(event.data));
};
```

The versioned Rust scene and public-state contracts are in
[`perception/contracts`](../perception/contracts/). Prediction and numbered
ball identity are not part of the current live-view contract.

## Browser assets

When started with a configured web root, the Rust gateway serves the built
HTML, JavaScript, and CSS bundle. Extensionless client routes fall back to the
application entry point; missing API routes and missing assets remain 404s.
The production browser client is standard web technology, not Rust/WASM.
