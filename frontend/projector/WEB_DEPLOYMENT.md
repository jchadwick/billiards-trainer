# Local Browser Build Guide

The projector can be packaged as a native LÖVE application or as a love.js
browser build. Packaging does not add a backend transport: the native app
receives its custom UDP protocol on `127.0.0.1:9999`, and the browser build has
no WebSocket client.

## Build the browser package

Prerequisites are Node.js and npm. The first build clones the public love.js
repository and installs its dependencies; subsequent builds reuse that local
checkout.

```bash
cd frontend/projector
./build_html.sh
```

The generated static site is written to `frontend/projector/build/web/`.

## Preview on this machine

Serve the output on loopback and open the shown local URL in a browser:

```bash
cd frontend/projector/build/web
python3 -m http.server --bind 127.0.0.1 8080
```

Then visit `http://127.0.0.1:8080`. Stop the server with Ctrl+C.

## Runtime and integration status

The native LÖVE app binds a UDP socket to `127.0.0.1:9999` and accepts the
custom payloads documented in [README.md](README.md). The backend API exposes
`/api/v1/websocket/ws`, but the projector has no WebSocket client and the
backend trajectory payload (`lines` with scaled `start`/`end` points) differs
from the projector UDP contract (`paths` and optional `ghostBalls`). Neither
the native app nor the browser package automatically connects to the backend.

The repository contains local package and preview workflows only. It does not
include a remote deployment or projector-control mechanism. Integrating live
backend data requires a separately designed adapter that selects a transport,
converts coordinate and trajectory schemas, and preserves the local receiver's
message contract.

## Troubleshooting

- **Build reports `npm` missing:** install Node.js and npm using the local
  operating system's package manager.
- **Browser shows a blank page:** check the browser console and confirm the
  local server is serving the generated `index.html` and companion assets.
- **No live backend updates:** no production backend publisher or browser
  transport is wired. Local test messages exercise only the native UDP receiver.
- **Graphics do not render:** check browser WebGL support and the browser
  console for rendering errors.
