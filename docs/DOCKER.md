# Docker deployment

The Compose image packages the Rust gateway, live detector, GStreamer capture
and camera discovery, the browser client, and the verified model. It does not
install Python.

## Build and run

From a checkout with the standard model assets, run:

```sh
docker compose up --build -d backend
```

The build verifies the ONNX export and checkpoint against the export manifest.
Set `MODEL_DIRECTORY` to select another directory with the standard layout:
`exports/billiards-yolov8n.onnx`, its sibling `.onnx.manifest.json`, and
`billiards-yolov8n.pt`. A public source checkout may require obtaining the model
assets before this step. The Dockerfile's `runtime` target is available for
gateway smoke tests without model assets.

Open `http://127.0.0.1:8000` to select a camera and calibrate the table. The
gateway starts even if no camera has been selected. The default host port is
8000 and can be changed with `API_PORT`. Set `START_LIVE=false` for a
gateway-only instance.

OpenVINO Runtime and its Intel GPU plugin must be available inside the container
for live inference. Use a Compose override to provide the matching runtime
libraries and access to the Intel render device. A local camera also requires
passing its video device to the container and granting access to the container
user. A supported network camera can be selected by its HTTP(S) MJPEG URL.

## Persistence

The `billiards-data` volume keeps camera setup, calibration, and the game
session across container replacement. Each service instance should own its own
table and data volume.

To use an existing configuration file, set `CONFIG_FILE` and bind-mount its
**directory** in a Compose override. The gateway needs directory write access
to update the file atomically. For example:

```yaml
services:
  backend:
    environment:
      CONFIG_FILE: /app/site-config/config.json
    volumes:
      - ./site-config:/app/site-config
```

The HTTP and WebSocket APIs have no authentication. The published port binds
to loopback by default.
