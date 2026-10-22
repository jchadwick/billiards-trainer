# Docker deployment

The Compose service builds one Rust application server per table. The image
contains the Rust gateway, Rust live-perception executables, native GStreamer
runtime components, and the built browser client. It does not install or launch
Python. The gateway serves the static HTML/CSS/JavaScript client, exposes HTTP
and WebSocket APIs, and supervises the live detector when `START_LIVE=true`.

## Prepare configuration and model files

Compose mounts `CONFIG_HOST_FILE` at `/app/config/config.json` and
`MODEL_DIRECTORY` at `/app/models` (read-only). The live Rust process requires a
verified ONNX export, its source checkpoint, and the export manifest named by
appending `.manifest.json` to the ONNX filename. Before starting live capture,
set these paths in the mounted configuration, relative to the application root
`/app`:

```json
{
  "perception": {
    "provider": "rust",
    "mode": "live",
    "rust_live": {
      "onnx_path": "models/billiards-yolov8n.onnx",
      "source_checkpoint_path": "models/billiards-yolov8n.pt"
    }
  }
}
```

Place `billiards-yolov8n.onnx.manifest.json` next to the ONNX file in the host
model directory. The example model filenames are illustrative: use a real,
qualified export and its matching checkpoint/manifest. The repository's
default configuration may not contain a qualified Rust live model pair, so a
live launch will fail clearly until one is supplied. There is no implicit
Python or alternate-model fallback.

Set the selected camera or network stream and any applied calibration in the
same configuration. A successful startup alone does not establish calibrated
coordinates, detector accuracy, freshness, or camera qualification.

## Build and run

```bash
docker compose build backend
docker compose up -d backend
docker compose logs -f backend
docker compose down
```

The HTTP service is exposed on host port 8000 by default; override it with
`API_PORT`:

```bash
API_PORT=8080 docker compose up -d backend
```

Set `START_LIVE=false` to run the Rust gateway without opening a camera. This
mode is useful for gateway/replay integration checks and does not perform live
detection. The scene-ingress TCP listener remains loopback-only inside the
container and is not published as a host port.

One service instance owns one table's capture and state. Run separate
containers/configurations for separate tables; do not scale one service into
multiple replicas against the same camera. The Compose file does not start a
projector process.

## Configuration persistence and data

`CONFIG_HOST_FILE` selects the host configuration file and
`MODEL_DIRECTORY` selects the host directory containing the ONNX, checkpoint,
and manifest. The configuration mount is writable because configuration API
updates persist to that file. Keep model artifacts read-only to the service.
The `billiards-data` volume retains application data between container
recreations.

The service is anonymous and has no authentication layer; API and WebSocket
clients send no credentials.
