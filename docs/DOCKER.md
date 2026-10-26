# Docker deployment

The image builds the Rust gateway and live detector, a GStreamer capture helper,
and the standard browser client. The runtime image does not install Python.

## Configure model and capture

Compose mounts `CONFIG_HOST_FILE` at `/app/config/config.json` and
`MODEL_DIRECTORY` at `/app/models` (read-only). Live inference requires a
verified ONNX export, its source checkpoint, and the sibling manifest named by
appending `.manifest.json` to the ONNX filename. Set the model paths and
machine-specific capture/calibration settings in the mounted configuration.

Set `api.stream.viewer_url` separately when the browser should display a camera
stream. The browser loads that URL directly; the state API does not proxy video.

## Build and run

```sh
docker compose build backend
docker compose up -d backend
docker compose logs -f backend
```

The service listens on host port 8000 by default. Set `API_PORT` to override
it. Set `START_LIVE=false` to run the gateway without opening a camera. The
scene-ingress listener remains loopback-only inside the container.

One service instance owns one table's capture and state. Use separate
configurations and instances for separate tables; do not scale one instance
against the same camera.

## Persistence and security boundary

`CONFIG_HOST_FILE` selects the writable host configuration file.
`MODEL_DIRECTORY` selects the read-only model directory. The `billiards-data`
volume retains application data when the container is recreated.

The HTTP and WebSocket APIs have no authentication. Restrict network access to
trusted clients.
