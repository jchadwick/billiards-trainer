# Local development

There is no `docker-compose.dev.yml`. The root `docker-compose.yml` runs the
backend image without source mounts or hot reload and exposes only the HTTP API.
Use it to verify the packaged backend locally:

```bash
docker compose up --build backend
```

The image uses CPU-only PyTorch and ONNX Runtime. CUDA acceleration requires a
separate image built with CUDA-compatible PyTorch and a matching container
runtime.

For backend hot reload, run the application from the repository root instead:

```bash
python -m backend.main --config config.json
```

The web client and projector are separate applications. Run the web client with
`npm ci && npm run dev` from `frontend/web`. Launch the LÖVE projector from
`frontend/projector` when needed. Its local UDP receiver binds to
`127.0.0.1:9999`; the backend does not publish the projector's custom protocol,
and the old Compose mapping for UDP port 5005 has been removed.

The backend WebSocket endpoint is
`ws://localhost:8000/api/v1/websocket/ws`. It carries state and trajectory
updates; WebSocket camera-frame delivery is disabled. Camera video is served
through the backend's HTTP stream routes.
