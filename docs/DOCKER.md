# Docker deployment

The repository Compose file builds one backend process per table from
`backend/Dockerfile`. It does not start Redis, the web frontend, or the LÖVE
projector. Redis is optional for distributed rate-limit admission; it does not
distribute camera ownership, game state, or broadcasts.

## Start the backend

```bash
docker compose build backend
docker compose up -d backend
docker compose logs -f backend
docker compose down
```

The API is exposed on host port 8000 by default. To use another host port, set
`API_PORT` when starting Compose; the container continues listening on port
8000:

```bash
API_PORT=8080 docker compose up -d backend
```

The image includes the repository's `config.json`; `CONFIG_FILE` selects the
file passed to `python -m backend.main`. The Compose service mounts
`./models` read-only at `/app/models` by default (`MODEL_DIRECTORY` can select
another host directory). Supply the detector model named by the selected
configuration. Startup fails clearly when that required model is missing. The
default configuration points to a CoreML `.mlpackage`; Linux deployments must
select a Linux-compatible model format and device in their mounted config.

To use a host-managed config, add a read-write bind mount and change
`CONFIG_FILE` in a Compose override. The file must be writable by the
container's application user because validated API updates persist atomically.

The container starts through `backend/entrypoint.sh`, which performs import
checks and launches `backend.main` as one process. Run one backend instance per
table. Do not increase worker count to scale multiple tables in one process;
start a separate instance for each table.

## Other components

The frontend runs separately from Compose. From `frontend/web`, install the
locked dependencies and start the Vite server with `npm ci` and `npm run dev`.

The projector is also separate. It listens on `127.0.0.1:9999` for its local
custom UDP payload; the backend does not publish that protocol. WebSocket video
delivery is disabled. The backend's HTTP video routes remain available on the
API port.

## Rate limiting

In-memory rate limiting is the default for this single-worker deployment. If
Redis storage is explicitly selected in configuration, provide a reachable
Redis service separately. The application fails readiness when the selected
Redis store is unavailable and rejects WebSocket admission during an outage;
it does not silently fall back to per-process memory limits.
