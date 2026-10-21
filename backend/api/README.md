# Backend API

The FastAPI application exposes versioned HTTP routes and one WebSocket stream.
The live OpenAPI document at `/openapi.json` is the authoritative HTTP contract;
[the API reference](../../docs/API_REFERENCE.md) documents coordinate and
transport behavior.

## Run locally

From the repository root:

```bash
python -m backend.main --config config.json
```

Interactive documentation is available at `http://localhost:8000/docs`, and the
OpenAPI schema is at `http://localhost:8000/openapi.json`.

## Current routes

- HTTP API routes use the `/api/v1` prefix. Health, readiness, configuration,
  game state, calibration, diagnostics, and video routes are registered by
  `backend/api/main.py`.
- The only WebSocket connection route is
  `ws://localhost:8000/api/v1/websocket/ws`.
- The LÖVE projector listens on UDP port 9999 for its own custom `paths` and
  `ghostBalls` payload. Production backend publishing to that receiver is not
  wired; backend WebSocket trajectories use a different schema.
- WebSocket video broadcast is disabled. Use the HTTP video and frame routes
  for camera output.

WebSocket clients subscribe by sending a message with a `type` of `subscribe`
and a `data.streams` array. Available stream names and payload schemas are
maintained in `backend/api/websocket/schemas.py`. State and trajectory image
coordinates carry explicit scale metadata; see the API reference for the full
coordinate contract.

## Configuration

The API reads the shared `backend.config.Config` singleton. The application
launcher selects the active JSON file before starting services. Configuration
updates go through the shared schema validator and atomic persistence store.

```python
from backend.config import Config

Config.set_config_file("config.json")
config = Config()
settings = config.get_all()
config.apply_updates({"vision": {"camera": {"fps": 30}}})
```

The HTTP configuration routes include validation-only updates and schema
retrieval. `force_update` can bypass warnings, not validation errors.

## Lifecycle and deployment

The application owns one camera/vision lifecycle and one in-process state for a
table. Run one backend worker per table. Redis can coordinate rate-limit
admission when explicitly selected, but does not distribute camera ownership,
game state, or broadcasts.

## Tests

Run API tests from the repository root with:

```bash
python -m pytest backend/tests/unit/api --no-cov
```

See [API specifications](SPECS.md) for requirement history and
[the backend README](../README.md) for repository-wide setup and verification.
