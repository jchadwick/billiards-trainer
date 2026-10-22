# Local Deployment Guide

The project supports local builds and operation. It does not include remote
login, authenticated deployment, registry publishing, or a production hosting
workflow. The application has no user or service authentication: HTTP,
WebSocket, configuration, and perception endpoints do not require credentials.
Run it on a machine and network where anonymous access is acceptable.

## Run with Docker Compose

From the repository root:

```bash
docker compose build
docker compose up -d backend
curl http://localhost:8000/api/v1/health/live
docker compose logs -f backend
docker compose down
```

See [Docker Build and Local Run](docker-deployment.md) for the current workflow
and distribution notes.

## Build a local distribution

```bash
make deploy
```

This creates a `dist/` directory in the current checkout. Install and run that
bundle on the same machine with `./dist/run.sh`. The script starts the Rust
gateway, which also serves the built browser client; it does not start an
Nginx service. The project does not copy the bundle to another host or provide
remote process control.

## Projector

The projector is a separate LÖVE application with a local UDP receiver. It is
not connected to the backend's production WebSocket publisher. Build and test
it locally using the instructions in
[`frontend/projector/TESTING.md`](../frontend/projector/TESTING.md). The
project does not provide a remote projector login or deployment helper.

## Configuration

The configuration API is anonymous. Use it only from clients on the intended
local machine or trusted network. There is no account setup, password, API key,
bearer token, session, or permission system to configure.
