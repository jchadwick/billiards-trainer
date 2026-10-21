# Projector Local Build and Status

The projector is a separate LÖVE application. Project documentation and scripts
support local installation, execution, and browser packaging only. Remote
SSH/SCP deployment and control helpers have been removed; the repository does
not provide projector login or authentication.

## Run locally

Install LÖVE using the operating system's package manager, then run:

```bash
cd frontend/projector
love .
```

For the local test sender and expected display behavior, see
[`frontend/projector/TESTING.md`](../frontend/projector/TESTING.md). For the
browser package, see
[`frontend/projector/WEB_DEPLOYMENT.md`](../frontend/projector/WEB_DEPLOYMENT.md).

## Integration status

The native projector accepts its custom UDP message format on loopback. The
backend does not currently publish that format, and the projector does not
include a WebSocket client or a production backend adapter. Its browser build is
packaging only and does not connect to the backend. Projector output should
therefore be treated as a standalone local application until an adapter is
implemented and tested.
