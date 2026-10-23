# Projector Local Build and Status

The projector is a separate native LÖVE application. Project documentation and
scripts support local installation and execution only. Remote SSH/SCP
deployment, browser packaging, and control helpers are not maintained; the
repository does not provide projector login or authentication.

## Run locally

Install LÖVE using the operating system's package manager, then run:

```bash
cd frontend/projector
love .
```

For coordinate contract checks, see
[`frontend/projector/TESTING.md`](../frontend/projector/TESTING.md).

## Integration status

The native projector accepts its custom UDP message format on loopback. The
backend does not currently publish that format, and the projector does not
include a WebSocket client or a production backend adapter. It remains a
standalone local application until an adapter is implemented and tested.
