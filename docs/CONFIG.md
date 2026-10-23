# Configuration Guide

The Rust `billiards-server` is the runtime configuration authority. It reads
the repository's `config.json` by default. Select a different file for a local
run with `RUST_CONFIG`:

```bash
RUST_CONFIG=/path/to/config.json make run
```

`make run-api` uses the same setting and starts the gateway without live
capture. The immutable Rust-owned schema and defaults are in
[`perception/contracts/configuration/v1`](../perception/contracts/configuration/v1/).
The older Python schema files are retained only for compatibility tests and
offline evaluation; they do not define the installed server's configuration.

## Validate and update configuration

Check the selected file without changing it:

```bash
RUST_CONFIG=config.json make config-check
```

The Rust HTTP API uses the same validator and persistence store:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/config` | Read active configuration |
| `PUT` | `/api/v1/config/` | Validate and apply a partial update |
| `POST` | `/api/v1/config/import` | Validate and import configuration data |
| `GET` | `/api/v1/config/export` | Export configuration data |
| `GET` | `/api/v1/config/schema` | Read supported keys and validation schema |

For example, validate an update without persisting it:

```bash
curl -X PUT 'http://localhost:8000/api/v1/config/?validate_only=true' \
  -H 'Content-Type: application/json' \
  -d '{"vision":{"camera":{"fps":30}}}'
```

The API is unauthenticated by design. Use it only from the intended local
machine or trusted network. Validation failures are reported without changing
the active configuration; successful updates preserve the store's revision
and restart-required semantics.

## Coordinate settings

Image-space values carry scale metadata, including each point, boundary, and
extent. Do not infer a missing scale from the configured resolution. See
[Coordinate Systems](COORDINATE_SYSTEMS.md) for the coordinate contract.

## Performance measurement

Performance measurements run through the explicit Rust audit commands; their
run options are command-line arguments rather than application settings. The
old Python-only `performance_audit` configuration and instrumentation were
removed because they measured a retired pipeline. See
[Performance Audit](PERFORMANCE_AUDIT.md) for the supported live and recorded
video workflows.
