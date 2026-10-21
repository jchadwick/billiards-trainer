# Configuration Guide

The backend uses one validated JSON configuration file at runtime. The default
launcher selects the repository-root `config.json`; pass `--config PATH` to
`backend.main` to select another file. The immutable
[`backend/config_schema_defaults.json`](../backend/config_schema_defaults.json)
provides supported keys, types, defaults, and validation rules. It is a schema
source, not another runtime configuration layer.

## Select and read configuration

```bash
python -m backend.main --config config.json
```

Select the file before services use the singleton:

```python
from backend.config import Config

Config.set_config_file("config.json")
config = Config()

camera = config.get("vision.camera", {})
frame_rate = config.get("vision.camera.fps", 30)
all_settings = config.get_all()
```

`Config.get(path, default)` reads a dot-separated path and returns an isolated
copy. The default is used only when the path is absent.

## Update and persistence

Use `set` for a single dotted path or `apply_updates` for a partial nested
object. Use `replace_configuration` only when replacing the full configuration.
All writes validate against the schema defaults, reject unknown keys and invalid
values, merge related settings before cross-field checks, and persist with an
atomic file replacement before updating memory. Persistence errors are raised
and do not publish the candidate configuration in memory.

```python
config.apply_updates({
    "vision": {"camera": {"fps": 30}},
    "api": {"websocket": {"delivery": {"send_timeout_seconds": 5.0}}},
})

# Re-read the selected file after an external edit.
config.reload()
```

The HTTP API uses the same validator and store:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/config` | Read the active configuration |
| `PUT` | `/api/v1/config/` | Validate and apply a partial update |
| `POST` | `/api/v1/config/import` | Validate and import configuration data |
| `GET` | `/api/v1/config/export` | Export configuration data |
| `GET` | `/api/v1/config/schema` | Read supported keys and validation schema |

A validation-only update performs no write:

```bash
curl -X PUT 'http://localhost:8000/api/v1/config/?validate_only=true' \
  -H 'Content-Type: application/json' \
  -d '{"vision":{"camera":{"fps":30}}}'
```

`force_update=true` can bypass warnings only; it never bypasses validation
errors. A successful update response reports changed fields and whether a
restart is required. Configuration imports and direct Python updates use the
same schema rules.

## Coordinate configuration

Image-space configuration values must carry scale metadata, including every
point, boundary, and extent. The default table corners and marker dots are
stored with their reference resolution and per-point scale. Do not infer a
missing scale from the configured resolution. See
[Coordinate Systems](COORDINATE_SYSTEMS.md) for the complete contract and
examples.

## Performance audit

Detailed timing and event collection is disabled by default. Enable it for one
launch without changing the saved configuration:

```bash
python -m backend.main --performance-audit
```

Or set `performance_audit.enabled` to `true`. The independent
`state_enabled`, `trajectory_enabled`, and `viewers_enabled` switches select
measurement domains. Defaults, validated bounds, sampling and export limits,
run commands, and result interpretation are documented in
[Performance Audit](PERFORMANCE_AUDIT.md).

## Validation

The API schema endpoint exposes the shared runtime schema:

```bash
curl 'http://localhost:8000/api/v1/config/schema'
```

The hardware-free configuration tests are in `backend/tests/unit/test_config_validation.py`
and `backend/tests/unit/test_config_compatibility.py`. The default launcher reads
`config.json` unless `--config` selects another path. `API_PORT` overrides only
the API server port when the application starts.
