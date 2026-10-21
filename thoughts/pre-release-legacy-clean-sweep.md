# Pre-release legacy clean sweep

**Status: implementation complete; quality gate open.** This audit removed
compatibility-only APIs and unimplemented stubs after checking repository
callers, runtime registration, and test coverage. The full suite passes, but
the repository's existing 80% coverage gate remains below threshold. Explicit
coordinate transforms and safe operational fallbacks remain documented below.

## Removed

- Public aliases for core handlers, assistance/prediction engines, physics
  models, tracker, and cache; callers now import canonical names.
- Deprecated subscription method aliases, old conversion wrappers used only by
  tests, unused alternative-shot helpers, and old config loading aliases.
- Old WebSocket connection route and unsupported message/payload aliases.
- The frontend Nginx `/ws` WebSocket alias; API and WebSocket connections now
  share the canonical `/api/v1/websocket/ws` path and API listener.
- Unconsumed `api.websocket_path` and `projector.network` configuration values;
  config validation now rejects these removed settings.
- Duplicate/orphan API models, route modules, vision configuration manager,
  detector factory, placeholder core validation module, no-op game-rules stub,
  unused logging metrics getter, and unimplemented calibration routes.
- Five uncollected top-level manual scripts for API, camera detection, and the
  4K migration, plus an unreferenced YAML config fixture; their current
  contracts are exercised by the maintained backend tests and replay harness.
- Obsolete WebSocket URLs in live frontend examples, diagnostics, and the
  development server; projector-only WebSocket configuration fields were
  removed from both runtime config and defaults.
- Projector WebSocket URL/state scaffolding and unsupported geometry metadata
  forms. The projector's local UDP `paths`/`ghostBalls` protocol remains a
  distinct, currently unconnected feature.
- Misleading deployment wiring: the dead UDP 5005 publication, projector
  `EXPOSE 5005`, unused video/config environment variables, unsupported worker
  count settings, obsolete Compose `version`, and multi-worker container launch
  override. Docker now launches the same single-process application entry
  point used by the systemd deployment. The retired separate WebSocket port was
  removed from container metadata and deployment preflight checks.
- Stale current documentation for removed routes, aliases, configuration APIs,
  and video-over-WebSocket behavior. Unwired GStreamer multi-consumer streaming
  proposals and old implementation reports are marked as historical instead
  of being rewritten as if their original claims were current.

## Coordinate contract

Image-space points, vectors, table corners, boundaries, boxes, radii, and other
extents carry explicit per-axis scale or source-resolution provenance at each
boundary. Source-resolution metadata is retained where the current capture and
observation contracts need to normalize arbitrary camera frames; it is not an
implicit migration from an assumed pre-4K format. Normalized, projector, and
physical-world coordinates remain separate named spaces. No scale is inferred
from coordinate magnitude.

The project-wide audit is recorded in
[`project-wide-coordinate-audit.md`](project-wide-coordinate-audit.md). Its
focused coordinate suite passed 82 tests, and a production-source AST scan
found no unscaled Python `Vector2D` constructors.

## Deliberately retained

- Core event manager's legacy subscriber bridge: active module coordination and
  handler delivery still use it. Removing it requires migrating that live event
  flow first.
- Standard resolution-name mappings such as `1080p` and `4K`: these are current
  human-readable inputs to the resolution utility and examples, not aliases for
  an older function signature or serialized coordinate schema.
- Explicit rejection of deprecated table field names: these guards prevent
  obsolete dimensions from being silently ignored; they do not convert or
  accept the old schema.
- Exact `source_resolution` inheritance in game-state/observation deserialization:
  it supplies the current enclosing frame provenance to child coordinates.
- Scoped geometry-scale inheritance in visualizer/projector payloads: children
  inherit only their explicitly declared containing geometry scale; unsupported
  legacy metadata names are rejected.
- Opt-in vision phase scheduling's pass-through behavior while scheduling
  profiles are disabled. It protects current full-perception operation until
  representative camera recordings validate the cadence profiles.
- Detector/model fallback modes and defensive numeric defaults: these are live
  operational behavior when optional hardware, models, or partial direct
  constructor settings are absent, not old serialized-schema adapters.
- Separate detection (`common.models.Ball`) and physics (`BallState`) models;
  their fields and lifecycle responsibilities differ.

## Verification and remaining quality issue

- Full hardware-free backend suite: 1,113 passed, 1 deselected because it is
  marked for model-asset replay. The no-cov run passed.
- Coordinate-focused backend suite: 82 passed. Frontend web suite: 34 passed.
- Latest configuration tests: 55 passed. Deployment launcher tests: 7 passed;
  `bash -n`, JSON parsing, and `docker compose config --quiet` passed.
- Full backend run with the repository's required coverage gate executes the
  tests successfully but reports 46.48% coverage against the unchanged 80%
  threshold. The gate was not lowered.
- Frontend WebSocket tests and type checking were run. The 34 frontend tests
  pass; TypeScript checking still reports unrelated pre-existing errors in
  accessibility components, generic store wrappers, and other application
  examples.
- Docker Compose configuration and the production image build pass. The image
  uses CPU-only PyTorch, imports the backend and ONNX/YOLO detector stack, and
  is 1.91 GB. The Docker context excludes frontend assets, test recordings,
  test suites, and nested development caches; unused display and test-runner
  packages are not installed in the production image. A full container startup
  still requires an installed detector model; the default CoreML model is not
  suitable for Linux and must be overridden for Linux deployment.
- Hardware-backed replay/30–60 FPS measurement still requires a camera and
  installed detector assets; no throughput guarantee is inferred from the
  hardware-free benchmark.
