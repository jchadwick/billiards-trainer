# API coordinate-contract implementation note

The API model layer now distinguishes scaled image-pixel geometry from physical table-plane geometry. `PositionWithScale` is the shared image-point contract: `scale=[sx, sy]` maps the source point into canonical 4K pixels. API `Coordinate2D`, game `Vector2DModel`, vision `Point2DModel`, pixel bounds, and the legacy WebSocket geometry schema require that metadata. Pixel radii, cue lengths, contour measurements, table pixel dimensions, and calibration pixel error also carry explicit scale factors. Calibration world points and request world positions instead require `unit="meters"`; physical table dimensions and ball/pocket radii remain meter-valued fields.

Request declarations were included in the audit: calibration screen/world points are now separately typed; calibration validation correspondences are typed; manual ball position and velocity updates require scaled points. The unused `validate_coordinate_list()` and `vector2d_to_list()` utilities were removed because both accepted/emitted ambiguous bare pairs and neither had production callers. `list_to_vector2d()` remains available only when the caller supplies scale. `vector2d_to_coordinate2d()` and its inverse preserve non-identity scale. Stale descriptions that called pixel-scaled WebSocket positions “meters” were corrected.

`backend.api.websocket.schemas` had no production constructors for its legacy `BallData`, `CueData`, `TableData`, or trajectory classes; production references are the stream enum and wrapper helpers. Their declared contract now requires scale-aware points and boundaries. Active dict-based producers still need migration in their owning stages:

- `backend/integration_service.py:_convert_detection_to_core_format` emits bare point objects for ball positions and velocities (lines 591 and 593), cue tip (609), table corners (620–622), and pocket positions (626–628). Use the detected frame size/source scale when serializing these.
- `backend/integration_service.py:_emit_multiball_trajectory` emits bare line endpoints and collision position arrays (lines 1752–1766).
- `backend/core/__init__.py` trajectory event construction emits bare line endpoints, collision arrays, and points (lines 344–362).
- `backend/core/physics/trajectory.py` trajectory export emits bare line endpoints and collision positions (lines 901–980).
- `backend/core/integration.py:_serialize_ball` emits bare ball position/velocity dictionaries (lines 353–366). `backend/core/events/handlers.py:_serialize_ball` does the same (lines 363–375), and its collision event carries bare coordinates (lines 338–341); cue serialization just after the ball serializer needs the same treatment.

These producers were intentionally left untouched because this task owns API model declarations/converters only; integration, core, and vision changes are sequenced separately. Frame width and height remain source-resolution context, not coordinates themselves.

## Verification

- Focused API, prediction-lifecycle, rate-limit, and real WebSocket delivery tests: **202 passed** (`--no-cov`).
- Ruff passed on all 11 changed source/test files.
- Strict mypy passed on all 9 changed API source modules.
- Black and isort checks passed on all 11 changed files.
