# Core projector coordinate transform

## Decision

Projector trajectories share a single table-meter coordinate frame. The origin
is the top-left playing-area corner; +X follows the top edge to the top-right
corner, and +Y follows the left edge to the bottom-left corner. Projector
calibration world points must explicitly name this origin and use meters.

The runtime table context maps canonical camera/image pixels to table meters
using the four ordered `TableState.playing_area_corners` and the explicit
`real_width_meters` / `real_height_meters`. `CoreModuleIntegrator` snapshots the
matrix, canonical corners, and physical dimensions into immutable tuple-valued
context when it sends trajectory updates. Missing corners or physical
dimensions omit the context, so the projector draws no trajectory.

Projector calibration scales each explicitly scaled source pixel point into
canonical projector pixels once, then calculates canonical-projector-pixel →
table-meter homography. Invalid, singular, or identity matrices are rejected;
calibration never stores an identity fallback. Trajectory coordinates use
camera-canonical-pixel → table-meter → projector-canonical-pixel transforms.
The final canonical projector coordinates are divided by the projector scale
and serialized as native projector pixels with that scale, so downstream
consumers can multiply exactly once. Explicit table-meter trajectory points
are accepted only with `coordinate_space=table_meters`, `coordinate_unit=meters`,
and the same origin context.

## Verification

- `pytest backend/tests/unit/test_core_coordinate_serializers.py -q --no-cov`:
  **20 passed**. Coverage includes resolution equivalence, a perspective
  projector mapping, meter-unit and origin requirements, missing geometry,
  identity calibration, and missing/identity/singular runtime matrices.
- Related Core coordinate invariants and integration conversion suites passed:
  **42 tests total**.
- Ruff, Black, and isort passed on `backend/core/integration.py` and the
  coordinate serializer tests.
- Strict mypy was run on the changed module and tests with Python 3.11. The test
  module is clean; the existing integration module still reports legacy strict
  typing findings outside this transform, and the repository's configured
  Python 3.9 target is unsupported by the installed mypy version.

## Runtime data limitation

The transform requires calibrated playing-area corners and explicit table
dimensions in meters, plus projector calibration points in the same meter
origin. When runtime state lacks any of these values, trajectory projection is
suppressed rather than guessed. WebSocket video remains outside this adapter.
