# Project-wide coordinate audit

Date: 2026-10-19

## Contract established

The canonical image plane is 3840 × 2160 pixels. Scale factors `[sx, sy]`
multiply source x/y values into that plane; identity is explicitly `[1, 1]`.
Points, vectors, bounds, corners, bounding-box extents, radii, and table pixel
extents require scale metadata appropriate to their own schema. Source
resolution is retained when needed for frame conversion. Coordinate magnitude
is never a scale or unit heuristic.

Image geometry is separate from normalized table coordinates, projector
coordinates, and physical table-plane measurements. Physical points and extents
are marked as meters and connected to image space through calibration. Core
physics game geometry is canonical pixel geometry; physical values such as mass
and optional real table dimensions remain SI.

## Coordinate-bearing repository inventory

- `backend/api/models/common.py`, `requests.py`, `responses.py`, and
  `websocket.py` define scale-aware image points, boxes, table pixel extents,
  and separate meter-valued models. API conversion is centralized in
  `backend/api/models/converters.py`.
- `backend/core/coordinates.py` and `backend/core/models.py` normalize game
  positions, table geometry, radii, corners, and bounds to canonical pixels.
- `backend/vision/coordinates.py` converts configured geometry from explicit
  canonical scales to the current frame. The configured table corners and
  marker dots in `config.json` and `backend/config_schema_defaults.json` are
  640 × 360 source points with per-point `[6, 6]` scale.
- `backend/vision/test_data/*.json` fixtures use a top-level canonical identity
  envelope; the fixture README documents inheritance for nested centers,
  boxes, radii, corners, and boundaries. YOLO labels remain normalized to
  their associated source image and are converted at the detector boundary.
- `frontend/web/src/utils/coordinates.ts` normalizes image points and extents
  to canonical coordinates, then maps them to the active frame and canvas.
  Related overlay, store, calibration, and type consumers are recorded in
  `thoughts/web-coordinate-consumers.md`.
- `frontend/projector/core/coordinate_contract.lua` and its local protocol
  tests enforce the projector's transport coordinate spaces. The parallel LÖVE
  visualizer and its implementation notes were retired under ADR-0086.
- The historical coordinate-scaling helper and example scripts referenced by
  earlier debugger notes have since been retired. Current geometry requirements
  are in `docs/COORDINATE_SYSTEMS.md` and are covered at runtime boundaries.

## Static asset and constructor scans

The independent repository JSON asset/config scan found no x/y point or
pixel-radius records lacking a scale field or explicit meter unit, allowing
coordinate metadata inherited from containing records. The scan excluded
caches and dependencies; it did not infer a missing per-record scale from
numeric values.

The independent Python AST scan of `Vector2D(...)` calls in `backend/` and
`tools/` found no production construction missing an explicit scale (including
valid third-positional-argument scales). The sole unscaled construction is the
intentional negative test at
`backend/tests/core/test_collision.py:1028`, which asserts that a missing scale
is rejected.

## Verification artifacts

- Backend coordinate implementation and tests: `thoughts/backend-vision-coordinate-scaling.md`, `thoughts/api-coordinate-contract.md`, `thoughts/api-core-geometry-units.md`, `thoughts/core-coordinate-runtime-fixes.md`, `thoughts/core-coordinate-serializers.md`, and `thoughts/core-validation-geometry-units.md`.
- Frontend implementation notes: `thoughts/web-coordinate-consumers.md` and `thoughts/projector-coordinate-scaling.md`.
- API request/response examples are checked against current models in `docs/API_REFERENCE.md`; the shared contract is documented in `docs/COORDINATE_SYSTEMS.md` and ADR-0007.

The audit note captures scoped scans and module verification records; it does
not claim that every dynamically constructed, runtime-only coordinate has been
statically enumerated. Future spatial fields must add explicit units/scales and
tests at their producer and consumer boundaries.
