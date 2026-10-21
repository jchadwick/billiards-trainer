# Core Coordinate Invariants

Core physics stores image-space geometry in canonical 4K pixels. `Vector2D`
retains source coordinates only at an input boundary, together with positive,
finite scale metadata or an exact source resolution. Its geometry operations
compare and calculate in canonical space. A declared world, meter, normalized,
or other unsupported coordinate space is rejected even when scale metadata is
present.

`BallState` and `TableState` normalize position, velocity, pocket, boundary,
extent, and radius geometry before physics uses it. Pixel extents and circle
radii carry explicit scale metadata on input and serialize with canonical pixel
units and identity scales. Circle radii reject anisotropic scales. Real table
dimensions, ball mass, cue dimensions, and cushion height remain physical SI
values with explicit units; no value-magnitude heuristic converts between
meters, millimeters, and pixels. Pixel-per-meter conversion is unavailable
until both real table dimensions are known.

`CoordinateMetadata.table_bounds` are normalized and serialized as canonical
pixel bounds with an explicit identity scale. Legacy camera-space bounds need
camera resolution or a scale; unqualified camera bounds are rejected as
ambiguous. Legacy core model dictionaries without extent scales retain the
historical canonical-pixel interpretation, while explicitly declared pixel
inputs require enough resolution or scale context to normalize them.

Calibration corner and pocket coordinates use their declared camera
resolution, and calibrated physical table dimensions remain in meters. New
cross-resolution tests compare 1080p and 4K ball/table geometry, extents,
pocket radii, bounds, serialization round trips, unsupported coordinate
spaces, and missing physical-dimension behavior.

Focused verification passed:

- 596 Core, coordinate, model, physics, state-validation, and invariant tests
  with coverage disabled.
- Ruff, Black, and isort on the touched Core modules and tests.
- Strict mypy on five touched Core modules with Python 3.11.
