# Core coordinate serializer audit

## Decision

Core transport boundaries must preserve the coordinate scale already carried by
`Vector2D` and model serializers. Pixel values also identify their units; physical
measurements retain their SI units. Legacy cue `tip_x`/`tip_y` values are accepted
only when the payload declares a source resolution, tip/cue/detection scale, or
canonical pixel space. No coordinate magnitude is used to guess a resolution or
unit. Collision event history validates the same scale-aware mapping before
retaining incoming positions.

Trajectory visualization endpoints and collision positions now serialize as
`{x, y, scale}` objects. The visualization exporter retains flattened `x`/`y`
and velocity aliases for compatibility, with explicit scales beside them. This
replaces the previously ambiguous bare arrays in line endpoints and collision
positions. Canonical units and resolution are stated in each trajectory
payload; ball radius, table extents, cue dimensions, and projector extents carry
their units and relevant scale metadata.

The Core projector adapter preserves scale-bearing points, attaches scale to
projector-native legacy calibration and text positions using the configured
resolution, and tags screen calibration coordinates as pixels and table
coordinates as meters. The Vision adapter rejects unscaled detection positions
and returns its stationary velocity estimate with inherited scale and an
explicit per-second pixel unit.

Game-state updates and exports use model `to_dict()` serializers rather than
recursive dataclass conversion so the canonical table, radius, and coordinate
metadata remain intact through transport and reload.

## Verification

- `pytest backend/tests/core backend/tests/unit/test_core_coordinate_invariants.py backend/tests/unit/test_core_coordinate_serializers.py backend/tests/unit/test_models.py backend/tests/unit/test_coordinates.py backend/tests/unit/test_integration_coordinate_conversion.py backend/tests/integration/test_pipeline_replay.py -q --no-cov`: **475 passed, 1 skipped**. The skip requires installed model assets.
- Ruff, Black, and isort passed on all changed Core modules and tests.
- `compileall` and `git diff --check` passed.
- Strict mypy (`--strict --python-version 3.11 --follow-imports=silent`) was run on the changed Core modules and new serializer test. It reports 126 existing Core typing findings across four files; none are in the new test or changed coordinate payload expressions. The configured Python 3.9 target is unsupported by the installed mypy, so the check explicitly selected Python 3.11.

## Follow-up audit finding

`backend/core/validation/physics.py` still puts a bare pixel `radius` and bare
`table_width`/`table_height` into out-of-bounds validation details. The position
itself is scale-aware. That validation module was outside this serializer task
and should add radius scale/unit plus table dimension scale/unit if these
details are part of the externally visible error contract.
