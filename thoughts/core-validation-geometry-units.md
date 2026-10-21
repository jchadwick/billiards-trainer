# Core physics validation geometry units

Physics validation details now keep canonical-pixel units and scale metadata attached to serialized positions, ball radii, table extents, and computed geometric distances. Existing numeric fields remain available for compatibility; out-of-bounds reports also include the ball radius and the normalized table-dimension scale so consumers can interpret the boundary check.

The validator's defensive handling of malformed legacy velocity/force/collision data now matches its runtime contract in typing, and validation-summary collections are explicitly typed. For list-based trajectories, empty-trajectory errors use `unknown` as the ball ID because a trajectory point does not contain that identity.

Verification: `backend/tests/unit/validation/test_physics.py` passes (76 tests, `--no-cov`); Ruff, Black, isort, and strict mypy on `backend/core/validation/physics.py` pass with Python 3.11.
