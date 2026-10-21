# Coordinate Tests

The current coordinate regression suite is
[`test_coordinates.py`](test_coordinates.py). It exercises `Vector2D` input
validation and factories, geometry and arithmetic, conversion between source
resolutions and the canonical 4K coordinate space, and serialization.

Run commands from the repository root so Python imports use the canonical
`backend.*` package paths:

```bash
python -m pytest --no-cov backend/tests/unit/test_coordinates.py
python -m pytest --collect-only --no-cov backend/tests
python -m pytest backend/tests -m "not slow and not hardware"
```

The test module imports `Vector2D` from `backend.core.coordinates`. Core
physics uses `backend.core.models.BallState`; vision detections use the
separate `backend.common.models.Ball` type. See the root `pyproject.toml` for
the shared pytest markers, timeout, and coverage threshold.
