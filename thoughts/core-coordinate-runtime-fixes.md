# Core coordinate runtime fixes

## Contract

`Vector2D` inputs can retain camera-resolution coordinates and scale metadata,
but `BallState` and `TableState` normalize their spatial values to canonical
4K pixels. Table dimensions, ball radii, and calibrated playing-area corners
therefore share canonical pixel units inside physics and correction code.
`ErrorCorrector._move_ball_to_table` accepts a scale-aware position, converts it
to canonical pixels, and returns an explicitly canonical point; its radius
argument is the already-canonical `BallState.radius`.

## Boundary behavior

For calibrated convex quadrilaterals, table containment measures the true
perpendicular distance from a ball center to each edge. The requested margin is
in canonical pixels, which makes `is_point_on_table(point, ball_radius)` a
whole-ball containment check even for skewed and rotated calibration results.
Correction clips the calibrated polygon against the same inward edge offsets,
then projects an out-of-bounds center to the nearest point on the remaining
feasible polygon. Uncalibrated tables use their canonical rectangular extent.
This replaces the prior axis-aligned clamp, which could leave a ball outside a
calibrated playing area.

## Regression coverage

`backend/tests/unit/test_core_coordinate_runtime.py` covers a camera-resolution
input clamped against a rotated calibration, perpendicular ball-radius margins,
and deterministic separation of coincident balls. The two outstanding core
Ruff `SIM103` findings were simplified without suppressions.
