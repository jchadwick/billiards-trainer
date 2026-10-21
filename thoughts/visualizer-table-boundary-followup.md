# Visualizer table-boundary projection follow-up

## Decision

Camera-space positions in the visualizer are first normalized into canonical
4K pixels, then mapped by a projective homography from the calibrated
playing-area corners (top-left, top-right, bottom-right, bottom-left) to the
local table plane `[0, 1] × [0, 1]`. Missing, malformed, crossed, or unscaled
table corners cause image-space projection to fail closed. The projector
calibration then maps the four local table corners to its four configured
screen corners using bilinear interpolation.

World positions use only explicit `real_world_dimensions` in meters and the
explicit `playing_area_top_left` meter origin, supplied on the point or table
state. Canonical pixel extents (`width` and `height`) are never used to scale
meter values. Meter radii and image-pixel radii are projected by transforming
their horizontal and vertical offsets at the point, so their size follows both
the playing-area homography and projector calibration.

## Implementation

- Added canonical-corner validation and a pivoted projective transform helper
  to `frontend/visualizer/core/coordinates.lua`.
- Routed trajectory, collision, aim, and ghost-ball positions through the
  playing-area mapping in `frontend/visualizer/modules/trajectory/init.lua`.
- Replaced projector calibration's two-axis interpolation with bilinear
  interpolation that honors all four configured corners. The direct
  `transformCanonical` entry point also requires table bounds.
- Added hardware-free coverage for four-corner correspondence, a non-rectangular
  projective quadrilateral, equivalent 1080p and 960p inputs, meter dimensions
  and origin requirements, and missing/invalid bounds.

The current table API exposes physical dimensions but does not yet include a
meter-origin field. Meter-space projection therefore remains disabled unless
an explicit origin is supplied by the point or table metadata. This is
intentional fail-closed behavior; the transport needs to carry the origin if a
producer sends meter-space trajectory points.

## Verification

From `frontend/visualizer`:

- `rg --files -g '*.lua' | xargs -n1 luac -p` — passed for all Lua files.
- `lua test_coordinates.lua` — passed (`All projector coordinate tests passed`).
- `git -c core.whitespace=cr-at-eol diff --check -- frontend/visualizer` — passed.
