# Visualizer coordinate scaling

## Decision

The Lua projector consumes image coordinates whose two-axis `scale` maps them
to canonical 3840 x 2160 pixels. The shared `core.coordinates` module applies
each scale exactly once at message ingestion and marks normalized values as
canonical. Scale is required even when it is `[1, 1]`; an optional
`coordinate_space` must describe a compatible image-pixel space. Calibration
then maps canonical pixels through the calibrated playing-area boundary to
normalized projector screen corners.

Projector calibration corners use normalized output fractions and carry
`coordinate_space: "normalized_projector"` with identity scale. Normalized
table and projector points and bounds remain in their declared planes and
require identity scale. Table sizes and coordinates explicitly marked in meters
remain physical values; meter-based trajectory points are divided by the
received table width and height only when projected. Pixel table extents require
an explicit `dimensions_scale`; scalar ball and pocket radii require a declared
meter or pixel unit, and pixel radii require explicit scale metadata. Rectangle
bounds, corners, pockets, trajectory endpoints, collisions, and aim endpoints
are normalized once before projection.

## Verification

- Exact commands from `frontend/visualizer`: `lua test_coordinates.lua` and
  `luac -p core/coordinates.lua core/state_manager.lua
  modules/trajectory/init.lua rendering/calibration.lua
  modules/network/websocket_example.lua test_coordinates.lua`. Both pass. The
  checks cover per-axis point/boundary/extent/radius scaling,
  coordinate-domain separation, transport ingestion, the executable
  WebSocket example payloads through a deterministic socket stub, and
  projector mapping.
- The CRLF-aware `git diff --check` passes for tracked visualizer changes. A
  trailing-whitespace scan also passes for the new coordinate module, Lua test,
  and this implementation note.
- Exact LOVE smoke command from `frontend/visualizer`: `cp main.lua
  main.lua.codex_backup; trap 'mv main.lua.codex_backup main.lua' EXIT; cp
  test_init.lua main.lua; love . --test-init`. Result: 7/8 modules load; the
  existing Network module cannot find `lib.websocket` in this checkout.
- The executable WebSocket example now declares camera-pixel coordinates and
  carries scales for ball positions and velocities, table corners, bounds,
  pocket locations, trajectory segments, collisions, and ghost-ball radii. Its
  state and statistics examples now connect correctly so their callbacks run.
- Pixel table dimensions are multiplied by `dimensions_scale` only when
  `dimensions_unit` explicitly marks pixel extents. Physical meter dimensions
  remain unchanged.
