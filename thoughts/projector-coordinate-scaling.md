# Projector coordinate contract

Projector UDP table geometry uses a strict contract: trajectory, collision,
aim, state, motion, event, and unfamiliar coordinate-bearing messages must
declare `coordinate_space: "normalized_table"` and
`coordinate_scale: [1, 1]` at the message boundary. Nested normalized points
inherit that owner metadata, and any nested override must repeat the same space
and identity scale. The receiver validates nested x/y points, aim endpoints,
cue tips, pockets, table corners and boundaries, bounds, radii, and table
extents before queue admission. Invalid values are rejected before modules or
registered callbacks can observe them.

Normalized point and bound values must fit in the unit table rectangle. Motion
velocity is a separate finite vector, tagged
`velocity_space: "normalized_table_per_second"` with identity
`velocity_scale: [1, 1]`; it is not constrained to the point range. Ball and
ghost-ball radii use projector pixels and require `radius_space:
"projector_pixels"` and identity `radius_scale`. A physical radius instead
requires `radius_unit: "meters"` and carries no projector-pixel scale. Other
extents require an explicit `extent_space` and identity `extent_scale`.

Physical records remain separate from normalized projector geometry. A physical
point explicitly tagged `unit: "meters"` (optionally
`coordinate_space: "world_meters"`) is checked for finite coordinates without
being transformed. `real_world_dimensions` requires positive finite dimensions
and `unit: "meters"`. Conflicting unit/space declarations and image scales on
physical records are rejected. Chessboard positions and square sizes use
projector pixels. Frame width and height use explicit image-pixel dimension
metadata.

Projector calibration corners remain an output boundary in
`normalized_projector` space with identity scale. Dragging converts display
pixels to normalized projector coordinates and clamps to the display bounds.
There is no fallback that interprets old or unscaled UDP coordinate payloads.

The hardware-free Lua tests cover supported message shapes, nested geometry,
malformed and missing metadata, non-finite/out-of-range values, physical-unit
separation, and rejection before queue/callback delivery. Python sender tests
verify normalized table payloads and projector-pixel chessboard messages.
