# Coordinate contract

The detector and live overlay share a canonical 3840 × 2160 image plane. Raw
source-frame dimensions remain part of every observed ball record. Rust scales
detector points from the actual source dimensions into the canonical plane and
serializes canonical points with `[1, 1]` scale. The browser converts them back
to the dimensions of the displayed video before drawing.

Each coordinate-bearing value must use the scale and unit defined by its
contract. Do not infer scale from a point's magnitude, the camera configuration,
or the current viewer dimensions. Ball overlays use `radius_extent` when a
detector box half-width and half-height are available; otherwise they use the
configured scalar `radius` and its own `radius_scale`. A point's scale does not
implicitly apply to a radius.

The public state contract distinguishes image-space points from calibrated
table-plane positions. Image positions are in canonical pixels. A calibrated
`table_position` is in millimeters with its origin at the center of the playing
surface. Missing table calibration means `table_position` is absent; the client
must not estimate it from image coordinates.

Image spaces are versioned in the scene contract (`raw` and `rectified`). The
live public overlay currently requires raw source coordinates and matching
source-frame dimensions. Calibration transforms are associated with explicit
revisions and must be applied by the Rust calibration path before table-plane
positions are published.

Relevant implementations and tests:

- Rust public-state conversion: `perception/src/public_state.rs`.
- Scene and public-state schemas/fixtures: `perception/contracts/public-state/`.
- Browser normalization and rendering: `frontend/web/src/utils/coordinates.ts`
  and `frontend/web/src/components/video/LiveView.tsx`.

When changing a spatial payload, update its versioned schema, Rust validation,
browser conversion, and contract tests together.
