# Coordinate Systems

**Audience:** contributors who read, write, transform, or render spatial data

## Contract

The backend's canonical image plane is **3840 × 2160 pixels**. Image-space values at module and transport boundaries carry explicit scale metadata. For a source point `{ "x": x, "y": y, "scale": [sx, sy] }`, canonical coordinates are `(x * sx, y * sy)`. Scale factors are positive, finite, and ordered `[scale_x, scale_y]`. Already-canonical values still declare `[1, 1]`; callers must not infer a scale from coordinate magnitude, camera settings, or expected table size.

The scale maps values into the canonical image plane. It does not name the source resolution; retain source-frame dimensions where a later conversion needs them. To convert canonical coordinates to a frame of width `W` and height `H`, use `x_frame = x_canonical * W / 3840` and `y_frame = y_canonical * H / 2160`.

```json
{
  "x": 100,
  "y": 50,
  "scale": [6, 6]
}
```

This point came from a 640 × 360 source, so it maps to `(600, 300)` in the canonical image plane. A point already in that plane uses `"scale": [1, 1]`.

## Scale every spatial value

Apply explicit scale to every coordinate-bearing value, including detections, ball centers, cue points, trajectory and collision points, calibration points, table corners, polygon vertices, bounds, bounding-box origins and extents, radii, and table dimensions. Positions do not supply a scale for an unrelated radius or extent: use that value's own field.

- Point and displacement objects use `scale: [sx, sy]`.
- Bounding boxes use a scale that applies to both the origin and width/height.
- Vision detections use their own `radius_scale`. The vision model permits independent x/y factors; an anisotropic transform represents the detected circular radius as an ellipse in canonical image space. Core/API game-state ball and pocket radii represent circles and reject unequal factors.
- Table pixel extents use `dimensions_scale`; table pixel pocket radii use `pocket_radius_scale`.
- Core table bounds use `table_bounds_scale` over `(min_x, min_y, max_x, max_y)`. Scale both edges on each axis; other bounds schemas must follow their own explicitly documented shared or per-edge scale fields.
- Coordinate arrays are not self-describing. Use an API model that requires scale, or convert an array only when its exact source image dimensions are separately known and the receiving interface explicitly supports that representation.

Missing or invalid scale is an input error. Do not treat an omitted value as canonical identity and do not select a scale by comparing coordinate magnitude with 3840 × 2160.

## Distinct coordinate spaces and units

| Space | Meaning | Conversion rule |
|---|---|---|
| Canonical image pixels | The shared 3840 × 2160 image plane used by core spatial state and API serialization | Image values include scale metadata; `[1, 1]` means already canonical |
| Source-image pixels | Pixels in a particular camera, crop, or resized image | Keep source dimensions and scale x/y independently into canonical pixels |
| Normalized table coordinates | Ratios relative to a declared table area, typically `[0, 1]` | Declare the normalized space; convert using calibrated table geometry, not pixel scales |
| Projector pixels | Pixels in a specific projector surface | Keep separate from image/table coordinates and apply the projector calibration transform |
| Physical table-plane coordinates | Physical positions or dimensions in SI units | Mark points/dimensions with `unit: "meters"`; use a calibrated mapping to image pixels |

`backend/core/coordinates.py` defines scale-aware image `Vector2D`. Core game-state positions, table bounds, ball/pocket radii, and physics calculations use canonical pixel geometry. Physical quantities such as mass, cushion height, calibration world points, and optional real table dimensions remain separate SI values. Do not label core image vectors as meters or combine pixel and meter values in arithmetic without a calibrated conversion.

Calibration requests keep their paired values explicit: `screen_position` is a scaled image point; `world_position` is an `{x, y, unit: "meters"}` table-plane point. The mapping between them comes from calibration, not a universal pixels-per-meter constant.

## Repository-specific inputs

- `config.json` and `backend/config_schema_defaults.json` define the configured table corners and marker dots as coordinates from a 640 × 360 reference image. Every point currently carries `[6, 6]`, and the table section records `calibration_resolution_width: 640` and `calibration_resolution_height: 360`. Vision consumers must use the point's own scale; the resolution settings alone do not authorize reconstructing a missing scale.
- YOLO label coordinates are normalized relative to the exact source image dimensions as required by the label format. Convert using those dimensions at the detector boundary; do not treat normalized labels as pixel positions or guess dimensions from the values.
- The web frontend normalizes received image points and extents to canonical image space, then maps canonical values into the active video frame and canvas. Keep rendering dimensions separate from source-image dimensions.
- The projector and visualizer have their own coordinate contracts for table-normalized and projector spaces. Do not reinterpret those values as image pixels; follow their declared space and scale metadata.

## Worked conversions

### API image point to a 1080p frame

```python
point = {"x": 100.0, "y": 50.0, "scale": [6.0, 6.0]}
canonical = (point["x"] * point["scale"][0], point["y"] * point["scale"][1])
# (600, 300) in 3840 x 2160 canonical pixels
frame_1080p = (canonical[0] * 1920 / 3840, canonical[1] * 1080 / 2160)
# (300, 150) in the 1920 x 1080 frame
```

### Pixel radius

If a detected radius is `4` pixels from the same 640 × 360 source, serialize it as `radius: 4` and `radius_scale: [6, 6]`. Its canonical radius is 24 pixels. Do not reuse a point scale implicitly; include the radius scale in the containing schema.

### Table geometry and physical dimensions

```json
{
  "width": 3200,
  "height": 1600,
  "dimensions_scale": [1, 1],
  "dimensions_unit": "canonical_pixels",
  "pocket_radius": 72,
  "pocket_radius_scale": [1, 1],
  "pocket_radius_unit": "canonical_pixels",
  "real_world_dimensions": {"width": 2.54, "height": 1.27, "unit": "meters"}
}
```

The pixel extents describe image geometry; optional real dimensions describe the physical table independently. Neither substitutes for the other's scale or unit metadata.

## Implementation references

- API point and physical-unit models: `backend/api/models/common.py`
- API, WebSocket, and calibration payload schemas: `backend/api/models/requests.py`, `backend/api/models/responses.py`, `backend/api/models/websocket.py`
- Shared API/core conversion: `backend/api/models/converters.py`
- Canonical resolution: `backend/core/constants_4k.py`
- Core scale-aware vector and state normalization: `backend/core/coordinates.py`, `backend/core/models.py`
- Configured vision geometry: `backend/vision/coordinates.py`
- Frontend image normalization: `frontend/web/src/utils/coordinates.ts`

## Review checklist

Before adding or changing a spatial payload, confirm:

1. What coordinate space and unit does each spatial value represent?
2. Does each point, boundary, radius, and extent carry the correct scale in the receiving schema?
3. Are source image dimensions retained where frame conversion or normalized labels need them?
4. Are physical meter values explicitly labeled and converted only through calibration?
5. Do tests cover identity and non-identity scales, boundaries, and invalid or missing metadata?
