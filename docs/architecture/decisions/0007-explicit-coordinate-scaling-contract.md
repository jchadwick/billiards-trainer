# ADR-0007: Explicit scaling for image-space geometry

- Status: accepted
- Date: 2026-10-19
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The system consumes geometry from camera frames, image crops, normalized detector
outputs, browser canvases, and projector surfaces. These values can have similar
numeric ranges while representing different spaces. The earlier documentation
also described core state as meters, although current game geometry and physics
use canonical image pixels. Missing scale metadata or magnitude-based unit
inference can silently displace detections, table boundaries, and predictions.
Physical calibration points still need real-world units.

## Decision

Use 3840 × 2160 canonical image pixels for image-space game geometry. Every
image-space point and vector crossing a typed API or module boundary must carry
positive finite `[scale_x, scale_y]` metadata that multiplies its values into
canonical coordinates; canonical values explicitly use `[1, 1]`. Scale
bounding-box origins and extents, table dimensions, radii, and every boundary
with the field defined for that geometry. Do not infer a coordinate space,
resolution, scale, or unit from numeric magnitude.

Keep normalized table and projector pixels as separate declared spaces. Keep
physical table-plane points and dimensions separate from image geometry and
mark physical values with `unit: "meters"`; connect them to pixels only through
explicit calibration.

## Consequences

### Positive

- Boundaries, detections, radii, and extents can be converted consistently
  between source frames and canonical image coordinates.
- Missing metadata is detectable at schema and conversion boundaries instead
  of becoming an invisible coordinate offset.
- Physical measurements remain meaningful without being conflated with pixels.

### Negative

- Producers must carry scale metadata through intermediate payloads and
  serialization, including identity scale for canonical values.
- Clients that previously supplied bare point arrays or meter-like core state
  must convert to the declared contracts.

## Alternatives considered

### Store all geometry in meters

This conflicts with the current camera-centered core model and requires a
calibrated image-to-world transform for every detection, boundary, radius, and
rendering path. Physical values are retained where they are intrinsically
physical, while image geometry remains in its canonical pixel plane.

### Infer scale from source dimensions or coordinate magnitudes

Camera settings can differ from the frame or reference image that produced a
point, and numeric ranges overlap across spaces. Inference would silently
accept ambiguous input. Producers instead declare scale and retain source
resolution when needed.

## Verification

- API model and conversion tests reject missing/invalid scales and preserve
  coordinate, radius, and dimension metadata across HTTP and WebSocket models.
- Calibration route tests verify that all corners carry scale consistent with
  declared resolution.
- Core coordinate and serializer tests verify canonical geometry and separate
  meter-valued dimensions.
- Vision, frontend, projector, visualizer, and tooling coordinate tests cover
  scaled points, bounds, extents, and radii at their respective boundaries.
- The project-wide coordinate audit note records the scoped Python-constructor
  and JSON asset scans and their exclusions.
