# Frontend API Geometry Unit Consumption

Date: 2026-10-19

The frontend now models the unit-bearing HTTP `TableInfo` and WebSocket/Core
`TableStateData` contracts alongside the existing vision table payload. Both
forms normalize table width and height into canonical image pixels exactly
once, using the shared per-axis `dimensions_scale` or the legacy independent
`width_scale` and `height_scale` values. Physical dimensions remain a separate,
validated `{ width, height, unit: "meters" }` property.

Table geometry accepts supplied `corners`, `playing_area_corners`, and
`pocket_positions`/`pockets`. When no boundary points arrive, the frontend keeps
the pixel extents and pocket centers but leaves corners empty and bounds absent;
it does not estimate a boundary from pocket centers. Unsupported explicit pixel
units discard pixel extents or radii rather than treating their numbers as
pixels. Ball and pocket radii use their own scale metadata, independently from
point coordinates.

Regression coverage in
`frontend/web/src/tests/integration/ScaledCoordinateConsumption.test.ts`
compares server-shaped 1080p `[2, 2]` and canonical `[1, 1]` payloads through
both `DataProcessingService` and `VideoStore`, including radius, table extents,
pocket radius, meter dimensions, explicit boundary handling, unsupported
units, and legacy per-axis extent scales.

Verification: all 23 frontend tests pass. TypeScript reports 79 existing
project-wide errors; none reference the changed frontend source or regression
test files.
