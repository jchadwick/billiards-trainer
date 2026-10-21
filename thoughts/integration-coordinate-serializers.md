# Integration coordinate boundary

## Contract decisions

- Vision geometry is converted once at the Vision-to-Core boundary. Pixel data uses each object's `source_resolution`; the exact `DetectionResult.frame_size` is a fallback only when the object declares pixel space but omits its own resolution. There is no configured camera-resolution guess in this conversion path.
- Pixel and `4k_canonical` are the only accepted image-coordinate spaces. World-space and unknown/missing coordinate-space values raise `CoordinateConversionError`; `_process_detection` records and skips the malformed frame without stopping the integration loop.
- Core continues receiving canonical numeric x/y values, with mandatory `scale: [1, 1]` on every point. Pixel radii, cue/table extents, and pocket radii are converted to canonical values and carry identity scale metadata; two-axis radius extents preserve the result when source-to-4K scaling is anisotropic. World-meter table dimensions remain separate and explicitly labeled in meters.
- Cue angle and length use the coordinate transform along the cue direction. Cue motion, shaft, contact, and aiming geometry are mapped with the same declared source transform. Ball position, velocity, acceleration, predicted position, and history use the same single transform.
- Trajectory line endpoints and collision positions are serialized as canonical `{x, y, scale}` objects. Legacy Lua visualizer consumers still read collision-root `x` and `y`, so those numeric fields remain with the same root-level scale metadata as the nested `position` point.

## Verification

- `backend/tests/unit/test_integration_coordinate_conversion.py` compares 1080p and 4K ball, cue, table, rail, playing-area, and pocket geometry; it also covers exact frame-resolution fallback, missing/unsupported coordinate spaces, world-space rejection, and trajectory-vector canonicalization.
- `backend/tests/integration/test_pipeline_replay.py::test_recorded_frame_replay_to_real_websocket` sends converted state and trajectory through a real local WebSocket. It asserts scale metadata on ball, cue, table, trajectory points, nested collision positions, and legacy collision-root coordinates.
- Focused replay, conversion, and prediction-lifecycle tests pass: **29 passed, 1 skipped** (`--no-cov`). The skipped test requires installed model assets.
- Ruff, Black, and isort pass on the integration source and touched tests. Strict mypy passes for both integration source modules (`--strict --python-version 3.11 --follow-imports=silent`). The repository's existing mypy configuration still warns that its configured Python 3.9 target is unsupported by the pinned mypy; the command overrides it to 3.11.

## Sequenced remaining work

- The replay previously exposed a broadcaster validation gap for scaled collision positions; the API/WebSocket task updated the validator. The required real-socket replay now passes with both nested and legacy collision-root points carrying scale metadata.
- Core parsing currently consumes canonical x/y values and ignores scale metadata, table rails, and the separate playing-area field. The sequenced Core parser/serializer task should validate scale metadata and preserve scaled points through state/event serialization.
- Frontend scale handling remains a contract gap: the API contract defines `scale` as per-axis multipliers, while `frontend/web/src/services/data-handlers.ts` documents it as coordinate-space dimensions and later ignores the metadata; `frontend/web/src/stores/VideoStore.ts` also copies trajectory x/y into app models without scale. Current integration points are canonical with `[1,1]`, so replay values match, but non-identity transport scales would be misinterpreted or lost. The Lua trajectory visualizers read legacy scalar collision-root x/y fields and currently ignore scale; integration preserves these with explicit root-level metadata for the sequenced frontend task.
