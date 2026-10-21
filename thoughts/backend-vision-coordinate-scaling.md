# Vision configured-geometry scaling

Configured `table.playing_area_corners` and `table.marker_dots` now require
per-point `[scale_x, scale_y]` metadata. Each coordinate is multiplied by that
metadata to reach canonical 4K image pixels, then independently mapped to the
current frame width and height. Vision no longer derives these transforms from
`table.calibration_resolution_width` or `table.calibration_resolution_height`.
Those settings can describe a camera mode that differs from the image in which
a point was recorded.

Ball exclusion, phase table bounds, marker filtering, and marker mask painting
share the conversion helpers in `backend/vision/coordinates.py`. Invalid or
missing scale metadata makes the affected configured geometry unusable; it is
not reconstructed from a guessed resolution. Cue-angle correction uses the
four scaled corners and is skipped when they do not form a finite, convex,
non-degenerate polygon, leaving the detector's angle unchanged.

The configuration currently gives `mask_radius_px` no independent scale
metadata. It is therefore interpreted in the same reference-pixel space as
the associated marker point, and scaled using that marker's `[scale_x,
scale_y]`. This produces an ellipse when source or target scaling is
anisotropic. The normalized-distance ball filter and painted mask use the same
per-axis radii, so their exclusion regions agree at different frame sizes.

Ground-truth JSON fixtures correspond to 3840×2160 PNG images and declare the
top-level envelope `coordinate_space: "4k_canonical"` and
`coordinate_scale: [1.0, 1.0]`. It covers every point, radius, box, boundary,
and extent in that JSON. The detector annotation XML remains in its native
format.

## Verification

`backend/tests/unit/test_vision_configured_geometry_scaling.py` verifies
multiple frame sizes, scale metadata that intentionally disagrees with the old
calibration-derived ratio, anisotropic radius transforms, mask painting,
ball/phase boundary use, cue-angle fallback for invalid corners, and fixture
metadata against image dimensions. All 69 `backend/tests/unit/test_vision*.py`
tests pass, and Ruff checks and formatting pass for the changed Python files.
The new conversion module passes strict mypy; strict mypy over the selected
vision modules follows imports into existing code and reports 407 findings
across 36 files, so that broader check remains a repository-level limitation.
