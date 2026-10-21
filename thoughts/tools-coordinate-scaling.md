# Tools Coordinate Scaling Implementation

## Scope

This change makes the developer tools consume explicit image-coordinate metadata
and convert image geometry at the frame boundary. The tools continue using their
existing domain models; this is a focused scaling layer for tool inputs, display,
and benchmark comparison.

## Decisions

- `tools/coordinate_scaling.py` is the shared contract for tool coordinate space,
  source resolution, scale validation, and point/extent/radius/boundary conversion.
  Missing or inconsistent metadata raises an error instead of inferring dimensions
  from coordinate values.
- The video debugger keeps calibrated table geometry in canonical 4K coordinates.
  It reads each corner's own configured scale and converts a display copy to the
  current frame, so rendering cannot mutate the table used for physics.
- Vision detections retain their declared coordinate space and source metadata.
  Ball centers/radii and cue endpoints are converted only when drawing. Cue-to-ball
  matching normalizes both geometries to canonical space and scales its existing
  source-pixel tolerance from the current frame resolution.
- The detector benchmark requires ground-truth coordinate space, source resolution,
  and canonical scale. It converts centers, optional rectangle edges, and scalar
  radii to the decoded video's actual frame dimensions. Benchmark overlays use the
  same conversion path for both detector outputs.
- OpenCV radius limits and ROI margins come from a config with explicit source
  resolution and scale. The config keeps pixel extents in the detector's nested
  `size_constraints` and `performance` sections; intensity and ratio thresholds
  remain dimensionless/image-value settings.

## Verification

Run the deterministic tools checks with:

```sh
.venv/bin/pytest tools/test_coordinate_scaling.py -q --no-cov
```

The tests cover scaling of centers, extents, radii, and rectangle boundaries;
missing and inconsistent source metadata; configuration shape and threshold
conversion; per-corner configured scale preservation; non-4K debugger rendering
and cue matching; and benchmark overlay normalization across canonical and pixel
detector outputs.

Formatting, lint, syntax, and example JSON were checked with Black, Ruff,
`py_compile`, and the standard JSON parser for both checked-in fixtures.
