# Backend example coordinate scaling

Backend examples now use the coordinate contracts supported by the runtime:

- `resolution_config_examples.py` uses real resolution helpers, scales source
  points, image bounds, extents, and radii to canonical 4K pixels, and keeps
  measured table dimensions as explicit SI fields on `TableState`.
- `example_cue_pointing.py` constructs detections from a declared 1080p source
  resolution or from canonical coordinates. It scales pixel tolerances before
  calling the low-level geometry helper, whose bare tuple contract is canonical
  4K pixels.
- `demo_hybrid_validator.py` now demonstrates the existing
  `BallPositionRefiner`; the former `HybridValidator`/`ValidationConfig` import
  referenced APIs that do not exist. Its synthetic detections record their frame
  resolution before conversion to canonical space.
- Hardware-free smoke tests import every example module, run the coordinate
  examples, check scaled geometry and physical units, and detect direct
  `Vector2D` constructions without scale metadata.

The video debugger and benchmark examples are outside this module's ownership;
their source geometry is handled in the separate tools coordinate stage.
