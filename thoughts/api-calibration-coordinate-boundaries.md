# API calibration coordinate boundaries

Table corners and marker dots cross the configuration API as source-image points
with required positive, finite `scale` metadata. When the calibration frame
resolution is supplied, the route verifies both dimensions and confirms that
the scale maps that frame to canonical 3840×2160 pixels before any config write.
The read endpoint returns the saved scale and frame resolution after reload.

Manual calibration routes now use the shared request models: image positions
carry scale metadata and table-plane positions declare meters. The session keeps
the source point and scale for transport while homography and validation use
canonical image pixels and meter-valued world coordinates. Current-calibration
and fisheye calibration responses likewise return scale-aware table corners.

Focused verification: config and calibration route tests pass with `--no-cov`;
Ruff, Black, isort, and strict mypy pass on the changed route and test modules.
