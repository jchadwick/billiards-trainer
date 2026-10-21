# Historical Resolution Configuration Summary (Superseded)

> **The original implementation summary is archived and no longer describes the current module.** It claimed that `resolution_config.py` provided table-size registries, coordinate-space classes, a `ResolutionConfig` utility, vector scaling, and a large test suite. Those APIs are not present in the current implementation, and the old file inventory is not current. Do not use the old summary as an API reference.

The current `backend/core/resolution_config.py` exports `StandardResolution`, `get_standard_resolution`, and `validate_point_in_bounds`. It looks up standard image resolutions and checks raw x/y values against rectangular bounds; it does not convert coordinate spaces or represent physical table dimensions.

Use [`RESOLUTION_CONFIG_README.md`](RESOLUTION_CONFIG_README.md) for the helper API, [`COORDINATE_CONVERTER_EXAMPLES.md`](COORDINATE_CONVERTER_EXAMPLES.md) for scale-aware core conversions, and [`docs/COORDINATE_SYSTEMS.md`](../../docs/COORDINATE_SYSTEMS.md) for the repository-wide spatial contract.
