# Historical Group 6: Analysis and Prediction 4K Migration (Superseded)

> **Historical record only.** This file captured an earlier migration pass and its assumptions. It is not evidence that every current analysis/configuration value has been calibrated, nor does it define current APIs, defaults, or compatibility guarantees. The authoritative coordinate contract is [`docs/COORDINATE_SYSTEMS.md`](../../../docs/COORDINATE_SYSTEMS.md).

The earlier work recorded converting several analysis thresholds and examples from a 1920 × 1080 assumption to the 3840 × 2160 canonical image plane. It also recorded using scale-aware vectors and keeping time, probabilities, mass, and other non-spatial values in their own units. Those notes describe the intent of that pass; current behavior must be checked against source and configuration.

Current core image geometry uses canonical 3840 × 2160 pixels. `Vector2D` values require scale metadata, and core physics/game-state positions, table pixel extents, bounds, and radii normalize to that image plane. Physical dimensions in meters remain separate and connect to image geometry through calibration. See [`backend/core/COORDINATE_CONVERTER_EXAMPLES.md`](../COORDINATE_CONVERTER_EXAMPLES.md) for current model examples.
