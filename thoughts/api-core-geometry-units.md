# API/Core geometry unit contract

API game-state transports now match the Core physics contract: ball radii,
table extents, and pocket radii are explicit pixel measurements with positive
scale pairs and the `canonical_pixels` unit. The physical table footprint is
carried separately as an optional `{width, height, unit: "meters"}` object.

Core serializes normalized geometry with unit scale `[1, 1]`. API conversion
preserves those fields and passes inbound scale metadata to Core exactly once,
where Core normalizes dimensions and radii. Round-trip tests compare API
metadata against Core serialization and verify physical dimensions survive.

Verification: 148 focused converter and coordinate-model tests passed with
`--no-cov`; Ruff, Black, and isort passed. Strict mypy passed for the changed
model and converter modules with Python 3.11 after annotating the shared unit
constants as their corresponding `Literal` types. The configured mypy target
still emits its existing Python 3.9 unsupported-version notice before applying
the explicit Python 3.11 override.
