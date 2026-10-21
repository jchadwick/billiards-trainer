# API table-boundary round-trip

The HTTP `TableInfo` and WebSocket `TableStateData` models now carry optional
`playing_area_corners` points using the same required `PositionWithScale`
contract as other image coordinates. When supplied, each boundary must contain
exactly four points; when omitted or `null`, an uncalibrated table stays
uncalibrated.

Both Core-to-transport converters emit the canonicalized Core corners with
their scale metadata, and both reverse converters restore them to `TableState`.
The Core model normalizes a source-resolution scale before API serialization,
so source points at scale `[2, 2]` and already-canonical points at `[1, 1]`
round-trip to identical canonical coordinates.

Verification: `backend/tests/unit/api/test_converters.py` and
`backend/tests/unit/api/test_coordinate_contract_models.py` pass (151 tests,
`--no-cov`). Ruff, Black, isort, and strict mypy for the changed model and
converter modules pass; mypy was invoked with Python 3.11 because the installed
version rejects the repository's configured Python 3.9 target.
