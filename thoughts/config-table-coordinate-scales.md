# Persisted table geometry scale contract

Table calibration points in `config.json` retain their raw `x` and `y` values in
`calibration_resolution_width` by `calibration_resolution_height` source space.
Every playing-area corner and marker dot now also requires a two-element
`scale` array. Each factor multiplies the matching raw coordinate to produce
canonical 3840 by 2160 image coordinates. The current calibration dimensions are
640 by 360, so the migrated factors are `[6.0, 6.0]`.

The bundled schema defaults and active project configuration carry the same
scale metadata. Configuration validation and the generated JSON Schema reject
missing or unknown point keys, scales of the wrong length, non-numeric or
boolean factors, non-finite factors, and factors less than or equal to zero.
Calibration resolution fields remain available to consumers that need to
compare the original raw coordinates to a current frame size.

Focused verification: `backend/tests/unit/test_config_validation.py` and
`backend/tests/unit/test_config_compatibility.py` pass with `--no-cov`; Ruff,
Black, isort, JSON parsing, and strict mypy for `backend/config.py` and the
updated validation test pass. Mypy prints the repository's existing warning
that the configured Python 3.9 target is unsupported by the installed checker;
the explicit Python 3.11 check passes.
