# Backend quality: core and vision stages

This stage removes strict typing and style findings in `backend/core/` and
`backend/vision/` without broad ignores or changing the configured check scope.
It also repairs several stale CoreModule calls found while validating types.

Core now returns its actual shot-analysis contract, builds suggestions from the
assistance engine's target recommendations, and creates outcome predictions
through the existing trajectory calculator. Assistance derives the richer
metrics it consumes from the basic shot analyzer and calibrated geometry.
Missing-spin copies preserve `None`, and cushion detection safely accepts a
missing table. Vision's package exports no longer reference missing modules;
those names had no repository callers. Python 3.9 remains the declared minimum,
and postponed annotations retain compatible syntax where Ruff modernized types.

Verification:

- Core-focused suite plus replay: 385 passed, 1 optional model-assets replay
  skipped.
- Core strict mypy: 32 source files passed.
- Vision focused suite: 69 passed; calibration and detection suites were also
  run by their scoped workers.
- Vision strict mypy: 29 source files passed with `--follow-imports=silent` to
  isolate this module; full-backend mypy remains the authoritative unsilenced
  repository check.
- Ruff, Black, and isort pass for both complete packages.
- Full hardware-free backend suite before the new Core facade contract tests:
  1,125 passed.

The unsilenced repository-wide strict mypy run still reports work in tools,
examples, streaming, system, video, development-server, utilities, and common
modules. Ruff and formatting are clean repository-wide except for the known
vision import-order fixes, which were completed in this stage.

## Tools stage

The four `backend/tools/` sources now pass Ruff, Black, isort, and scoped strict
mypy. The diagnostics script had duplicated and monkey-patched the vision
pipeline using detector/config attributes removed by the current architecture;
it now uses the production `VisionModule` profiler instead. Benchmark frame
inputs are normalized to the detector's uint8 contract, and optional detector
and tracker components are checked before use. PEP 604 annotations in tools are
postponed so the package's declared Python 3.9 syntax floor remains supported.

Verification: scoped strict mypy passed 4 files; the benchmark and coordinate
tool tests passed (7 total); Python compilation and CLI help paths passed. Full
camera/model/video diagnostics remain hardware and asset dependent.
