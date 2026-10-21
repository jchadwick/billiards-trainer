# Backend quality stage: video module

## Changes

- Added strict annotations for the POSIX and file-backed shared-memory handles,
  protocol header decoding, writer statistics, context managers, and the video
  process lifecycle. Python 3.9 grammar compatibility was checked explicitly.
- Narrowed optional camera, calibration, writer, and reader state before use.
  Calibration that reports success without camera parameters now cleanly disables
  distortion correction.
- Fixed a shared-memory lifecycle bug found during round-trip verification:
  `SharedMemoryHeader` retained an exported memory view, which prevented POSIX
  mappings from closing and left segments registered as leaked. Reader detach and
  writer cleanup now release the header view first. Failed attachment to malformed
  segments also releases its mapping.

## Verification

- Scoped strict mypy (`backend/video`, Python 3.11, silent imports): 5 source files,
  no findings.
- Ordinary strict mypy: only the imported finding in
  `backend/common/models.py:443` remains (`Any` returned as `str`); it is outside
  this stage's ownership.
- Ruff, Black, isort, compileall, and Python 3.9 grammar parsing passed.
- Focused pipeline replay, prediction lifecycle, and vision phase integration tests:
  32 passed, 1 optional installed-model replay skipped.
- POSIX and forced file-backed shared-memory frame round-trips passed, including
  metadata, fresh-frame behavior, cleanup, and malformed-header cleanup.

No direct shared-memory tests were present in `backend/tests`; the IPC checks above
were run as hardware-free smoke tests. Adding persistent unit coverage remains a
follow-up for the test-owning stage.
