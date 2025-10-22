## Deprecated Test Cleanup Plan

### Objective
Remove legacy test suites that still depend on modules which were permanently deleted during the recent architecture overhaul. This prevents pytest collection failures while keeping the active codebase green.

### Subagents & Tasks
1. **Backend Test Cleanup Agent**
   - Delete backend-level tests importing legacy modules such as `DirectCameraModule`, `CoordinateSpace`, and legacy API middleware.
   - Targets include:
     - `backend/test_direct_camera.py`
     - `backend/test_vision_integration.py`
     - `backend/tests/core/test_resolution_config.py`
     - `backend/tests/test_multiball_trajectory_simple.py`
     - `backend/tests/unit/test_api.py`
     - `backend/tests/unit/test_config.py`
     - `backend/tests/unit/test_coordinate_converter.py`
     - `backend/tests/unit/test_multiball_trajectory.py`

2. **Auxiliary Test Cleanup Agent**
   - Remove additional tests outside the backend module that still reference the removed subsystems.
   - Targets include:
     - `backend/tests/vision/run_tests.py` dependencies (confirm necessity after backend cleanup)
     - `backend/tests/vision/test_detector_adapter.py` (verify imports)
     - `tests/vision/tracking/test_integration.py`
     - `tools/test_backend_equivalence.py`

### Verification Strategy
- Re-run `pytest` after all targeted deletions to confirm the suite collects and executes without import errors.
- Investigate any new failures and iterate until the suite is clean.

### Notes
- All removed tests touch legacy APIs that no longer exist; keeping them blocks the pipeline.
- No production logic is altered—only the obsolete test artifacts are pruned.
- If future modules reinstate similar functionality, fresh tests should be authored alongside the new implementations.
