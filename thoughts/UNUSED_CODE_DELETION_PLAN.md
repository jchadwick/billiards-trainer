# Unused Code Deletion Plan for Backend

**Generated**: 2025-10-29
**Analysis Method**: 5 parallel subagents analyzing codebase execution paths

---

## Executive Summary

`make run` executes two processes:
1. **Video Module**: `python -m backend.video` (background process)
2. **API Server**: `uvicorn backend.api.main:app` (foreground process)

Analysis identified **60+ files** (approximately 30-40% of backend codebase) that are NOT executed during `make run`. These fall into several categories:

- ✅ **Safe to delete**: Example scripts, deprecated code, duplicate entry points
- ⚠️ **Manual tools**: Development/debug tools (keep for dev workflows)
- ⚠️ **Test files**: Keep for CI/CD
- ⚠️ **Documentation**: Keep for reference

---

## What IS Executed by `make run`

### Video Module Execution Path
Starting from: `python -m backend.video`

**Core Files (15):**
- `backend/video/__main__.py` - Entry point
- `backend/video/process.py` - Main orchestrator
- `backend/video/ipc/shared_memory.py` - Triple-buffered IPC
- `backend/vision/capture.py` - Camera abstraction
- `backend/vision/kinect2_capture.py` - Kinect v2 support (optional)
- `backend/vision/calibration/camera.py` - Camera calibration
- `backend/config.py` - Configuration system
- `backend/vision/__init__.py` - Vision package init
- `backend/vision/calibration/__init__.py` - Calibration package init
- `backend/video/__init__.py` - Video package init
- `backend/video/ipc/__init__.py` - IPC package init
- `backend/__init__.py` - Backend package init

**Configuration:**
- `config.json` - Main configuration

**Dependencies:**
- Standard library, numpy, cv2, optional pylibfreenect2

---

### API Server Execution Path
Starting from: `uvicorn backend.api.main:app`

**API Module (40+ files):**
- `backend/api/main.py` - FastAPI app creation
- `backend/api/dependencies.py` - Dependency injection
- `backend/api/shutdown.py` - Graceful shutdown
- `backend/api/middleware/*.py` (5 files) - Error handling, logging, tracing, performance, metrics
- `backend/api/models/*.py` (10 files) - Pydantic models and converters
- `backend/api/routes/*.py` (12 files) - All API endpoints
- `backend/api/websocket/*.py` (7 files) - WebSocket system

**Core Module (50+ files):**
- `backend/core/__init__.py` - Core orchestration
- `backend/core/coordinates.py` - Vector2D coordinate system
- `backend/core/constants_4k.py` - 4K resolution constants
- `backend/core/game_state.py` - Game state management
- `backend/core/models.py` - Core data models
- `backend/core/resolution_converter.py` - Resolution conversion
- `backend/core/rules.py` - Game rules
- `backend/core/analysis/*.py` (3 files) - Assistance, prediction, shot analysis
- `backend/core/collision/geometric_collision.py` - Geometric collision
- `backend/core/events/*.py` (2 files) - Event system
- `backend/core/physics/*.py` (4 files) - Engine, collision, spin, trajectory
- `backend/core/utils/*.py` (3 files) - Cache, geometry, math utilities
- `backend/core/validation/*.py` (4 files) - Physics, state, table validation

**Integration Service (2 files):**
- `backend/integration_service.py` - Module coordinator
- `backend/integration_service_conversion_helpers.py` - Data conversion

**Vision Module (30+ files):**
- `backend/vision/__init__.py` - Vision module main
- `backend/vision/models.py` - Vision data models
- `backend/vision/performance_profiler.py` - Performance monitoring
- `backend/vision/calibration/*.py` (5 files) - Camera, color, geometry calibration
- `backend/vision/detection/*.py` (5 files) - YOLO, balls, hybrid validator, detector adapter
- `backend/vision/stream/video_consumer.py` - Shared memory consumer
- `backend/vision/tracking/*.py` (2 files) - Kalman filter, object tracker

**Streaming Module (2 files):**
- `backend/streaming/__init__.py` - Package init
- `backend/streaming/enhanced_camera_module.py` - Camera with preprocessing (used by stream.py)

**System Module (2 files):**
- `backend/system/__init__.py` - Package init
- `backend/system/health_monitor.py` - Health monitoring

**Utils (2 files):**
- `backend/utils/__init__.py` - Package init
- `backend/utils/logging.py` - Logging configuration

**Total Production Files Executed: ~150 files**

---

## What is NOT Executed by `make run`

### Category 1: Alternative Entry Points (SAFE TO DELETE)

#### `backend/main.py` (89 lines)
- **Purpose**: Alternative unified entry point (combines video + API)
- **Why Not Used**: `make run` launches video and API separately
- **Imports**: `backend.api.main`, `backend.config`
- **Recommendation**: ⚠️ **KEEP** - Useful alternative for single-process deployment
- **Notes**: Configures VAAPI GPU acceleration, could be valuable for Docker/production

---

### Category 2: Development Server (SAFE TO DELETE if dev_server unused)

#### `backend/dev_server.py`
- **Purpose**: Simplified development server
- **Why Not Used**: `make run` uses full production API
- **Recommendation**: ⚠️ **EVALUATE** - Only used by `make run-dev` (separate target)
- **Decision**: Keep if dev workflow uses `make run-dev`, otherwise delete

---

### Category 3: Example/Demo Scripts (SAFE TO DELETE)

#### `backend/projector_example.py`
- **Purpose**: Example usage of projector integration
- **Status**: References removed projector module (likely deprecated)
- **Recommendation**: ✅ **DELETE** - Appears to be broken/outdated
- **Imports**: Would fail if projector module removed

#### `backend/core/resolution_config_examples.py`
- **Purpose**: 8 examples showing resolution_config usage
- **Recommendation**: ⚠️ **MOVE TO DOCS** or ✅ **DELETE**
- **Alternative**: Convert to unit tests or inline documentation

#### `backend/core/utils/example_cue_pointing.py`
- **Purpose**: 5 examples of find_ball_cue_is_pointing_at
- **Recommendation**: ⚠️ **MOVE TO DOCS** or ✅ **DELETE**
- **Alternative**: Convert to unit tests or inline documentation

#### `backend/vision/config_manager_video_example.py`
- **Purpose**: VisionConfigurationManager examples
- **Recommendation**: ⚠️ **MOVE TO DOCS** or ✅ **DELETE**
- **Alternative**: Convert to unit tests

#### `backend/vision/detection/demo_hybrid_validator.py`
- **Purpose**: HybridValidator demonstration
- **Recommendation**: ⚠️ **MOVE TO DOCS** or ✅ **DELETE**
- **Alternative**: Convert to unit tests

#### `backend/vision/detection/test_tpu_detector.py`
- **Purpose**: TPU detector testing
- **Recommendation**: ⚠️ **MOVE TO tests/** or ✅ **DELETE**
- **Alternative**: Convert to pytest test

---

### Category 4: Verification Scripts (SAFE TO DELETE after migration complete)

#### `backend/core/verify_group1.py`
- **Purpose**: Verify 4K migration Group 1
- **Status**: Part of 4K migration verification
- **Recommendation**: ✅ **DELETE** once migration verified complete
- **Alternative**: Convert findings to unit tests

#### `backend/core/collision/verify_group5.py`
- **Purpose**: Verify 4K migration Group 5
- **Status**: Part of 4K migration verification
- **Recommendation**: ✅ **DELETE** once migration verified complete
- **Alternative**: Convert findings to unit tests

---

### Category 5: Manual Test Scripts (EVALUATE)

#### Root-Level Test Scripts (5 files)

**`backend/test_api.py`**
- **Purpose**: Basic API connectivity test
- **Recommendation**: ⚠️ **CONVERT TO PYTEST** or ✅ **DELETE**
- **Alternative**: Move to `tests/integration/test_api_connectivity.py`

**`backend/test_api_endpoints.py`**
- **Purpose**: HTTP endpoint integration tests
- **Recommendation**: ⚠️ **CONVERT TO PYTEST** or ✅ **DELETE**
- **Alternative**: Move to `tests/integration/test_api_endpoints.py`

**`backend/test_detection.py`**
- **Purpose**: Ball detection with live camera
- **Recommendation**: ⚠️ **KEEP FOR HARDWARE TESTING** or convert to pytest with hardware marker
- **Notes**: Requires camera hardware

**`backend/test_detection_simple.py`**
- **Purpose**: Simplified detection test
- **Recommendation**: ⚠️ **CONVERT TO PYTEST** or ✅ **DELETE**
- **Alternative**: Move to `tests/integration/test_detection_simple.py`

**`backend/test_4k_functionality.py`**
- **Purpose**: 4K coordinate system tests
- **Recommendation**: ⚠️ **CONVERT TO PYTEST** or ✅ **DELETE**
- **Alternative**: Move to `tests/unit/test_4k_coordinates.py`

---

### Category 6: Development/Debug Tools (KEEP)

These are valuable for development workflows:

#### `backend/tools/performance_diagnostics.py`
- **Purpose**: Vision pipeline performance analysis
- **Recommendation**: ✅ **KEEP** - Actively used for optimization

#### `backend/tools/quick_perf_check.py`
- **Purpose**: Quick performance verification
- **Recommendation**: ✅ **KEEP** - Useful dev tool

#### `backend/tools/watch_performance.py`
- **Purpose**: Real-time performance monitoring dashboard
- **Recommendation**: ✅ **KEEP** - Active monitoring tool

---

### Category 7: Calibration Tools (KEEP)

These are run manually during setup:

#### `backend/vision/calibrate_from_grid.py`
- **Purpose**: Camera distortion calibration from grid
- **Recommendation**: ✅ **KEEP** - Essential setup tool

#### `backend/vision/calibrate_distortion_from_balls.py`
- **Purpose**: Distortion calibration from ball grid
- **Recommendation**: ✅ **KEEP** - Essential setup tool

---

### Category 8: Test Files (KEEP for CI/CD)

All test files should be kept:
- `backend/tests/core/*.py` (5 physics tests)
- `backend/tests/unit/*.py` (6 unit tests)
- Test configuration and fixtures

**Total**: 17 test files - All testing active production code

---

### Category 9: Verified Unused Modules (SAFE TO DELETE)

#### `backend/system/utils.py` ✅ CONFIRMED UNUSED
- **Status**: Empty stub class (only contains `class SystemUtils: pass`)
- **Usage**: Exported by `system/__init__.py` but never instantiated
- **Verification**: No usage of `SystemUtils()` or `SystemUtils.` found in codebase
- **Recommendation**: ✅ **DELETE** - Remove file and update `system/__init__.py`
- **Impact**: None - dead code

---

### Category 10: Verified Active Modules (DO NOT DELETE)

These modules were initially suspected as unused but are verified as ACTIVE:

#### `backend/core/validation.py` ❌ DO NOT DELETE
- **Status**: ACTIVE - Stub providing `ValidationResult` class
- **Used by**:
  - `backend/core/game_state.py`
  - `backend/vision/calibration/__init__.py`
  - `backend/vision/calibration/manager.py`
- **Recommendation**: ✅ **KEEP** - Active dependency

#### `backend/core/integration.py` ❌ DO NOT DELETE
- **Status**: ACTIVE - Integration interfaces for module communication
- **Used by**: `backend/vision/tracking/__init__.py`
- **Recommendation**: ✅ **KEEP** - Active dependency

#### `backend/system/monitoring.py` ❌ DO NOT DELETE
- **Status**: ACTIVE - SystemMetrics dataclass heavily used
- **Used by**:
  - `backend/api/middleware/performance.py`
  - `backend/api/routes/health.py`
  - `backend/api/routes/diagnostics.py`
  - `backend/api/models/responses.py`
  - `backend/api/websocket/endpoints.py`
- **Recommendation**: ✅ **KEEP** - Critical for monitoring

---

## Deletion Plan Summary

### Phase 1: Safe Immediate Deletions

```bash
# Delete broken/deprecated example
rm backend/projector_example.py

# Delete empty stub module
rm backend/system/utils.py
# Update system/__init__.py to remove SystemUtils import and export

# Delete verification scripts (after confirming 4K migration complete)
rm backend/core/verify_group1.py
rm backend/core/collision/verify_group5.py
```

### Phase 2: Convert Examples to Documentation

```bash
# Move to examples directory or convert to inline docs
mkdir -p backend/examples
mv backend/core/resolution_config_examples.py backend/examples/
mv backend/core/utils/example_cue_pointing.py backend/examples/
mv backend/vision/config_manager_video_example.py backend/examples/
mv backend/vision/detection/demo_hybrid_validator.py backend/examples/
```

### Phase 3: Convert Manual Tests to Pytest

```bash
# Convert to proper pytest tests in tests/ directory
# Then delete originals:
rm backend/test_api.py
rm backend/test_api_endpoints.py
rm backend/test_detection_simple.py
rm backend/test_4k_functionality.py
# Keep test_detection.py for hardware testing
```

### Phase 4: ~~Verify and Delete Unused Modules~~ VERIFICATION COMPLETE

✅ **Verification Complete** - All suspected modules have been verified:
- ✅ `backend/system/utils.py` - CONFIRMED UNUSED → Delete in Phase 1
- ❌ `backend/core/validation.py` - ACTIVE → Keep
- ❌ `backend/core/integration.py` - ACTIVE → Keep
- ❌ `backend/system/monitoring.py` - ACTIVE → Keep

No additional deletions needed in this phase.

### Phase 5: Evaluate Alternative Entry Points

Decision needed:
- **Keep** `backend/main.py` if used for single-process deployment
- **Keep** `backend/dev_server.py` if `make run-dev` is used
- **Delete** if not needed

---

## Files to Keep (Not for Deletion)

### Essential Production Code
- All files in execution path (~150 files) ✅
- `backend/streaming/enhanced_camera_module.py` ✅ (used by stream.py)

### Development Tools
- `backend/tools/*.py` (3 files) ✅
- `backend/vision/calibrate_*.py` (2 files) ✅

### Tests
- `backend/tests/**/*.py` (17 files) ✅
- `backend/test_detection.py` ✅ (hardware testing)

### Documentation
- All `*.md` files (38 files) ✅
- `README.md`, `SPECS.md`, etc.

---

## Impact Analysis

### Estimated Deletions

| Category | Files | Lines | Risk Level |
|----------|-------|-------|------------|
| Example Scripts | 6 | ~800 | ✅ Low risk |
| Verification Scripts | 2 | ~200 | ✅ Low risk |
| Manual Tests | 4 | ~600 | ⚠️ Medium risk (convert first) |
| Empty Stub Modules | 1 | ~12 | ✅ Low risk (verified unused) |
| **Total Safe Deletions** | **13** | **~1612** | **✅ Low-Medium risk** |
| **Conversion/Relocation** | **4** | **~600** | **⚠️ Medium risk** |

### Potential Space Savings
- **Source code**: ~2000 lines
- **Test files**: Keep all (needed for CI/CD)
- **Documentation**: Keep all (reference material)

### Risks
- ❌ **High Risk**: Deleting unverified modules (Phase 4)
- ⚠️ **Medium Risk**: Deleting manual tests before conversion (Phase 3)
- ✅ **Low Risk**: Deleting examples and verification scripts (Phases 1-2)

---

## Recommended Execution Order

1. ✅ **Immediate**: Delete `projector_example.py` (clearly broken)
2. ✅ **Short-term**: Move examples to `backend/examples/` or docs (Phase 2)
3. ⚠️ **Medium-term**: Convert manual tests to pytest (Phase 3)
4. ⚠️ **Before deletion**: Verify unused modules with grep/usage analysis (Phase 4)
5. ⚠️ **Final decision**: Keep or delete alternative entry points (Phase 5)

---

## Additional Configuration Cleanup

### Unused Config Files to Evaluate

After code deletion, check for orphaned config:
- YAML files in `backend/vision/models/training_runs/*/args.yaml` (18 files)
- Check if all training runs are still referenced

---

## Next Steps

1. Review this plan with the team
2. Run comprehensive tests before any deletions
3. Create feature branch for cleanup work
4. Execute phases 1-2 immediately (low risk)
5. Create conversion tasks for phase 3 (manual tests → pytest)
6. Conduct thorough grep analysis for phase 4 modules
7. Make architectural decision on phase 5 entry points
8. Run full test suite after each phase
9. Update documentation to reflect changes
10. Consider archiving deleted code in separate branch before deletion

---

## Verification Commands

Before deleting any file, verify it's not imported:

```bash
# Check if file is imported anywhere
FILE="backend/projector_example.py"
grep -r "from ${FILE%.py}" backend/ || echo "Not imported"
grep -r "import ${FILE%.py}" backend/ || echo "Not imported"

# Check if module is used
MODULE="backend.core.validation"
grep -r "from $MODULE" backend/ || echo "Not used"
grep -r "import $MODULE" backend/ || echo "Not used"
```

---

## Conclusion

Approximately **16 files** can be deleted or relocated, representing ~2000 lines of code. However, this should be done carefully with proper verification and testing at each phase. The majority of the "unused" code is actually valuable (tests, tools, documentation) and should be kept or reorganized rather than deleted.

**Recommended approach**: Focus on Phases 1-2 (low-risk deletions and reorganization) and defer high-risk deletions until thorough verification is complete.
