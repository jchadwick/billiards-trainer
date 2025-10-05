# Build Failure Quick Reference

**Run ID:** 18117672492 | **Status:** 🔴 0/7 Jobs Passing | **Analyzed:** 2025-09-29 23:26 EDT

## TL;DR - Fast Fix

```bash
cd backend
./fix_build_failures.sh  # Automated fix for 6/7 issues
# Then manually fix CameraFrame in tests/unit/test_vision.py
pytest tests/unit/ -v
git add tests/ system/
git commit -m "fix: resolve import errors and formatting from CI build 18117672492"
git push
```

---

## Critical Issues (Must Fix First)

| # | Issue | File:Line | Fix | Priority |
|---|-------|-----------|-----|----------|
| 1 | `api.middleware.auth` not found | `tests/unit/test_api.py:9` | Change to `authentication` | P0 |
| 2 | `PersistenceManager` → `ConfigPersistence` | `tests/unit/test_config.py:10` | Rename class | P0 |
| 3 | `ShotAssistant` → `AssistanceEngine` | `tests/unit/test_core.py:6` | Rename class | P0 |
| 4 | `projector.calibration.color` missing | `tests/unit/test_projector.py:8` | Use `vision.calibration.color` | P0 |
| 5 | `CameraFrame` not in `vision.models` | `tests/unit/test_vision.py:13` | **MANUAL FIX NEEDED** | P0 |
| 6 | `Ball` → `BallState` | Integration tests | Rename class | P0 |
| 7 | `BallTracker` → `ObjectTracker` | Performance tests | Rename class | P0 |
| 8 | Black formatting | `system/orchestrator.py:764` | Run `black` | P1 |

---

## Error Categories

### 🔴 Import Errors (Blocking All Tests)
- **Count:** 7 errors across 9 test files
- **Impact:** Zero tests can execute
- **Root Cause:** Class/module renames without test updates

### 🟡 Code Quality (Blocking Merge)
- **Count:** 1 formatting violation
- **Impact:** CI blocks merge
- **Root Cause:** Line too long for Black formatter

### 🟠 Runtime Errors (System Tests Only)
- **Count:** 7 setup errors + 5 test failures
- **Impact:** System tests fail
- **Root Cause:** Missing method implementations

---

## Manual Fix Required: CameraFrame

**Location:** `tests/unit/test_vision.py:13`

**Current:**
```python
from vision.models import CameraFrame  # ❌ Does not exist
```

**Options:**

### Option A: Use FrameStatistics (Quick)
```python
from vision.models import FrameStatistics
# Update all test references from CameraFrame to FrameStatistics
```

### Option B: Implement CameraFrame (Complete)
Add to `vision/models.py`:
```python
class CameraFrame:
    """Represents a frame captured from the camera."""
    def __init__(self, image: NDArray, timestamp: float, frame_id: int):
        self.image = image
        self.timestamp = timestamp
        self.frame_id = frame_id
        self.statistics: Optional[FrameStatistics] = None
```

**Recommendation:** Start with Option A to unblock CI, then implement Option B properly.

---

## Test Results After Fix

### Expected (Phase 1 Complete)
```bash
$ pytest tests/unit/ --collect-only
collected 50 items  # ✅ No import errors
```

### Expected (Phase 2 Complete)
```bash
$ black --check .
All done! ✨ 🍰 ✨
```

### Expected (Full Fix)
```bash
$ pytest tests/
Unit: ✅ 45 passed
Integration: ✅ 12 passed
Performance: ✅ 8 passed
System: ⚠️ Some may still fail (separate issues)
```

---

## Rollback

If fixes cause problems:
```bash
cd backend
# Backups are in: backup_YYYYMMDD_HHMMSS/
cp -r backup_*/tests/ .
cp backup_*/orchestrator.py system/
```

---

## CI Job Status Reference

| Job | Before | After Fix | Duration |
|-----|--------|-----------|----------|
| Unit Tests | ❌ Import errors | ✅ Should pass | 1m45s |
| Code Quality | ❌ Black format | ✅ Should pass | 51s |
| Performance Tests | ❌ Import error | ✅ Should pass | 1m9s |
| Integration Tests | ❌ Import errors | ✅ Should pass | 1m15s |
| System Tests | ❌ Multiple errors | ⚠️ Needs Phase 3 | 1m4s |
| Performance Benchmark | ❌ Import error | ✅ Should pass | 43s |
| Test Matrix | ❌ Import errors | ✅ Should pass | 1m17s |

---

## Class Name Changes Reference

| Test Expects | Actual Code | Location |
|--------------|-------------|----------|
| `PersistenceManager` | `ConfigPersistence` | `config/storage/persistence.py:35` |
| `ShotAssistant` | `AssistanceEngine` | `core/analysis/assistance.py:105` |
| `ColorCalibrator` from `projector.calibration.color` | `ColorCalibrator` from `vision.calibration.color` | `vision/calibration/color.py` |
| `CameraFrame` | `FrameStatistics` | `vision/models.py:196` |
| `Ball` | `BallState` | `core/models.py:130` |
| `BallTracker` | `ObjectTracker` | `vision/tracking/tracker.py:170` |

---

## Full Documentation

For detailed analysis, see: [BUILD_FAILURE_ANALYSIS.md](./BUILD_FAILURE_ANALYSIS.md)

For automation, run: `./fix_build_failures.sh`
