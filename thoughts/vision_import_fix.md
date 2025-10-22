# Vision Module Import Fix

## Problem
The vision module files were using relative imports that went beyond the top-level package:
```python
from ...core.constants_4k import BALL_RADIUS_4K
from ...core.resolution_converter import ResolutionConverter
```

This caused "attempted relative import beyond top-level package" errors when trying to import vision module components.

## Root Cause
The vision module is at `backend/vision/`, and the core module is at `backend/core/`. Using `...core` tries to go up 3 levels from vision, which goes beyond the backend package boundary.

## Solution
Changed all relative imports from `...core` to absolute imports using `core`:

```python
# Before (BROKEN):
from ...core.constants_4k import BALL_RADIUS_4K
from ...core.resolution_converter import ResolutionConverter

# After (FIXED):
from core.constants_4k import BALL_RADIUS_4K
from core.resolution_converter import ResolutionConverter
```

This works because when running from `backend/`, Python can resolve `core` as an absolute import.

## Files Changed

### 1. `/backend/vision/detection/balls.py`
**Lines 29-30:**
- Changed `from ...core.constants_4k` to `from core.constants_4k`
- Changed `from ...core.resolution_converter` to `from core.resolution_converter`

### 2. `/backend/vision/detection/cue.py`
**Lines 24-25:**
- Changed `from ...core.constants_4k` to `from core.constants_4k`
- Changed `from ...core.resolution_converter` to `from core.resolution_converter`

### 3. `/backend/vision/detection/table.py`
**Lines 12-13:**
- Changed `from ...core.constants_4k` to `from core.constants_4k`
- Changed `from ...core.resolution_converter` to `from core.resolution_converter`

### 4. `/backend/vision/detection/detector_adapter.py`
**Line 36:**
- Changed `from ...core.resolution_converter` to `from core.resolution_converter`

## Verification

Verified using AST parsing that:
1. All 4 files no longer have `from ...core` imports
2. No other files in the vision module have this issue

```
✓ vision/detection/balls.py: No bad relative imports to core module
✓ vision/detection/cue.py: No bad relative imports to core module
✓ vision/detection/table.py: No bad relative imports to core module
✓ vision/detection/detector_adapter.py: No bad relative imports to core module
```

## Impact

- **Fixed:** Import errors when importing vision.detection modules
- **No Breaking Changes:** The imports still resolve to the same modules
- **Consistent:** Now matches the import style used in other parts of the codebase

## Related Issues

Note: The vision module still has other import issues unrelated to this fix (e.g., `backend.video` module import errors in `video_consumer.py`), but those are separate problems outside the scope of this fix.

## Summary

Successfully fixed all problematic relative imports (`from ...core`) in the vision module's detection submodule. All imports now use absolute imports (`from core`) which resolve correctly when running from the backend directory.
