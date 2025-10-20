# Legacy Code Removal - Shared Memory IPC Only

## Summary

Removed all dual-mode logic and legacy camera code paths. The system now **ONLY** uses shared memory IPC for video frame access. The feature flag approach was removed in favor of a single, clean implementation.

## Changes Made

### 1. VisionModule (backend/vision/__init__.py)

**Removed:**
- `_use_shared_memory` field and all checks for it
- CameraCapture import from capture module
- All dual-mode initialization logic
- Legacy camera mode code paths in `_initialize_components()`
- Dual-mode logic in `start_capture()`
- Dual-mode logic in `stop_capture()`
- Dual-mode logic in `_capture_loop()`
- Feature flag checks and branching
- `camera_connected` from statistics
- `video_mode` from statistics (was "shared_memory" or "camera")

**Simplified:**
- Now ONLY creates VideoConsumer in `_initialize_components()`
- `start_capture()` ONLY uses `VideoConsumer.start()`
- `stop_capture()` ONLY uses `VideoConsumer.stop()`
- `_capture_loop()` ONLY uses `VideoConsumer.get_frame()`
- `process_frame()` simplified - no single-frame camera mode
- `calibrate_camera()` now uses `get_current_frame()` instead of direct camera access
- Statistics now only track VideoConsumer status

**Added:**
- VideoConsumer and VideoModuleNotAvailableError to exports
- Clear documentation that shared memory IPC is the only mode

### 2. API Streaming (backend/api/routes/stream.py)

**Removed:**
- Feature flag check `video.use_shared_memory` in `/stream/video` endpoint
- All legacy EnhancedCameraModule code path
- Dual-mode branching logic
- `vision_module` dependency from `/stream/video` endpoint

**Simplified:**
- `/stream/video` endpoint ONLY uses `generate_mjpeg_stream_from_shm()`
- Documentation updated to reflect shared memory only
- Removed misleading comments about "two modes"

### 3. Configuration (config.json)

**Removed:**
- `"use_shared_memory": true` line from `video` section

This is no longer needed because there's only one mode now.

## Architecture

### Before (Dual-Mode)
```
┌─────────────────┐
│  VisionModule   │
└────────┬────────┘
         │
    ┌────┴─────┐
    │ Feature  │
    │  Flag?   │
    └────┬─────┘
         │
    ┌────┴──────────────┐
    │                   │
┌───▼────┐      ┌──────▼────────┐
│ Video  │      │ CameraCapture │
│Consumer│      │ (Legacy Mode) │
└────────┘      └───────────────┘
```

### After (Shared Memory Only)
```
┌─────────────────┐
│  VisionModule   │
└────────┬────────┘
         │
         │
    ┌────▼────┐
    │ Video   │
    │Consumer │
    └─────────┘
```

## Benefits

1. **Simpler Code**: No more branching logic, no more feature flags
2. **Single Source of Truth**: Video Module is THE source for frames
3. **Clearer Intent**: Code clearly shows we use IPC, not direct camera access
4. **Better Errors**: If Video Module isn't running, you get a clear error immediately
5. **Easier Maintenance**: One code path = less bugs, easier to reason about
6. **Better Architecture**: Separation of concerns - Video Module handles camera, Vision Module handles detection

## Migration Notes

### For Users
- **MUST** run Video Module before starting Vision Module or API server
- Start with: `python -m backend.video`
- Old `video.use_shared_memory` config flag is ignored (removed from config.json)

### For Developers
- `VisionModule` no longer has a `camera` attribute
- Use `VideoConsumer` for all frame access
- No more `if self._use_shared_memory:` checks needed
- CameraCapture is still available in `backend.vision.capture` for Video Module use

## What Still Uses CameraCapture?

The Video Module process (`backend/video/process.py`) still uses CameraCapture - it's the **producer** side of the shared memory IPC. VisionModule is the **consumer** side.

```
┌──────────────────┐
│  Video Module    │
│  (Producer)      │
│                  │
│  CameraCapture   │
│       ↓          │
│  Shared Memory   │
└────────┬─────────┘
         │
         │ IPC
         │
┌────────▼─────────┐
│  Vision Module   │
│  (Consumer)      │
│                  │
│  VideoConsumer   │
└──────────────────┘
```

## Testing

After these changes:
1. Start Video Module: `python -m backend.video`
2. Start API server: `python -m backend.api`
3. Check video stream: http://localhost:8000/stream/video
4. Verify VisionModule can attach and process frames

## Files Modified

1. `/Users/jchadwick/code/billiards-trainer/backend/vision/__init__.py`
2. `/Users/jchadwick/code/billiards-trainer/backend/api/routes/stream.py`
3. `/Users/jchadwick/code/billiards-trainer/config.json`

## Lines of Code Removed

- Removed ~150 lines of dual-mode branching logic
- Removed ~80 lines of legacy camera initialization code
- Simplified ~40 lines of statistics and validation code

**Total: ~270 lines of complexity removed** ✨
