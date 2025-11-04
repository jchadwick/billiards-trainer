# Cue Detection Flakiness Fix

## Problem
Cue detection was extremely flaky - sometimes working, sometimes not detecting the cue at all.

## Root Cause
Recent performance optimization added a frame-level detection cache using `id(frame.data)` (memory address) as the cache key.

**The Issue:** Camera/video systems reuse frame buffers for performance. When the same memory address is reused for different frames:
- Old frame with cue → YOLO runs → cue detected → cached with key `0x12345`
- New frame without cue reuses buffer `0x12345` → cache hit → returns old "cue detected" result ❌
- OR vice versa: frame without cue cached → new frame with cue → cache returns "no cue" ❌

This caused the detector to return stale detections from previous frames.

## Solution
Changed the cache key from `id(frame.data)` to `(id(frame.data), content_hash)` where `content_hash` is a hash of a small center region of the frame.

This ensures:
1. Fast lookup using buffer address
2. Content verification prevents false cache hits from buffer reuse
3. Minimal performance impact (only hashes 20x20 pixel region)

## Changes Made
- `backend/vision/detection/yolo_detector.py:1271-1277` - Updated cache key generation
- `backend/vision/detection/yolo_detector.py:211` - Updated cache type annotation

## Testing
Should verify:
- Cue detection now works consistently
- Cache still provides performance benefits (avoids double YOLO inference per frame)
- No performance regression from the small region hashing
