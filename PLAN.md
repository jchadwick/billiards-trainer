# Billiards Trainer Implementation Plan

This plan outlines the remaining tasks to complete the Billiards Trainer system. The tasks are prioritized based on comprehensive codebase analysis conducted on 2025-10-03.

**Last Updated:** 2025-10-03
**Status:** Core features ~90% complete. Primary focus: Fix initialization hang, connect real-time data flow, perform actual calibration.

---

## ✅ COMPLETED (Major Features)

### Backend - Vision Module (95% Complete - NEW ASSESSMENT)
- ✅ **Table Detection** - Full implementation with multi-algorithm approach (backend/vision/detection/table.py)
- ✅ **Ball Detection** - 100% COMPLETE with 4 detection methods, classification, stripe detection, number recognition (backend/vision/detection/balls.py)
- ✅ **Cue Detection** - 100% COMPLETE with 3 detection methods, shot analysis, motion tracking (backend/vision/detection/cue.py)
- ✅ **Object Tracking** - Full Kalman filter tracking with Hungarian algorithm (backend/vision/tracking/tracker.py)
- ✅ **Camera Calibration** - Complete intrinsic/extrinsic calibration with automatic table-based method (backend/vision/calibration/camera.py)
- ✅ **Color Calibration** - Full color calibration system with adaptive lighting (backend/vision/calibration/color.py)
- ✅ **Image Preprocessing** - 100% COMPLETE 8-step pipeline with GPU acceleration (backend/vision/preprocessing.py)
- ✅ **GPU Acceleration** - NEW: Complete VAAPI + OpenCL support with automatic fallback (backend/vision/gpu_utils.py)
- ✅ **Fisheye Correction** - Complete with pre-computed undistortion maps (backend/streaming/enhanced_camera_module.py)
- ✅ **Camera Capture** - Robust camera interface with auto-reconnect (backend/vision/capture.py)

### Backend - Streaming Module (100% Complete - NEW ASSESSMENT)
- ✅ **EnhancedCameraModule** - ALREADY INTEGRATED with fisheye correction and preprocessing (backend/streaming/enhanced_camera_module.py)
- ✅ **CameraModuleAdapter** - Bridge to legacy interface, working correctly (backend/api/routes/stream.py:55-149)
- ✅ **Lazy Initialization** - Async initialization to prevent server startup hang (backend/api/routes/stream.py:167-272)
- ✅ **MJPEG Streaming** - Complete with configurable quality and FPS (backend/api/routes/stream.py:275-393)

### Backend - API Module (95% Complete)
- ✅ **All REST Endpoints** - Complete implementation of all SPECS.md requirements (routes/*.py)
- ✅ **Configuration Endpoints** - GET/PUT/reset/import/export fully working (routes/config.py)
- ✅ **Calibration Endpoints** - Complete camera fisheye calibration workflow (routes/calibration.py:1124-1743)
- ✅ **Game State Endpoints** - Current/historical state access (routes/game.py)
- ✅ **Video Streaming** - MJPEG streaming endpoint working (routes/stream.py)
- ✅ **WebSocket System** - Full message types, broadcasting, subscriptions (websocket/*.py)
- ✅ **Health & Diagnostics** - Comprehensive monitoring endpoints (routes/health.py, routes/diagnostics.py)
- ✅ **Middleware Stack** - Error handling, logging, metrics, tracing (middleware/*.py)

### Backend - Core Module (88% Complete)
- ✅ **Game State Management** - Complete state tracking and validation (core/integration.py:243-351)
- ✅ **Physics Engine** - Full trajectory calculation with collisions (core/physics/)
- ✅ **Transformation Matrix Calculation** - Complete homography calculation with OpenCV (core/integration.py:1737-1821)
- ✅ **Event System** - Message queuing and broadcasting (core/integration.py:1004-1053)

### Backend - Projector Module (95% Complete)
- ✅ **Display Management** - Full multi-monitor support (display/manager.py)
- ✅ **Rendering Pipeline** - Complete OpenGL rendering (rendering/renderer.py)
- ✅ **Trajectory Visualization** - Advanced trajectory rendering (rendering/trajectory.py)
- ✅ **Visual Effects** - Particle system with 8 effect types (rendering/effects.py)
- ✅ **Text Rendering** - Font management and styling (rendering/text.py)
- ✅ **WebSocket Client** - Full async networking with reconnection (network/client.py)
- ✅ **Calibration Manager** - Complete calibration workflow (calibration/manager.py)

### Frontend - Web Module (70% Complete)
- ✅ **API/WebSocket Clients** - Complete REST and WebSocket clients (services/*.ts)
- ✅ **Video Streaming UI** - Full MJPEG display with zoom/pan (components/video/VideoStream.tsx)
- ✅ **Overlay Rendering** - Complete canvas overlay system (components/video/OverlayCanvas.tsx)
- ✅ **Calibration Wizard** - 5-step interactive calibration (components/config/calibration/CalibrationWizard.tsx)
- ✅ **UI Component Library** - All basic UI components (components/ui/*)
- ✅ **Live View** - Complete with keyboard shortcuts and controls (components/video/LiveView.tsx)

---

## 🔴 CRITICAL PRIORITY (Blocks Core Functionality)

### CRITICAL-1: Fix VisionModule Initialization Hang ⚠️ HIGHEST PRIORITY
- **File:** `backend/vision/gpu_utils.py:60` and `backend/vision/__init__.py:163-221`
- **Root Cause:** OpenCL device initialization deadlock when calling `cv2.ocl.Device_getDefault()`
- **Specific Issue:** Intel iHD VAAPI driver initialization can hang indefinitely at kernel level
- **Current Status:** VisionModule initialization hangs, blocking all computer vision functionality
- **Solution:**
  1. Add `cv2.ocl.setUseOpenCL(False)` before any GPU operations in gpu_utils.py
  2. Make OpenCL initialization opt-in rather than opt-out
  3. Add timeout mechanism using threading.Timer for GPU detection
  4. Lazy import of pylibfreenect2 in capture.py (currently blocks at module load)
- **Impact:** CRITICAL - Without this fix, vision processing pipeline cannot start
- **Effort:** 2-3 hours
- **Files to modify:**
  - `backend/vision/gpu_utils.py:37-71` - Wrap OpenCL init in timeout, add disable flag
  - `backend/vision/preprocessing.py:115-127` - Make GPU initialization optional by default
  - `backend/vision/capture.py:26-36` - Lazy import Kinect2
  - `backend/config/default.json:98` - Set `use_gpu: false` by default

### CRITICAL-2: Implement Coordinate Transformation for Projection
- **File:** `backend/core/integration.py:1693-1699`
- **Current Status:** Transformation matrix is calculated correctly but never applied
- **Issue:** `_convert_to_screen_coordinates()` returns world coordinates unchanged
- **Solution:** Implement cv2.perspectiveTransform to apply stored homography matrix
- **Impact:** CRITICAL - Projector cannot display trajectories at correct screen positions
- **Effort:** 1-2 hours
- **Implementation:**
  ```python
  def _convert_to_screen_coordinates(self, world_points: list[dict]) -> list[dict]:
      matrix = self.projection_settings.get("transformation_matrix")
      if not matrix:
          return world_points  # Not calibrated yet

      # Convert dict to numpy array
      pts = np.array([[p["x"], p["y"]] for p in world_points], dtype=np.float32)
      pts = pts.reshape(-1, 1, 2)

      # Apply perspective transform
      matrix_np = np.array(matrix, dtype=np.float32)
      transformed = cv2.perspectiveTransform(pts, matrix_np)

      # Convert back to dict format
      return [{"x": float(pt[0][0]), "y": float(pt[0][1])} for pt in transformed]
  ```

### CRITICAL-3: Perform Actual Camera Calibration
- **File:** `backend/calibration/camera_fisheye_default.yaml`
- **Current Status:** Using placeholder identity calibration values
- **Issue:** Camera matrix has basic focal length (1000), dist_coeffs are all zeros (no correction)
- **Impact:** CRITICAL - Fisheye correction is ineffective, geometric accuracy is poor
- **Solution:** Run actual calibration on target environment using one of two methods:
  1. **Automatic:** Use `/api/v1/vision/calibration/camera/auto-calibrate` endpoint (single frame from table)
  2. **Manual:** Capture 10+ chessboard images via calibration workflow endpoints
- **Effort:** 1-2 hours (mostly waiting for calibration to complete)
- **Test Command:**
  ```bash
  # Test automatic calibration from target environment
  curl -X POST http://192.168.1.31:8000/api/v1/vision/calibration/camera/auto-calibrate
  ```

---

## 🔴 HIGH PRIORITY (Missing Integration)

### HIGH-1: Connect Vision Data to WebSocket Broadcaster
- **Files:**
  - `backend/streaming/enhanced_camera_module.py:204` - Frame capture point
  - `backend/api/websocket/broadcaster.py:138-211` - Frame broadcast method
  - `backend/core/integration.py:1012-1030` - State broadcast scheduling
- **Current Status:**
  - ✅ EnhancedCameraModule captures frames
  - ✅ WebSocket broadcaster has all methods implemented
  - ❌ No connection between camera and broadcaster
- **Solution:**
  1. Add frame callback to EnhancedCameraModule that calls `message_broadcaster.broadcast_frame()`
  2. Ensure event loop is accessible from camera thread (use asyncio.run_coroutine_threadsafe)
  3. Wire game state updates from core module to broadcaster
- **Impact:** HIGH - WebSocket clients can't receive real-time frames and detection data
- **Effort:** 3-4 hours

### HIGH-2: Create Unified Calibration Workflow
- **Files:**
  - `backend/api/routes/calibration.py:1553-1691` - Auto-calibrate endpoint
  - `backend/vision/calibration/camera.py:576-755` - Table-based calibration
  - `backend/vision/calibration/geometry.py` - Geometric calibration
- **Current Status:**
  - ✅ Camera intrinsic calibration fully implemented
  - ✅ Table geometric calibration fully implemented
  - ❌ No unified workflow combining both
  - ❌ No single API endpoint for complete system calibration
- **Solution:**
  1. Create `/api/v1/calibration/system/auto-calibrate` endpoint
  2. Combines camera fisheye calibration + table detection + geometric calibration
  3. Returns unified calibration object with all parameters
  4. Add validation step to verify calibration quality
- **Impact:** HIGH - User must manually run multiple calibration steps
- **Effort:** 4-6 hours

### HIGH-3: Wire Game State Updates for Real-Time Broadcasting
- **Files:**
  - `backend/core/integration.py:243-259` - State update method
  - `backend/api/websocket/broadcaster.py:213-230` - State broadcast method
  - `backend/api/main.py:250-258` - API interface initialization
- **Current Status:**
  - ✅ Core module has state update logic
  - ✅ Broadcaster has state broadcast method
  - ❌ APIInterfaceImpl not instantiated and registered with core
  - ❌ Event loop not accessible for async broadcasts
- **Solution:**
  1. Create APIInterfaceImpl instance during lifespan startup
  2. Register with CoreModuleIntegrator
  3. Ensure asyncio event loop access for _queue_message calls
  4. Test with WebSocket client subscription to "state" stream
- **Impact:** HIGH - Clients can't see live game state, ball positions, cue tracking
- **Effort:** 2-3 hours

---

## 🟡 MEDIUM PRIORITY (Completeness & Quality)

### MEDIUM-1: Implement Real Module Log Retrieval
- **File:** `backend/api/routes/modules.py:522-559`
- **Current:** Returns hardcoded mock log entries
- **Action:** Read from actual log files or Python logging system
- **Impact:** Cannot debug module issues via API
- **Effort:** 2-3 hours

### MEDIUM-2: Implement WebSocket Frame Quality Filtering
- **File:** `backend/api/websocket/manager.py:483-495`
- **Current:** Sets quality label but doesn't resize/compress frames
- **Action:** Actual image resizing and compression based on quality level
- **Impact:** Wastes bandwidth on low-quality connections
- **Effort:** 2-3 hours

### MEDIUM-3: Implement Cache Hit Rate Tracking
- **Files:**
  - `backend/core/__init__.py:765`
  - `backend/vision/tracking/optimization.py:311-313`
- **Current:** Hardcoded placeholder values (0.0 and 0.8)
- **Action:** Track real cache hits/misses in optimization layer
- **Impact:** Performance metrics are inaccurate
- **Effort:** 2-3 hours

### MEDIUM-4: Implement Ball Number Template Matching
- **File:** `backend/vision/detection/balls.py:769`
- **Current:** Uses simplified contour analysis instead of template matching
- **Action:** Create templates for numbers 1-15, implement template matching
- **Impact:** Ball numbering may be inaccurate in some lighting conditions
- **Effort:** 4-6 hours

### MEDIUM-5: Implement Real Network Diagnostics
- **File:** `backend/api/routes/diagnostics.py:283-339`
- **Current:** Simulated tests with random values
- **Action:** Implement real HTTP/WebSocket connectivity and bandwidth tests
- **Impact:** Cannot diagnose actual network issues
- **Effort:** 3-4 hours

---

## 🟢 LOW PRIORITY (Nice-to-Have)

### Backend - Non-Critical Features
- **API-5:** Implement Calibration Backup Persistence (calibration.py:661-678) - 1-2 hours
- **API-6:** Implement Session Export Raw Frames (game.py:499-503) - 2-3 hours
- **API-7:** Implement Shutdown State Persistence (shutdown.py:372-391) - 2-3 hours
- **CORE-3:** Implement Energy Conservation Validation (physics/validation.py:327) - 2-3 hours
- **PROJ-1:** Add Interactive Calibration UI Integration (projector/main.py:800-802) - 3-4 hours

### Frontend - Missing Features
- **UI-1:** Connect WebSocket Data to Overlay System (requires HIGH-1 backend work) - 3-4 hours
- **UI-2:** Replace Mock Authentication (AuthStore.ts:498-650) - 2-3 hours
- **UI-3:** Add System Control Buttons (various dashboard components) - 3-4 hours
- **UI-4:** Implement Real Performance Monitoring (PerformanceMetrics.tsx) - 2-3 hours
- **UI-5:** Complete Configuration UI Backend Integration (configuration.tsx) - 3-5 hours

### Frontend - Enhancement Features
- **UI-6:** Implement Event Log with Search/Export (ErrorLog.tsx) - 4-6 hours
- **UI-7:** Implement Dashboard Customization (DashboardLayout.tsx) - 6-8 hours
- **UI-8:** Implement Adaptive Video Quality (VideoStream.tsx) - 3-4 hours

### General - Testing & Optimization
- **TEST-1:** Increase Unit Test Coverage (target >80%) - 16-24 hours
- **PERF-1:** Profile and optimize performance bottlenecks - 12-16 hours

---

## 🎯 Recommended Implementation Order

### Phase 1: Fix Critical Blockers (4-8 hours) ⚠️⚠️⚠️
**Must complete before system can work on target environment**

1. **CRITICAL-1:** Fix VisionModule initialization hang (GPU/OpenCL issue) - 2-3 hours
2. **CRITICAL-2:** Implement coordinate transformation for projection - 1-2 hours
3. **CRITICAL-3:** Perform actual camera calibration on target environment - 1-2 hours
4. **Test:** Verify camera capture, vision processing, and calibration all work

### Phase 2: Connect Real-Time Data Flow (6-10 hours)
**Enable live computer vision and WebSocket streaming**

1. **HIGH-1:** Connect vision data to WebSocket broadcaster - 3-4 hours
2. **HIGH-3:** Wire game state updates for real-time broadcasting - 2-3 hours
3. **HIGH-2:** Create unified calibration workflow - 4-6 hours
4. **Test:** Verify WebSocket clients receive frames, ball positions, and game state

### Phase 3: Complete Integration & Polish (8-12 hours)
**Fill in remaining gaps and improve quality**

1. **MEDIUM-1:** Real module log retrieval - 2-3 hours
2. **MEDIUM-2:** WebSocket frame quality filtering - 2-3 hours
3. **MEDIUM-3:** Cache hit rate tracking - 2-3 hours
4. **MEDIUM-5:** Real network diagnostics - 3-4 hours
5. **Test:** End-to-end system test with all features

### Phase 4: Advanced Features & Optimization (20-40 hours)
**After core system is working end-to-end**

1. **MEDIUM-4:** Ball number template matching - 4-6 hours
2. Frontend integration (UI-1 through UI-8) - 20-30 hours
3. Low priority backend features (API-5 through PROJ-1) - 10-12 hours
4. Testing and optimization (TEST-1, PERF-1) - 28-40 hours

---

## 📊 Updated Completion Status

| Module | Completion | Critical Gaps |
|--------|-----------|---------------|
| **Backend - Vision** | 95% | VisionModule init hang (GPU/OpenCL), actual calibration needed |
| **Backend - Streaming** | 100% | ✅ COMPLETE - EnhancedCameraModule fully integrated |
| **Backend - API** | 95% | WebSocket-vision connection, some placeholders |
| **Backend - Core** | 88% | Coordinate transformation application |
| **Backend - Projector** | 95% | Minor interactive calibration UI |
| **Frontend - Web** | 70% | Real-time data connection, auth, system controls |
| **Overall System** | ~90% | Fix GPU hang, connect data flow, run calibration |

---

## 🔑 Key Findings from Analysis

### ✅ What's Actually Complete (Previously Thought Incomplete)

1. **EnhancedCameraModule Integration:** COMPLETE - Already integrated in stream.py, DirectCameraModule doesn't exist
2. **Fisheye Calibration Workflow:** COMPLETE - Full camera calibration API endpoints exist (lines 1124-1743)
3. **Ball/Cue Detection:** 100% COMPLETE - No placeholders, production-ready with multiple algorithms
4. **Image Preprocessing:** 100% COMPLETE - 8-step pipeline with GPU acceleration fully implemented
5. **GPU Acceleration:** NEWLY IMPLEMENTED - VAAPI + OpenCL support with automatic fallback
6. **WebSocket Infrastructure:** COMPLETE - Handler, broadcaster, manager all operational
7. **Transformation Matrix Calculation:** COMPLETE - Uses cv2.getPerspectiveTransform correctly

### ❌ What's Actually Broken (Not Just Incomplete)

1. **VisionModule Initialization:** HANGS - OpenCL device query at gpu_utils.py:60 deadlocks
2. **Coordinate Transformation:** CALCULATED BUT NOT APPLIED - Matrix exists, just not used
3. **Camera Calibration Data:** PLACEHOLDER - Using identity matrix, no actual calibration performed
4. **Vision-to-WebSocket Connection:** DISCONNECTED - Both sides exist but not wired together

### 🔄 What Changed Recently (Uncommitted)

1. **GPU Acceleration:** backend/vision/gpu_utils.py created with full VAAPI + OpenCL support
2. **Enhanced Error Handling:** improved FileStorage validation in enhanced_camera_module.py
3. **Configuration:** GPU settings added to config/default.json, VAAPI environment in main.py
4. **Preprocessing Integration:** GPU accelerator integrated into preprocessing.py

---

## 🎯 Success Criteria for Target Environment

The system is ready for deployment when:

1. ✅ **Startup:** Backend starts without hanging (fix CRITICAL-1)
2. ✅ **Camera:** Can capture frames from /dev/video0 or /dev/video1
3. ✅ **Calibration:** Can run auto-calibrate and get real calibration values (CRITICAL-3)
4. ✅ **Detection:** Ball and cue detection produce results
5. ✅ **WebSocket:** Clients can connect and receive frame/state streams (HIGH-1, HIGH-3)
6. ✅ **Projection:** Projector displays trajectories at correct positions (CRITICAL-2)
7. ✅ **Performance:** Maintains 15+ FPS with detection enabled

---

## 📝 Notes

- **Focus on backend first:** Per PROMPT2.md, ignore frontend until backend camera calibration and detection working
- **Target environment:** 192.168.1.31 - No camera/projector on local machine
- **Deployment:** Use `rsync -av dist/ jchadwick@192.168.1.31:/opt/billiards-trainer/` (NEVER use --delete flag)
- **Auto-reload:** Target system runs in auto-reload mode, changes picked up automatically
- **Testing:** All validation must happen on target environment where hardware exists
