# Billiards Trainer Implementation Plan

This plan outlines the remaining tasks to complete the Billiards Trainer system. The tasks are prioritized based on comprehensive codebase analysis and development priorities in the `SPECS.md` files.

**Last Updated:** 2025-10-02
**Status:** Most core features implemented (~85% complete). Focus on integration, calibration workflow, and critical placeholders.

---

## ✅ COMPLETED (Major Features)

### Backend - Vision Module (~85% Complete)
- ✅ **Table Detection** - Full implementation with multi-algorithm approach (backend/vision/detection/table.py)
- ✅ **Ball Detection** - Complete with type classification, stripe detection, number recognition (backend/vision/detection/balls.py)
- ✅ **Cue Detection** - Advanced cue stick detection with shot analysis (backend/vision/detection/cue.py)
- ✅ **Object Tracking** - Full Kalman filter tracking with Hungarian algorithm (backend/vision/tracking/tracker.py)
- ✅ **Camera Calibration** - Complete intrinsic/extrinsic calibration (backend/vision/calibration/camera.py)
- ✅ **Color Calibration** - Full color calibration system with adaptive lighting (backend/vision/calibration/color.py)
- ✅ **Image Preprocessing** - Comprehensive 7-step pipeline (backend/vision/preprocessing.py)
- ✅ **Camera Capture** - Robust camera interface with auto-reconnect (backend/vision/capture.py)

### Backend - API Module (~95% Complete)
- ✅ **All REST Endpoints** - Complete implementation of all SPECS.md requirements (routes/*.py)
- ✅ **Configuration Endpoints** - GET/PUT/reset/import/export fully working (routes/config.py)
- ✅ **Calibration Endpoints** - Real OpenCV homography calculations (routes/calibration.py)
- ✅ **Game State Endpoints** - Current/historical state access (routes/game.py)
- ✅ **Video Streaming** - MJPEG streaming endpoint working (routes/stream.py)
- ✅ **WebSocket System** - Full message types and broadcasting (websocket/*.py)
- ✅ **Health & Diagnostics** - Comprehensive monitoring endpoints (routes/health.py, routes/diagnostics.py)
- ✅ **Middleware Stack** - Error handling, logging, metrics, tracing (middleware/*.py)

### Backend - Projector Module (~95% Complete)
- ✅ **Display Management** - Full multi-monitor support (display/manager.py)
- ✅ **Rendering Pipeline** - Complete OpenGL rendering (rendering/renderer.py)
- ✅ **Trajectory Visualization** - Advanced trajectory rendering (rendering/trajectory.py)
- ✅ **Visual Effects** - Particle system with 8 effect types (rendering/effects.py)
- ✅ **Text Rendering** - Font management and styling (rendering/text.py)
- ✅ **WebSocket Client** - Full async networking with reconnection (network/client.py)
- ✅ **Calibration Manager** - Complete calibration workflow (calibration/manager.py)

### Frontend - Web Module (~70% Complete)
- ✅ **API/WebSocket Clients** - Complete REST and WebSocket clients (services/*.ts)
- ✅ **Video Streaming UI** - Full MJPEG display with zoom/pan (components/video/VideoStream.tsx)
- ✅ **Overlay Rendering** - Complete canvas overlay system (components/video/OverlayCanvas.tsx)
- ✅ **Calibration Wizard** - 5-step interactive calibration (components/config/calibration/CalibrationWizard.tsx)
- ✅ **UI Component Library** - All basic UI components (components/ui/*)
- ✅ **Live View** - Complete with keyboard shortcuts and controls (components/video/LiveView.tsx)

---

## 🔴 HIGH PRIORITY (Critical Functionality Gaps)

### Backend - Integration & Streaming
- **STREAM-1: Integrate EnhancedCameraModule** (HIGHEST PRIORITY)
  - File: `backend/streaming/enhanced_camera_module.py`
  - Action: Replace DirectCameraModule with EnhancedCameraModule for fisheye correction
  - Location: `backend/api/routes/stream.py:76-92`
  - Impact: Adds fisheye correction and preprocessing to camera stream
  - Effort: 2-4 hours
  - **This is mentioned in your prompt as special focus area**

- **STREAM-2: Create Calibration Workflow for Camera**
  - Files: `backend/api/routes/calibration.py` (new endpoints)
  - Action: Add endpoints for fisheye calibration mode (start/capture/process/apply)
  - Integration: Connect with CalibrationWizard.tsx
  - Effort: 4-6 hours
  - **Critical for calibration workflow mentioned in your prompt**

- **CORE-1: Fix Calibration Transformation Matrix**
  - File: `backend/core/integration.py:1738-1745`
  - Current: Returns identity matrix placeholder
  - Action: Implement real perspective transformation using OpenCV
  - Impact: Calibration transformations won't work correctly without this
  - Effort: 2-3 hours

- **API-1: Connect Real WebSocket Handler to Main Endpoint**
  - File: `backend/api/main.py:379-394`
  - Current: Simple echo implementation at `/ws`
  - Action: Replace with full WebSocketHandler
  - Impact: Real-time data streaming not working on main endpoint
  - Effort: 1-2 hours

- **VISION-1: Fix VisionModule Initialization Hang**
  - File: `backend/vision/__init__.py`
  - Current: Hangs during detector initialization (per CAMERA_STREAM_STATUS.md)
  - Action: Isolate which detector causes hang, fix or simplify
  - Impact: Cannot use full vision processing pipeline
  - Effort: 4-8 hours (debugging required)

### Frontend - Real-Time Integration
- **UI-1: Connect WebSocket Data to Overlay System**
  - Files: `frontend/web/src/stores/VisionStore.ts`, `components/video/LiveView.tsx`
  - Current: Overlay components exist but no live data feed
  - Action: Subscribe to WebSocket detection frames and update overlays
  - Impact: Detection overlays don't show real-time data
  - Effort: 3-4 hours

- **UI-2: Replace Mock Authentication**
  - File: `frontend/web/src/stores/AuthStore.ts:498-650`
  - Current: Uses hardcoded mock tokens
  - Action: Connect to real backend authentication
  - Impact: Authentication not secure
  - Effort: 2-3 hours

---

## 🟡 MEDIUM PRIORITY (Completeness & Quality)

### Backend - Placeholders & Integration
- **API-2: Implement Frame Quality Filtering**
  - File: `backend/api/websocket/manager.py:483-495`
  - Current: Sets quality label but doesn't resize/compress frames
  - Action: Actual image resizing and compression based on quality level
  - Impact: Wastes bandwidth on low-quality connections
  - Effort: 2-3 hours

- **API-3: Real Network Diagnostics**
  - File: `backend/api/routes/diagnostics.py:283-339`
  - Current: Simulated tests with random values
  - Action: Implement real HTTP/WebSocket connectivity and bandwidth tests
  - Impact: Cannot diagnose actual network issues
  - Effort: 3-4 hours

- **API-4: Real Module Log Retrieval**
  - File: `backend/api/routes/modules.py:522-559`
  - Current: Returns hardcoded mock log entries
  - Action: Read from actual log files or logging system
  - Impact: Cannot debug module issues via API
  - Effort: 2-3 hours

- **CORE-2: Implement Cache Hit Rate Tracking**
  - Files: `backend/core/__init__.py:765`, `backend/vision/tracking/optimization.py:311-313`
  - Current: Hardcoded placeholder values
  - Action: Track real cache hits/misses
  - Impact: Performance metrics are inaccurate
  - Effort: 2-3 hours

- **CORE-3: Implement Energy Conservation Validation**
  - File: `backend/core/physics/validation.py:327`
  - Current: Always returns True
  - Action: Calculate kinetic energy before/after collisions
  - Impact: Cannot detect physics calculation errors
  - Effort: 2-3 hours

### Frontend - Missing Features
- **UI-3: Add System Control Buttons**
  - Files: Various dashboard components
  - Current: Missing UI for start/stop detection, reset game, assistance levels, projector on/off
  - Action: Add control buttons that call existing API endpoints
  - Impact: Users cannot control system from UI
  - Effort: 3-4 hours

- **UI-4: Implement Real Performance Monitoring**
  - Files: `frontend/web/src/components/monitoring/PerformanceMetrics.tsx`
  - Current: Components exist but no real data collection
  - Action: Poll backend health/metrics endpoints
  - Impact: Cannot monitor system performance
  - Effort: 2-3 hours

- **UI-5: Complete Configuration UI Backend Integration**
  - Files: `frontend/web/src/routes/configuration.tsx`, config components
  - Current: UI components exist but backend sync unclear
  - Action: Verify and complete backend save/load for all config panels
  - Impact: Configuration changes may not persist
  - Effort: 3-5 hours

---

## 🟢 LOW PRIORITY (Nice-to-Have)

### Backend - Non-Critical Features
- **API-5: Implement Calibration Backup Persistence**
  - File: `backend/api/routes/calibration.py:661-678`
  - Current: Minimal backup to `/tmp/`
  - Action: Proper backup storage with full calibration data
  - Effort: 1-2 hours

- **API-6: Implement Session Export Raw Frames**
  - File: `backend/api/routes/game.py:499-503`
  - Current: Placeholder README instead of frames
  - Action: Export actual frame data when `include_raw_frames=true`
  - Effort: 2-3 hours

- **API-7: Implement Shutdown State Persistence**
  - File: `backend/api/shutdown.py:372-391`
  - Current: Sleep placeholders instead of real save
  - Action: Implement actual state save and resource cleanup
  - Effort: 2-3 hours

- **PROJ-1: Add Interactive Calibration UI Integration**
  - File: `backend/projector/main.py:800-802`
  - Current: Returns default corners without user input
  - Action: Integrate interactive UI for manual calibration point selection
  - Effort: 3-4 hours

### Frontend - Enhancement Features
- **UI-6: Implement Event Log with Search/Export**
  - Files: `frontend/web/src/components/monitoring/ErrorLog.tsx`
  - Current: Basic component, no search/filter/export
  - Action: Add searchable event history and export functionality
  - Effort: 4-6 hours

- **UI-7: Implement Dashboard Customization**
  - File: `frontend/web/src/components/monitoring/DashboardLayout.tsx`
  - Current: Layout exists but no drag-and-drop
  - Action: Add drag-and-drop panel customization with persistence
  - Effort: 6-8 hours

- **UI-8: Implement Adaptive Video Quality**
  - File: `frontend/web/src/components/video/VideoStream.tsx`
  - Current: Fixed quality streaming
  - Action: Add bandwidth detection and automatic quality adjustment
  - Effort: 3-4 hours

### General - Testing & Optimization
- **TEST-1: Increase Unit Test Coverage**
  - Current: Basic test coverage
  - Action: Write additional unit tests for all modules
  - Target: >80% coverage per SPECS.md
  - Effort: 16-24 hours

- **PERF-1: Add Performance Optimizations**
  - Current: CPU-based processing
  - Action: Profile code and add GPU acceleration where applicable
  - Impact: Improved frame rates and lower latency
  - Effort: 12-16 hours

---

## 📋 SPECIAL FOCUS: Backend/Streaming Integration (Per Your Prompt)

### New Files to Integrate
1. **backend/streaming/enhanced_camera_module.py** - Enhanced camera with fisheye correction
2. **backend/streaming/gstreamer_consumer.py** - GStreamer shared memory consumer
3. **backend/streaming/SPECS.md** - Streaming service specification
4. **backend/streaming/IMPLEMENTATION_GUIDE.md** - GStreamer multi-consumer guide

### Integration Tasks (HIGHEST PRIORITY)
- **STREAM-1**: Replace DirectCameraModule with EnhancedCameraModule (2-4 hours) ⭐
- **STREAM-2**: Create fisheye calibration workflow (4-6 hours) ⭐
- **STREAM-3**: Test camera streaming on target environment (2-3 hours) ⭐
- **STREAM-4**: Document streaming architecture decision (EnhancedCamera vs GStreamer) (1 hour)

### Calibration Workflow Integration (Per Your Prompt)
- **CAL-1**: Backend - Add calibration mode endpoints (start/capture/process/apply) ⭐
- **CAL-2**: Frontend - Connect CalibrationWizard to new endpoints ⭐
- **CAL-3**: Test full calibration workflow on target environment ⭐
- **CAL-4**: Create calibration documentation/guide ⭐

---

## 🎯 Recommended Implementation Order

### Phase 1: Critical Streaming & Calibration (8-16 hours) ⭐⭐⭐
1. STREAM-1: Integrate EnhancedCameraModule (HIGHEST)
2. STREAM-2: Create camera calibration workflow
3. STREAM-3: Test on target environment
4. CORE-1: Fix calibration transformation matrix
5. CAL-1 & CAL-2: Complete calibration workflow integration

### Phase 2: Real-Time Data Flow (6-10 hours)
1. API-1: Connect real WebSocket handler
2. UI-1: Connect WebSocket data to overlays
3. VISION-1: Fix VisionModule initialization hang
4. UI-2: Replace mock authentication

### Phase 3: Quality & Completeness (10-15 hours)
1. API-2: Frame quality filtering
2. API-3: Real network diagnostics
3. API-4: Real module logs
4. CORE-2: Cache hit rate tracking
5. UI-3: System control buttons
6. UI-4: Performance monitoring
7. UI-5: Config UI backend integration

### Phase 4: Polish & Testing (20-30 hours)
1. Low priority features (API-5 through UI-8)
2. TEST-1: Increase test coverage
3. PERF-1: Performance optimizations

---

## 📊 Overall Completion Status

| Module | Completion | Critical Gaps |
|--------|-----------|---------------|
| **Backend - Vision** | 85% | VisionModule init hang, streaming integration |
| **Backend - API** | 95% | WebSocket handler, some placeholders |
| **Backend - Projector** | 95% | Minor interactive calibration UI |
| **Backend - Core** | 90% | Transformation matrix, validation |
| **Backend - Streaming** | 0% | NEW - needs integration ⭐ |
| **Frontend - Web** | 70% | Real-time data, auth, system controls |
| **Overall System** | ~80% | Integration, calibration workflow ⭐ |

---

## 🔑 Key Notes

1. **Most detection algorithms are fully implemented** - The original PLAN.md listed these as incomplete, but they are actually complete and production-ready.

2. **Streaming module is new** - The backend/streaming/ directory contains alternative camera implementations that need integration, especially for fisheye correction.

3. **Integration is the main gap** - Individual modules are largely complete, but integration points need work (WebSocket data flow, calibration workflow, real-time updates).

4. **Frontend needs real data** - UI components exist but many lack backend data connections.

5. **Target environment focus** - Per CLAUDE.local.md, primary focus is getting this working on target environment (192.168.1.31).

6. **No camera/projector on local system** - Testing hardware features requires target environment.
