# Billiards Trainer Implementation Plan

*Updated based on comprehensive codebase analysis by 10 specialized agents (Jan 2025)*

## Executive Summary

Detailed analysis by 10 specialized agents reveals **high-quality implementation** with substantial progress. The system is approximately **80-85% complete** overall, with most modules showing well-architected implementations using real algorithms rather than placeholders.

**Key Findings**:
- Individual modules are largely complete with real implementations
- **Integration wiring has critical gaps** preventing end-to-end functionality
- WebSocket infrastructure exists but execution paths are unreliable
- Frontend testing is critically inadequate (only 2 test files)
- Several ML features use heuristics instead of proper implementations

**Note**: This is a single-user training system with no authentication requirements.

---

## Current Implementation Status by Module

### Backend Modules (Verified by Agents)

| Module | Completion | Status | Critical Gaps |
|--------|-----------|--------|---------------|
| **Vision** | 90% | ✅ Strong | Ball number recognition (heuristic), format converters (placeholder) |
| **Core** | 92% | ✅ Excellent | WebSocket broadcast execution paths, cache hit ratio (1 line) |
| **API** | 90% | ✅ Excellent | Session monitor metrics (2 fields), module log reading (mock data) |
| **Config** | 95% | ✅ Excellent | Profile conditions stub, type converter edge cases |
| **Projector** | 90% | ✅ Strong | Text rendering (texture-to-screen), coordinate transformation |
| **System** | 95% | ✅ Excellent | Memory threshold config (1-line fix) |
| **Integration** | 92% | ⚠️ Good | WebSocket execution paths unreliable in non-async contexts |

### Frontend Module

| Component | Completion | Status | Critical Gaps |
|-----------|-----------|--------|---------------|
| **Web UI** | 80-85% | ✅ Strong | Video feed placeholder, simulated metrics |
| **Components** | 100% | ✅ Complete | All UI components fully implemented |
| **Stores** | 100% | ✅ Complete | MobX stores complete, state migration not implemented |
| **Routing** | 100% | ✅ Complete | All routes working |
| **WebSocket Client** | 100% | ✅ Complete | Full implementation with auto-reconnect |

### Testing Infrastructure

| Category | Status | Coverage | Notes |
|----------|--------|----------|-------|
| **Backend Unit Tests** | ✅ Good | 314 test classes/functions | Across 70+ files |
| **Backend Integration** | ✅ Excellent | Comprehensive | API, Core, Vision well-tested |
| **Backend E2E** | ⚠️ Limited | Basic | 2-3 test files |
| **Performance Tests** | ✅ Good | Real timing | Vision, physics validated |
| **Frontend Tests** | ❌ CRITICAL | 2 files only | No component tests, no store tests |
| **Projector Tests** | ❌ Poor | ~20% | No rendering tests |
| **System Tests** | ❌ Poor | ~10% | No orchestrator tests |

### Calibration System

| Feature | Status | Automation Level |
|---------|--------|------------------|
| **Camera Calibration** | ✅ Complete | 20% auto (manual chessboard) |
| **Geometric Calibration** | ✅ Complete | 30% auto (manual corners) |
| **Color Calibration** | ✅ Complete | 40% auto (K-means + manual) |
| **Projector Calibration** | ✅ Complete | 10% auto (manual keystone) |
| **Adaptive Recalibration** | ✅ Complete | Detects 30% lighting changes |
| **Video Feed in Wizard** | ❌ MISSING | Placeholder canvas only |

**Setup Time**: 20-45 minutes with ~30 manual actions
**Ease of Setup**: 4/10 (Moderate-Difficult)

---

## Priority 1: CRITICAL Integration Issues


### 🔴 **CRITICAL: WebSocket State Broadcasting**
**Severity**: BLOCKS real-time data flow
**Estimated Effort**: 4-6 hours

#### Issues:
1. **Execution path reliability** (`backend/core/integration.py:998-1080`)
   - `message_broadcaster.broadcast_game_state()` only works in async contexts
   - Falls back to queueing but queue flush sends to wrong destination
   - Queue flush calls `event_manager.send_api_message()` NOT WebSocket (line 1091)

2. **Frame streaming unreliable** (`backend/system/orchestrator.py:754-791`)
   - Async task creation with RuntimeError fallback
   - Only works when event loop is running
   - No sync fallback that actually broadcasts

#### Tasks:
- [ ] **Fix queue flush destination** (2-3 hours)
  - Modify `APIInterfaceImpl._flush_message_queue()` to call `message_broadcaster`
  - Ensure queued messages reach WebSocket clients
  - Add fallback for non-async contexts

- [ ] **Ensure event loop for frame streaming** (2-3 hours)
  - Verify event loop running in orchestrator context
  - Add sync fallback that queues for async broadcast
  - Test frame delivery in both contexts

**Verification**: Start system without async context, verify frames and state updates reach WebSocket clients

---

### 🟠 **HIGH: Frontend Video Feed in Calibration Wizard**
**Severity**: BLOCKS usable calibration
**Estimated Effort**: 2-4 hours

#### Issue:
- `CalibrationWizard.tsx:62-102` draws placeholder grid pattern
- `videoStore` imported but never used (line 60)
- Users calibrating blind without visual feedback
- `VideoStream` component and `videoStore.currentFrameImage` available but not integrated

#### Tasks:
- [ ] **Integrate camera feed** (2-3 hours)
  - Replace canvas placeholder with `VideoStream` component or `videoStore.currentFrameImage`
  - Overlay calibration points on video feed
  - Add real-time detection overlay

- [ ] **Test calibration workflow** (1 hour)
  - Verify corner selection on real video
  - Verify color picker on real video

**Verification**: Open calibration wizard, see live video feed with overlay points

---

### 🟠 **HIGH: Ball Number Recognition**
**Severity**: AFFECTS game state accuracy
**Estimated Effort**: 8-16 hours

#### Issue:
- `detection/balls.py:283-898` uses heuristic shape analysis
- Template matching returns shape guesses (lines 651-683)
- OCR uses geometric features without ML (lines 731-837)
- Cannot distinguish ball 1 from ball 9

#### Tasks:
- [ ] **Implement ML-based recognition** (6-12 hours)
  - Integrate Tesseract OCR or train ML model
  - Use real billiard ball image dataset
  - Replace heuristic methods

- [ ] **Test with real footage** (2-4 hours)
  - Validate >90% accuracy requirement
  - Test in various lighting conditions

**Verification**: Detect balls on real table, verify numbers are correctly identified

---

## Priority 2: High-Impact Completions

### 🟡 **MEDIUM: Projector Text Rendering**
**Severity**: TEXT OVERLAYS incomplete
**Estimated Effort**: 4-6 hours

#### Issue:
- `rendering/text.py:448` - "Fallback to rectangle placeholder"
- Surface-to-texture conversion exists but incomplete
- Projection matrix handling incomplete (lines 740-756)

#### Tasks:
- [ ] **Complete texture rendering** (4-5 hours)
  - Fix projection matrix setup in `_render_texture_at_position()`
  - Handle all pygame surface formats
  - Add error recovery

- [ ] **Test text overlays** (1 hour)
  - Verify readable text on projector
  - Test various font sizes and colors

---

### 🟡 **MEDIUM: Frontend Test Coverage**
**Severity**: QUALITY RISK - Regression potential
**Estimated Effort**: 40-60 hours

#### Current Status:
- Only 2 test files (`api-integration.test.ts`, `DetectionOverlayIntegration.test.ts`)
- 0 component tests (81 components untested)
- 0 store tests (19 stores untested)
- ~5% total coverage

#### Tasks:
- [ ] **Add component tests** (24-32 hours)
  - React Testing Library tests for all major components
  - Focus: LiveView, CalibrationWizard, VideoStream, config forms
  - Target: 60% component coverage minimum

- [ ] **Add store tests** (16-20 hours)
  - Test all MobX stores with isolation
  - Mock dependencies appropriately
  - Test state mutations and side effects

- [ ] **Add integration tests** (8-12 hours)
  - End-to-end user workflows
  - WebSocket integration flows
  - Error handling scenarios

---

### 🟡 **MEDIUM: Backend Test Coverage Gaps**
**Severity**: QUALITY RISK
**Estimated Effort**: 40-50 hours

#### Gaps:
- **Projector**: ~20% coverage (no rendering tests)
- **System**: ~10% coverage (no orchestrator tests)
- **API Middleware**: ~40% missing (rate limiting, security, performance untested)

#### Tasks:
- [ ] **Add projector module tests** (20-24 hours)
  - Mock OpenGL rendering tests
  - Display manager tests
  - Trajectory rendering tests
  - Text rendering tests

- [ ] **Add system module tests** (16-20 hours)
  - Orchestrator lifecycle tests
  - Health monitoring tests
  - Recovery mechanism tests
  - Resource management tests

- [ ] **Add API middleware tests** (8-12 hours)
  - Rate limiting edge cases
  - Security middleware tests
  - Session monitoring tests
  - Performance middleware tests

---

### 🟡 **MEDIUM: Vision Format Converters**
**Severity**: INTEGRATION limitation
**Estimated Effort**: 6-8 hours

#### Issue:
- `tracking/integration.py:535-550` - All format converters return input unchanged
- Cannot integrate YOLO or other detection formats

#### Tasks:
- [ ] **Implement format converters** (6-8 hours)
  - OpenCV format (bounding boxes, keypoints)
  - YOLO format (center_x, center_y, width, height)
  - Custom format based on requirements

---

### 🟡 **MEDIUM: Stripe Ball Detection**
**Severity**: CLASSIFICATION accuracy
**Estimated Effort**: 8-12 hours

#### Issue:
- `detection/balls.py:604-609` - Uses variance threshold only
- Cannot reliably distinguish striped from solid balls

#### Tasks:
- [ ] **Enhance stripe detection** (8-12 hours)
  - Add edge detection and frequency analysis
  - Implement pattern recognition
  - Calibrate thresholds on real balls

---

## Priority 3: Feature Completions

### Frontend Enhancements (20-30 hours)

- [ ] **YAML/TOML config import** (2-3 hours)
  - `ConfigImportExport.tsx:196-207` - Currently throws errors
  - Add js-yaml and @iarna/toml packages

- [ ] **State migration** (4-6 hours)
  - `persistence.ts:270-274` - Currently clears state
  - Implement version migration logic

- [ ] **Performance metrics** (8-10 hours)
  - `PerformanceMetrics.tsx:52-63` - Uses simulated data
  - Wire to real backend metrics API

- [ ] **Service management UI** (4-6 hours)
  - `ServiceManagement.tsx:134-174` - API not wired
  - Connect to backend service control endpoints

### API Module Completions (12-16 hours)

- [ ] **Module log reading** (2-4 hours)
  - `routes/modules.py:522-560` - Returns mock logs
  - Implement actual log file reading

- [ ] **Calibration session persistence** (4-6 hours)
  - `routes/calibration.py:670` - Database operations missing
  - Wire to CalibrationSessionDB model

- [ ] **Frame export** (3-4 hours)
  - `routes/game.py:510-514` - Placeholder README
  - Implement actual frame export to ZIP

- [ ] **WebSocket quality reduction** (3-4 hours)
  - `websocket/manager.py:483` - Placeholder
  - Implement adaptive quality based on bandwidth

### Config Module Enhancements (6-10 hours)

- [ ] **Profile conditions** (4-6 hours)
  - `profiles/conditions.py:22` - Minimal stub
  - Implement condition evaluation for auto-profile selection

- [ ] **Type converter edge cases** (2-4 hours)
  - `utils/converter.py` - Multiple pass statements in exception handlers
  - Complete fallback logic for complex types

### Core Module Enhancements (8-12 hours)

- [ ] **Cache hit ratio tracking** (2-3 hours)
  - `__init__.py:765` - Placeholder 0.0
  - `tracking/optimization.py:312` - Returns 0.8 or 0.0
  - Add actual hit/miss counting

- [ ] **Physics validation** (4-6 hours)
  - `physics/validation.py:327` - Passed=True placeholder
  - Implement energy conservation calculations

- [ ] **Projector coordinate transformation** (2-3 hours)
  - `integration.py:1627-1631` - Returns points unchanged
  - Implement actual calibration matrix transformation

### System Module Enhancements (6-9 hours)

- [ ] **Memory threshold config** (1 hour)
  - `monitoring.py:343-345` - Pass statement
  - Add MB-to-percentage conversion using total memory

- [ ] **Shutdown state persistence** (4-6 hours)
  - `api/shutdown.py:373-386` - Sleep placeholder
  - Implement actual state serialization

- [ ] **Cleanup placeholders** (2-3 hours)
  - Various pass statements in exception handlers
  - Review and enhance as needed

---

## Priority 4: Production Readiness

### Load Testing (16-24 hours)

- [ ] **Sustained load tests** (8-12 hours)
  - 30 FPS for hours
  - 100+ concurrent WebSocket connections
  - Memory leak detection

- [ ] **Performance benchmarking** (4-6 hours)
  - Response time profiling
  - Database query optimization
  - Frame processing optimization

- [ ] **Stress testing** (4-6 hours)
  - Resource exhaustion scenarios
  - Connection flood tests
  - Rapid state change tests

### Deployment & Infrastructure (16-24 hours)

- [ ] **Docker configuration** (8-12 hours)
  - Multi-stage Docker builds
  - Docker Compose for full stack
  - Health checks and monitoring

- [ ] **CI/CD pipeline** (4-6 hours)
  - Automated testing
  - Coverage reporting
  - Deployment automation

- [ ] **Documentation** (4-6 hours)
  - Deployment guide
  - API documentation
  - User manual updates

---

## Implementation Strategy

### **Phase 1: Critical Integration (Week 1 - 15-25 hours)**
**Goal**: Functional system with end-to-end data flow

**Days 1-2** (8-12 hours):
1. Fix WebSocket execution paths (4-6 hours) - **CRITICAL**
2. Add video feed to calibration wizard (2-4 hours) - **HIGH**
3. Test end-to-end data flow (2-3 hours)

**Days 3-5** (8-16 hours):
4. Implement ball number recognition (8-16 hours) OR defer to Phase 2

**Outcome**: Fully functional system with real-time data flow

---

### **Phase 2: Testing & Feature Completions (Week 2-3 - 60-80 hours)**
**Goal**: Production-quality with comprehensive testing

**Week 2** (40-50 hours):
1. Add frontend test coverage (40-60 hours priority subset)
   - Critical component tests (LiveView, VideoStream, CalibrationWizard) - 12-16 hours
   - Store tests (VideoStore, AuthStore, ConfigStore) - 8-12 hours
   - Integration tests - 4-6 hours

2. Add backend test coverage (20-30 hours priority subset)
   - Projector rendering tests (critical) - 8-12 hours
   - System orchestrator tests - 8-12 hours
   - API middleware tests - 4-6 hours

**Week 3** (20-30 hours):
3. Complete high-priority features:
   - Projector text rendering (4-6 hours)
   - Vision format converters (6-8 hours)
   - Stripe ball detection (8-12 hours) OR defer to Phase 3

**Outcome**: Well-tested system with critical features complete

---

### **Phase 3: Feature Completions & Polish (Week 4-5 - 40-60 hours)**
**Goal**: Feature-complete system

**Week 4** (20-30 hours):
1. Frontend enhancements (20-30 hours)
2. API module completions (12-16 hours subset)
3. Config module enhancements (6-10 hours)

**Week 5** (20-30 hours):
4. Core module enhancements (8-12 hours)
5. System module completions (6-9 hours)
6. Calibration auto-setup improvements (4-6 hours)

**Outcome**: Feature-complete system with enhanced usability

---

### **Phase 4: Production Deployment (Week 5-6 - 28-48 hours)**
**Goal**: Production-deployed system

**Week 5** (12-20 hours):
1. Load testing (12-20 hours)

**Week 6** (16-28 hours):
2. Docker configuration (8-12 hours)
3. CI/CD pipeline (4-6 hours)
4. Documentation (4-6 hours)
5. Final integration testing (4-8 hours)

**Outcome**: Production-deployed, monitored system

---

## Revised Completion Estimates

| Milestone | Completion % | Time Required | Cumulative |
|-----------|-------------|---------------|------------|
| **Current Status** | ~80-85% | - | - |
| **After Phase 1** | ~88-90% | 1 week | 1 week |
| **After Phase 2** | ~93-95% | 2-3 weeks | 3-4 weeks |
| **After Phase 3** | ~97-98% | 1-2 weeks | 4-5 weeks |
| **After Phase 4** | ~99% | 1 week | 5-6 weeks |

**Time to Functional System**: 1 week (Phase 1) - **PRIORITY**
**Time to Production-Quality**: 3-4 weeks (Phases 1+2)
**Time to Feature-Complete**: 4-5 weeks (Phases 1+2+3)
**Time to Production-Deployed**: 5-6 weeks (All Phases)

---

## Key Strengths (Verified by 10 Agents)

- ✅ **Real Implementations**: OpenCV homography, physics engine with RK4 integration, actual WebSocket infrastructure
- ✅ **Production Architecture**: Comprehensive error handling, logging, health monitoring
- ✅ **Modern Stack**: TypeScript, React 19, FastAPI, OpenGL, MobX 6, Vite 6, ModernGL
- ✅ **Backend Testing**: 314 test classes/functions across 70+ files
- ✅ **Sophisticated Patterns**: Factory, Observer, Strategy, Repository patterns throughout
- ✅ **Excellent Documentation**: Clear docstrings, inline comments, specifications

## Critical Weaknesses (10-Agent Findings)

- ❌ **Integration: WebSocket execution paths** - Unreliable in non-async contexts, queue flush destination wrong
- ❌ **Frontend Testing: Only 2 test files** - No component tests, no store tests, ~5% coverage
- ❌ **Backend Testing Gaps**: Projector (20%), System (10%), API middleware (40%)
- ❌ **Calibration UX**: Video feed placeholder, 30-minute manual setup
- ❌ **Ball Recognition**: Heuristic-based, needs ML implementation
- ❌ **Text Rendering**: Texture-to-screen incomplete

---

## Detailed Module Assessments

### Vision Module: 90% Complete
**Strengths**:
- Real OpenCV algorithms (Hough, Kalman filters, homography)
- Sophisticated multi-method detection (Hough, contour, blob, combined)
- Excellent test coverage (245 tests)
- Performance exceeds requirements (46+ FPS)

**Gaps**:
- Ball number recognition uses heuristics (needs ML/OCR)
- Stripe detection simplified (variance threshold only)
- Format converters are placeholders

### Core Module: 92% Complete
**Strengths**:
- Physics implementation exceeds specs (RK4 integration)
- Complete game state management
- Comprehensive event system
- Excellent integration architecture

**Gaps**:
- WebSocket broadcast execution paths unreliable
- Cache hit ratio tracking placeholder (1 line)
- Physics validation energy conservation placeholder

### API Module: 90% Complete
**Strengths**:
- Complete authentication system (JWT, RBAC, API keys, database)
- Real OpenCV calibration (not identity matrices)
- Full WebSocket infrastructure (broadcaster, manager, handlers)
- Comprehensive middleware stack (9 layers)

**Gaps**:
- Session monitor metrics (2 fields)
- Module log reading (mock data)
- Frame export (placeholder)
- WebSocket quality reduction (placeholder)

### Config Module: 95% Complete
**Strengths**:
- Comprehensive loading (file, env, CLI, inheritance)
- Full validation (schema, rules, types)
- Atomic persistence with backups
- Hot reload with rollback
- 14 test files with good coverage

**Gaps**:
- Profile conditions stub (not critical)
- Type converter edge cases (exception handlers)

### Projector Module: 90% Complete
**Strengths**:
- Complete OpenGL renderer with shaders
- Full calibration system (keystone, geometric)
- Comprehensive WebSocket client with handlers
- Complete effects and trajectory rendering

**Gaps**:
- Text rendering incomplete (texture-to-screen)
- Coordinate transformation simplified
- Very poor test coverage (~20%)

### System Module: 95% Complete
**Strengths**:
- Production-ready orchestration
- Comprehensive health monitoring
- Full auto-recovery system
- Resource and process management

**Gaps**:
- Memory threshold MB conversion (1 line)
- Shutdown state persistence (placeholder)
- Very poor test coverage (~10%)

### Frontend Web UI: 75-80% Complete
**Strengths**:
- 100% component library complete
- 100% MobX stores complete
- 100% routing complete
- Full WebSocket client with auto-reconnect

**Gaps**:
- Video feed placeholder in calibration
- Simulated metrics data
- YAML/TOML import not implemented
- State migration not implemented
- Service management UI not wired
- **CRITICAL: Only 2 test files** (no component/store tests)

---

## Bottom Line

This is a **well-architected, substantially implemented system** requiring:

1. **IMMEDIATE**: Fix WebSocket execution paths (4-6 hours) - **BLOCKS FUNCTIONALITY**
2. **Week 1**: Complete Phase 1 critical items (15-25 hours total)
3. **Weeks 2-4**: Add comprehensive testing and complete features (100-140 hours)
4. **Weeks 5-6**: Production deployment readiness (44-64 hours)

**Total Effort to Production**: **160-230 hours** (4-6 weeks full-time)

The system uses **real algorithms throughout** (OpenCV, RK4 physics, actual WebSocket) and needs **focused integration and testing work**, not fundamental rewrites.

**Note**: This is a single-user training system with no authentication requirements.
