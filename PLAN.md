# Billiards Trainer Implementation Plan

*Updated based on comprehensive codebase analysis by 10 specialized agents (Jan 2025)*

## Executive Summary

Detailed analysis by 10 specialized agents reveals **high-quality implementation** with substantial progress. The system is approximately **80-85% complete** overall, with most modules showing well-architected implementations using real algorithms rather than placeholders.

**Critical Finding**: Individual modules are largely complete, but **integration wiring has gaps** preventing end-to-end functionality. WebSocket infrastructure exists but frame streaming isn't fully wired. Frontend needs video feed integration in calibration wizard.

## Current Implementation Status by Module

### Backend Modules (Verified by Agents)

| Module | Completion | Status | Critical Gaps |
|--------|-----------|--------|---------------|
| **Vision** | 85% | ✅ Strong | Ball number recognition (heuristic-based), camera-to-table transform partial |
| **Core** | 75% | ✅ Solid | Integration interfaces not wired, spin physics needs calibration |
| **API** | 90% | ✅ Excellent | Minor placeholders (frame export, session metrics), OpenCV calibration is real |
| **Config** | 85% | ✅ Strong | Type converter incomplete, schema validator needs verification |
| **Projector** | 85% | ✅ Solid | Text rendering incomplete, interactive calibration partial |
| **System** | 95% | ✅ Excellent | Memory threshold config (1-line fix), otherwise production-ready |
| **Integration** | 75% | ⚠️ Partial | WebSocket state broadcast not wired, frame streaming missing |

### Frontend Module

| Component | Completion | Status | Critical Gaps |
|-----------|-----------|--------|---------------|
| **Web UI** | 85% | ✅ Strong | Video feed in calibration wizard (placeholder canvas), monitoring uses simulated data |
| **Components** | 100% | ✅ Complete | All UI components fully implemented |
| **Stores** | 100% | ✅ Complete | MobX stores complete, state migration not implemented |
| **Routing** | 100% | ✅ Complete | All routes working |

### Testing Infrastructure

| Category | Status | Coverage |
|----------|--------|----------|
| **Backend Unit Tests** | ✅ Good | ~1,863 tests across 89 files |
| **Backend Integration** | ✅ Excellent | 304 tests across 12 files |
| **Backend E2E** | ⚠️ Limited | 12 test classes (basic coverage) |
| **Performance Tests** | ✅ Excellent | 15 test classes with real timing |
| **Frontend Tests** | ❌ Poor | Only 56 integration tests, no component tests |

### Calibration System

| Feature | Status | Automation Level |
|---------|--------|------------------|
| **Camera Calibration** | ✅ Complete | 20% auto (needs manual chessboard capture) |
| **Geometric Calibration** | ✅ Complete | 30% auto (manual corner selection) |
| **Color Calibration** | ✅ Complete | 40% auto (K-means + manual HSV tuning) |
| **Projector Calibration** | ✅ Complete | 10% auto (manual keystone adjustment) |
| **Adaptive Recalibration** | ✅ Complete | Detects 30% lighting changes |
| **Video Feed in Wizard** | ❌ Missing | Placeholder canvas only |

**Setup Time**: 20-45 minutes with ~30 manual actions
**Ease of Setup**: 4/10 (Moderate-Difficult)

---

## Priority 1: Critical Blockers (IMMEDIATE - Prevents Basic Functionality)

### 🔴 **CRITICAL: WebSocket Integration Wiring**
**Severity**: BLOCKS real-time data flow
**Estimated Effort**: 4-7 hours

#### Issues:
1. **WebSocket state broadcasting not wired** (`backend/core/integration.py:998`)
   - `message_broadcaster.broadcast_game_state()` exists but never called
   - Integration layer routes through `websocket_manager` only
   - Frontend won't receive state updates

2. **Frame streaming not wired** (`backend/system/orchestrator.py:747`)
   - Vision frames never reach WebSocket broadcaster
   - `message_broadcaster.broadcast_frame()` exists but uncalled
   - No frame extraction in detection callback

3. **Frontend WebSocket connection incomplete** (`frontend/web/src/stores/VideoStore.ts:187`)
   - `VideoStore.connect()` uses HTTP API, not WebSocket
   - No WebSocket stream subscription on mount
   - Backend endpoint exists at `ws://localhost:8000/ws`

#### Tasks:
- [ ] **Wire WebSocket state broadcast** (1-2 hours)
  - Modify `integration.py:998` to call `message_broadcaster.broadcast_game_state()`

- [ ] **Wire frame streaming** (2-3 hours)
  - Add frame extraction in `orchestrator.py:747` callback
  - Call `message_broadcaster.broadcast_frame()` with frame data

- [ ] **Wire frontend WebSocket** (1-2 hours)
  - Modify `VideoStore.ts:187` to create WebSocket connection
  - Subscribe to `state` and `frame` streams

**Verification**: Start system, open frontend, confirm state updates and video frames appear in WebSocket messages

---

### 🟠 **HIGH: Frontend Video Feed in Calibration Wizard**
**Severity**: BLOCKS usable calibration
**Estimated Effort**: 2-4 hours

#### Issue:
- `CalibrationWizard.tsx:62-102` draws placeholder grid pattern
- No actual camera feed integration
- Users calibrating blind without visual feedback

#### Tasks:
- [ ] **Integrate camera feed** (2-3 hours)
  - Replace placeholder canvas with MJPEG stream or WebSocket frames
  - Add real-time detection overlay

- [ ] **Test calibration workflow** (1 hour)
  - Verify corner selection on real video
  - Verify color picker on real video

---

### 🟠 **HIGH: Vision Ball Number Recognition**
**Severity**: AFFECTS game state accuracy
**Estimated Effort**: 8-16 hours

#### Issue:
- `detection/balls.py:283-898` uses heuristic guesses
- Template matching returns shape-based guesses (lines 651-683)
- OCR uses shape analysis without real ML (lines 731-837)
- Cannot accurately identify specific balls

#### Tasks:
- [ ] **Implement real number recognition** (6-12 hours)
  - Add ML-based template matching or OCR
  - Train on real billiard ball images
  - Integrate with existing detection pipeline

- [ ] **Test with real footage** (2-4 hours)
  - Validate >90% accuracy requirement

---

### 🟡 **MEDIUM: Config Module Type Converter**
**Severity**: MAY AFFECT type coercion
**Estimated Effort**: 4-6 hours

#### Issue:
- `/Users/jchadwick/code/billiards-trainer/backend/config/utils/converter.py` has multiple `pass` statements
- Type conversion logic incomplete (lines 14, 121, 127, 133, 139, 146, 262, 288)
- Affects validation and auto-correction features

#### Tasks:
- [ ] **Implement type converter** (3-5 hours)
  - Complete conversion logic for all types
  - Add unit tests

- [ ] **Verify config validation** (1 hour)
  - Test with various config formats

---

### 🟡 **MEDIUM: Projector Text Rendering**
**Severity**: TEXT OVERLAYS won't work
**Estimated Effort**: 6-8 hours

#### Issue:
- `rendering/text.py:448` - "Fallback to rectangle placeholder"
- Actual text rendering to texture not implemented
- Affects shot statistics, probabilities, instructions display

#### Tasks:
- [ ] **Implement text rendering** (5-7 hours)
  - Surface-to-texture conversion after line 150
  - GPU text rendering
  - Multi-line text layout

- [ ] **Test text overlays** (1 hour)
  - Verify readable text on projector

---

## Priority 2: High-Impact Completions (Core Features)

### Frontend Testing Infrastructure (8-12 hours)
**Current**: Only 56 integration tests, no component tests
**Target**: 80%+ component coverage

- [ ] **Add React component tests** (6-8 hours)
  - Test all major components with React Testing Library
  - Focus on LiveView, CalibrationWizard, VideoStream

- [ ] **Add store tests** (2-4 hours)
  - Test all MobX stores
  - Test state updates and side effects

### Backend Test Coverage (6-10 hours)
**Gaps**: Projector module (90% missing), API middleware (60% missing)

- [ ] **Add projector module tests** (4-6 hours)
  - OpenGL renderer tests (with mocking)
  - Display manager tests
  - Trajectory rendering tests

- [ ] **Add API middleware tests** (2-4 hours)
  - Test rate limiting, security, logging
  - Test session monitoring

### Calibration Auto-Setup Improvements (4-6 hours)
**Goal**: Reduce setup time from 30 min to 5 min

- [ ] **Add default calibration profiles** (1-2 hours)
  - Ship with "Standard 9-Foot Table" preset
  - Include common lighting profiles

- [ ] **Implement auto-load last calibration** (1-2 hours)
  - Load most recent on startup
  - Show calibration status indicator

- [ ] **Add quick setup mode** (2 hours)
  - Express setup with defaults + 4 corner clicks
  - Skip camera calibration for initial testing

### Vision Module Completions (14-20 hours)

- [ ] **Format converters** (`tracking/integration.py:535-550`) - 6-8 hours
  - Implement OpenCV/YOLO/custom format conversion

- [ ] **Stripe ball detection** (`detection/balls.py:604-609`) - 8-12 hours
  - Enhance stripe vs solid detection with pattern recognition

### API Module Completions (9-14 hours)

- [ ] **Calibration session persistence** (`routes/calibration_original.py:105-114`) - 4-6 hours
  - Implement database load/save

- [ ] **Session monitor metrics** (`middleware/session_monitor.py:878-879`) - 3-4 hours
  - Implement blocked attempts tracking

- [ ] **Module log reading** (`routes/modules.py:522-560`) - 2-4 hours
  - Replace mock logs with actual file reading

---

## Priority 3: Feature Completions (Enhanced Functionality)

### Frontend Enhancements (3-4 days)

- [ ] **Configuration format support** (2-3 hours)
  - Add YAML/TOML parsing (`ConfigImportExport.tsx:201-207`)

- [ ] **State migration** (4-6 hours)
  - Implement version migration (`persistence.ts:271`)

- [ ] **Monitoring visualization** (8-10 hours)
  - Replace simulated metrics with real backend data
  - Chart.js already in dependencies

### Core Module Enhancements (13-17 hours)

- [ ] **Physics validation completeness** (6-8 hours)
  - Implement energy conservation checks (`physics/validation.py:327`)

- [ ] **Cache hit ratio tracking** (1 hour)
  - Add hit/miss counting (`__init__.py:765`)

- [ ] **Advanced physics** (6-8 hours)
  - Enhance masse shots, multi-ball optimization

### System Module Completions (6-9 hours)

- [ ] **State persistence on shutdown** (4-6 hours)
  - Implement actual state save (`api/shutdown.py:373-386`)

- [ ] **Auto-recovery testing** (2-3 hours)
  - Verify module restart functionality

---

## Priority 4: Production Readiness (4-7 days)

### Security Testing (2-3 days)
**Current**: No dedicated security tests

- [ ] **Add security tests**
  - Authentication bypass tests
  - Authorization tests
  - Input validation tests (SQLi, XSS, CSRF)

### Load Testing (2-3 days)

- [ ] **Sustained load tests**
  - 30 FPS for hours
  - 100+ concurrent WebSocket connections
  - Memory leak testing over days

### Docker Configuration (1-2 days)

- [ ] **Complete containerization**
  - Multi-stage Docker builds
  - Docker Compose for full stack
  - Health checks and monitoring

---

## Implementation Strategy

### **Phase 1: Critical Integration (Week 1 - 20-30 hours)**
**Goal**: End-to-end data flow working

**Days 1-2**:
1. Wire WebSocket integration (4-7 hours)
2. Add video feed to calibration wizard (2-4 hours)
3. Test end-to-end data flow (2-3 hours)
4. Fix any integration issues (2-4 hours)

**Days 3-5**:
5. Implement ball number recognition (8-16 hours) OR defer to Phase 2
6. Complete config type converter (4-6 hours)
7. Fix projector text rendering (6-8 hours)

**Outcome**: Fully functional system with all core features operational

---

### **Phase 2: Testing & Polish (Week 2 - 20-30 hours)**
**Goal**: Production-ready quality

1. Add frontend tests (8-12 hours)
2. Add backend test coverage (6-10 hours)
3. Implement calibration auto-setup (4-6 hours)
4. Complete API/Vision module gaps (20-30 hours)

**Outcome**: Well-tested, easy-to-setup system

---

### **Phase 3: Production Deployment (Week 3+ - flexible)**
**Goal**: Production deployment readiness

1. Security testing (2-3 days)
2. Load testing (2-3 days)
3. Docker configuration (1-2 days)
4. Documentation updates (1-2 days)

**Outcome**: Production-deployed system

---

## Key Strengths (Verified by Agents)

- ✅ **Real Implementations**: OpenCV homography, physics engine with RK4 integration, actual WebSocket
- ✅ **Production Architecture**: Comprehensive error handling, logging, health monitoring
- ✅ **Modern Stack**: TypeScript, React 19, FastAPI, OpenGL, MobX 6, Vite 6
- ✅ **Comprehensive Backend Testing**: 1,863 unit tests, 304 integration tests, 15 performance tests
- ✅ **Sophisticated Patterns**: Factory, Observer, Strategy patterns throughout
- ✅ **Excellent Documentation**: Clear docstrings, inline comments, specifications

## Critical Weaknesses (Agent Findings)

- ❌ **Integration Wiring**: WebSocket state broadcast, frame streaming not connected
- ❌ **Frontend Testing**: Only 56 tests, no component tests
- ❌ **Calibration UX**: Video feed placeholder, 30-minute manual setup
- ❌ **Ball Recognition**: Heuristic-based, needs ML implementation
- ❌ **Text Rendering**: Incomplete, affects projector overlays

---

## Completion Estimates

| Milestone | Completion % | Time Required |
|-----------|-------------|---------------|
| **Current Status** | ~80-85% | - |
| **After Phase 1** | ~92-94% | 1 week |
| **After Phase 2** | ~96-97% | 2 weeks |
| **After Phase 3** | ~99% | 3+ weeks |

**Time to Basic Functional System**: 1 week (Phase 1)
**Time to Production-Ready**: 2-3 weeks (Phases 1+2)
**Time to Feature-Complete**: 4+ weeks (All Phases)

---

## Revised Assessment

The comprehensive 10-agent analysis reveals a **well-architected system with substantial implementation** requiring targeted integration work rather than fundamental development.

**Key Insight**: Individual modules are largely complete with real algorithms (OpenCV, physics, rendering). The critical gap is **integration wiring** - connecting WebSocket broadcasters, adding video feeds to frontend, and completing a few incomplete implementations (ball recognition, text rendering, type converter).

**Most Critical Next Steps** (Priority Order):
1. Wire WebSocket integration (4-7 hours) - **BLOCKS real-time data**
2. Add video feed to calibration wizard (2-4 hours) - **BLOCKS usable calibration**
3. Implement ball number recognition (8-16 hours) - **AFFECTS game accuracy**
4. Complete config type converter (4-6 hours) - **MAY AFFECT validation**
5. Fix projector text rendering (6-8 hours) - **BLOCKS text overlays**

**Bottom Line**: This system is ~20-30 hours of focused work away from being fully functional, and 2-3 weeks away from production deployment with comprehensive testing.
