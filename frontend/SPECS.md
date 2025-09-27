# Frontend Web Application Specification

## Module Purpose

The Frontend Web Application provides a zero-installation, browser-based interface for system control, monitoring, and configuration. It delivers real-time visualization of the pool table, trajectory predictions, and system settings through an intuitive, responsive web interface accessible from any device.

## Functional Requirements

### 1. Live View

#### 1.1 Video Streaming
- **FR-UI-001**: Display real-time camera feed at 30+ FPS
- **FR-UI-002**: Support adaptive video quality based on bandwidth
- **FR-UI-003**: Show overlay of detected objects on video
- **FR-UI-004**: Enable full-screen video mode
- **FR-UI-005**: Provide zoom and pan controls

#### 1.2 Detection Overlay
- **FR-UI-006**: Render ball positions and IDs on video
- **FR-UI-007**: Display table boundary detection
- **FR-UI-008**: Show cue stick position and angle
- **FR-UI-009**: Highlight pocket locations
- **FR-UI-010**: Display confidence scores for detections

#### 1.3 Trajectory Display
- **FR-UI-011**: Render predicted ball paths in real-time
- **FR-UI-012**: Show collision points and bounces
- **FR-UI-013**: Display different trajectory types with colors
- **FR-UI-014**: Animate trajectory updates smoothly
- **FR-UI-015**: Show probability/confidence for predictions

### 2. System Control

#### 2.1 Calibration Interface
- **FR-UI-016**: Provide camera calibration wizard
- **FR-UI-017**: Allow table corner adjustment
- **FR-UI-018**: Support color threshold tuning
- **FR-UI-019**: Display calibration test patterns
- **FR-UI-020**: Save and load calibration profiles

#### 2.2 Configuration Management
- **FR-UI-021**: Display all system settings in organized categories
- **FR-UI-022**: Provide form-based configuration editing
- **FR-UI-023**: Validate settings before applying
- **FR-UI-024**: Show setting descriptions and help text
- **FR-UI-025**: Support configuration import/export

#### 2.3 System Operations
- **FR-UI-026**: Start/stop detection processing
- **FR-UI-027**: Reset game state
- **FR-UI-028**: Clear trajectory predictions
- **FR-UI-029**: Switch between assistance levels
- **FR-UI-030**: Control projector output on/off

### 3. Monitoring Dashboard

#### 3.1 Performance Metrics
- **FR-UI-031**: Display real-time FPS counter
- **FR-UI-032**: Show processing latency graph
- **FR-UI-033**: Monitor CPU and memory usage
- **FR-UI-034**: Track detection accuracy metrics
- **FR-UI-035**: Display network bandwidth usage

#### 3.2 System Status
- **FR-UI-036**: Show connection status for all components
- **FR-UI-037**: Display camera and projector health
- **FR-UI-038**: List active WebSocket connections
- **FR-UI-039**: Show error and warning messages
- **FR-UI-040**: Provide system uptime information

#### 3.3 Event Log
- **FR-UI-041**: Display scrollable event history
- **FR-UI-042**: Filter events by type and severity
- **FR-UI-043**: Search events by keyword
- **FR-UI-044**: Export event logs
- **FR-UI-045**: Clear event history

### 4. User Interface

#### 4.1 Layout Management
- **FR-UI-046**: Provide responsive layout for all screen sizes
- **FR-UI-047**: Support customizable dashboard panels
- **FR-UI-048**: Enable drag-and-drop panel arrangement
- **FR-UI-049**: Save and restore layout preferences
- **FR-UI-050**: Support dark and light themes

#### 4.2 Navigation
- **FR-UI-051**: Provide main navigation menu
- **FR-UI-052**: Support keyboard shortcuts
- **FR-UI-053**: Enable breadcrumb navigation
- **FR-UI-054**: Provide context-sensitive help
- **FR-UI-055**: Support browser back/forward

#### 4.3 Accessibility
- **FR-UI-056**: Support screen readers (ARIA labels)
- **FR-UI-057**: Provide keyboard-only navigation
- **FR-UI-058**: Support high contrast mode
- **FR-UI-059**: Enable font size adjustment
- **FR-UI-060**: Provide alternative text for images

### 5. Multi-Device Support

#### 5.1 Responsive Design
- **FR-UI-061**: Adapt layout for mobile phones
- **FR-UI-062**: Optimize for tablet displays
- **FR-UI-063**: Support desktop widescreen layouts
- **FR-UI-064**: Handle portrait and landscape orientations
- **FR-UI-065**: Provide touch-friendly controls

#### 5.2 Cross-Browser Compatibility
- **FR-UI-066**: Support Chrome 90+
- **FR-UI-067**: Support Firefox 88+
- **FR-UI-068**: Support Safari 14+
- **FR-UI-069**: Support Edge 90+
- **FR-UI-070**: Graceful degradation for older browsers

## Non-Functional Requirements

### Performance Requirements
- **NFR-UI-001**: Initial page load < 2 seconds
- **NFR-UI-002**: Time to interactive < 3 seconds
- **NFR-UI-003**: 60 FPS UI animations
- **NFR-UI-004**: < 100ms response to user interactions
- **NFR-UI-005**: Smooth video playback without stuttering

### Scalability Requirements
- **NFR-UI-006**: Support 50+ concurrent viewers
- **NFR-UI-007**: Handle 10MB/s video streaming per client
- **NFR-UI-008**: Efficient WebSocket message handling
- **NFR-UI-009**: Lazy loading of components
- **NFR-UI-010**: CDN-ready static assets

### Reliability Requirements
- **NFR-UI-011**: Automatic reconnection on connection loss
- **NFR-UI-012**: Graceful error handling with user feedback
- **NFR-UI-013**: Local storage for offline capability
- **NFR-UI-014**: Session persistence across refreshes
- **NFR-UI-015**: No memory leaks during extended use

### Security Requirements
- **NFR-UI-016**: Secure WebSocket connections (WSS)
- **NFR-UI-017**: HTTPS-only deployment
- **NFR-UI-018**: XSS prevention
- **NFR-UI-019**: CSRF protection
- **NFR-UI-020**: Content Security Policy headers

## Interface Specifications

### Component Architecture

```typescript
// Core interfaces for the frontend application

interface SystemState {
  connection: ConnectionState;
  camera: CameraState;
  projector: ProjectorState;
  detection: DetectionState;
  performance: PerformanceMetrics;
}

interface ConnectionState {
  status: 'connected' | 'connecting' | 'disconnected' | 'error';
  latency: number;
  lastHeartbeat: Date;
  reconnectAttempts: number;
}

interface CameraState {
  streaming: boolean;
  resolution: [number, number];
  fps: number;
  exposure: number;
  brightness: number;
}

interface DetectionState {
  balls: Ball[];
  cue: CueStick | null;
  table: Table;
  trajectories: Trajectory[];
  timestamp: number;
  frameNumber: number;
}

interface Ball {
  id: string;
  position: Point2D;
  radius: number;
  type: 'cue' | 'solid' | 'stripe' | 'eight';
  number?: number;
  velocity: Vector2D;
  confidence: number;
}

interface CueStick {
  tipPosition: Point2D;
  angle: number;
  elevation: number;
  detected: boolean;
  confidence: number;
}

interface Trajectory {
  ballId: string;
  points: Point2D[];
  collisions: Collision[];
  type: 'primary' | 'reflection' | 'collision';
  probability: number;
}

interface Configuration {
  system: SystemConfig;
  camera: CameraConfig;
  vision: VisionConfig;
  display: DisplayConfig;
  assistance: AssistanceConfig;
}

interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  layout: LayoutConfig;
  videoQuality: 'auto' | 'high' | 'medium' | 'low';
  notifications: boolean;
  shortcuts: KeyboardShortcuts;
}
```

### API Client Interface

```typescript
// WebSocket and REST API client

class APIClient {
  private ws: WebSocket;
  private rest: RestClient;

  // WebSocket methods
  connect(url: string): Promise<void>;
  disconnect(): void;
  subscribe(event: string, callback: Function): void;
  unsubscribe(event: string): void;
  send(type: string, data: any): void;

  // REST methods
  getConfig(): Promise<Configuration>;
  updateConfig(config: Partial<Configuration>): Promise<void>;
  startCalibration(): Promise<CalibrationSession>;
  getSystemStatus(): Promise<SystemState>;
  exportData(format: 'json' | 'csv'): Promise<Blob>;
}

// WebSocket message types
type WSMessage =
  | { type: 'frame'; data: FrameData }
  | { type: 'state'; data: DetectionState }
  | { type: 'trajectory'; data: TrajectoryData }
  | { type: 'metrics'; data: PerformanceMetrics }
  | { type: 'alert'; data: Alert }
  | { type: 'config'; data: Configuration };

// REST endpoints
const API_ENDPOINTS = {
  health: '/api/v1/health',
  config: '/api/v1/config',
  calibration: '/api/v1/calibration',
  gameState: '/api/v1/game/state',
  system: '/api/v1/system',
  profiles: '/api/v1/profiles'
};
```

### Component Structure

```typescript
// Main React/Vue component structure

// React Example
interface AppProps {
  apiUrl: string;
  wsUrl: string;
}

const App: React.FC<AppProps> = ({ apiUrl, wsUrl }) => {
  return (
    <ThemeProvider theme={theme}>
      <Router>
        <Layout>
          <Header />
          <Navigation />
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/live" element={<LiveView />} />
            <Route path="/config" element={<Configuration />} />
            <Route path="/calibration" element={<Calibration />} />
            <Route path="/monitoring" element={<Monitoring />} />
          </Routes>
        </Layout>
      </Router>
    </ThemeProvider>
  );
};

// Vue Example
export default defineComponent({
  name: 'App',
  setup() {
    const store = useStore();
    const api = useAPI();

    onMounted(() => {
      api.connect();
      store.dispatch('initialize');
    });

    return {
      // Component logic
    };
  }
});
```

### State Management

```typescript
// Redux/Vuex store structure

interface RootState {
  system: SystemState;
  detection: DetectionState;
  config: Configuration;
  ui: UIState;
  user: UserState;
}

// Actions
const actions = {
  // System
  CONNECT: 'system/connect',
  DISCONNECT: 'system/disconnect',
  UPDATE_STATUS: 'system/updateStatus',

  // Detection
  UPDATE_FRAME: 'detection/updateFrame',
  UPDATE_TRAJECTORIES: 'detection/updateTrajectories',
  RESET_STATE: 'detection/reset',

  // Configuration
  LOAD_CONFIG: 'config/load',
  UPDATE_CONFIG: 'config/update',
  SAVE_CONFIG: 'config/save',

  // UI
  SET_THEME: 'ui/setTheme',
  TOGGLE_PANEL: 'ui/togglePanel',
  SET_LAYOUT: 'ui/setLayout'
};

// Selectors
const selectors = {
  isConnected: (state: RootState) => state.system.connection.status === 'connected',
  getCurrentFPS: (state: RootState) => state.system.performance.fps,
  getDetectedBalls: (state: RootState) => state.detection.balls,
  getActiveTheme: (state: RootState) => state.ui.theme
};
```

## UI/UX Design Specifications

### Visual Design System

```scss
// Design tokens and theme

:root {
  // Colors
  --color-primary: #2E7D32;      // Green (pool table)
  --color-secondary: #1976D2;    // Blue
  --color-success: #4CAF50;
  --color-warning: #FF9800;
  --color-error: #F44336;

  // Neutrals
  --color-bg: #FFFFFF;
  --color-surface: #F5F5F5;
  --color-text: #212121;
  --color-text-secondary: #757575;

  // Spacing
  --space-xs: 4px;
  --space-sm: 8px;
  --space-md: 16px;
  --space-lg: 24px;
  --space-xl: 32px;

  // Typography
  --font-primary: 'Inter', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;

  // Shadows
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.12);
  --shadow-md: 0 4px 6px rgba(0,0,0,0.16);
  --shadow-lg: 0 10px 20px rgba(0,0,0,0.19);

  // Transitions
  --transition-fast: 150ms ease;
  --transition-base: 250ms ease;
  --transition-slow: 350ms ease;
}

// Dark theme overrides
[data-theme="dark"] {
  --color-bg: #121212;
  --color-surface: #1E1E1E;
  --color-text: #FFFFFF;
  --color-text-secondary: #B0B0B0;
}
```

### Layout Structure

```html
<!-- Main layout template -->
<div class="app-container">
  <!-- Header -->
  <header class="app-header">
    <div class="logo">Billiards Trainer</div>
    <nav class="main-nav">
      <a href="/">Dashboard</a>
      <a href="/live">Live View</a>
      <a href="/config">Configuration</a>
    </nav>
    <div class="header-actions">
      <button class="connection-status">Connected</button>
      <button class="theme-toggle">🌙</button>
    </div>
  </header>

  <!-- Main Content -->
  <main class="app-main">
    <!-- Sidebar (optional) -->
    <aside class="app-sidebar">
      <!-- Context menu -->
    </aside>

    <!-- Content Area -->
    <section class="app-content">
      <!-- Page content -->
    </section>

    <!-- Info Panel (optional) -->
    <aside class="app-panel">
      <!-- Additional info -->
    </aside>
  </main>

  <!-- Footer -->
  <footer class="app-footer">
    <div class="status-bar">
      <span>FPS: 30</span>
      <span>Latency: 15ms</span>
      <span>Detection: Active</span>
    </div>
  </footer>
</div>
```

## Success Criteria

### Functional Success Criteria

1. **Live View**
   - Video displays at consistent 30+ FPS
   - Overlays render without lag
   - Trajectories update in real-time
   - All detections visible and labeled

2. **Configuration**
   - All settings accessible and editable
   - Changes apply immediately
   - Validation prevents invalid settings
   - Import/export works reliably

3. **Monitoring**
   - Metrics update every second
   - Historical graphs display correctly
   - Alerts appear within 1 second
   - Logs searchable and filterable

4. **Responsive Design**
   - Works on all screen sizes
   - Touch controls function properly
   - Layout adapts appropriately
   - No horizontal scrolling on mobile

### Performance Success Criteria

1. **Loading Performance**
   - First Contentful Paint < 1.5s
   - Time to Interactive < 3s
   - Largest Contentful Paint < 2.5s
   - Cumulative Layout Shift < 0.1

2. **Runtime Performance**
   - 60 FPS for UI animations
   - < 100ms interaction response
   - No jank during scrolling
   - Memory usage < 200MB

3. **Network Performance**
   - WebSocket latency < 50ms
   - Video streaming adaptive
   - Efficient data transfer
   - Automatic reconnection < 5s

### Usability Success Criteria

1. **User Experience**
   - Intuitive navigation
   - Clear visual hierarchy
   - Consistent interactions
   - Helpful error messages

2. **Accessibility**
   - WCAG 2.1 AA compliance
   - Keyboard navigation complete
   - Screen reader compatible
   - Color contrast sufficient

## Testing Requirements

### Unit Testing
- Test all components individually
- Mock API responses
- Test state management
- Validate prop types
- Coverage target: 80%

### Integration Testing
- Test component interactions
- Verify API communication
- Test routing and navigation
- Validate form submissions
- Test error scenarios

### E2E Testing
- Test complete user flows
- Verify video streaming
- Test configuration changes
- Validate calibration process
- Test multi-device scenarios

### Performance Testing
- Measure loading times
- Profile rendering performance
- Test with slow networks
- Validate memory usage
- Stress test with multiple users

### Cross-Browser Testing
- Test on Chrome, Firefox, Safari, Edge
- Verify mobile browsers
- Test different viewports
- Validate touch interactions
- Check feature compatibility

## Implementation Guidelines

### Project Structure
```
frontend/
├── src/
│   ├── components/
│   │   ├── common/       # Reusable components
│   │   ├── live/         # Live view components
│   │   ├── config/       # Configuration components
│   │   └── monitoring/   # Dashboard components
│   ├── views/
│   │   ├── Dashboard.tsx
│   │   ├── LiveView.tsx
│   │   ├── Configuration.tsx
│   │   └── Calibration.tsx
│   ├── services/
│   │   ├── api.ts        # API client
│   │   ├── websocket.ts  # WebSocket handler
│   │   └── auth.ts       # Authentication
│   ├── store/
│   │   ├── index.ts      # Store setup
│   │   ├── slices/       # Redux slices / Vuex modules
│   │   └── selectors.ts  # Selectors
│   ├── hooks/            # Custom React hooks
│   ├── utils/            # Utility functions
│   ├── styles/           # Global styles
│   └── types/            # TypeScript definitions
├── public/               # Static assets
├── tests/               # Test files
└── package.json
```

### Key Dependencies
- **React/Vue**: UI framework
- **TypeScript**: Type safety
- **Redux/Vuex**: State management
- **Socket.io-client**: WebSocket
- **Axios**: HTTP client
- **Chart.js**: Graphs
- **Tailwind/MUI**: UI components

### Development Priorities
1. Set up project structure
2. Implement WebSocket connection
3. Create live view with video
4. Add detection overlays
5. Build configuration interface
6. Implement monitoring dashboard
7. Add calibration wizard
8. Optimize performance