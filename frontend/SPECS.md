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

#### 2.1a Advanced Camera Calibration
- **FR-UI-016a**: Interactive geometric calibration with click-to-set points
- **FR-UI-016b**: ROI (Region of Interest) configuration with visual feedback
- **FR-UI-016c**: Perspective correction controls with real-time preview
- **FR-UI-016d**: Multiple camera backend selection and configuration
- **FR-UI-016e**: Camera exposure and white balance fine-tuning

#### 2.1b Projector Calibration Management
- **FR-UI-017a**: Interactive geometric calibration for projector alignment
- **FR-UI-017b**: Keystone correction controls with corner adjustment
- **FR-UI-017c**: Multi-monitor display configuration and selection
- **FR-UI-017d**: Perspective correction with test pattern overlay
- **FR-UI-017e**: Alignment validation with accuracy measurements

#### 2.1c Color & Detection Calibration
- **FR-UI-018a**: HSV color range calibration with live preview
- **FR-UI-018b**: Ball detection sensitivity adjustment
- **FR-UI-018c**: Table boundary detection parameter tuning
- **FR-UI-018d**: Cue stick detection configuration
- **FR-UI-018e**: Lighting condition adaptation controls

#### 2.2 Configuration Management
- **FR-UI-021**: Display all system settings in organized categories
- **FR-UI-022**: Provide form-based configuration editing
- **FR-UI-023**: Validate settings before applying
- **FR-UI-024**: Show setting descriptions and help text
- **FR-UI-025**: Support configuration import/export

#### 2.2a Advanced Configuration Features
- **FR-UI-021a**: Hot reload configuration without system restart
- **FR-UI-021b**: Configuration profile inheritance and merging
- **FR-UI-021c**: Real-time validation with detailed error feedback
- **FR-UI-021d**: Configuration rollback and version history
- **FR-UI-021e**: Diff views for configuration changes

#### 2.2b Environment & Source Management
- **FR-UI-022a**: Environment variable management interface
- **FR-UI-022b**: CLI argument visualization and editing
- **FR-UI-022c**: Configuration source precedence display
- **FR-UI-022d**: Multi-source configuration merging preview
- **FR-UI-022e**: Configuration override management

#### 2.2c Profile & Template Management
- **FR-UI-023a**: Named configuration profile creation and management
- **FR-UI-023b**: Profile switching with validation checks
- **FR-UI-023c**: Configuration template library
- **FR-UI-023d**: Conditional profile activation rules
- **FR-UI-023e**: Profile export/import with dependency tracking

#### 2.3 System Operations
- **FR-UI-026**: Start/stop detection processing
- **FR-UI-027**: Reset game state
- **FR-UI-028**: Clear trajectory predictions
- **FR-UI-029**: Switch between assistance levels
- **FR-UI-030**: Control projector output on/off

#### 2.3a Module-Specific Controls
- **FR-UI-026a**: Vision module camera device selection and switching
- **FR-UI-026b**: Physics engine parameter tuning (gravity, friction, spin)
- **FR-UI-026c**: Force estimation and collision detection controls
- **FR-UI-026d**: Projector display quality and effects configuration
- **FR-UI-026e**: Real-time parameter adjustment with live preview

#### 2.3b Advanced System Control
- **FR-UI-027a**: Hot reload configuration without system restart
- **FR-UI-027b**: Module dependency management and startup ordering
- **FR-UI-027c**: Performance profiling and optimization controls
- **FR-UI-027d**: Emergency shutdown procedures with data preservation
- **FR-UI-027e**: System recovery and diagnostic modes

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

### 6. Module Orchestration & System Control

#### 6.1 Module Management
- **FR-UI-071**: Display all backend modules with health status
- **FR-UI-072**: Start/stop individual modules (API, Core, Vision, Projector, Config)
- **FR-UI-073**: Restart modules with graceful shutdown procedures
- **FR-UI-074**: Show module initialization sequence and dependencies
- **FR-UI-075**: Display module startup/shutdown logs in real-time

#### 6.2 System Health Dashboard
- **FR-UI-076**: Show system-wide health overview with color-coded status
- **FR-UI-077**: Display inter-module communication status
- **FR-UI-078**: Monitor module resource usage (CPU, memory, GPU)
- **FR-UI-079**: Show module dependency graph with status indicators
- **FR-UI-080**: Alert on module failures with automatic restart options

#### 6.3 Service Management
- **FR-UI-081**: Control core services (authentication, database, WebSocket)
- **FR-UI-082**: Manage background processes and scheduled tasks
- **FR-UI-083**: Configure service discovery and load balancing
- **FR-UI-084**: Monitor service performance metrics
- **FR-UI-085**: Handle service failover and recovery procedures

### 7. System Diagnostics & Troubleshooting

#### 7.1 Automated Diagnostics
- **FR-UI-086**: Run automated system health checks
- **FR-UI-087**: Execute diagnostic procedures for each module
- **FR-UI-088**: Validate hardware connections and configuration
- **FR-UI-089**: Test network connectivity and bandwidth
- **FR-UI-090**: Perform end-to-end system validation

#### 7.2 Interactive Troubleshooting
- **FR-UI-091**: Provide step-by-step troubleshooting wizards
- **FR-UI-092**: Display system repair suggestions based on detected issues
- **FR-UI-093**: Guide users through calibration recovery procedures
- **FR-UI-094**: Offer automated fixes for common problems
- **FR-UI-095**: Provide escalation paths for complex issues

#### 7.3 Hardware Detection & Validation
- **FR-UI-096**: Auto-detect cameras, projectors, and displays
- **FR-UI-097**: Validate hardware compatibility and drivers
- **FR-UI-098**: Test camera capture capabilities and settings
- **FR-UI-099**: Verify projector output and alignment
- **FR-UI-100**: Monitor hardware health and temperature

#### 7.4 Network & Performance Diagnostics
- **FR-UI-101**: Test WebSocket connection stability
- **FR-UI-102**: Measure system latency and throughput
- **FR-UI-103**: Identify performance bottlenecks
- **FR-UI-104**: Monitor resource usage patterns
- **FR-UI-105**: Generate performance reports

### 8. Data Management & Persistence

#### 8.1 Backup Management
- **FR-UI-106**: Schedule automatic backups of configuration and calibration data
- **FR-UI-107**: Create manual backup snapshots with descriptions
- **FR-UI-108**: Restore from backup with preview and validation
- **FR-UI-109**: Manage backup retention policies
- **FR-UI-110**: Monitor backup integrity and status

#### 8.2 Data Migration & Export
- **FR-UI-111**: Export configuration in multiple formats (JSON, YAML, XML)
- **FR-UI-112**: Import configuration with validation and conflict resolution
- **FR-UI-113**: Migrate data between system versions
- **FR-UI-114**: Generate system reports and documentation
- **FR-UI-115**: Archive historical data with compression

#### 8.3 Storage Management
- **FR-UI-116**: Monitor disk usage and available space
- **FR-UI-117**: Clean up temporary files and logs
- **FR-UI-118**: Manage log rotation and retention
- **FR-UI-119**: Compress and archive old data
- **FR-UI-120**: Configure storage quotas and alerts

### 9. User Management & Access Control

#### 9.1 Role-Based Access Control
- **FR-UI-121**: Create and manage user roles (Admin, Operator, Viewer)
- **FR-UI-122**: Assign permissions for specific system functions
- **FR-UI-123**: Configure module-level access restrictions
- **FR-UI-124**: Manage API access keys and tokens
- **FR-UI-125**: Audit permission changes and access attempts

#### 9.2 Session & Activity Management
- **FR-UI-126**: Monitor active user sessions
- **FR-UI-127**: Configure session timeouts and security policies
- **FR-UI-128**: Log user activities and system changes
- **FR-UI-129**: Generate audit trails and compliance reports
- **FR-UI-130**: Manage concurrent session limits

#### 9.3 User Preferences & Profiles
- **FR-UI-131**: Save and restore user interface preferences
- **FR-UI-132**: Manage personal configuration profiles
- **FR-UI-133**: Configure notification preferences
- **FR-UI-134**: Customize dashboard layouts per user
- **FR-UI-135**: Synchronize preferences across devices

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
  modules: ModuleState[];
  services: ServiceState[];
  diagnostics: DiagnosticsState;
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
  modules: ModuleConfig[];
  profiles: ConfigurationProfile[];
}

// New interfaces for system management
interface ModuleState {
  id: string;
  name: 'api' | 'core' | 'vision' | 'projector' | 'config';
  status: 'running' | 'stopped' | 'starting' | 'stopping' | 'error';
  health: 'healthy' | 'warning' | 'critical';
  uptime: number;
  resources: ResourceUsage;
  dependencies: string[];
  logs: LogEntry[];
}

interface ServiceState {
  id: string;
  name: string;
  type: 'authentication' | 'database' | 'websocket' | 'background';
  status: 'active' | 'inactive' | 'failed';
  pid?: number;
  port?: number;
  startTime: Date;
}

interface DiagnosticsState {
  lastCheck: Date;
  overallHealth: 'healthy' | 'warning' | 'critical';
  checks: DiagnosticCheck[];
  recommendations: string[];
}

interface DiagnosticCheck {
  id: string;
  name: string;
  category: 'hardware' | 'network' | 'performance' | 'configuration';
  status: 'pass' | 'warning' | 'fail';
  message: string;
  details?: any;
}

interface ResourceUsage {
  cpu: number;
  memory: number;
  gpu?: number;
  network: NetworkUsage;
}

interface NetworkUsage {
  bytesIn: number;
  bytesOut: number;
  connectionsActive: number;
}

interface ConfigurationProfile {
  id: string;
  name: string;
  description: string;
  active: boolean;
  inherits?: string[];
  settings: Record<string, any>;
  created: Date;
  modified: Date;
}

interface BackupInfo {
  id: string;
  name: string;
  description: string;
  timestamp: Date;
  size: number;
  type: 'automatic' | 'manual';
  integrity: 'verified' | 'corrupted' | 'unknown';
}

interface UserRole {
  id: string;
  name: 'admin' | 'operator' | 'viewer';
  permissions: Permission[];
  modules: string[];
}

interface Permission {
  resource: string;
  actions: ('read' | 'write' | 'execute' | 'delete')[];
}

interface UserSession {
  id: string;
  userId: string;
  startTime: Date;
  lastActivity: Date;
  ipAddress: string;
  userAgent: string;
  active: boolean;
}

interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  layout: LayoutConfig;
  videoQuality: 'auto' | 'high' | 'medium' | 'low';
  notifications: boolean;
  shortcuts: KeyboardShortcuts;
  dashboardLayout: DashboardLayout;
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

  // System management methods
  getModules(): Promise<ModuleState[]>;
  startModule(moduleId: string): Promise<void>;
  stopModule(moduleId: string): Promise<void>;
  restartModule(moduleId: string): Promise<void>;
  getModuleLogs(moduleId: string, limit?: number): Promise<LogEntry[]>;

  // Service management
  getServices(): Promise<ServiceState[]>;
  controlService(serviceId: string, action: 'start' | 'stop' | 'restart'): Promise<void>;

  // Diagnostics
  runDiagnostics(): Promise<DiagnosticsState>;
  runSpecificCheck(checkId: string): Promise<DiagnosticCheck>;
  getSystemHealth(): Promise<{ overall: string; details: any }>;

  // Configuration profiles
  getProfiles(): Promise<ConfigurationProfile[]>;
  createProfile(profile: Omit<ConfigurationProfile, 'id' | 'created' | 'modified'>): Promise<string>;
  updateProfile(id: string, updates: Partial<ConfigurationProfile>): Promise<void>;
  deleteProfile(id: string): Promise<void>;
  activateProfile(id: string): Promise<void>;

  // Backup management
  getBackups(): Promise<BackupInfo[]>;
  createBackup(name: string, description?: string): Promise<string>;
  restoreBackup(backupId: string): Promise<void>;
  deleteBackup(backupId: string): Promise<void>;

  // User management
  getUserSessions(): Promise<UserSession[]>;
  terminateSession(sessionId: string): Promise<void>;
  getUserRoles(): Promise<UserRole[]>;
  updateUserRole(userId: string, roleId: string): Promise<void>;
}

// WebSocket message types
type WSMessage =
  | { type: 'frame'; data: FrameData }
  | { type: 'state'; data: DetectionState }
  | { type: 'trajectory'; data: TrajectoryData }
  | { type: 'metrics'; data: PerformanceMetrics }
  | { type: 'alert'; data: Alert }
  | { type: 'config'; data: Configuration }
  | { type: 'moduleStatus'; data: ModuleState }
  | { type: 'serviceStatus'; data: ServiceState }
  | { type: 'diagnostics'; data: DiagnosticsState }
  | { type: 'resourceUsage'; data: ResourceUsage }
  | { type: 'systemHealth'; data: { overall: string; details: any } }
  | { type: 'backup'; data: BackupInfo }
  | { type: 'userSession'; data: UserSession }
  | { type: 'configurationChange'; data: { profileId: string; changes: any } };

// REST endpoints
const API_ENDPOINTS = {
  health: '/api/v1/health',
  config: '/api/v1/config',
  calibration: '/api/v1/calibration',
  gameState: '/api/v1/game/state',
  system: '/api/v1/system',
  profiles: '/api/v1/profiles',

  // System management endpoints
  modules: '/api/v1/modules',
  moduleControl: '/api/v1/modules/{id}/control',
  moduleLogs: '/api/v1/modules/{id}/logs',

  services: '/api/v1/services',
  serviceControl: '/api/v1/services/{id}/control',

  diagnostics: '/api/v1/diagnostics',
  diagnosticsRun: '/api/v1/diagnostics/run',
  diagnosticsCheck: '/api/v1/diagnostics/check/{id}',

  configProfiles: '/api/v1/config/profiles',
  configProfile: '/api/v1/config/profiles/{id}',
  profileActivate: '/api/v1/config/profiles/{id}/activate',

  backups: '/api/v1/backups',
  backup: '/api/v1/backups/{id}',
  backupRestore: '/api/v1/backups/{id}/restore',

  users: '/api/v1/users',
  userSessions: '/api/v1/users/sessions',
  userSession: '/api/v1/users/sessions/{id}',
  userRoles: '/api/v1/users/roles'
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
