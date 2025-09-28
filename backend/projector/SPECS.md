# Projector Module Specification

## Module Purpose

The Projector module is responsible for rendering and displaying augmented reality overlays directly onto the pool table surface via a physical projector. It receives trajectory and game state data from the backend, performs geometric corrections for proper alignment, and generates real-time visual feedback that enhances the playing experience.

## Functional Requirements

### 1. Display Management

#### 1.1 Projector Interface
- **FR-PROJ-001**: Initialize projector display output (fullscreen or window)
- **FR-PROJ-002**: Detect and configure available display devices
- **FR-PROJ-003**: Support multiple display resolutions (720p, 1080p, 4K)
- **FR-PROJ-004**: Handle projector disconnection and reconnection
- **FR-PROJ-005**: Provide projector status and health monitoring

#### 1.2 Rendering Pipeline
- **FR-PROJ-006**: Render trajectory lines with configurable styles
- **FR-PROJ-007**: Display collision indicators at impact points
- **FR-PROJ-008**: Show ball path predictions with fade effects
- **FR-PROJ-009**: Render aiming assistance guides
- **FR-PROJ-010**: Display success probability indicators

### 2. Geometric Calibration

#### 2.1 Perspective Correction
- **FR-PROJ-011**: Perform 4-point perspective transformation
- **FR-PROJ-012**: Support keystone correction for angled projection
- **FR-PROJ-013**: Compensate for barrel/pincushion distortion
- **FR-PROJ-014**: Handle non-perpendicular projector positioning
- **FR-PROJ-015**: Save and load calibration profiles

#### 2.2 Alignment Calibration
- **FR-PROJ-016**: Display calibration grid pattern
- **FR-PROJ-017**: Allow manual corner point adjustment
- **FR-PROJ-018**: Provide automatic alignment detection
- **FR-PROJ-019**: Support fine-tuning with arrow keys
- **FR-PROJ-020**: Validate calibration accuracy with test patterns

### 3. Visual Rendering

#### 3.1 Trajectory Visualization
- **FR-PROJ-021**: Draw primary ball trajectory lines
- **FR-PROJ-022**: Show reflection paths off cushions
- **FR-PROJ-023**: Display collision transfer trajectories
- **FR-PROJ-024**: Indicate pocket entry paths
- **FR-PROJ-025**: Show spin effect curves

#### 3.2 Visual Effects
- **FR-PROJ-026**: Apply gradient colors to indicate force/speed
- **FR-PROJ-027**: Animate trajectory appearance/disappearance
- **FR-PROJ-028**: Pulse or highlight recommended shots
- **FR-PROJ-029**: Show ghost balls for aiming reference
- **FR-PROJ-030**: Display impact zones with transparency

#### 3.3 Information Overlays
- **FR-PROJ-031**: Show shot difficulty indicators
- **FR-PROJ-032**: Display angle measurements
- **FR-PROJ-033**: Present force/power recommendations
- **FR-PROJ-034**: Show success probability percentages
- **FR-PROJ-035**: Display system messages and alerts

### 4. Real-time Updates

#### 4.1 Data Reception
- **FR-PROJ-036**: Connect to backend via WebSocket
- **FR-PROJ-037**: Receive trajectory updates in real-time
- **FR-PROJ-038**: Handle game state changes
- **FR-PROJ-039**: Process configuration updates
- **FR-PROJ-040**: Manage connection failures and reconnection

#### 4.2 Synchronization
- **FR-PROJ-041**: Synchronize display with camera frame rate
- **FR-PROJ-042**: Maintain smooth animation at 60 FPS
- **FR-PROJ-043**: Minimize latency between detection and display
- **FR-PROJ-044**: Handle frame dropping gracefully
- **FR-PROJ-045**: Interpolate between updates for smoothness

### 5. Customization

#### 5.1 Visual Preferences
- **FR-PROJ-046**: Configure line thickness and styles
- **FR-PROJ-047**: Adjust color schemes and themes
- **FR-PROJ-048**: Set transparency/opacity levels
- **FR-PROJ-049**: Choose animation speeds
- **FR-PROJ-050**: Select information display modes

#### 5.2 Assistance Levels
- **FR-PROJ-051**: Show/hide different trajectory types
- **FR-PROJ-052**: Adjust assistance based on skill level
- **FR-PROJ-053**: Toggle specific visual aids on/off
- **FR-PROJ-054**: Configure information density
- **FR-PROJ-055**: Support practice vs competition modes

## Non-Functional Requirements

### Performance Requirements
- **NFR-PROJ-001**: Render at consistent 60 FPS minimum
- **NFR-PROJ-002**: Display latency < 16ms (one frame at 60 FPS)
- **NFR-PROJ-003**: Support 4K resolution output
- **NFR-PROJ-004**: Use < 500MB memory for rendering
- **NFR-PROJ-005**: GPU utilization < 50% on modern hardware

### Visual Quality Requirements
- **NFR-PROJ-006**: Anti-aliased line rendering
- **NFR-PROJ-007**: Smooth gradient transitions
- **NFR-PROJ-008**: No visible flicker or tearing
- **NFR-PROJ-009**: Consistent brightness across projection
- **NFR-PROJ-010**: Color accuracy within 10% of target

### Reliability Requirements
- **NFR-PROJ-011**: No display crashes during operation
- **NFR-PROJ-012**: Automatic recovery from GPU errors
- **NFR-PROJ-013**: Graceful degradation under load
- **NFR-PROJ-014**: Maintain calibration across restarts
- **NFR-PROJ-015**: Handle projector power cycles

### Compatibility Requirements
- **NFR-PROJ-016**: Support Windows, Linux, macOS
- **NFR-PROJ-017**: Work with OpenGL 3.3+ or DirectX 11+
- **NFR-PROJ-018**: Compatible with common projector models
- **NFR-PROJ-019**: Support HDMI, DisplayPort, VGA outputs
- **NFR-PROJ-020**: Handle multiple GPU configurations

## Interface Specifications

### Projector Module Interface

```python
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

class DisplayMode(Enum):
    FULLSCREEN = "fullscreen"
    WINDOW = "window"
    BORDERLESS = "borderless"

class LineStyle(Enum):
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    ARROW = "arrow"

class RenderQuality(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

@dataclass
class Point2D:
    x: float
    y: float

@dataclass
class Line:
    """Rendered line segment"""
    start: Point2D
    end: Point2D
    color: Tuple[int, int, int, int]  # RGBA
    width: float
    style: LineStyle
    glow: bool = False
    animated: bool = False

@dataclass
class Circle:
    """Rendered circle (for balls, impact points)"""
    center: Point2D
    radius: float
    color: Tuple[int, int, int, int]
    filled: bool
    width: float = 1.0

@dataclass
class Text:
    """Rendered text overlay"""
    position: Point2D
    content: str
    size: int
    color: Tuple[int, int, int, int]
    background: Optional[Tuple[int, int, int, int]] = None
    anchor: str = "center"  # center, left, right

@dataclass
class CalibrationPoints:
    """Projector calibration corner points"""
    top_left: Point2D
    top_right: Point2D
    bottom_right: Point2D
    bottom_left: Point2D

@dataclass
class RenderFrame:
    """Complete frame to be rendered"""
    trajectories: List[Line]
    collision_points: List[Circle]
    ghost_balls: List[Circle]
    text_overlays: List[Text]
    highlight_zones: List[Dict]
    timestamp: float

class ProjectorModule:
    """Main projector interface"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize projector module with configuration"""
        pass

    def start_display(self, mode: DisplayMode = DisplayMode.FULLSCREEN) -> bool:
        """Start projector display output"""
        pass

    def stop_display(self) -> None:
        """Stop projector display"""
        pass

    def calibrate(self, interactive: bool = True) -> CalibrationPoints:
        """Run calibration procedure"""
        pass

    def set_calibration(self, points: CalibrationPoints) -> None:
        """Apply calibration points"""
        pass

    def render_frame(self, frame: RenderFrame) -> None:
        """Render a complete frame"""
        pass

    def render_trajectory(self,
                         points: List[Point2D],
                         color: Tuple[int, int, int, int],
                         width: float = 2.0,
                         style: LineStyle = LineStyle.SOLID) -> None:
        """Render a trajectory path"""
        pass

    def render_collision(self,
                        position: Point2D,
                        radius: float = 20,
                        color: Tuple[int, int, int, int] = (255, 0, 0, 128)) -> None:
        """Render collision indicator"""
        pass

    def render_text(self,
                   text: str,
                   position: Point2D,
                   size: int = 24,
                   color: Tuple[int, int, int, int] = (255, 255, 255, 255)) -> None:
        """Render text overlay"""
        pass

    def clear_display(self) -> None:
        """Clear all rendered content"""
        pass

    def set_render_quality(self, quality: RenderQuality) -> None:
        """Set rendering quality level"""
        pass

    def get_display_info(self) -> Dict:
        """Get display device information"""
        pass

    def connect_to_backend(self, url: str) -> bool:
        """Connect to backend WebSocket"""
        pass

    def disconnect_from_backend(self) -> None:
        """Disconnect from backend"""
        pass
```

### Configuration Schema

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class DisplayConfig(BaseModel):
    """Display output configuration"""
    mode: str = "fullscreen"  # fullscreen, window, borderless
    monitor_index: int = 0  # Which monitor to use
    resolution: List[int] = [1920, 1080]
    refresh_rate: int = 60
    vsync: bool = True
    gamma: float = Field(1.0, ge=0.5, le=2.0)
    brightness: float = Field(1.0, ge=0, le=2.0)
    contrast: float = Field(1.0, ge=0, le=2.0)

class CalibrationConfig(BaseModel):
    """Geometric calibration configuration"""
    calibration_points: Optional[List[List[float]]] = None
    keystone_horizontal: float = Field(0.0, ge=-1.0, le=1.0)
    keystone_vertical: float = Field(0.0, ge=-1.0, le=1.0)
    rotation: float = Field(0.0, ge=-180, le=180)
    barrel_distortion: float = Field(0.0, ge=-1.0, le=1.0)
    edge_blend: Dict[str, float] = {
        "left": 0, "right": 0, "top": 0, "bottom": 0
    }

class VisualConfig(BaseModel):
    """Visual rendering configuration"""
    # Line rendering
    trajectory_width: float = Field(3.0, ge=1, le=10)
    trajectory_color: Tuple[int, int, int] = (0, 255, 0)
    collision_color: Tuple[int, int, int] = (255, 0, 0)
    reflection_color: Tuple[int, int, int] = (255, 255, 0)

    # Effects
    enable_glow: bool = True
    glow_intensity: float = Field(0.5, ge=0, le=1)
    enable_animations: bool = True
    animation_speed: float = Field(1.0, ge=0.1, le=5.0)
    fade_duration: float = Field(0.5, ge=0, le=2.0)

    # Transparency
    trajectory_opacity: float = Field(0.8, ge=0, le=1)
    ghost_ball_opacity: float = Field(0.4, ge=0, le=1)
    overlay_opacity: float = Field(0.6, ge=0, le=1)

class RenderingConfig(BaseModel):
    """Rendering engine configuration"""
    renderer: str = "opengl"  # opengl, directx, vulkan, software
    quality: str = "high"  # low, medium, high, ultra
    antialiasing: str = "4x"  # none, 2x, 4x, 8x, 16x
    texture_filtering: str = "anisotropic"
    max_fps: int = 60
    buffer_frames: int = 2
    use_gpu: bool = True

class NetworkConfig(BaseModel):
    """Network connection configuration"""
    backend_url: str = "ws://localhost:8000/ws"
    reconnect_interval: int = 5  # seconds
    heartbeat_interval: int = 30  # seconds
    buffer_size: int = 1024 * 1024  # 1MB
    compression: bool = True

class AssistanceDisplayConfig(BaseModel):
    """Assistance feature display configuration"""
    show_primary_trajectory: bool = True
    show_collision_trajectories: bool = True
    show_ghost_balls: bool = True
    show_impact_points: bool = True
    show_angle_measurements: bool = False
    show_force_indicators: bool = True
    show_probability: bool = True
    show_alternative_shots: bool = False
    max_trajectory_bounces: int = 3

class ProjectorConfig(BaseModel):
    """Complete projector configuration"""
    display: DisplayConfig
    calibration: CalibrationConfig
    visual: VisualConfig
    rendering: RenderingConfig
    network: NetworkConfig
    assistance: AssistanceDisplayConfig
    debug: bool = False
    log_level: str = "INFO"
```

## Rendering Pipeline

### Frame Rendering Process

```python
def render_pipeline(frame_data: Dict) -> None:
    """
    Complete frame rendering pipeline

    1. Receive Data
       - Parse trajectory data from backend
       - Extract collision points and predictions
       - Get visual configuration parameters

    2. Geometric Transform
       - Apply calibration transformation matrix
       - Convert table coordinates to projector space
       - Apply keystone and distortion corrections

    3. Render Primitives
       - Draw table boundary (optional)
       - Render trajectory lines with style
       - Draw collision indicators
       - Render ghost balls
       - Add text overlays

    4. Apply Effects
       - Add glow effects to lines
       - Apply transparency/opacity
       - Animate elements (fade in/out)
       - Apply color gradients

    5. Composite and Display
       - Combine all rendered elements
       - Apply final color correction
       - Present to display device
       - Swap buffers (double buffering)
    """
    pass
```

### Calibration Process

```python
def calibration_sequence() -> CalibrationPoints:
    """
    Interactive calibration procedure

    1. Display calibration grid
       - Show crosshairs at corners
       - Display alignment guides
       - Show current transformation

    2. User adjustment
       - Arrow keys for fine adjustment
       - Mouse drag for coarse adjustment
       - Number keys for corner selection

    3. Test pattern
       - Display test trajectories
       - Show reference grid
       - Verify alignment

    4. Save calibration
       - Calculate transformation matrix
       - Store calibration points
       - Save to configuration file

    5. Apply calibration
       - Load transformation matrix
       - Set up rendering pipeline
       - Confirm successful calibration
    """
    pass
```

## Success Criteria

### Display Success Criteria

1. **Visual Quality**
   - Smooth, anti-aliased line rendering
   - No visible flicker or tearing
   - Consistent color reproduction
   - Clear text legibility at all sizes

2. **Alignment Accuracy**
   - Projection aligned within 5mm of physical table
   - Stable calibration without drift
   - Correct perspective at all table positions
   - Accurate trajectory overlay on actual paths

3. **Real-time Performance**
   - Consistent 60 FPS rendering
   - < 16ms frame rendering time
   - Smooth animations without stuttering
   - No dropped frames during normal operation

### Functional Success Criteria

1. **Trajectory Display**
   - All trajectory types rendered correctly
   - Collision points clearly visible
   - Smooth trajectory updates during aiming
   - Correct fade-in/fade-out effects

2. **Calibration**
   - Calibration completes in < 2 minutes
   - Calibration persists across restarts
   - Easy fine-tuning adjustments
   - Visual confirmation of accuracy

3. **Customization**
   - All visual preferences apply immediately
   - Settings persist between sessions
   - Assistance levels work as configured
   - Color schemes apply correctly

### Performance Success Criteria

1. **Rendering Performance**
   - 60 FPS with full visual effects
   - < 500MB GPU memory usage
   - < 30% GPU utilization typical
   - Smooth degradation under load

2. **Latency**
   - < 50ms total system latency
   - < 16ms rendering latency
   - Immediate response to updates
   - No perceptible lag in animations

3. **Stability**
   - No crashes during 24-hour operation
   - Automatic recovery from errors
   - Graceful handling of disconnections
   - Consistent performance over time

## Testing Requirements

### Unit Testing
- Test geometric transformation functions
- Validate calibration calculations
- Test rendering primitive functions
- Verify color space conversions
- Coverage target: 80%

### Integration Testing
- Test complete rendering pipeline
- Verify WebSocket data reception
- Test calibration procedures
- Validate configuration updates
- Test error recovery

### Visual Testing
- Verify rendering quality
- Test all visual effects
- Validate color accuracy
- Check text rendering
- Test animation smoothness

### Performance Testing
- Benchmark frame rates
- Measure rendering latency
- Test GPU memory usage
- Profile CPU/GPU usage
- Stress test with complex scenes

### Hardware Testing
- Test with different projectors
- Verify multiple resolutions
- Test various GPU configurations
- Validate on all platforms
- Test with different cable types

## Implementation Guidelines

### Code Structure
```python
projector/
├── __init__.py
├── main.py              # Main entry point
├── display/
│   ├── __init__.py
│   ├── manager.py      # Display management
│   ├── window.py       # Window/fullscreen handling
│   └── monitor.py      # Monitor detection
├── rendering/
│   ├── __init__.py
│   ├── renderer.py     # Main rendering engine
│   ├── opengl/         # OpenGL implementation
│   ├── primitives.py   # Basic shape rendering
│   ├── effects.py      # Visual effects
│   └── text.py         # Text rendering
├── calibration/
│   ├── __init__.py
│   ├── interactive.py  # Interactive calibration
│   ├── transform.py    # Geometric transforms
│   └── persistence.py  # Save/load calibration
├── network/
│   ├── __init__.py
│   ├── client.py       # WebSocket client
│   └── protocol.py     # Message handling
├── config/
│   ├── __init__.py
│   └── settings.py     # Configuration management
└── utils/
    ├── __init__.py
    ├── colors.py       # Color utilities
    ├── geometry.py     # Geometric helpers
    └── performance.py  # Performance monitoring
```

### Key Dependencies
- **PyOpenGL**: OpenGL rendering
- **pygame/pyglet**: Window management
- **moderngl**: Modern OpenGL wrapper
- **websocket-client**: WebSocket connection
- **numpy**: Matrix operations
- **Pillow**: Image processing

### Development Priorities
1. Implement basic display output
2. Add simple line rendering
3. Implement calibration system
4. Add WebSocket client
5. Implement full rendering pipeline
6. Add visual effects
7. Optimize performance
8. Add advanced features
