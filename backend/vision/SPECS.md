# Vision Module Specification

## Module Purpose

The Vision module is the core computer vision engine responsible for analyzing video frames to detect game elements (table, balls, cue stick), and providing processed visual data to other system components. This module uses a hybrid approach combining YOLOv8 deep learning for robust object detection with OpenCV for tracking, refinement, and geometric analysis. It consumes video streams from the separate Streaming Service, eliminating direct camera access conflicts.

## Functional Requirements

### 1. Stream Consumption Requirements

#### 1.1 Streaming Service Interface
- **FR-VIS-001**: Connect to Streaming Service via shared memory or network stream
- **FR-VIS-002**: Consume video frames at required frame rate (30 FPS for analysis)
- **FR-VIS-003**: Support multiple stream sources (shared memory, RTSP, HTTP)
- **FR-VIS-004**: Handle stream disconnection and reconnection gracefully
- **FR-VIS-005**: Monitor stream health and latency

#### 1.2 Image Preprocessing
- **FR-VIS-006**: Convert color spaces (BGR to HSV/LAB) for robust detection
- **FR-VIS-007**: Apply noise reduction and image smoothing
- **FR-VIS-008**: Perform automatic exposure and white balance correction
- **FR-VIS-009**: Crop to region of interest (ROI) for performance optimization
- **FR-VIS-010**: Handle varying lighting conditions adaptively

### 2. Table Detection Requirements

#### 2.1 Boundary Detection
- **FR-VIS-011**: Detect pool table edges using color and edge detection
- **FR-VIS-012**: Identify table corners with sub-pixel accuracy
- **FR-VIS-013**: Distinguish table surface from surrounding environment
- **FR-VIS-014**: Handle partial table visibility and occlusions
- **FR-VIS-015**: Validate detected table dimensions against expected ratios

#### 2.2 Pocket Detection
- **FR-VIS-016**: Locate all six pockets on the table
- **FR-VIS-017**: Determine pocket size and shape
- **FR-VIS-018**: Track pocket positions relative to table boundaries
- **FR-VIS-019**: Handle different pocket styles (corner vs side)

### 3. Ball Detection Requirements

#### 3.1 Ball Recognition (YOLO-Based Detection)
- **FR-VIS-020**: Detect all balls on the table surface using YOLOv8 deep learning model
- **FR-VIS-021**: Distinguish between different ball types (cue, solid, stripe, 8-ball) via trained classes
- **FR-VIS-022**: Identify ball numbers/patterns when visible using hybrid YOLO + OpenCV approach
- **FR-VIS-023**: Track ball positions with ±2 pixel accuracy using YOLO detection + OpenCV refinement
- **FR-VIS-024**: Measure ball radius for size validation from YOLO bounding boxes

#### 3.2 Ball Tracking (OpenCV-Based)
- **FR-VIS-025**: Track ball movement across frames using Kalman filters
- **FR-VIS-026**: Predict ball positions during fast movement with physics-based prediction
- **FR-VIS-027**: Handle ball occlusions using track history and prediction
- **FR-VIS-028**: Detect stationary vs moving balls via velocity thresholds
- **FR-VIS-029**: Calculate ball velocity and acceleration from track history

#### 3.3 Deep Learning Detection
- **FR-VIS-056**: Support YOLOv8-nano model for real-time detection (30+ FPS on CPU)
- **FR-VIS-057**: Provide fallback to OpenCV detection when YOLO model unavailable
- **FR-VIS-058**: Support model hot-swapping without system restart
- **FR-VIS-059**: Enable hybrid validation combining YOLO detection with OpenCV verification
- **FR-VIS-060**: Support custom trained models in ONNX format

### 4. Cue Stick Detection Requirements

#### 4.1 Cue Recognition
- **FR-VIS-030**: Detect cue stick using YOLOv8 model or fallback to line detection
- **FR-VIS-031**: Determine cue angle relative to cue ball using bounding box orientation
- **FR-VIS-032**: Track cue tip position with sub-pixel accuracy
- **FR-VIS-033**: Detect cue movement patterns (aiming vs striking) via temporal analysis
- **FR-VIS-034**: Handle multiple cue sticks in frame through class-based detection

#### 4.2 Shot Detection
- **FR-VIS-035**: Identify when cue contacts ball
- **FR-VIS-036**: Estimate strike force from cue velocity
- **FR-VIS-037**: Determine strike point on cue ball
- **FR-VIS-038**: Detect English/spin application

### 5. Calibration Requirements

#### 5.1 System Calibration
- **FR-VIS-039**: Perform automatic camera calibration on first startup
- **FR-VIS-040**: Calculate camera intrinsic parameters at native resolution
- **FR-VIS-041**: Determine camera-to-table transformation
- **FR-VIS-042**: Automatically detect and compensate for fisheye/lens distortion
- **FR-VIS-043**: Support manual calibration adjustment

#### 5.2 Fisheye Correction (Calibration Wizard)
- **FR-VIS-048**: Provide calibration wizard endpoint for fisheye correction
- **FR-VIS-049**: Capture calibration frame at full camera resolution (1920x1080)
- **FR-VIS-050**: Automatically detect table geometry to derive distortion parameters
- **FR-VIS-051**: Compute fisheye correction from table's rectangular shape
- **FR-VIS-052**: Apply fisheye correction to all video streams in real-time
- **FR-VIS-053**: Save calibration parameters for reuse across server restarts
- **FR-VIS-054**: Return calibration quality metrics and before/after preview
- **FR-VIS-055**: Support manual recalibration when table position changes

#### 5.3 Color Calibration
- **FR-VIS-044**: Auto-detect optimal color thresholds
- **FR-VIS-045**: Adapt to ambient lighting changes
- **FR-VIS-046**: Provide color picker interface
- **FR-VIS-047**: Save and load calibration profiles

## Non-Functional Requirements

### Performance Requirements
- **NFR-VIS-001**: Process frames at minimum 30 FPS on standard hardware (CPU-only)
- **NFR-VIS-002**: Detection latency < 33ms per frame (for 30 FPS)
- **NFR-VIS-009**: YOLOv8-nano model size < 10MB (ONNX format)
- **NFR-VIS-010**: Support both CPU and GPU inference with automatic selection
- **NFR-VIS-003**: Memory usage < 1GB during operation
- **NFR-VIS-004**: CPU usage < 60% on quad-core processor
- **NFR-VIS-005**: Support GPU acceleration when available

### Accuracy Requirements
- **NFR-VIS-006**: Ball detection accuracy > 98%
- **NFR-VIS-007**: False positive rate < 1%
- **NFR-VIS-008**: Position accuracy within 2mm on standard table
- **NFR-VIS-009**: Angle measurement accuracy within 1 degree
- **NFR-VIS-010**: Color classification accuracy > 95%

### Robustness Requirements
- **NFR-VIS-011**: Handle lighting variations (500-5000 lux)
- **NFR-VIS-012**: Work with different table colors (green, blue, red)
- **NFR-VIS-013**: Tolerate camera vibration and minor movement
- **NFR-VIS-014**: Recover from temporary occlusions
- **NFR-VIS-015**: Operate with partial table visibility (minimum 75%)

### Compatibility Requirements
- **NFR-VIS-016**: Support common webcam resolutions (720p, 1080p, 4K)
- **NFR-VIS-017**: Work with USB, network, and integrated cameras
- **NFR-VIS-018**: Compatible with OpenCV 4.5+
- **NFR-VIS-019**: Support multiple color formats (RGB, YUV, MJPEG)
- **NFR-VIS-020**: Support Kinect V2

## Interface Specifications

### Module Interface

```python
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
from enum import Enum

class BallType(Enum):
    CUE = "cue"
    SOLID = "solid"
    STRIPE = "stripe"
    EIGHT = "eight"

@dataclass
class Ball:
    """Detected ball information"""
    position: Tuple[float, float]  # (x, y) in pixels
    radius: float  # pixels
    ball_type: BallType
    number: Optional[int]  # 1-15 for numbered balls
    confidence: float  # 0.0-1.0 detection confidence
    velocity: Tuple[float, float]  # (vx, vy) pixels/second
    is_moving: bool

@dataclass
class CueStick:
    """Detected cue stick information"""
    tip_position: Tuple[float, float]
    angle: float  # degrees from horizontal
    length: float  # pixels
    is_aiming: bool
    confidence: float

@dataclass
class Table:
    """Detected table information"""
    corners: List[Tuple[float, float]]  # 4 corners
    pockets: List[Tuple[float, float]]  # 6 pockets
    width: float  # pixels
    height: float  # pixels
    surface_color: Tuple[int, int, int]  # Average HSV

@dataclass
class DetectionResult:
    """Complete frame detection results"""
    frame_number: int
    timestamp: float
    balls: List[Ball]
    cue: Optional[CueStick]
    table: Table
    processing_time: float  # milliseconds

class VisionModule:
    """Main vision processing interface"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration"""
        pass

    def start_capture(self) -> bool:
        """Start camera capture"""
        pass

    def stop_capture(self) -> None:
        """Stop camera capture"""
        pass

    def process_frame(self) -> Optional[DetectionResult]:
        """Process single frame and return detections"""
        pass

    def get_current_frame(self) -> np.ndarray:
        """Get latest captured frame"""
        pass

    def calibrate_camera(self) -> bool:
        """Perform camera calibration"""
        pass

    def calibrate_colors(self, sample_image: np.ndarray) -> Dict:
        """Auto-calibrate color thresholds"""
        pass

    def set_roi(self, corners: List[Tuple[int, int]]) -> None:
        """Set region of interest"""
        pass

    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        pass
```

### Configuration Schema

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class CameraSettings(BaseModel):
    device_id: int = 0
    backend: str = "auto"  # auto, v4l2, dshow, gstreamer
    resolution: List[int] = [1920, 1080]
    fps: int = 30
    exposure_mode: str = "auto"  # auto, manual
    exposure_value: Optional[float] = None
    gain: float = Field(1.0, ge=0, le=10)
    buffer_size: int = 1

class ColorThresholds(BaseModel):
    hue_min: int = Field(0, ge=0, le=179)
    hue_max: int = Field(179, ge=0, le=179)
    saturation_min: int = Field(0, ge=0, le=255)
    saturation_max: int = Field(255, ge=0, le=255)
    value_min: int = Field(0, ge=0, le=255)
    value_max: int = Field(255, ge=0, le=255)

class DetectionSettings(BaseModel):
    # Table detection
    table_color: ColorThresholds
    table_edge_threshold: float = Field(0.7, ge=0, le=1)
    min_table_area: float = 0.3  # Fraction of image

    # Ball detection
    ball_colors: Dict[str, ColorThresholds]
    min_ball_radius: int = 10
    max_ball_radius: int = 40
    ball_detection_method: str = "hough"  # hough, contour, blob
    ball_sensitivity: float = Field(0.8, ge=0, le=1)

    # Cue detection
    cue_detection_enabled: bool = True
    min_cue_length: int = 100
    cue_line_threshold: float = Field(0.6, ge=0, le=1)
    cue_color: Optional[ColorThresholds] = None

class ProcessingSettings(BaseModel):
    use_gpu: bool = False
    enable_preprocessing: bool = True
    blur_kernel_size: int = 5
    morphology_kernel_size: int = 3
    enable_tracking: bool = True
    tracking_max_distance: int = 50
    frame_skip: int = 0  # Process every Nth frame

class VisionConfig(BaseModel):
    camera: CameraSettings
    detection: DetectionSettings
    processing: ProcessingSettings
    debug: bool = False
    save_debug_images: bool = False
    debug_output_path: str = "/tmp/vision_debug"
```

## Processing Pipeline

### Frame Processing Flow

```python
def process_frame_pipeline(frame: np.ndarray) -> DetectionResult:
    """
    Complete frame processing pipeline

    1. Preprocessing
       - Color space conversion (BGR → HSV)
       - Noise reduction (Gaussian blur)
       - Morphological operations

    2. Table Detection
       - Color thresholding for table surface
       - Contour detection for boundaries
       - Corner point extraction
       - Pocket location identification

    3. Ball Detection
       - Circle detection (Hough circles)
       - Color classification
       - Number recognition (if visible)
       - Tracking association

    4. Cue Detection
       - Line detection (Hough lines)
       - Length and angle filtering
       - Tip position calculation

    5. Post-processing
       - Coordinate transformation
       - Velocity calculation
       - Confidence scoring
       - Result packaging
    """
    pass
```

### Detection Algorithms

```python
# Ball Detection Algorithm
def detect_balls(image: np.ndarray, config: DetectionSettings) -> List[Ball]:
    """
    1. Apply color mask for each ball type
    2. Find contours or circles
    3. Filter by size and circularity
    4. Classify by color histogram
    5. Track across frames
    6. Calculate velocity from position history
    """
    pass

# Cue Detection Algorithm
def detect_cue(image: np.ndarray, config: DetectionSettings) -> Optional[CueStick]:
    """
    1. Apply Canny edge detection
    2. Detect lines using Hough transform
    3. Filter lines by length and proximity
    4. Find best matching line pair
    5. Calculate tip position and angle
    """
    pass

# Table Detection Algorithm
def detect_table(image: np.ndarray, config: DetectionSettings) -> Table:
    """
    1. Apply color threshold for table cloth
    2. Find largest contour
    3. Approximate to quadrilateral
    4. Detect pockets as dark regions at corners/edges
    5. Validate geometry
    """
    pass
```

## Success Criteria

### Detection Success Criteria

1. **Ball Detection**
   - Detect 95%+ of visible balls consistently
   - No ghost detections when table is clear
   - Maintain tracking during normal play speed
   - Correctly classify ball types 90%+ of the time

2. **Cue Detection**
   - Detect cue when visible 85%+ of frames
   - Angle accuracy within 2 degrees
   - Stable detection without flickering
   - Distinguish cue from other linear objects

3. **Table Detection**
   - Detect table in first 5 frames after startup
   - Maintain stable boundary detection
   - Adapt to lighting changes without recalibration
   - Work with 80%+ of standard table colors

### Performance Success Criteria

1. **Frame Rate**
   - Maintain 30+ FPS with 1080p input
   - Maintain 15+ FPS with 4K input
   - No frame drops during continuous operation
   - Graceful degradation under high CPU load

2. **Latency**
   - Frame capture to detection < 33ms
   - Processing pipeline < 30ms per frame
   - Total end-to-end latency < 50ms
   - Consistent performance without drift

3. **Resource Usage**
   - Memory usage < 1GB steady state
   - No memory leaks over 24-hour operation
   - CPU usage < 60% on modern quad-core
   - GPU memory < 2GB when GPU enabled

### Robustness Success Criteria

1. **Environmental Tolerance**
   - Operate in 500-5000 lux illumination
   - Handle shadows from players
   - Tolerate reflections on balls
   - Work with worn/dirty table surfaces

2. **Error Recovery**
   - Recover from camera disconnection in < 5 seconds
   - Continue operation with partial occlusions
   - Automatically recalibrate after lighting changes
   - Gracefully handle invalid frames

## Testing Requirements

### Unit Testing
- Test each detection algorithm independently
- Mock camera input with test images
- Validate color space conversions
- Test edge cases (empty table, clustered balls)
- Coverage target: 85%

### Integration Testing
- Test complete pipeline with pre-recorded videos
- Verify frame rate consistency
- Test with various lighting conditions
- Validate tracking across frames
- Test calibration procedures

### Performance Testing
- Benchmark processing times per component
- Stress test with maximum ball count
- Test with different resolutions
- Measure resource usage over time
- Profile for bottlenecks

### Accuracy Testing
- Ground truth comparison with known positions
- Measure detection rates across scenarios
- Validate angle measurements
- Test color classification accuracy
- Compare with manual annotations

## Implementation Guidelines

### Code Structure
```python
vision/
├── __init__.py
├── capture.py           # Camera interface and capture
├── preprocessing.py     # Image preprocessing pipeline
├── detection/
│   ├── __init__.py
│   ├── table.py        # Table detection algorithms
│   ├── balls.py        # Ball detection and classification
│   ├── cue.py          # Cue stick detection
│   └── utils.py        # Common detection utilities
├── tracking/
│   ├── __init__.py
│   ├── tracker.py      # Object tracking across frames
│   └── kalman.py       # Kalman filter for prediction
├── calibration/
│   ├── __init__.py
│   ├── camera.py       # Camera calibration
│   ├── color.py        # Color threshold calibration
│   └── geometry.py     # Geometric calibration
├── models.py           # Data models and types
└── utils/
    ├── __init__.py
    ├── visualization.py # Debug visualization
    ├── metrics.py      # Performance metrics
    └── transforms.py   # Coordinate transforms
```

### Key Dependencies
- **opencv-python**: Core computer vision
- **numpy**: Numerical operations
- **scikit-image**: Additional image processing
- **numba**: JIT compilation for performance
- **Pillow**: Image format handling

### Development Priorities
1. Implement camera capture interface
2. Develop table detection
3. Implement ball detection
4. Add cue detection
5. Integrate object tracking
6. Add calibration routines
7. Optimize performance
8. Add GPU acceleration
