# Tracking System Documentation

The tracking system provides comprehensive multi-object tracking and prediction capabilities for billiards ball detection. It combines advanced Kalman filtering with Hungarian algorithm association to deliver smooth, reliable tracking with prediction capabilities.

## Features

### Core Tracking
- **Multi-object tracking** with unique ID assignment and management
- **Kalman filter prediction** for smooth position and velocity estimation
- **Hungarian algorithm association** for optimal detection-to-track matching
- **Track state management** (tentative, confirmed, lost, deleted)
- **Velocity and acceleration estimation** from position history
- **Track validation and confidence scoring** based on measurement quality

### Advanced Capabilities
- **Trajectory prediction** for multiple time steps into the future
- **Missing detection prediction** during temporary occlusions
- **Track loss and recovery** handling for robust operation
- **Identity management** with track history preservation
- **Edge case handling** for balls entering/leaving the frame

### Performance Optimization
- **Parallel processing** for track predictions on multi-core systems
- **Vectorized cost computation** using NumPy for efficient association
- **Memory optimization** with automatic cleanup and caching
- **Adaptive algorithm selection** based on problem size
- **Performance monitoring** with comprehensive metrics

### Integration Features
- **Trajectory smoothing** for natural motion visualization
- **Interpolation** for consistent frame-to-frame motion
- **Format conversion** utilities for different detection modules
- **Comprehensive configuration** with adaptive parameter tuning

## Architecture

```
tracking/
├── kalman.py          # Kalman filter implementation
├── tracker.py         # Multi-object tracker
├── optimization.py    # Performance optimization utilities
├── integration.py     # Integration and smoothing utilities
└── __init__.py        # Module exports
```

### Core Components

#### KalmanFilter
Implements a 6-state Kalman filter for position, velocity, and acceleration tracking:

```python
from backend.vision.tracking import KalmanFilter

# Initialize with position
kf = KalmanFilter(
    initial_position=(100.0, 200.0),
    process_noise=1.0,
    measurement_noise=10.0,
    initial_velocity=(10.0, 5.0)  # Optional
)

# Predict next state
predicted_pos = kf.predict(dt=0.033)  # 30 FPS

# Update with measurement
kf.update(measured_position=(105.0, 198.0))

# Get current estimates
position = kf.get_position()
velocity = kf.get_velocity()
acceleration = kf.get_acceleration()
```

#### ObjectTracker
Multi-object tracker using Hungarian algorithm for association:

```python
from backend.vision.tracking import ObjectTracker, Ball, BallType

# Configure tracker
config = {
    'max_age': 30,          # Max frames to keep lost tracks
    'min_hits': 3,          # Min detections to confirm track
    'max_distance': 50.0,   # Max association distance
    'process_noise': 1.0,
    'measurement_noise': 10.0
}

tracker = ObjectTracker(config)

# Create detections
balls = [
    Ball(position=(100.0, 200.0), radius=15.0, ball_type=BallType.CUE),
    Ball(position=(300.0, 400.0), radius=15.0, ball_type=BallType.SOLID)
]

# Update tracking
tracked_balls = tracker.update_tracking(balls, frame_number=1)

# Get predictions and statistics
predictions = tracker.predict_positions(time_delta=0.033)
velocities = tracker.get_object_velocities()
stats = tracker.get_tracking_statistics()
```

#### IntegratedTracker
High-level interface with optimization and smoothing:

```python
from backend.vision.tracking import create_integrated_tracker

# Create optimized tracker
tracker = create_integrated_tracker({
    'enable_optimization': True,
    'smooth_trajectories': True,
    'predict_missing_detections': True,
    'performance_monitoring': True
})

# Process frame
result = tracker.process_frame(detections, frame_number=1)

# Access comprehensive results
tracked_objects = result.tracked_objects
trajectories = result.track_trajectories
predictions = result.predictions
performance = result.performance_metrics
```

## Configuration

### Basic Configuration
```python
config = {
    # Core tracking parameters
    'max_age': 30,                    # Frames to keep lost tracks
    'min_hits': 3,                    # Detections needed to confirm
    'max_distance': 50.0,             # Max association distance (pixels)
    'process_noise': 1.0,             # Kalman process noise
    'measurement_noise': 10.0,        # Kalman measurement noise

    # Performance optimization
    'enable_optimization': True,       # Enable performance optimizations
    'parallel_processing': True,       # Use parallel track prediction
    'max_threads': 4,                 # Max threads for parallel processing
    'memory_limit_mb': 512,           # Memory usage limit

    # Advanced features
    'smooth_trajectories': True,       # Apply trajectory smoothing
    'predict_missing_detections': True, # Predict during occlusions
    'adaptive_tuning': True,          # Adapt parameters automatically
    'performance_monitoring': True    # Enable performance monitoring
}
```

### Adaptive Parameters
The system automatically adapts parameters based on performance:

- **Low FPS**: Reduces computational load by decreasing `max_distance` and `max_age`
- **High FPS, Low Accuracy**: Increases precision by adjusting noise parameters
- **Large Problem Size**: Switches to efficient algorithms automatically

## Performance

### Benchmarks
- **Small scenes** (1-5 objects): ~0.5ms processing time
- **Medium scenes** (6-15 objects): ~2ms processing time
- **Large scenes** (16+ objects): ~5ms processing time
- **Memory usage**: <50MB for typical scenarios

### Optimization Features
- **Parallel prediction**: 2-4x speedup on multi-core systems
- **Vectorized computation**: 5-10x faster distance calculations
- **Adaptive algorithms**: Automatic selection based on problem size
- **Memory pooling**: Reduced allocation overhead
- **Caching**: Reuse of cost matrices for similar problem sizes

## Testing

Run the comprehensive test suite:

```bash
# Run all tracking tests
python tests/vision/tracking/test_standalone.py
python tests/vision/tracking/test_integration.py

# Run specific component tests
python tests/vision/tracking/test_kalman_simple.py
```

### Test Coverage
- **Kalman filter**: Prediction, update, validation, trajectory estimation
- **Object tracker**: Association, state management, statistics
- **Integration**: Smoothing, optimization, performance monitoring
- **Error handling**: Invalid inputs, edge cases, recovery

## API Reference

### Core Classes

#### KalmanFilter
- `__init__(initial_position, process_noise, measurement_noise, ...)`
- `predict(dt)` → predicted position
- `update(measured_position)` → None
- `get_position()` → (x, y)
- `get_velocity()` → (vx, vy)
- `get_acceleration()` → (ax, ay)
- `predict_trajectory(time_steps, dt)` → List of positions
- `is_valid_measurement(position)` → bool

#### ObjectTracker
- `__init__(config)`
- `update_tracking(detections, frame_number)` → tracked objects
- `predict_positions(time_delta)` → Dict[track_id, position]
- `get_object_velocities()` → Dict[track_id, velocity]
- `get_tracking_statistics()` → statistics dict
- `reset()` → None

#### IntegratedTracker
- `__init__(config)`
- `process_frame(detections, frame_number)` → TrackingResult
- `get_performance_summary()` → performance dict
- `reset()` → None

### Utility Functions

#### Factory Functions
- `create_integrated_tracker(config)` → IntegratedTracker

#### Format Conversion
- `convert_detection_format(detections, source_format)` → converted detections

## Integration Examples

### With OpenCV Detection
```python
import cv2
from backend.vision.tracking import create_integrated_tracker

# Setup
tracker = create_integrated_tracker()

# In video processing loop
frame = cv2.imread('frame.jpg')
detections = detect_balls_opencv(frame)  # Your detection function

# Track
result = tracker.process_frame(detections, frame_number)
tracked_balls = result.tracked_objects

# Visualize
for ball in tracked_balls:
    cv2.circle(frame, ball.position, ball.radius, (0, 255, 0), 2)
```

### With Custom Detection Format
```python
from backend.vision.tracking import convert_detection_format, create_integrated_tracker

# Convert from custom format
detections = get_custom_detections()
converted = convert_detection_format(detections, 'custom')

# Track
tracker = create_integrated_tracker()
result = tracker.process_frame(converted, frame_number)
```

## Advanced Usage

### Custom Kalman Configuration
```python
from backend.vision.tracking import KalmanFilter

# High precision tracking
precise_kf = KalmanFilter(
    initial_position=(100, 200),
    process_noise=0.1,        # Low process noise
    measurement_noise=1.0     # Low measurement noise
)

# Fast moving objects
fast_kf = KalmanFilter(
    initial_position=(100, 200),
    process_noise=10.0,       # High process noise
    measurement_noise=5.0
)
```

### Performance Monitoring
```python
tracker = create_integrated_tracker({'performance_monitoring': True})

# Process frames...
result = tracker.process_frame(detections, frame_number)

# Check performance
summary = tracker.get_performance_summary()
print(f"FPS: {summary['current_fps']:.1f}")
print(f"Processing time: {summary['average_processing_time']:.3f}ms")
print(f"Memory usage: {summary['memory_usage_mb']:.1f}MB")
```

### Trajectory Analysis
```python
# Get track trajectories
trajectories = tracker.get_track_trajectories()

for track_id, positions in trajectories.items():
    # Analyze trajectory
    distances = [np.linalg.norm(np.array(pos) - np.array(positions[0]))
                for pos in positions]
    max_distance = max(distances)
    print(f"Track {track_id} max displacement: {max_distance:.1f} pixels")
```

## Troubleshooting

### Common Issues

**Low tracking accuracy**:
- Increase `min_hits` parameter
- Decrease `max_distance` for stricter association
- Adjust Kalman noise parameters based on detection quality

**Poor performance**:
- Enable optimization: `enable_optimization=True`
- Use parallel processing: `parallel_processing=True`
- Reduce `max_age` to limit track count

**Track fragmentation**:
- Decrease `min_hits` for faster confirmation
- Increase `max_distance` for looser association
- Enable missing detection prediction

**Memory usage**:
- Reduce `max_age` to limit track history
- Disable trajectory smoothing for simple tracking
- Lower `memory_limit_mb` for automatic cleanup

### Debug Mode
```python
tracker = create_integrated_tracker({
    'debug_mode': True,
    'performance_monitoring': True
})

# Check tracking statistics
stats = tracker.get_tracking_statistics()
print(f"Active tracks: {stats['confirmed_tracks']}")
print(f"Total created: {stats['total_tracks_created']}")
```

## Future Enhancements

- **Deep learning integration** for advanced association
- **Physics-based prediction** using billiards dynamics
- **Multi-camera tracking** with 3D reconstruction
- **Real-time parameter adaptation** based on scene analysis
- **GPU acceleration** for large-scale tracking
