# Phase 3 Implementation Plan: Enhanced Ball Detection

**Estimated Effort**: 60-75 hours (8-10 days)
**Risk Level**: MEDIUM
**Dependencies**: None (can start immediately)
**Parallelizable with**: Phase 2 (Shared Memory IPC)

---

## Objectives

1. Implement multi-path OpenCV detection (Color, Hough, Contours, Watershed)
2. Keep existing YOLO as 5th detection path
3. Implement sophisticated NMS fusion across all paths
4. Add advanced filtering (blur detection, edge exclusion, temporal)
5. Improve detection accuracy from ~40% to 70-95%
6. Maintain configurability for all detection parameters

---

## Architecture Overview

```
Before (Current):
┌─────────────────────────────────────────┐
│  Ball Detection Pipeline                │
│                                         │
│  Frame → YOLO Detection                 │
│          ├─ Position (bbox)             │
│          └─ Confidence                  │
│               ↓                         │
│         OpenCV Classification           │
│          └─ Ball type/number            │
│               ↓                         │
│         Tracking (Kalman)               │
│               ↓                         │
│         DetectionResult                 │
└─────────────────────────────────────────┘

After (Phase 3):
┌─────────────────────────────────────────────────────────┐
│  Enhanced Multi-Path Detection Pipeline                 │
│                                                         │
│  Frame → Preprocessing (CLAHE, Gradient)                │
│            ↓                                            │
│     ┌──────┴──────┬──────────┬──────────┬──────────┐   │
│     ↓             ↓          ↓          ↓          ↓   │
│  Color-Based  HoughCircles Contours Watershed  YOLO    │
│  (Cassapa)    (Shape)     (Circular) (Cluster) (ML)    │
│     │             │          │          │          │   │
│     └──────┬──────┴──────────┴──────────┴──────────┘   │
│            ↓                                            │
│      NMS Fusion (Confidence-weighted)                   │
│            ↓                                            │
│      Advanced Filtering                                 │
│      ├─ Edge Exclusion (5% margin)                      │
│      ├─ Blur Detection (Laplacian)                      │
│      ├─ Gradient Validation (radial)                    │
│      └─ Temporal Filtering (multi-frame)                │
│            ↓                                            │
│      Tracking (Kalman + Lost Prediction)                │
│            ↓                                            │
│      DetectionResult                                    │
└─────────────────────────────────────────────────────────┘
```

---

## Task Breakdown

### Task 3.1: Extract OpenCV Detector from V2 (40-50h)

**Objective**: Implement pure OpenCV multi-path detection

#### Subtasks:

**3.1.1 Copy Base OpenCV Detector (3-4h)**

**From**: `/Users/jchadwick/code/billiards-trainer-v2/backend/vision/detection/opencv_detector.py`
**To**: `/Users/jchadwick/code/billiards-trainer/backend/vision/detection/opencv_detector.py`

**Adaptations Needed**:
- Replace v2's ConfigManager with current Config singleton
- Update imports
- Remove v2-specific dependencies

**3.1.2 Implement Color-Based Detection (Cassapa Method) (8-10h)**

**File**: `backend/vision/detection/opencv_detector.py` (lines 951-1101 from v2)

**Core Algorithm**:
```python
def _detect_color_based(self, frame: np.ndarray, table_mask: np.ndarray) -> List[RawBallDetection]:
    """
    Color-based detection using felt color inversion
    Balls = NOT green (felt color)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Invert felt color to get balls
    lower_green = np.array([35, 60, 40])
    upper_green = np.array([95, 255, 255])
    felt_mask = cv2.inRange(hsv, lower_green, upper_green)
    ball_mask = cv2.bitwise_not(felt_mask)

    # Apply table mask
    ball_mask = cv2.bitwise_and(ball_mask, table_mask)

    # Morphological operations: opening (remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Estimate radius from area
        radius = np.sqrt(area / np.pi)

        # Calculate confidence from gradient strength
        confidence = self._calculate_confidence(frame, (cx, cy), radius)

        detections.append(RawBallDetection(
            position=(cx, cy),
            radius=radius,
            confidence=confidence,
            source="color_based"
        ))

    return detections
```

**Configuration Keys**:
```json
{
  "vision": {
    "detection": {
      "opencv": {
        "color_based": {
          "felt_hsv_lower": [35, 60, 40],
          "felt_hsv_upper": [95, 255, 255],
          "morph_kernel_size": 5,
          "min_ball_area": 200,
          "max_ball_area": 2000
        }
      }
    }
  }
}
```

**3.1.3 Implement HoughCircles Detection (6-8h)**

**File**: `backend/vision/detection/opencv_detector.py` (lines 858-949 from v2)

```python
def _detect_hough_circles(self, frame: np.ndarray, table_mask: np.ndarray) -> List[RawBallDetection]:
    """Shape-based circle detection using Hough transform"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply table mask
    gray = cv2.bitwise_and(gray, gray, mask=table_mask)

    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(expected_radius * 1.5),
        param1=50,  # Canny edge threshold
        param2=30,  # Accumulator threshold
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        return []

    detections = []
    for (x, y, r) in circles[0]:
        # Calculate confidence from gradient strength
        confidence = self._evaluate_circle_candidate(frame, (x, y), r)

        detections.append(RawBallDetection(
            position=(int(x), int(y)),
            radius=float(r),
            confidence=confidence,
            source="hough_circles"
        ))

    return detections
```

**Configuration Keys**:
```json
{
  "vision": {
    "detection": {
      "opencv": {
        "hough": {
          "dp": 1.2,
          "min_dist_multiplier": 1.5,
          "param1": 50,
          "param2": 30,
          "gaussian_kernel": 9
        }
      }
    }
  }
}
```

**3.1.4 Implement Contour-Based Detection (6-8h)**

**File**: `backend/vision/detection/opencv_detector.py`

```python
def _detect_contours(self, frame: np.ndarray, table_mask: np.ndarray) -> List[RawBallDetection]:
    """Contour detection with circularity filtering"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=table_mask)

    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        # Calculate circularity: 4π*area/perimeter²
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)

        if circularity < 0.7 or circularity > 1.2:
            continue  # Not circular enough

        # Get bounding circle
        (x, y), radius = cv2.minEnclosingCircle(contour)

        # Confidence based on circularity
        confidence = min(circularity, 1.0)

        detections.append(RawBallDetection(
            position=(int(x), int(y)),
            radius=float(radius),
            confidence=confidence,
            source="contour"
        ))

    return detections
```

**3.1.5 Implement Watershed Segmentation (8-10h)**

**File**: `backend/vision/detection/opencv_detector.py` (lines 1103-1248 from v2)

```python
def _detect_watershed(self, frame: np.ndarray, table_mask: np.ndarray) -> List[RawBallDetection]:
    """
    Watershed segmentation for clustered/touching balls
    Most complex method, used as last resort
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=table_mask)

    # Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Distance transform to find centers
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # Find local maxima (ball centers)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Dilate to find regions
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(frame, markers)

    # Extract individual regions
    detections = []
    for label in range(2, markers.max() + 1):
        mask = np.uint8(markers == label) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(contour)

        if radius < min_radius or radius > max_radius:
            continue

        confidence = 0.7  # Watershed has moderate confidence

        detections.append(RawBallDetection(
            position=(int(x), int(y)),
            radius=float(radius),
            confidence=confidence,
            source="watershed"
        ))

    return detections
```

**3.1.6 Implement Table Segmentation (4-5h)**

**File**: `backend/vision/detection/table_segmenter.py` (copy from v2)

- Calibrated boundary polygon (preferred)
- Felt color detection (fallback)
- Morphological operations for cleanup

**3.1.7 Implement Preprocessing (3-4h)**

**File**: `backend/vision/detection/opencv_detector.py` (lines 304-357 from v2)

- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gradient map computation (Sobel filters)
- Gaussian blur for noise suppression

**3.1.8 Testing (2-3h)**
- Test each detection method independently
- Measure precision/recall for each
- Verify configuration loading

**Deliverables**:
- ✅ All 4 OpenCV detection methods implemented
- ✅ Table segmentation working
- ✅ Preprocessing pipeline functional
- ✅ All methods configurable

---

### Task 3.2: Integrate YOLO as 5th Path (20-25h)

**Objective**: Keep existing YOLO detector, integrate into multi-path framework

#### Subtasks:

**3.2.1 Adapt YOLO to Common Interface (5-6h)**

**File**: `backend/vision/detection/yolo_detector.py`

**Changes**:
```python
# Add method to return RawBallDetection objects
def detect_balls_raw(self, frame: np.ndarray) -> List[RawBallDetection]:
    """
    Returns RawBallDetection objects for fusion with OpenCV methods
    """
    # Existing YOLO detection
    detections = self.detect_balls_with_classification(frame)

    # Convert to RawBallDetection
    raw_detections = []
    for ball in detections:
        raw_detections.append(RawBallDetection(
            position=ball.position,
            radius=ball.radius,
            confidence=ball.confidence,
            source="yolo",
            ball_type=ball.ball_type,  # YOLO provides this
            ball_number=ball.number
        ))

    return raw_detections
```

**3.2.2 Create Unified BallDetector (8-10h)**

**File**: `backend/vision/detection/ball_detector.py`

```python
class BallDetector:
    """
    Unified ball detector with multi-path fusion
    Combines YOLO (ML) + 4 OpenCV methods
    """

    def __init__(self, config: Config):
        self.config = config

        # Initialize all detectors
        self.yolo_detector = YOLODetector(config) if config.get("vision.detection.yolo.enabled", True) else None
        self.opencv_detector = OpenCVDetector(config) if config.get("vision.detection.opencv.enabled", True) else None

        # NMS parameters
        self.nms_threshold = config.get("vision.detection.nms_threshold", 0.3)
        self.confidence_weights = config.get("vision.detection.confidence_weights", {
            "yolo": 1.0,
            "color_based": 0.9,
            "hough_circles": 0.85,
            "contour": 0.8,
            "watershed": 0.7
        })

    def detect_balls(self, frame: np.ndarray) -> List[Ball]:
        """
        Multi-path detection with NMS fusion
        """
        all_detections = []

        # Path 1: YOLO (if enabled)
        if self.yolo_detector:
            yolo_detections = self.yolo_detector.detect_balls_raw(frame)
            all_detections.extend(yolo_detections)

        # Paths 2-5: OpenCV methods (if enabled)
        if self.opencv_detector:
            opencv_detections = self.opencv_detector.detect_balls(frame)
            all_detections.extend(opencv_detections)

        # Weighted NMS fusion
        fused_detections = self._nms_fusion(all_detections)

        # Advanced filtering
        filtered_detections = self._apply_filters(fused_detections, frame.shape)

        # Convert to Ball objects with tracking info
        balls = self._to_ball_objects(filtered_detections)

        return balls

    def _nms_fusion(self, detections: List[RawBallDetection]) -> List[RawBallDetection]:
        """
        Non-Maximum Suppression with confidence weighting
        """
        if not detections:
            return []

        # Apply source-specific confidence weights
        weighted_detections = []
        for det in detections:
            weight = self.confidence_weights.get(det.source, 1.0)
            weighted_conf = det.confidence * weight
            weighted_detections.append(det._replace(confidence=weighted_conf))

        # Sort by confidence (highest first)
        weighted_detections.sort(key=lambda d: d.confidence, reverse=True)

        # NMS algorithm
        kept_detections = []
        while weighted_detections:
            # Keep highest confidence detection
            best = weighted_detections.pop(0)
            kept_detections.append(best)

            # Remove overlapping detections
            weighted_detections = [
                det for det in weighted_detections
                if self._calculate_iou(best, det) < self.nms_threshold
            ]

        return kept_detections

    def _calculate_iou(self, det1: RawBallDetection, det2: RawBallDetection) -> float:
        """Calculate Intersection over Union for circular detections"""
        # Distance between centers
        dx = det1.position[0] - det2.position[0]
        dy = det1.position[1] - det2.position[1]
        distance = np.sqrt(dx**2 + dy**2)

        # If circles don't overlap, IoU = 0
        if distance >= det1.radius + det2.radius:
            return 0.0

        # If one circle contains the other
        if distance <= abs(det1.radius - det2.radius):
            smaller_area = np.pi * min(det1.radius, det2.radius) ** 2
            larger_area = np.pi * max(det1.radius, det2.radius) ** 2
            return smaller_area / larger_area

        # Partial overlap (use approximation)
        # For simplicity, use distance-based overlap metric
        overlap_ratio = 1 - (distance / (det1.radius + det2.radius))
        return max(0.0, overlap_ratio)
```

**Configuration**:
```json
{
  "vision": {
    "detection": {
      "yolo": {
        "enabled": true,
        "confidence": 0.15
      },
      "opencv": {
        "enabled": true,
        "methods": ["color_based", "hough_circles", "contour", "watershed"]
      },
      "nms_threshold": 0.3,
      "confidence_weights": {
        "yolo": 1.0,
        "color_based": 0.9,
        "hough_circles": 0.85,
        "contour": 0.8,
        "watershed": 0.7
      }
    }
  }
}
```

**3.2.3 Testing Fusion Algorithm (4-5h)**
- Test with all paths enabled
- Test with selective paths disabled
- Verify NMS removes duplicates correctly
- Measure combined precision/recall

**3.2.4 Performance Optimization (3-4h)**
- Profile each detection path
- Identify bottlenecks
- Add early exit conditions
- Parallelize independent paths (future enhancement)

**Deliverables**:
- ✅ YOLO integrated into multi-path framework
- ✅ NMS fusion working correctly
- ✅ Configurable path selection
- ✅ Performance benchmarked

---

### Task 3.3: Advanced Filtering (10-12h)

**Objective**: Implement sophisticated filtering for false positive reduction

#### Subtasks:

**3.3.1 Edge Exclusion Filter (2-3h)**

```python
def _filter_edge_exclusion(self, detections: List[RawBallDetection], frame_shape: tuple) -> List[RawBallDetection]:
    """
    Exclude detections near frame edges
    V2 reports 92% false positive reduction with 5% margin
    """
    height, width = frame_shape[:2]
    margin = self.config.get("vision.detection.edge_exclusion_margin", 0.05)

    min_x = width * margin
    max_x = width * (1 - margin)
    min_y = height * margin
    max_y = height * (1 - margin)

    filtered = []
    for det in detections:
        x, y = det.position
        if min_x <= x <= max_x and min_y <= y <= max_y:
            filtered.append(det)

    return filtered
```

**3.3.2 Blur Detection (3-4h)**

```python
def _filter_blur_detection(self, frame: np.ndarray, detections: List[RawBallDetection]) -> List[RawBallDetection]:
    """
    Reduce confidence for blurred balls (motion blur)
    Uses Laplacian variance for sharpness measurement
    """
    enable_blur = self.config.get("vision.detection.enable_blur_detection", False)
    if not enable_blur:
        return detections

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    filtered = []
    for det in detections:
        # Extract ball region
        x, y = det.position
        r = int(det.radius)
        x1, y1 = max(0, x-r), max(0, y-r)
        x2, y2 = min(frame.shape[1], x+r), min(frame.shape[0], y+r)

        region = gray[y1:y2, x1:x2]
        if region.size == 0:
            continue

        # Calculate Laplacian variance (sharpness metric)
        laplacian = cv2.Laplacian(region, cv2.CV_64F)
        variance = laplacian.var()

        # Threshold for blur
        blur_threshold = self.config.get("vision.detection.blur_threshold", 100)

        if variance < blur_threshold:
            # Ball is blurred - reduce confidence
            det = det._replace(confidence=det.confidence * 0.3)

        filtered.append(det)

    return filtered
```

**3.3.3 Gradient Validation (3-4h)**

```python
def _filter_gradient_validation(self, frame: np.ndarray, detections: List[RawBallDetection]) -> List[RawBallDetection]:
    """
    Validate gradients point radially (toward/away from center)
    Rejects shadows, pockets, non-circular objects
    """
    enable_gradient = self.config.get("vision.detection.enable_gradient_validation", False)
    if not enable_gradient:
        return detections

    # Compute gradients
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    filtered = []
    for det in detections:
        x, y = det.position
        r = det.radius

        # Sample 36 points around perimeter
        angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
        valid_count = 0

        for angle in angles:
            px = int(x + r * np.cos(angle))
            py = int(y + r * np.sin(angle))

            if px < 0 or px >= frame.shape[1] or py < 0 or py >= frame.shape[0]:
                continue

            # Get gradient at this point
            grad_x = gx[py, px]
            grad_y = gy[py, px]

            # Expected radial direction
            expected_x = np.cos(angle)
            expected_y = np.sin(angle)

            # Dot product (alignment)
            alignment = abs(grad_x * expected_x + grad_y * expected_y)

            if alignment > 0.5:  # Threshold
                valid_count += 1

        # Require 70% of points have radial gradients
        if valid_count / len(angles) > 0.7:
            filtered.append(det)

    return filtered
```

**3.3.4 Temporal Filtering (2-3h)**

```python
def _filter_temporal(self, detections: List[RawBallDetection]) -> List[RawBallDetection]:
    """
    Require detection across multiple frames
    Reduces transient false positives
    """
    enable_temporal = self.config.get("vision.detection.enable_temporal_filtering", False)
    if not enable_temporal:
        return detections

    # This requires maintaining history - integrate with tracker
    # For now, placeholder implementation
    return detections
```

**Deliverables**:
- ✅ All 4 filters implemented
- ✅ Configurable enable/disable per filter
- ✅ Measured false positive reduction

---

### Task 3.4: Integration with Tracking (8-10h)

**Objective**: Enhance existing tracking with lost ball prediction

#### Subtasks:

**3.4.1 Adapt SimpleTracker from V2 (5-6h)**

**File**: `backend/vision/tracking/simple_tracker.py` (from v2)

**Key Features**:
- Nearest-neighbor matching
- Lost ball prediction using constant velocity
- Adaptive match distance (increases for lost balls)
- Track expiration after timeout

**3.4.2 Integrate with BallDetector (2-3h)**

**File**: `backend/vision/__init__.py`

```python
# In _process_single_frame():
raw_detections = self.ball_detector.detect_balls(frame)
tracked_balls = self.ball_tracker.update(raw_detections)
```

**3.4.3 Testing (1-1h)**
- Test tracking across occlusions
- Test lost ball prediction
- Verify ID consistency

**Deliverables**:
- ✅ Enhanced tracking with prediction
- ✅ Lost ball handling
- ✅ Stable ball IDs

---

## Testing Strategy

### Unit Tests (per detection method)
- `test_color_based_detection.py`
- `test_hough_circles_detection.py`
- `test_contour_detection.py`
- `test_watershed_detection.py`
- `test_yolo_integration.py`

### Integration Tests
- `test_multi_path_fusion.py` - Test NMS across all paths
- `test_advanced_filtering.py` - Test all filters together
- `test_tracking_integration.py` - End-to-end tracking

### Performance Tests
- Measure detection accuracy (precision/recall)
- Measure latency per method
- Profile CPU usage
- Compare with current YOLO-only approach

### Acceptance Criteria
- ✅ Detection accuracy >70% (ideally 85-95%)
- ✅ False positive rate <10%
- ✅ Latency <100ms per frame
- ✅ All detection paths configurable
- ✅ NMS fusion working correctly

---

## Configuration Schema

Complete configuration for all detection methods:

```json
{
  "vision": {
    "detection": {
      "yolo": {
        "enabled": true,
        "model_path": "models/yolov8n-pool.pt",
        "confidence": 0.15,
        "nms_threshold": 0.45,
        "device": "cpu"
      },
      "opencv": {
        "enabled": true,
        "downscale_factor": 0.5,
        "methods": ["color_based", "hough_circles", "contour", "watershed"],
        "color_based": {
          "felt_hsv_lower": [35, 60, 40],
          "felt_hsv_upper": [95, 255, 255],
          "morph_kernel_size": 5,
          "min_ball_area": 200,
          "max_ball_area": 2000
        },
        "hough": {
          "dp": 1.2,
          "min_dist_multiplier": 1.5,
          "param1": 50,
          "param2": 30,
          "gaussian_kernel": 9,
          "min_radius": 10,
          "max_radius": 50
        },
        "contour": {
          "min_circularity": 0.7,
          "max_circularity": 1.2,
          "adaptive_threshold_block_size": 11,
          "adaptive_threshold_c": 2
        },
        "watershed": {
          "enabled": false,
          "distance_transform_threshold": 0.5
        }
      },
      "nms_threshold": 0.3,
      "confidence_weights": {
        "yolo": 1.0,
        "color_based": 0.9,
        "hough_circles": 0.85,
        "contour": 0.8,
        "watershed": 0.7
      },
      "filters": {
        "edge_exclusion_margin": 0.05,
        "enable_blur_detection": false,
        "blur_threshold": 100,
        "enable_gradient_validation": false,
        "enable_temporal_filtering": false
      }
    },
    "tracking": {
      "max_match_distance": 50.0,
      "lost_threshold_frames": 30,
      "expire_timeout_sec": 5.0
    }
  }
}
```

---

## Migration Strategy

### Step 1: Implement Alongside Current (Week 1-2)
- Add new OpenCVDetector without touching YOLODetector
- Feature flag: `USE_ENHANCED_DETECTION=false` initially

### Step 2: A/B Testing (Week 2-3)
- Run both detection systems in parallel
- Compare results on same frames
- Measure accuracy, latency, false positives

### Step 3: Tuning (Week 3)
- Adjust confidence weights
- Tune NMS threshold
- Enable/disable specific paths based on results

### Step 4: Gradual Rollout (Week 4)
- Enable for test dataset
- Monitor accuracy metrics
- Enable for production

---

## Success Metrics

### Accuracy Metrics
- **Precision**: >85% (detections that are actual balls)
- **Recall**: >90% (actual balls that are detected)
- **F1 Score**: >87%
- **False Positive Rate**: <10%

### Performance Metrics
- **Latency**: <100ms per frame (all paths combined)
- **CPU Usage**: <60% of one core
- **Memory**: <50MB additional

### Quality Metrics
- **NMS Effectiveness**: <5% duplicate detections
- **Tracking Stability**: >95% ID consistency
- **Configuration Coverage**: All parameters configurable

---

## Documentation Deliverables

1. **Algorithm Documentation**: Explanation of each detection method
2. **Configuration Guide**: How to tune each parameter
3. **Performance Comparison**: Before/after metrics
4. **Troubleshooting Guide**: Common issues and solutions

---

## Timeline

**Week 1**: Task 3.1.1-3.1.4 (Color, Hough, Contour detections)
**Week 2**: Task 3.1.5-3.1.8 (Watershed, Table, Preprocessing, Testing)
**Week 3**: Task 3.2 (YOLO integration + fusion)
**Week 4**: Task 3.3 + 3.4 (Filtering + Tracking)

**Total**: 4 weeks (60-75 hours)

---

## Risk Mitigation

### Risk: Detection accuracy doesn't improve
**Mitigation**: A/B test early, adjust confidence weights

### Risk: Latency too high (>100ms)
**Mitigation**: Disable expensive methods (watershed), optimize preprocessing

### Risk: Too many false positives
**Mitigation**: Enable all filters, increase NMS threshold

### Risk: Configuration too complex
**Mitigation**: Provide presets (low/medium/high accuracy)

---

## Next Steps

1. Review this plan with team
2. Create feature branch: `feature/enhanced-detection`
3. Begin Task 3.1.1 (Extract OpenCV Detector)
4. Set up accuracy benchmarking pipeline
5. Schedule weekly demo of detection improvements
