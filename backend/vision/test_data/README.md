# Vision Module Test Data

## ⚠️ CRITICAL: Primary Quality Validation Dataset

**This directory contains the PRIMARY quality validation dataset for the entire vision system.**

These curated test images and their ground truth annotations are the authoritative reference for:
- ✅ Validating all vision system changes before merging
- ✅ Ensuring detection accuracy meets SPECS.md requirements (>98%)
- ✅ Preventing regressions in ball/cue detection
- ✅ Quality gating for production deployments

**Before modifying the vision system, you MUST:**
1. Run the ground truth test suite: `python -m pytest backend/vision/test_vision_with_test_data.py -v`
2. Ensure all tests pass with required accuracy thresholds
3. Add new test cases for any new scenarios or edge cases
4. Update annotations if detection algorithms legitimately improve accuracy

**DO NOT:**
- ❌ Modify annotations to make tests pass (fix the code instead)
- ❌ Remove test cases that fail (fix the detection algorithms)
- ❌ Lower quality thresholds without team approval
- ❌ Skip running these tests before submitting changes

## Obtaining Test Images

### Option 1: Use Your Own Videos/Images

If you have billiards videos or images:

1. Extract frames from videos:
   ```bash
   ffmpeg -i your_video.mp4 -vf fps=1 backend/vision/tests/test_data/frame_%04d.jpg
   ```

2. Rename frames to match the required test cases below

### Option 2: Download from Open Source Datasets

Download sample images from Roboflow Universe:
- [Billiard Ball Detection Dataset](https://universe.roboflow.com/billiard-ball-data-set/billiard-ball-detection-aeo1m)
- [8 Ball Pool Tool Dataset](https://universe.roboflow.com/8-ball-pool-tool/8-ball-pool-tool)

### Option 3: Public Domain/Creative Commons Images

Search for billiards images on:
- Pexels (https://www.pexels.com/search/billiards/)
- Unsplash (https://unsplash.com/s/photos/pool-table)
- Pixabay (https://pixabay.com/images/search/billiards/)

Note: Ensure images are appropriately licensed for your use.

## Required Test Images

### Basic Test Cases
- `empty_table.jpg` - Empty billiards table for false positive testing
- `single_ball.jpg` - Single ball on table for basic detection
- `multiple_balls.jpg` - Multiple balls (3-5) for multi-object detection
- `full_table.jpg` - All 15 balls for stress testing

### Cue Detection
- `frame_with_cue.jpg` - Frame with cue stick visible
- `cue_aiming.jpg` - Player aiming with cue
- `cue_shot.jpg` - Cue during shot execution

### Edge Cases
- `clustered_balls.jpg` - Tightly clustered balls for separation testing
- `dim_lighting.jpg` - Poor lighting conditions
- `shadows.jpg` - Frame with shadows on table
- `motion_blur.jpg` - Fast-moving ball with motion blur
- `partial_visibility.jpg` - Balls partially out of frame

### Calibration
- `calibration_straight_on.jpg` - Table from optimal angle
- `calibration_angled.jpg` - Table from angled perspective
- `table_corners_visible.jpg` - Clear view of all table corners

## Ground Truth Data

For each test image that requires accuracy validation, provide a corresponding JSON file with ground truth data:

JSON files follow this schema:

```json
{
  "image": "frame.png",
  "balls": [
    {
      "id": 1,
      "center": {"x": 450.0, "y": 320.0},
      "radius": 15.5,
      "bbox": {"xtl": 430.0, "ytl": 300.0, "xbr": 470.0, "ybr": 340.0}
    }
  ],
  "cues": [
    {
      "id": 1,
      "center": {"x": 900.0, "y": 400.0},
      "bbox": {"xtl": 850.0, "ytl": 350.0, "xbr": 950.0, "ybr": 450.0}
    }
  ]
}
```

Use `python -m backend.vision.tests.tools.convert_annotations --annotations annotations.xml --output-dir .`
to convert CVAT XML dumps into this format automatically.

## Test Data Inventory

### Current Test Cases (With Ground Truth Annotations)

| Image | Purpose | Ground Truth | Quality Requirement |
|-------|---------|--------------|---------------------|
| `empty_table.png` | False positive testing | 0 balls, 0 cues | 0 false positives (NFR-VIS-007: <1% FP rate) |
| `multiple_balls.png` | Multi-object detection | 5 balls with positions/radii | >98% detection accuracy (NFR-VIS-006) |
| `clustered_balls.png` | Ball separation | 16 balls, some clustered | >95% detection (challenging case) |
| `full_table.png` | Stress testing | 15 balls | >98% detection accuracy |
| `frame_with_cue.png` | Cue detection | Balls + 1 cue | Cue detected with >70% confidence |
| `cue_aiming.png` | Aiming state | 6 balls + 1 cue (aiming) | Cue angle and position accurate |
| `motion_blur.png` | Motion handling | Balls with motion blur | >85% detection (degraded conditions) |
| `calibration_straight_on.png` | Optimal view | Calibration reference | >98% detection, optimal accuracy |
| `table_corners_visible.png` | Calibration | All corners visible | Table geometry detection |

### Performance Requirements (from SPECS.md)

- **Detection Accuracy**: >98% on standard scenarios (NFR-VIS-006)
- **Position Accuracy**: ±2 pixels (FR-VIS-023, NFR-VIS-008)
- **False Positive Rate**: <1% (NFR-VIS-007)
- **Color Classification**: >95% accuracy (NFR-VIS-010)
- **Radius Accuracy**: Within 15% tolerance

## Usage in Tests

### Running Ground Truth Tests

```bash
# Run all ground truth validation tests
python -m pytest backend/vision/test_vision_with_test_data.py -v

# Run specific test
python -m pytest backend/vision/test_vision_with_test_data.py::TestVisionWithGroundTruth::test_empty_table_no_false_positives -v

# Run with detailed output
python -m unittest backend/vision/test_vision_with_test_data.TestVisionWithGroundTruth -v
```

### Loading Test Data Programmatically

```python
import json
from pathlib import Path
import cv2

# Load test image
test_data_dir = Path(__file__).parent / "test_data"
image = cv2.imread(str(test_data_dir / "multiple_balls.png"))

# Load ground truth annotations
with open(test_data_dir / "multiple_balls.json") as f:
    ground_truth = json.load(f)

# Access ball annotations
for ball in ground_truth["balls"]:
    center = (ball["center"]["x"], ball["center"]["y"])
    radius = ball["radius"]
    bbox = ball["bbox"]  # {"xtl", "ytl", "xbr", "ybr"}
```

## Adding New Test Data

### When to Add New Test Cases

Add new test cases when:
1. You discover a scenario that causes detection failures
2. You implement a new feature that needs validation
3. You find edge cases not covered by existing tests
4. You want to test specific lighting/environmental conditions

### Process for Adding Test Data

1. **Capture or create the test image**
   - Use representative real-world scenarios
   - Ensure image quality is consistent with production use
   - Name descriptively (e.g., `low_light_conditions.png`)

2. **Create ground truth annotations**
   - Manually annotate using CVAT or similar tool
   - Export as XML, then convert to JSON:
     ```bash
     python -m backend.vision.tests.tools.convert_annotations \
       --annotations annotations.xml --output-dir .
     ```
   - Verify annotation accuracy (positions, radii, bboxes)

3. **Add test case to test suite**
   - Add new test method in `test_vision_with_test_data.py`
   - Define appropriate quality thresholds
   - Document the scenario and requirements

4. **Update documentation**
   - Add entry to the table above
   - Describe the test scenario and its purpose
   - Document expected behavior and tolerances

5. **Verify test works**
   - Run the new test in isolation
   - Ensure it passes with current codebase (or identifies real issues)
   - Verify test fails appropriately when detection is broken

### Ground Truth JSON Schema

```json
{
  "image": "filename.png",
  "balls": [
    {
      "id": 1,
      "center": {"x": 450.0, "y": 320.0},
      "radius": 15.5,
      "bbox": {"xtl": 430.0, "ytl": 300.0, "xbr": 470.0, "ybr": 340.0}
    }
  ],
  "cues": [
    {
      "id": 1,
      "center": {"x": 900.0, "y": 400.0},
      "bbox": {"xtl": 850.0, "ytl": 350.0, "xbr": 950.0, "ybr": 450.0}
    }
  ]
}
```

## Maintaining Test Data Quality

### Regular Maintenance Tasks

- [ ] **Quarterly Review**: Review all test cases for continued relevance
- [ ] **Annotation Accuracy**: Verify annotations are still accurate as algorithms improve
- [ ] **Coverage Analysis**: Identify gaps in test scenario coverage
- [ ] **Performance Baseline**: Update baseline metrics as system improves

### Quality Checklist for Test Data

Before committing test data changes:
- [ ] Images are high quality and representative
- [ ] Annotations are pixel-accurate (verified manually)
- [ ] JSON files validate against schema
- [ ] Test cases are documented in this README
- [ ] Tests pass on current codebase (or document intentional failures)
- [ ] No sensitive or private information in images
