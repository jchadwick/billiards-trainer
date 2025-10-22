# Calibration Page Enhancements

## Summary

Added two new features to the calibration page at http://localhost:3000/calibration:

1. **Crosshair lines** spanning X & Y axis for each corner point to help align with table rails
2. **Table marker dots** feature to mark spots/markings on the table for masking during ball detection

## Changes Made

### Frontend Changes

**File:** `frontend/web/src/components/config/calibration/CalibrationWizard.tsx`

#### 1. Added TypeScript Interfaces
- `MarkerDot` interface for table marker dot positions (x, y, id)

#### 2. Crosshair Lines
- Modified the `drawFrame` function in `VideoFeedCanvas` component
- Added dashed blue crosshair lines (X & Y axis) that span the entire canvas for each corner point
- Lines are semi-transparent (`rgba(59, 130, 246, 0.5)`) with 5px dashes
- Helps users align corner points precisely with table rails

#### 3. Table Marker Dots UI
- Added `markerDots` state to track marker dot positions
- Added `selectionMode` state to toggle between 'corners' and 'markers' mode
- Added drawing logic for marker dots:
  - Red circles with X mark inside
  - Outer glow effect for visibility
  - Numbered labels (M1, M2, M3, etc.)
  - Badge showing count of marker dots
- Added mode selection buttons:
  - "Set Corners (X/4)" button
  - "Mark Table Dots (X)" button
- Added management buttons for each mode:
  - Corner mode: "Remove Last Corner", "Clear All Corners"
  - Marker mode: "Remove Last Marker", "Clear All Markers"
- Updated instruction text based on current mode
- Modified `handlePointSelect` to handle both corner placement and marker dot placement

#### 4. Data Persistence
- Updated `applyCalibration` function to send marker dots to backend
- Updated `loadExistingCalibration` to load saved marker dots from backend
- Marker dots are saved alongside corners in the playing area configuration

### Backend Changes

**File:** `backend/api/routes/config.py`

#### 1. Data Models
- Added `MarkerDot` Pydantic model with x and y coordinates
- Updated `PlayingAreaCornersRequest` model to include optional `marker_dots` field

#### 2. POST Endpoint `/api/v1/config/table/playing-area`
- Added validation for marker_dots array
- Saves marker_dots to config at `table.marker_dots`
- Returns marker_dots and marker_dots_count in response

#### 3. GET Endpoint `/api/v1/config/table/playing-area`
- Retrieves marker_dots from config
- Returns marker_dots array and marker_dots_count

#### 4. Configuration Storage
- Marker dots are stored in `config.json` at `table.marker_dots` path
- Format: Array of objects with x and y coordinates
- Example:
```json
{
  "table": {
    "marker_dots": [
      {"x": 100, "y": 150},
      {"x": 200, "y": 250}
    ]
  }
}
```

## User Workflow

### Setting Corners with Crosshairs
1. Navigate to http://localhost:3000/calibration
2. Select "Set Corners" mode (default)
3. Click on the 4 corners of the playing area in order:
   - Top-left
   - Top-right
   - Bottom-right
   - Bottom-left
4. Blue crosshair lines will appear through each corner point, spanning the entire canvas
5. Use these lines to precisely align with the table rails
6. If corners are already set, clicking updates the nearest corner

### Marking Table Dots
1. After setting corners, click "Mark Table Dots" button
2. Click on each table marker dot (spot/marking) that should be masked
3. Red circles with X marks will appear on each marked location
4. Click "Remove Last Marker" to undo the last marker dot
5. Click "Clear All Markers" to remove all marker dots

### Saving Configuration
1. Click "Apply Calibration" to save both corners and marker dots
2. Configuration is persisted to the backend config.json file
3. Both corners and marker dots will be loaded automatically on next visit

## Technical Details

### Visual Design
- **Corners**: Blue (#3b82f6) with numbered labels (1, 2, 3, 4)
- **Crosshairs**: Semi-transparent blue dashed lines
- **Marker Dots**: Red (#ef4444) circles with white X marks and labels (M1, M2, etc.)
- **Saved Calibration**: Magenta (#d946ef) overlay showing previously saved configuration

### Drawing Order
1. Video feed
2. Saved calibration overlay (magenta)
3. New corners being set (blue) with crosshairs
4. Connecting lines between corners
5. Marker dots (red)
6. Status badges (top-left corner)
7. Instruction text (bottom center)

### Error Handling
- Frontend validates that 4 corners are set before allowing save
- Backend validates corner count (must be exactly 4)
- Backend validates marker dot structure (must have x and y)
- Empty marker_dots array is valid (optional feature)

## Testing

To test the new features:

1. Restart the backend if needed to pick up the changes:
   ```bash
   # The backend should auto-reload, but if not, restart it
   cd /Users/jchadwick/code/billiards-trainer
   # Kill existing backend and restart
   ```

2. Visit http://localhost:3000/calibration

3. Test corner placement with crosshairs:
   - Click on 4 corners
   - Verify crosshair lines appear
   - Verify lines help align with table edges

4. Test marker dot placement:
   - Switch to "Mark Table Dots" mode
   - Click on several table marker locations
   - Verify red circles with X marks appear
   - Test "Remove Last Marker" button
   - Test "Clear All Markers" button

5. Test persistence:
   - Click "Apply Calibration"
   - Refresh the page
   - Verify corners and marker dots are loaded correctly

6. Test API endpoints:
   ```bash
   # GET endpoint
   curl http://localhost:8000/api/v1/config/table/playing-area

   # POST endpoint
   curl -X POST http://localhost:8000/api/v1/config/table/playing-area \
     -H 'Content-Type: application/json' \
     -d '{"corners":[{"x":37,"y":45},{"x":606,"y":39},{"x":604,"y":326},{"x":40,"y":322}],"marker_dots":[{"x":100,"y":150}]}'
   ```

## Future Enhancements

Potential improvements:
1. Add ability to click on existing marker dots to remove them individually
2. Add drag-and-drop support for moving marker dots
3. Add mask radius configuration for each marker dot
4. Integrate marker dots into the ball detection pipeline for actual masking
5. Add visual preview of masked regions
6. Add import/export functionality for calibration configurations
7. Add ability to save multiple calibration profiles

## Notes

- The crosshair lines are only shown for corners being actively set (blue), not for saved calibration (magenta)
- Marker dots are stored in screen coordinates (not world coordinates)
- The marker dots feature is completely optional - calibration works without any marker dots
- Backend auto-reload with uvicorn may take a few seconds to pick up changes
- TypeScript compilation had some pre-existing errors in other files, but the calibration code compiles successfully
