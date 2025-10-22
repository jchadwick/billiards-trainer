# Table Dimensions Coordinate System Mismatch

## Problem

The warning message indicates:
```
WARNING:core.game_state:State validation warnings:
Table width 3840 differs from 4K standard 3200;
Table height 2160 differs from 4K standard 1600
```

## Root Cause

There's a conceptual confusion between two different concepts:

### 1. **4K Resolution** (3840×2160)
- This is the **coordinate system canvas**
- The full frame/workspace where everything is positioned
- Defined in `coordinates.py:3` and `constants_4k.py:14`

### 2. **Table Dimensions in 4K Space** (3200×1600)
- This is the **pool table** within that canvas
- Represents the actual playing area
- Defined in `constants_4k.py:26-27`
- Maintains 2:1 aspect ratio for standard 9-foot pool table
- Centered in the 4K frame with margins around it

## What's Happening

1. **Vision Detection** (`vision/detection/table.py:943-955`)
   - Detects table corners in camera pixels
   - Calculates table width/height from detected corners
   - Returns dimensions in **camera resolution** (often 3840×2160 for 4K camera)

2. **Integration Service** (`integration_service.py:530-531`)
   - Passes through `table.width` and `table.height` from vision detection
   - These are the **measured pixel dimensions** of the table in camera space

3. **Game State** (`game_state.py:443-444`)
   - Receives table dimensions from integration service
   - Uses these dimensions directly in TableState
   - **Problem:** These are camera pixel measurements, not canonical 4K table dimensions

4. **Validation** (`validation/state.py:213-220`)
   - Expects table dimensions to be `TABLE_WIDTH_4K` (3200) and `TABLE_HEIGHT_4K` (1600)
   - Issues warning when dimensions don't match

## The Conceptual Model

```
┌─────────────────────────────────────────────┐
│   4K Resolution Canvas (3840×2160)          │
│                                             │
│    ┌───────────────────────────────────┐   │
│    │  Table Playing Area (3200×1600)   │   │
│    │                                   │   │
│    │  (This is the pool table)        │   │
│    │                                   │   │
│    └───────────────────────────────────┘   │
│                                             │
│  Margins: Left/Right: 320px, Top/Bottom: 280px │
└─────────────────────────────────────────────┘
```

## Why This Matters

1. **Physics calculations** use table dimensions for bounds checking
2. **Validation** expects canonical dimensions for consistency
3. **Ball positioning** needs correct table bounds for pocket detection
4. **Coordinate transformations** rely on known table dimensions

## Solution Options

### Option 1: Use Standard Table Dimensions (Recommended)
- Replace detected table dimensions with canonical constants
- Use detected corners only for coordinate transformation/warping
- Pros: Consistent, predictable, matches validation expectations
- Cons: Ignores actual measured table size

### Option 2: Update Validation to Accept Variable Dimensions
- Make validation accept any reasonable table dimensions
- Update constants to be configurable
- Pros: Flexible, works with different table sizes
- Cons: Loses canonical coordinate system benefits

### Option 3: Scale Table to Canonical Dimensions
- Scale detected table to match canonical dimensions
- Apply transformation matrix to all coordinates
- Pros: Preserves detected geometry while maintaining standards
- Cons: More complex coordinate transformations

## Recommended Fix

Use **Option 1**: Always use canonical table dimensions from constants_4k.py:

```python
# In game_state.py:_extract_table_state()
# Instead of using table_data.get("width", ...) and table_data.get("height", ...)
# Always use canonical dimensions:
from .constants_4k import TABLE_WIDTH_4K, TABLE_HEIGHT_4K

table_width = TABLE_WIDTH_4K  # Always 3200
table_height = TABLE_HEIGHT_4K  # Always 1600
```

The detected table corners are still useful for:
- Perspective transformation
- Determining table orientation
- Detecting pocket positions

But the dimensions should always be the canonical 4K table size.

## Files to Modify

1. `backend/core/game_state.py:443-444` - Use canonical dimensions
2. `backend/core/models.py:523-538` - Update `standard_9ft_table()` to use 4K pixel dimensions
3. `backend/integration_service.py` - Document that table width/height are informational only

## Test Cases to Update

- Ensure validation passes with canonical dimensions
- Verify ball positions use correct table bounds
- Check pocket detection with standard table size
