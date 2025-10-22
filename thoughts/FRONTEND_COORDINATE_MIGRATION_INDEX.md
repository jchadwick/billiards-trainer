# Frontend Coordinate Migration - Master Index

**Created**: 2025-10-21
**Status**: Analysis Complete
**Impact**: Low-Medium (6 files to update, 13+ files unaffected)

## Purpose

The backend is migrating from array-based coordinates `[x, y]` to dictionary-based coordinates `{x, y, space, resolution}`. This analysis documents all coordinate usage in the frontend and provides a complete migration guide.

## Documentation Files

### 1. Frontend Coordinate Analysis (COMPREHENSIVE)
**File**: `frontend_coordinate_analysis.md`
**Purpose**: Complete technical analysis of all coordinate references

Includes:
- Type definitions (API vs Internal)
- Critical conversion logic locations
- All rendering components
- Coordinate transformation utilities
- Integration test patterns
- Search patterns used
- Complete file list

**Read this for**: Deep understanding of the coordinate architecture

### 2. Frontend Coordinate Summary (QUICK START)
**File**: `frontend_coordinate_summary.md`
**Purpose**: Executive summary and migration checklist

Includes:
- Quick reference of current vs new formats
- Critical conversion point (VideoStore)
- 6 files requiring changes
- 13+ files that don't need changes
- Migration steps
- Risk assessment
- Testing checklist
- Rollback plan

**Read this for**: Fast overview and action items

### 3. Frontend Files Detailed (IMPLEMENTATION)
**File**: `frontend_files_detailed.md`
**Purpose**: File-by-file code changes with before/after examples

Includes:
- Complete before/after code for each file
- Line numbers for every change
- Exact change patterns
- Migration checklist by phase
- Metadata handling options

**Read this for**: Actual implementation work

## Quick Navigation

### "I need to understand the architecture"
→ Read: `frontend_coordinate_analysis.md`
→ Sections: Type Definitions, Critical Conversion Logic

### "I need to start the migration"
→ Read: `frontend_coordinate_summary.md`
→ Section: Migration Steps

### "I'm ready to write code"
→ Read: `frontend_files_detailed.md`
→ Section: All files requiring changes

### "I want to verify my changes"
→ Read: `frontend_coordinate_summary.md`
→ Section: Testing Checklist

## Key Findings

### Current Architecture
```
Backend (arrays) → API Types (arrays) → VideoStore (conversion) → Internal Types (objects) → Rendering
                                            ↑
                                    SINGLE CONVERSION POINT
```

### Files to Update (6)
1. `/types/api.ts` - Type definitions
2. `/stores/VideoStore.ts` - Conversion logic (CRITICAL)
3. `/services/data-handlers.ts` - Array indexing
4. `/tests/integration/DetectionOverlayIntegration.test.ts` - Test conversion
5. `/components/video/LiveView.tsx` - Bug fix
6. (Optional) Add metadata types

### Files Already Compatible (13+)
All overlay rendering, coordinate transformation, and internal store files already use object format and will work immediately after conversion updates.

## Migration Strategy

### Phase 1: Type Definitions
Update `/types/api.ts` to use `Position2D` objects instead of arrays.

### Phase 2: Conversion Logic
Update `/stores/VideoStore.ts` to access `.x` and `.y` instead of `[0]` and `[1]`.

### Phase 3: Data Processing
Update `/services/data-handlers.ts` array indexing throughout.

### Phase 4: Tests & Fixes
Update tests and fix LiveView bug.

### Phase 5: Verification
Run full test suite and manual testing.

## Risk Assessment

**Overall Risk**: LOW

**Reasons**:
- ✅ Single conversion point (VideoStore)
- ✅ Clean architecture with type separation
- ✅ Well-abstracted coordinate transformations
- ✅ No hardcoded coordinate values
- ✅ Good test coverage
- ✅ Most components already use object format

**Known Issues**:
- ⚠️ LiveView has existing type mismatch (will be fixed)

## Success Criteria

After migration:
- [ ] TypeScript compiles without errors
- [ ] All tests pass
- [ ] Ball positions render correctly
- [ ] Cue stick renders in correct position
- [ ] Table corners and pockets align
- [ ] Trajectory lines render properly
- [ ] Collision points appear correctly
- [ ] No console errors
- [ ] LiveView bug is fixed

## Contact Points

### Critical Files
- **VideoStore.ts**: Contains ALL conversion logic
- **api.ts**: Defines backend data structure
- **data-handlers.ts**: Processes coordinate data

### Safe Files
- All `/components/video/overlays/*` files
- `/utils/coordinates.ts`
- All rendering components

## Next Steps

1. Review all three documentation files
2. Start with type updates in `/types/api.ts`
3. Update conversion in `/stores/VideoStore.ts`
4. Update data processing in `/services/data-handlers.ts`
5. Run tests and verify
6. Fix LiveView bug
7. Final verification

## Compatibility Notes

### Backward Compatibility Option

If needed, support both formats temporarily:

```typescript
function normalizePosition(pos: [number, number] | Position2D): Position2D {
  if (Array.isArray(pos)) {
    return { x: pos[0], y: pos[1] };
  }
  return { x: pos.x, y: pos.y };
}
```

### Metadata Preservation

The new format includes `space` and `resolution` metadata. Consider:
- Storing for debugging
- Displaying in dev overlays
- Using for validation
- Ignoring if not needed

## Related Documents

- `/backend/COORDINATE_SYSTEMS.md` - Backend coordinate system documentation
- `/backend/RESOLUTION_CONFIG_SUMMARY.md` - Resolution configuration
- `/backend/MODELS_COORDINATE_MIGRATION.md` - Backend migration details
- `/backend/VECTOR2D_MIGRATION_COMPLETE.txt` - Backend migration status

## Version History

- 2025-10-21: Initial analysis complete
  - Searched all TypeScript files
  - Documented all coordinate references
  - Created migration plan
  - Identified 6 files to update
  - Found LiveView bug
  - Confirmed low risk assessment

---

## File Structure

```
thoughts/
├── FRONTEND_COORDINATE_MIGRATION_INDEX.md     ← You are here
├── frontend_coordinate_analysis.md            ← Complete technical analysis
├── frontend_coordinate_summary.md             ← Quick start guide
└── frontend_files_detailed.md                 ← Implementation details
```

---

**Start Here**: Read the summary first, then dive into specific files as needed.
