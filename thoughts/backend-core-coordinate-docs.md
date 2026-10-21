# Backend core coordinate documentation pass

Date: 2026-10-19

The core documentation now describes the implemented coordinate contract:
image-space geometry is stored in canonical 3840 × 2160 pixels, source points
carry explicit scale metadata, and physical measurements remain separate SI
values connected to image space through calibration. Physics validation
examples identify pixel positions, pixel-per-second velocities, SI mass/force,
and seconds explicitly.

The former `CoordinateConverter` and meter-storage examples did not match the
current core API. They have been replaced with current `Vector2D`,
`ResolutionConverter`, `BallState`, and `TableState` examples. Old migration
summaries remain as historical records with a clear superseded status and a
link to the current contract.

Verification performed:

- Current executable Python examples were run against the repository modules.
- Illustrative snippets in `backend/core/SPECS.md` were syntax-checked.
- The core documentation was scanned for bare `Vector2D(x, y)` examples; the
  only direct construction now passes explicit scale metadata.
- The edits passed a CRLF-aware `git diff --check`.

## Follow-up project documentation scan

The two coordinate investigation notes under `backend/thoughts/` are now clearly
marked historical and link to the current coordinate contract. The integration
test report now requires explicit scale or source resolution when converting a
raw tuple to `Vector2D`. The API model link in the conversion examples was
corrected. Local links resolve, and the CRLF-aware diff check passes.
