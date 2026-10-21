# Frontend web coordinate consumers

The web client now uses one internal image plane: canonical 3840×2160 pixels. Transport image points carry per-axis `scale` multipliers into that plane; `[1, 1]` means the coordinates are already canonical. Points without scale metadata are accepted only at APIs that also provide a known source resolution. Malformed, non-finite, zero, or negative scales are rejected instead of being drawn as if they were canonical.

Normalization happens at the data-handler and `VideoStore` boundaries. Positions, velocity vectors, cue endpoints and lengths, ball/pocket radii and extents, table corners, bounds, rails, trajectory endpoints, collisions, calibration points, and marker dots are normalized before use. Rendering converts canonical points and extents to the active video resolution. Physical table dimensions remain meter-valued metadata and are displayed with meter units only when the API explicitly identifies them as meters.

The cross-resolution test compares equivalent 1920×1080 points with `[2, 2]` scale and canonical 4K points with `[1, 1]`, covering both `VideoStore` and `DataProcessingService`, including table boundaries, radii, velocities, cue geometry, trajectories, and collisions. It also checks equivalent canvas positions and invalid-scale rejection.

Verification from `frontend/web`:

- `npm test -- --reporter=dot` — passed: 5 test files, 20 tests.
- `npm run typecheck` (runs `tsc --noEmit`) — fails on 79 existing TypeScript errors across 28 unrelated files in accessibility, config, diagnostics, layout, monitoring, navigation, hooks, API services, and stores. The coordinate-stage files do not appear in the diagnostics.
- `./node_modules/.bin/prettier --check <touched frontend files>` — reports formatting warnings across the existing touched files; the package has no formatting or lint script, and no ESLint executable is installed.
- `git -c core.whitespace=cr-at-eol diff --check -- <touched frontend files>` — passed; these files use CRLF line endings.

The independently running `LiveView` also normalizes its points and radii before applying its active frame dimensions. Existing synthetic coordinate examples and fixture payloads were kept explicitly canonical or updated to include scale metadata.
