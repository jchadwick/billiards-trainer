# Practice camera placement implementation

Date: 2026-11-18

## Scope

This implementation changes only the practice module and its scoped presentation rules:

- `frontend/web/src/components/practice/PracticePage.tsx`
- `frontend/web/src/components/practice/PracticeCameraOverlay.tsx`
- `frontend/web/src/components/practice/practice-placement.ts`
- Practice component tests in the same directory
- The `.practice-*` section of `frontend/web/src/styles.css`

The pre-edit workspace snapshot is saved at `/tmp/billiards-practice-baseline-20261118` (practice directory, `AGENTS.md`, git status, and the pre-existing scoped diff). The shared RootStore, receiver host, live routes, and perception modules were not edited.

## Camera and geometry contract

Practice borrows `RootStore.nativeVideoElement`, the app-owned native MJPEG image. It moves that image only from `#shared-video-receiver-parking`, and returns it to the exact original parent on unmount. If another visible surface owns the image, practice leaves it in place and reports the placement state as unverified.

Target slots use positive-origin millimeters for the existing practice geometry. The producer projection is still center-origin millimeters. Clicks on the camera overlay pass through `pixel_to_table_mm` to position a slot; rendering inverts the same producer homography with `tableMmToImage`. The overlay accepts only a compatible raw-image projection and checks known camera profile, calibration, rectification, producer configuration, displayed-image aspect, and per-ball source context. There are no guessed image locations or fabricated calibration values.

## Placement and feedback contract

- Default center-distance tolerance is 45 mm. `VITE_PRACTICE_PLACEMENT_TOLERANCE_MM` accepts 5–150 mm; invalid or absent values use the documented default.
- The matcher finds a deterministic maximum-cardinality one-to-one assignment. A measured center can satisfy only one target even when tolerance areas overlap.
- The current exercise defines a subset through its editable target slots. Extra compatible physical centers do not block Ready; the UI reports them and unassigned centers remain shot-path blockers.
- Only current-context physical table centers can match. Extrapolated centers are rendered dimly when available, but cannot match. A remembered `last_observed` center can match only while producer layout trust, current quiet-motion continuity, state connection, camera stream, and mapping context all remain valid.
- Motion age is rebased from producer `event_sequence` using the existing `advanceEvidenceAgeMs` policy. The practice page does not maintain its own ball-position cache and does not expire a trusted stationary layout merely because a full ball scan is old.
- `Ready` is withdrawn for stale motion, source silence, disconnect, shot/recovery phases, missing stream/image, source-generation reconnect with unchanged producer identity, or projection/context mismatch. An absent/empty physical-center set stays explicitly unverified.
- Cue/object roles belong to the player-selected target slots. The UI does not infer ball numbers, stripes, solids, or cue-ball identity from detection IDs.
- There is no standalone SVG table in the practice flow. Target regions, detected-center markers, and direct-pot guidance render on the live camera image; coordinate editing and the target list remain usable without a camera.

## Verification

The focused practice suite covers tolerance parsing, threshold boundaries, overlap-safe matching, extras, context/freshness gate withdrawal, camera overlay projection, projected shot lines, shared-image release, ownership conflicts, zero-ball honesty, synthetic Ready feedback, target editing, and route navigation.

Commands run during implementation:

```text
cd frontend/web
npx vitest run src/components/practice
npm run typecheck
npm run build
```

The focused practice suite passed (16 tests), the full frontend suite passed (152 tests across 32 files), typecheck passed, and the production build completed. Vite reported its existing large-chunk advisory for the configuration/Monaco chunks.

## Live review boundary and remaining integration

The preview browser reported unavailable, so the authorized read-only fallback was Chrome at `http://192.168.1.31/practice`. That server still serves the pre-change simulation page; the task explicitly forbids deployment before parent review. On the live coach route, the UI showed 12 last-known centers, `shot_underway`, and no usable camera image. The practice route header reported video loaded with freshness unavailable. The browser console contained only “WebSocket connected” and no JavaScript error.

This is not a live verification of the new overlay or green placement result. The implementation consumes the shared native image and producer `table_projection`/`table_position` contracts. Live green feedback still depends on the separate camera-recovery and perception work producing a current compatible projection, physical centers, and quiet-layout evidence. No camera service settings were changed or restarted, and nothing was deployed.
