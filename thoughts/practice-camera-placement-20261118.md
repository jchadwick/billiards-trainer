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

At that pre-promotion checkpoint, this was not a live verification of the new overlay or green placement result. The implementation consumes the shared native image and producer `table_projection`/`table_position` contracts. Live green feedback still depends on the separate camera-recovery and perception work producing a current compatible projection, physical centers, and quiet-layout evidence. No camera service settings were changed or restarted at that checkpoint.


## Production browser verification after promotion — 2026-11-19

The independently reviewed practice implementation and candidate 7 recovery
were deployed together at 2026-11-19 03:42:47 UTC. Production used the camera
image from `http://192.168.1.31:8001/stream`, with natural dimensions
3840×2160, plus a current trusted projection and 12 observed physical centers.
The app used one shared image through the SPA route cycle `/live → /practice →
/coach → /live`; the image remained loaded and moved between its expected
camera host and practice overlay slot. The `/coach` route remained a separate
live coaching view.

On `/practice`, the target editor, camera overlay, two target markers, and two
±45 mm tolerance regions were visible. A fresh browser API read supplied the
current detected centers; the UI form set the cue-position target to 872,
693 mm and Target 1 to 995, 694 mm. Those corresponded to the two selected
measured centers near 872.482, 693.305 mm and 994.771, 693.934 mm on the
active 2,235.2 × 1,117.6 mm table projection. The page reported `READY`, with
the measured centers 1 mm and 0 mm from their target slots. Ten other detected
centers remained visible and did not prevent this two-target exercise from
becoming ready. The route showed the live image overlay rather than the old
simulation sandbox.

Moving Target 1 to 39, 39 mm put it more than 1,059 mm from the nearest
current physical center, and the page withdrew Ready (`NOT VERIFIED`, 1/2
slots placed). Restoring the target to 995, 694 mm restored `READY` (2/2).
This was a browser-only target edit: no ball was moved or described as
physically placed, and no synthetic producer response was injected. The
isolated Playwright browser reported no page or console errors and no failed
HTTP responses. Screenshots are retained in `/tmp` as
`billiards-candidate7-production-live.png`,
`billiards-candidate7-production-practice-ready.png`, and
`billiards-candidate7-production-coach.png`.

The camera viewer URL correction remains persisted in production config, whose
SHA-256 is `4fb40299e23a59151f61af01a4724cd5dd5d72d79e35fead8083c00184dbf5e3`.
The previous dark snapshot was taken with Playroom Lights off; lights were
turned on and remain on. The current production browser confirms the actual
camera image and 12 aligned markers, distinguishing that original lighting
condition and prior service state from the now-ready detection pipeline.

The user declined the physical roll trial. Motion-to-settled reacquisition,
pocketing, and pickup therefore remain unverified on a live shot; this does
not block the authorized deployment.
