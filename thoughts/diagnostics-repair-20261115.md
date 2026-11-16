# Live diagnostics repair — 2026-11-15/16

## Root cause

`/live` renders `Diagnostics`, whose camera preview embeds `LiveView`. `LiveView`
had become the full live-coach page. The diagnostics CSS hid its header and ball
panel, but the coach status, instructions, cue/target/pocket inputs, alignment
message, and advice remained inside the camera card. This pushed the video down
and moved the rest of the diagnostic page below the fold.

## Repair

- Added `LiveView.presentation`, defaulting to `coach`; `/coach` keeps its
  existing full coaching view.
- `Diagnostics` requests the `diagnostics` presentation. It renders the same
  shared video pane and overlay canvas while omitting coach-only controls and
  the coach-only ball list. The diagnostics page retains its pause, expand,
  overlay, observation, and interpretation controls.
- Removed CSS that hid pieces of the coach page. The diagnostics view selects
  its markup explicitly instead of clipping it.
- The marker contract stays independent of frame IDs: compatible ball centers
  use solid markers for trusted layouts and dashed, translucent markers for
  untrusted layouts. No API, perception, backend, or camera code changed.

## Independent review and final correction

The independent reviewer approved the layout change and found one remaining
control issue: with native camera delivery, the overlay checkbox was still
disabled and the helper text implied exact frame pairing. The final correction
removed that native-delivery disable condition, leaving the checkbox disabled
only while inspection is frozen. Its explanation now reads “Ball centers show
trusted or last-known table positions.” The Diagnostics test mock uses native
delivery and verifies the checkbox can hide and restore overlays. The
`LiveViewFreshness` test verifies diagnostics mode keeps the independent-frame
overlay behavior and removes coach controls.

## Verification

- `npm test`: 30 files and 138 tests passed.
- `npm run typecheck`: passed.
- `npm run build -- --outDir /tmp/billiards-diagnostics-repair-build`: passed.
  The build includes the current diagnostics/admin frontend files from the
  working tree. Vite emitted its existing large-chunk advisory; the build
  completed successfully.
- The T3 preview browser reported unavailable. I used Playwright with installed
  Chrome and SSH forwards to the deployed host's HTTP service and camera
  stream.
- `/live`, `/coach`, `/practice`, and the new hashed JavaScript/CSS assets
  returned HTTP 200. The served index and assets matched the production build
  hashes recorded below.
- Before repair, the deployed `/live` screenshot showed coach controls inside
  the diagnostic camera card: those controls measured 301 px tall, the preview
  958 px tall, and video began at y=764. The retained pre-promotion frontend
  showed no coach controls, a 579 px preview, and video beginning at y=385.
  Evidence: `/tmp/billiards-production-current-live-before.png` and
  `/tmp/billiards-prepromote-live-before.png`.
- On repaired deployed `/live`, at a 1600×1100 viewport, the camera preview measured
  1190×669 px at y=385. There was no coach header, ball panel, coaching region,
  “Live coach” text, or “Confirm cue ball” control. The enabled, checked
  “Detection overlays” checkbox hid the canvas (`visibility: hidden`) and
  restored it (`visibility: visible`). Expand changed the preview from
  1190×669 to 1534×715 px and collapse restored the interpretation panel.
- The deployed camera image loaded at 3840×2160. Pause correctly reported
  “This camera does not permit browser image capture. Inspection remains
  live.” No uncaught browser page errors occurred. One WebSocket cancellation
  was logged when the verification navigated between routes; one stream request
  was aborted by that same navigation.
- On `/coach`, the live header, ball panel, live-shot coaching region, and coach
  title remained present. No live-ball success is claimed: while the camera
  image visibly contained balls, the producer reported zero ball observations,
  phase `recovering`, and an untrusted layout. Cue observation was unavailable.

## Frontend-only deployment

Promoted at **2026-11-16 03:52:38 UTC** to the active user service on
`hp-prodesk`. The app serves static frontend files from
`/home/jchadwick/apps/billiards-trainer/frontend`; the active unit was
`systemd --user billiards-trainer.service`. Promotion copied the fresh
immutable assets and static files, then atomically replaced `index.html`. No
service restart or `update.sh` run was needed.

Before and after promotion, these values matched:

- Approved `billiards-server` SHA-256:
  `6b7eee4dd01f877ab7d0694179333c4133f7f38dc320e23eec82214f42dca832`
- Runtime `config.json` SHA-256:
  `178e221274ad2146bd64187f726634858a436099e07e7e8048f0b16392da8b8f`
- `calibration_table.json` SHA-256:
  `58789941b1fd46e1b9215346c4e5dace4d5fe44136782ed97da1b2e66c9b507f`
- Camera process PID: `1651` (`ustreamer`, port 8001)

A frontend-only rollback copy is retained at
`/home/jchadwick/apps/.billiards-diagnostics-prepromote-20261116T035238Z/frontend`.
The existing server binary, native artifacts, configuration, calibration,
models, data, environment, and services were preserved.

Build asset hashes:

- `index.html`: `203e818b00ac7ab05b215b7df30694223ffcd14f4ae111424e09d53ac732c1cb`
- `assets/index-CK9z3W6I.js`:
  `657adfc4acc694a2d34328ce8a2e7cccd40fcf504c09dcb1d662ca88df37a515`
- `assets/LiveView-BR8Za3L7.js`:
  `9ae7a95e421527733f85180e9c62a81cfbc4c2180d06cacfec585fbfa95c8b8a`
- `assets/index-Wvoo9HFZ.css`:
  `1b3521c07dda6026a2fb0670f26c793fdec27837ef90b09fa8d0fd019865692f`

Repaired deployed screenshots and SHA-256:

- `/tmp/billiards-diagnostics-repair-deployed-live.png` —
  `9c4d06ba238ac7c8c7568718d9c3deaeca78ab9eacb0d25d62dbdae8faaa510a`
- `/tmp/billiards-diagnostics-repair-deployed-expanded.png` —
  `1368d3ba33d215f17467bdd2114438cc452872de50ad3e6679f46f59a8090c9e`

## Working-tree scope

The repository had extensive unrelated modified and untracked work. The
Diagnostics component and its test were already untracked, and `styles.css`
contained unrelated edits. I left those files in the working tree without
staging their complete contents. `thoughts/diagnostics-repair-integration.patch`
records only the task-owned Diagnostics, Diagnostics-test, and CSS hunks.
The narrow tracked `LiveView.tsx` and `LiveViewFreshness.test.tsx` changes and
this repair record are committed separately.
