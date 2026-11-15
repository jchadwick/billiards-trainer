# Deployment localization fix — 2026-11-14

## Diagnosis and change

The failed deployment completed inference but never trusted a layout: its
bounded phase diagnostics recorded 876 detector calls, zero detector failures,
876 obsolete results, and zero trusted-layout milliseconds. A separate staged
run with motion diagnostics showed quiet motion samples being replaced with
`OccludedOrUncertain` by table-visibility assessment. The saved appearance
reference differed from the live camera by about 120 luminance levels, yielding
`ExtremeBrightnessShift` and `ExtremeColorShift`. Those are exposure/reference
mismatches, not positive evidence that the table is occluded. The runtime then
reset quiet continuity on every motion check and rejected every completed scan.

`capture_runtime.rs` now exposes those exposure/color results as visibility
`unknown` and lets the existing context, whole-table motion continuity, dwell,
and completed-scan admission rules decide trust. Strong cloth/rail conflicts
and incompatible visibility-reference geometry still invalidate continuity.
The appearance-evidence module comment now documents its separate role.

## Regression and verification

Added `exposure_shifted_visibility_reference_does_not_block_live_scan_publication`
to `perception/tests/live_capture_runtime.rs`. It sends stable captured frames
whose brightness differs from the stored reference by more than the configured
limit, then asserts trusted localization and a ball-bearing public-state
delivery over the production v3 scene transport. It also asserts visibility is
reported as unknown with the exposure reason.

- Targeted regression: passed.
- `make test-rust`: passed (424 library tests and all targets).
- `make test-rust-camera`: passed (436 library tests and all targets).
- `make lint-rust`, `make format-check`, and `make config-check`: passed.

## Target staging

Only `perception/src/capture_runtime.rs` was synced to the retained source
staging tree. Corrected native binaries were built using the target Rust 1.100
toolchain and cached cargo target, then copied into the separate package at
`/home/jchadwick/build/billiards-deploy-20261114/fix-package`.

On the actual camera in the isolated staging app (HTTP 18000, scene 49331,
preview 49332), `/api/v1/health/ready` returned HTTP 200 with fresh
localization. `/api/v1/game/state` reported frame 26, 11 balls, fresh state,
`layout_trusted=true`, phase `aiming`, and table visibility `unknown` with
`extreme_brightness_shift` (observation frame 275). This demonstrates live scan
admission despite the real exposure mismatch. Subsequent camera samples were
classified as moving/reference drift and the state correctly downgraded to
stale. The WebSocket was exercised on staging, but its bounded audit started
after that downgrade and did not capture a fresh trusted message; the local
end-to-end regression verifies trusted publication over the same scene
transport. No physical shot or accuracy claim was made.

Corrected source SHA-256:
`5cf5001d851a959f62ef92fb9327ba03491e9965666bdadd85e2630140df45b0`.

Corrected staging binary SHA-256 values:

- `billiards-perception-live`: `86453809ad08bafa618f9c23c7aa1ba56905348ce757eb74e63eed3802bc00b1`
- `billiards-server`: `7d296c344036d34e6e0a1a2e09b20e75102d3b637d7ad392882a309be343d4be`
- `billiards-perception-gstreamer-probe`:
  `de1f5dcda642b28cb3132a84909d26091321f0642c424e9a0191cf0f15e09055`
- `billiards-perception-camera-discovery`:
  `0e05f183f3a8a607978c463240d96d41a5a2a95eb24434d466ebdbe1452b4f42`

The active user service remains the rolled-back payload: `active`, zero
restarts, `/api/v1/health/live` HTTP 200. No staging process remains running;
the corrected package has not been promoted.

## Review correction before promotion

Independent review found that exposure checks returned before rail-contrast
checks. That could hide simultaneous structural loss. The final revision
therefore evaluates the existing rail-contrast threshold first; only a scene
whose rails remain legible may treat a large global exposure/color mismatch as
inconclusive. No thresholds were added or relaxed.

The runtime transport regression now derives a valid structured reference and
preserves rail structure under a 120-level exposure offset. A second regression
acquires balls, covers the view, and makes the injected detector return empty:
the previous balls must remain stale, never become a trusted empty replacement.
Both runtime regressions and the assessment suite passed. The review approved
this correction for rebuilding and actual-target verification.

The staging observations and binary hashes above describe the earlier exposure
candidate. The final rail-guard revision has now been built and verified on the
target. Its sustained live check failed readiness, so the working rollback
payload remains active and the candidate was not promoted.

## Final rail-guard target retry — 2026-11-14

The final build source overlay was captured in
`thoughts/deployment-railguard-source-manifest-20261114.json` (SHA-256
`72dedff0d46d781b55752009367902a455eb11609581757309abcadeaa68089c`). It used
base revision `50236b6a3adc355d95f1f60518ef889e6e6f1c89`, plus the exact current
`capture_runtime.rs`, `table_visibility.rs`, and two regression test files
listed in that manifest. The source was built from the retained validated
dirty snapshot without changing unrelated working files.

The native release build succeeded on hp-prodesk with Rust 1.100 and
`camera-gstreamer-source`; its log is
`/home/jchadwick/build/billiards-deploy-20261114/railguard-build.log`. All four
binaries were rebuilt and staged in
`/home/jchadwick/build/billiards-deploy-20261114/fix-package-railguard`.
`ldd` found no unresolved dependencies. The live producer diagnostics confirmed
OpenVINO `GPU.0` execution with
`compiled_model_execution_devices_gpu_only`; scheduled inference reported zero
failed jobs.

On the actual camera in the isolated candidate staging root, one initial
completed scan published a trusted localization and table projection. That
was not sustained. A 30-second concurrent check collected 15 HTTP live-health
samples (all 200), 15 readiness samples (all 503/not-ready/stale), 15 HTTP
game-state samples (all stale), 320 WebSocket state messages, and 2,037 producer
diagnostic messages. The WebSocket transport sequence advanced from 6658 to
6977 while all 320 messages retained the same localized source frame 1246, six
cached balls, zero currently observed balls, and `layout_trusted=false`. Every
producer diagnostic reported localization stalled with
`localization_expired`; inference itself remained healthy with zero failed
jobs. Visibility samples varied between `not_visible / lost_rail_contrast` and
`unknown / extreme_brightness_shift`. A camera still showed a heavily
underexposed table view, so the lost-rail result was consistent with the
captured image rather than a reason to bypass the guard.

Staged `/practice` and `/live` both returned 200 and matched the candidate HTML
hash; the referenced JS and CSS assets also returned 200 and matched their
package hashes. Production was not interrupted: `billiards-trainer.service`
remained active at PID 68521 with zero restarts, and
`billiards-camera-stream.service` remained active at PID 1651. Installed
config/calibration hashes remained at their baseline values. No final
localization candidate was installed.

Final rail-guard package binary SHA-256 values:

- `billiards-server`: `59c3462dff26130adda9179a6e4d139c5c1e619bd4347a55d1c7ee72cad78b1`
- `billiards-perception-live`: `eae28194795d0f44038777fba6f5ea366b5b44e0f9645774e9c8a94b9401ce33`
- `billiards-perception-gstreamer-probe`:
  `ce46d614983efc47aa214fa6451dd1c457101c54491b2b5df96678ee104a3f80`
- `billiards-perception-camera-discovery`:
  `76bf0a1014b4ebfbb251c70e9d68f998033e77d1e437e7eab29b43c38eb4fa68`

The remaining deployment blocker is sustained trustworthy visibility and
localization from the real camera view. Do not promote based on the initial
trusted scan.

## Final status

This record describes the earlier rail-guard attempt, not the active payload.
The subsequent reviewed motion-noise and partial-observation retention fixes
were deployed at 2026-11-15 00:00:58 UTC. See
[the deployment outcome](deployment-20261114.md) and
[the root-cause and retention record](deployment-perception-rootcause-fix-20261114.md).
The camera was black during promotion; lit-table ball acceptance remains
pending, independently of software deployment and transport verification.
