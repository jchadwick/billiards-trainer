# Perception root-cause candidate — 2026-11-14

## Root cause

The previous actual-camera staging run collected 30 HTTP samples and 1,025
WebSocket messages over 60 seconds. Capture frame IDs advanced from 1191 to
2265, but every sample was stale, had zero balls, and reported
`not_visible / lost_rail_contrast`. A prior motion diagnostic from the same
camera showed `reference_drift` on a stationary-looking scene with only 0–1 of
14,400 consecutive samples changing per check; 43–73 reference-mask samples
flickered. The configured movement fraction treated that sparse reference-mask
jitter as motion. After filtering that signal, the appearance result still
overrode the independent motion classification and prevented scans.

## Implementation

- `motion_monitor.rs`: filter only the temporal reference-difference mask for
  spatially isolated noise. Consecutive-frame motion and broad-change checks
  remain unfiltered; the default neighborhood criterion is part of
  `MotionMonitorConfig`.
- `capture_runtime.rs` and `phase_scheduler.rs`: keep appearance conflicts out
  of cheap motion classification. A visibility conflict marks an existing
  whole-table layout stale and invalidates in-flight work, while retaining the
  quiet-motion dwell for later observations.
- `capture_pipeline.rs`: allow measured non-empty detections from an unconfirmed
  view to update tracks as partial observations, preserving unmatched centers
  as stale. Reject empty results from an unconfirmed view so they cannot clear
  the remembered layout or establish trusted emptiness.
- Tests cover reference-mask flicker, stale layout with retained quiet motion,
  non-empty unconfirmed observations, the existing exposure-shift publication,
  and the covered-empty preservation contract.

These changes keep the whole-table trusted/stale model in ADR-0114. Unconfirmed
non-empty measurements are published in the stale layout; an unconfirmed empty
result is not an authoritative replacement. No front-end, calibration,
runtime-config, model, camera-service, or production-service files were changed
for this candidate.

## Local verification

- `make test-rust`: passed (425 library tests; all targets passed).
- `make test-rust-camera`: passed (437 library tests; all targets passed).
- `make lint-rust`, `make format-check`, `make config-check`, and
  `git diff --check`: passed.
- The camera-feature suite had one transient timing failure when run directly
  after the complete suite under load; the failing motion-worker test passed
  alone and the complete camera-feature suite then passed on rerun.

## Native target build and camera staging

The candidate source overlay and binary hashes are recorded in
`deployment-perception-rootcause-source-manifest-20261114.json` (SHA-256 `7cd8c2b2a6545cc5fa3fcb6968afd3dad870f58b7ffb91eeab81996c93f4f772`). Six perception
files were overlaid on the retained dirty source snapshot at
`/home/jchadwick/build/billiards-deploy-20261114/source`, based on revision
`50236b6a3adc355d95f1f60518ef889e6e6f1c89`. The native Linux x86-64 release
build used the retained Cargo target cache and completed successfully; log:
`/home/jchadwick/build/billiards-deploy-20261114/perception-rootcause-build.log`.
Staging ran from
`/home/jchadwick/build/billiards-deploy-20261114/diagnose-perception-noise`
on isolated HTTP 18000, scene 49341, and frame-preview 49342.

With the table lights on, a 60-second actual-camera collection recorded 30/30
HTTP readiness and state responses as HTTP 200. Every HTTP sample was fresh,
trusted, and `visible / reference_match`, with 12 measured observations, 12
balls, and zero cached centers. The source frame advanced from 884 to 2049
(frame number 92 to 208). The WebSocket delivered 1,509 messages with no
errors; all were fresh and trusted with 12 measured observations, 12 balls,
and zero cached centers. Its source frame advanced from 884 to 2059 across 112
unique source frames (frame number 92 to 209; sequence 1247 to 2877).
Collection evidence is retained at
`/home/jchadwick/build/billiards-deploy-20261114/diagnose-perception-noise/live-evidence-lit.json`.
Producer diagnostics for the 122-second process lifetime reported about 19.8
captured FPS, 11.5 motion checks per second, 2 detector calls per second, zero
detector failures, two obsolete results, and 118,789 ms of trusted-layout
duration.

After collection, the staging process was terminated; no process from its
staging root remained and ports 18000, 49341, and 49342 were closed. The
production user service remained active as PID 68521 with zero restarts, and
the camera stream remained active as PID 1651 on port 8001. The candidate was
not promoted. This run verifies sustained quiet, visible-scene observations;
it does not claim a physical moving-ball or covered-table camera trial.

The earlier dim-camera run remains at
`/home/jchadwick/build/billiards-deploy-20261114/diagnose-perception-noise/live-evidence.json`.

The staged native binary SHA-256 values are:

- `billiards-server`: `5eae3b7e3a8bdf692dd6d6d6eb7770e178d840679a51bee5f654a0ebdc9f2413`
- `billiards-perception-live`: `ca1c9a22b14afcfbff14148709794019485e2b5919eef06f9d4b4ffe9c212cfc`
- `billiards-perception-gstreamer-probe`: `8f6a716a44a77748f892c0f9f713e5d403d5a64485941192591fc222be211350`
- `billiards-perception-camera-discovery`: `bfc94dfe302c240cfcbd2aa92d8ef30539f293ab313b347398a7c7ecfda0cc47`

## Independent review and retention correction

The source reviewer found that partial scans still used legacy track expiry,
which could remove unmatched remembered centers after one second and recycle
IDs on slow reobservations. The corrected partial-layout tracking path skips age
expiry and wholesale removal, while preserving source/context/order checks and
configured capacity bounds. Legacy callers retain their original age expiry.
The new `partial_layout_preserves_unmatched_centers_and_ids_beyond_legacy_expiry`
regression checks both matched and unmatched IDs beyond expiry, retained raw
measurement/capture provenance, and removal by a later clean scan. All 18
tracking integration tests pass. Independent source recheck approved the fix;
the target artifact must be rebuilt before promotion.


## Approved retention-fix rebuild and final staging attempt — 2026-11-14

The retention correction was committed at `41212de72ede41be4e75317205f521aaebf1c75f`
and synced to the retained native source snapshot. The rebuild completed at
2026-11-14 23:47:21 UTC. Exact binary and nine-file source hashes are recorded
in `deployment-perception-rootcause-source-manifest-20261114.json`. The local
seven-command verification runner completed with all exit codes zero, and the
commit hook passed format, Clippy, and Rust tests.

Final staging sampled the actual camera for 60 seconds (23:49:42–23:50:42 UTC).
All 30 `/api/v1/game/state` reads returned HTTP 200, but all 30 readiness reads
returned HTTP 503; state was stale with zero balls and zero measured centers.
Visibility reported `not_visible / lost_rail_contrast` throughout. The WebSocket
transport delivered 1,130 messages with no socket errors, but no localization
was available. A separate read-only camera frame was nearly black. The staged
server was stopped and all three isolated ports were closed. The candidate was
not promoted; the production trainer and camera services stayed active at PIDs
68521 and 1651 with zero restarts. Runtime config and calibration hashes stayed
unchanged. This run does not test localization under an illuminated scene and
does not qualify aiming or physical shooting accuracy.

## Authorized production promotion — 2026-11-15

The approved retention-fix candidate was promoted at 2026-11-15 00:00:58 UTC
under the user's existing instruction to deploy. The coordinating agent
removed its earlier readiness gate after inspecting the black camera image
and comparing the old production behavior. That gate is superseded for
software deployment: the current camera and previous production both show a black
frame, zero balls, a null frame number, and untrusted layout. Software
operation was verified independently from lighted-scene ball acceptance.

See `thoughts/deployment-20261114.md` and the updated source manifest for
installed hashes, frontend correction hashes, preservation checks, routes,
WebSocket and capture-diagnostics evidence, and rollback locations. No trust or
readiness logic was changed, and no synthetic data was injected. Deployment is
complete; ball acceptance under illumination remains pending.

Updated deployment manifest SHA-256: `6fc6afac59f637d79b54f873a6c035f617b3aece825269b92eb1bc0210c2653c`. Local bounded production evidence: `thoughts/deployment-production-evidence-20261115.json`.
