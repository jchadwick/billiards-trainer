# Ball Detection Recovery — 2026-11-18

## Current outcome — promoted to production

Candidate 7 and the separately reviewed practice camera overlay are deployed
to the production gateway on HTTP port 80. Promotion completed at
2026-11-19 03:42:47 UTC. The user service now runs `billiards-server` PID
311066, live producer PID 311074, and capture helper PID 311093 under wrapper
PID 311059. Systemd reports the service active with `NRestarts=0`. The camera
service remains PID 1651 on port 8001; isolated staging on ports 18000, 49341,
and 49342 has been stopped.

Production `/api/v1/health/live` returned `alive`; `/api/v1/health/ready`
returned `{"freshness":"fresh","localization_available":true,"status":"ready"}`.
The live game state reported 12/12 observed centers, a trusted `aiming`
layout, visible table appearance, and fresh state. A production browser loaded
the actual `http://192.168.1.31:8001/stream` image at 3840×2160 and showed all
12 markers over the visible balls. Its state WebSocket delivered 20 messages
with 12 observed balls in every message and advancing source frame IDs.

The deployed `/practice` route uses that same camera image, target markers,
and tolerance regions; it no longer presents the simulation sandbox. In an
isolated Playwright browser context, two editable target slots set through the
page form to current measured centers showed `READY`. Moving one target to a
point more than 1,059 mm from every current center withdrew Ready; restoring
the measured position restored Ready. `/live → /practice → /coach → /live`
retained exactly one loaded camera image through the route transitions. The
browser recorded no page errors, console errors, or HTTP failures. No physical
ball was moved during this check.

The active camera viewer URL remains
`http://192.168.1.31:8001/stream`. The production config hash remains
`4fb40299e23a59151f61af01a4724cd5dd5d72d79e35fead8083c00184dbf5e3`; the
calibration table hash remains
`58789941b1fd46e1b9215346c4e5dace4d5fe44136782ed97da1b2e66c9b507f`. Model,
calibration assets, runtime environment, camera process, launcher, and systemd
drop-ins were preserved. The existing OpenVINO `LD_LIBRARY_PATH` is still
active.

Promoted native SHA-256 values:

- `billiards-server`: `f5026270da82d09ec5aeee137a4dbf9700bf71d4f3e5e2b57b80eb47f6cd452b`
- `billiards-perception-live`: `ff817005046353dae175d4724f2eed7f233a8938cc700b66bb9fb193f219a9e5`
- `billiards-perception-gstreamer-probe`: `4678dd75bc27b4b422ddbbda2c2909a30cfeca3d7c0a9e4dcf70caa695512e2a`
- `billiards-perception-camera-discovery`: `b421b7283345904ad26c08d3bda7aab0463517b8ec89d6644cf9b229eac2c404`

The native source manifest is deployed at
`/home/jchadwick/apps/billiards-trainer/candidate7-native-source-manifest.json`
(SHA-256 `5da8aaa30deee89e01b46858aca9887600194eabf8319663702ff67ed7f6c9d1`).
The 32-file frontend manifest is at
`/home/jchadwick/apps/billiards-trainer/candidate7-frontend-manifest-20261119T033811Z.sha256`
(SHA-256 `d9bf06826bde059f426258a669e379ef3ad1ca22f0ef0134d53455d2abe9f828`).
The frontend was built from repository HEAD `cc0c398323db86b7170bb232c3f9d677bcb3022f`
plus the complete dirty frontend worktree, which had 114 modified/untracked
paths at build time. `candidate7-build-info.txt` records that provenance and
the typecheck, 152-test, and production-build results. The fresh frontend
passed typecheck, all 152 tests across 32 files, and Vite production build.
Vite printed its existing large-chunk advisory.

Full pre-promotion native and frontend backups are retained at
`/home/jchadwick/apps/.billiards-candidate7-rollback-20261119T033759Z`.
Earlier statements in this note that candidate 7 was staged but not promoted
describe the pre-promotion checkpoint and are superseded by this outcome.

## Goal

Deliver useful live aiming feedback from the real camera. Keep the last good ball centers available between scans, publish fresh measured centers when available, and keep phase/shot gates as cadence controls rather than hard stops on ball work.

## Findings

- The parent corrected the production viewer URL to `http://192.168.1.31:8001/stream`; the copied staging config still contains that value.
- Home Assistant API reported `light.playroom_lights` was `off`. I turned it on at 2026-11-18 20:50:21 UTC and verified the entity state became `on`. The direct 1280×720 camera snapshot changed from a very dark table to a well-lit image with 12 visible balls.
- On the lit snapshot, the native OpenVINO GPU provider (`GPU.0`) loaded the production ONNX model and returned 12 class-1 `ball` detections plus one class-0 `cue` detection in each of five repeated still-image frames. Live model settings were image size 1280, confidence 0.30, IoU 0.20, and maximum detections 16. Ball confidence ranged from about 0.88 to 0.95. This isolates the previous dark-scene shortfall from a model/provider startup failure.
- The still-running production service remained stale after the lights were restored: HTTP state showed 12 cached balls, zero current observations, phase `shot_underway`, `trust_reason=motion_detected`, and `/api/v1/health/ready` returned 503. Its configured `shot.ball_hz` is 0. The production user service was not restarted or modified during this investigation (`MainPID=251340`, `NRestarts=0`).
- A 60-second isolated run after the illumination change, before restarting that isolated process, also showed zero fresh observations while retaining 12 positions. After starting a fresh isolated candidate against the same bright, static camera scene, it immediately produced 12 measured centers and regained valid guidance. This means a fresh process recovers; recovery of an already-running process after a major lighting transition still needs review and a targeted follow-up. Do not present the process restart as proof of automatic recovery.

## Candidate changes

- `perception/src/phase_scheduler.rs`: a zero ball cadence in any phase now falls back to the lowest positive configured ball cadence; all-zero configuration falls back to the default pre-shot rate. Zero motion and cue cadences remain disabled. This keeps a `shot_underway` phase from suppressing ball scans.
- `perception/src/capture_runtime.rs`: a fresh nonempty scan from a disallowed motion generation or unsettled phase is published as an untrusted partial result. It updates measured centers while retaining unmatched last-known tracks; only a clean current scan can restore whole-layout trust. Empty or stale results still cannot erase the last known layout.
- `perception/src/motion_monitor.rs` and `perception/tests/motion_monitor.rs`: reference-mask hysteresis filters local flicker and the finite reference-drift dwell can settle after gradual brightness/layout mismatch. Regressions cover quiet recovery, slow ball-sized motion, and threshold flicker.
- `perception/src/scene_transport.rs`: an unexpired ball-bearing capture scene now survives a later phase-only update in the single pending transport slot; phase updates follow it. Deterministic tests cover pending and handshake cases.
- `perception/tests/live_capture_runtime.rs`: the blocked-scan motion regression now requires a fresh nonempty scan to publish as an untrusted partial observation, and synchronizes motion with the in-flight detector deterministically.

### Independent review fixes

- The Sol High review found that a stable brightness offset from the old reference could leave motion evidence uncertain forever. The monitor now keeps abrupt frame-to-frame shifts uncertain, then starts the existing finite settle dwell once consecutive frames stabilize despite the old reference's brightness offset.
- Detector admission now rejects over-age results whenever motion or a confirmed table-appearance conflict limits a result to partial use. A slow clean scan captured during uninterrupted quiet can still verify the layout. When appearance monitoring is configured, an empty result cannot clear positions while visibility is `Unknown` or `NotVisible`; nonempty detections remain usable.
- Gateway lifecycle output keeps per-ball freshness from a recent nonempty partial scan even when whole-layout trust is false. Guidance stays invalid, cached unmatched tracks stay unobserved, and an untrusted empty payload still reports positions as unavailable.
- The live scene writer now has two bounded coalescing slots: one latest scene and one latest phase-only follow-up. A later scene supersedes an older event; otherwise a queued event follows the scene so current guidance invalidation is not lost.

These changes do not loosen detector confidence or localization precision checks and do not fabricate detections.

## Verification

- Local `make test-rust`: passed (430 library tests and all targets).
- Local `make test-rust-camera`: passed (442 library tests and all camera-feature targets).
- `make lint-rust`, `make format-check`, `make config-check`, and `git diff --check`: passed.
- Target-native Linux regressions with `camera-gstreamer-source`: `motion_monitor` 12/12, `live_capture_runtime` 14/14, and `scene_transport::tests` 15/15 passed.
- After the review fixes, local `make test-rust` passed (432 library tests and all targets), and `make test-rust-camera` passed (444 library tests and all camera-feature targets). Focused `motion_monitor` (13), `live_capture_runtime` (14), `public_state_lifecycle` (10), `capture_runtime` (4), and `scene_transport::tests` (16) suites passed. Clippy with `-D warnings`, formatting, configuration validation, and `git diff --check` also passed.
- Final lit-camera candidate soak, 2026-11-18 20:58:11–20:59:12 UTC: HTTP state 29/29 fresh, trusted, and nonempty with exactly 12 observed centers per sample; readiness 29/29 returned ready. WebSocket delivered 1,494/1,494 fresh, trusted, nonempty states with zero empty states. Each of 12 tracks advanced through 130 unique source frames (129 advances), frame 2693 to 3900. Ball observation age was 403–743 ms in the first/last HTTP sample. Diagnostics recorded 1,154 expected and completed jobs, with zero failed jobs.
- T3 browser loaded the unmodified app route at `http://127.0.0.1:18080/live`. It loaded the real camera URL `http://192.168.1.31:8001/stream` with HTTP 200 and 3840×2160 natural dimensions. No response override/interception was used. The screenshot showed 12 markers aligned with the 12 physical balls, `Layout trust=Usable`, `Guidance validity=Valid`, and fresh motion/cue state.
- User declined the requested slow-roll trial for now, so moving-to-settled behavior has not been checked on the real table.

## Candidate artifacts

Retained Linux source and Cargo cache:

- `/home/jchadwick/build/billiards-deploy-20261114/source`
- `/home/jchadwick/build/billiards-deploy-20261114/cargo-target`

Isolated candidate directory:

- `/home/jchadwick/build/billiards-deploy-20261114/diagnose-ball-detection-20261118`
- Build command: `cargo build --release --locked --features camera-gstreamer-source --bins`
- Native candidate SHA-256:
  - `billiards-server`: `318729b46c5ff7e49a8a2d2af00effa431032f5880fa52db4de4b70739b368e0`
  - `billiards-perception-live`: `5436e81fe3f198924d594cc94d7db209ced2e63b6c7e8f4b68cbd53c94c5fd99`
  - `billiards-perception-gstreamer-probe`: `12b1ad4e8a425c1052f2bff425b4103fbe8096e60b4ee088b19240dfc31dc3d5`
  - `billiards-perception-camera-discovery`: `7d9f06de8febef8749e8e2f30875870dc92a125404435d780cd9f745c6c2ee8e`
- `ldd` found no missing shared libraries.
- Raw detector input and report are under `evidence/raw-detector-lit-20261118/`; the JPEG SHA-256 is `bc5ad088b03c61f3d8fb690694401a8ae8cea731faa97d583de9444fb5b8e2c1`.
- Original isolated stage binaries are saved in `bin-backup-before-post-review-candidate-20261118/`.

## Handoff / remaining work

The lit, static-table workflow succeeded on the previous isolated candidate after a fresh process start, and the raw detector saw all 12 balls. Production still runs the prior binary and remains stale in `shot_underway`; the viewer URL correction and light change alone did not restore current detections there. The post-review candidate is being rebuilt and rechecked against the lit camera before another independent review. No code is promoted to production. Leave the Playroom Lights on. Re-run the real-table movement-to-settled check when the user is ready.

No candidate binary was promoted to production. The production service and camera processes were left unchanged.

## Review 2 races and candidate verification — 2026-11-18

The second Sol High review found two visibility races and a lighting settle
bypass. The candidate now applies these corrections:

- An empty detector result can replace the layout only when its own capture
  frame positively shows the calibrated table. A newer table-wide
  `NotVisible` assessment also makes an in-flight result partial; measured
  centers still publish and whole-layout guidance becomes invalid.
- A newer `NotVisible` frame invalidates trust even if the scheduler was
  already untrusted when that frame arrived. Tests cover a delayed empty scan
  and a `NotVisible` detector frame between slower visibility checks.
- Every abrupt lighting shift starts a new settle dwell, including a return
  to the previously verified brightness.

The pool-specific absence rule stays separate from table-wide visibility. A
clean, settled full-table scan removes every unmatched track. This naturally
covers a pocketed ball near a pocket and a ball picked up elsewhere. Partial
or uncertain scans retain last-good centers until a clean full scan confirms
what remains on the table. There is no pocket-proximity heuristic, detector
confidence change, or fabricated observation.

Final verification:

- `make test-rust`: passed (433 library tests and all targets).
- `make test-rust-camera`: passed (445 library tests and all camera-feature
  targets).
- After making the reverse-order visibility test independent of frame timing,
  the final `live_capture_runtime` target passed 17/17 locally and 17/17 on
  native Linux. `motion_monitor` passed 15/15 on both; the scene retry
  regression passed on native Linux.
- `make lint-rust`, `make format-check`, `make config-check`, and scoped
  `git diff --check`: passed.
- Native Linux release candidate rebuilt from the retained dirty source at
  `/home/jchadwick/build/billiards-deploy-20261114/source` using the shared
  Cargo target at `/home/jchadwick/build/billiards-deploy-20261114/cargo-target`.
  The isolated candidate is under
  `/home/jchadwick/build/billiards-deploy-20261114/diagnose-ball-detection-20261118/bin/`.
  Server SHA-256 is
  `cd0d574312e0a094fcabfb3dc5118aa014d95a69b4665358b63e9a3b1863a063`;
  live producer SHA-256 is
  `816ebed05602dccab6729442824366642f8c8a13521bd0e6a7bb913b59166554`;
  the GStreamer probe SHA-256 is
  `eb2fe9bdeb7b2f54db17e104a06debd7ed4f1e9700f682e10e2270cee8459fa8`;
  camera discovery SHA-256 is
  `825415d26f8045b1eb48ff1a9b8f224dfa9e7964fa5cf72225d1803b7f0cf9ce`.
  All four binaries passed `ldd` without unresolved libraries.
- The staged config SHA-256 remains
  `93b89b4065e8495a0b6b891d753aa3a54150eb05c9da654b1b4ffcbce0bceb01`;
  its viewer URL is `http://192.168.1.31:8001/stream`. The producer uses the
  same camera service at `http://localhost:8001/stream` on the target host.
  A real producer preview descriptor reported RGB8, raw image space,
  1280×720, source frame `4525`; its JPEG preview hash is
  `9506c12db814117b55efd6b0f2c79f66d4f456265983a98b84f674b7c7d41cd0`.
- The no-interception browser check loaded `http://127.0.0.1:18080/live`,
  received the actual 3840×2160 MJPEG stream with HTTP 200, and showed 12
  measured markers aligned with the 12 balls. HTTP state showed 12/12
  observed balls, trusted layout, and valid guidance. The browser labels video
  freshness unknown because the MJPEG stream has no frame timestamp; ball
  state freshness was available. Screenshot:
  `/tmp/billiards-stage-browser-review3-20261118.png`.
- A 60-second static-camera soak completed at
  `/home/jchadwick/build/billiards-deploy-20261114/diagnose-ball-detection-20261118/evidence/review3-soak/websocket/20261118-224354-3f5f5aa6/`.
  HTTP sampling reported 30/30 ready, trusted, guidance-valid, nonempty states
  with 12/12 observed balls; sampled ball ages ranged from 263 to 954 ms. The
  WebSocket audit received 1,513/1,513 complete fresh state messages, with
  zero invalid JSON, zero stale or cached/failed updates, and 136 distinct
  advancing source frames. Reported state publication age was p50 250 ms and
  p95 254 ms; this measures read-completion-to-publication age, not
  cross-process camera age.
- Producer diagnostics reported OpenVINO `GPU.0`, 7,559 completed scheduled
  jobs, zero failed jobs, and one ball detection in flight. At shutdown, the
  runtime had captured 9,444 frames, run 4,981 motion checks and 1,100 ball
  detections, with zero detector failures, zero obsolete results, zero
  visibility-rejected empty scans, and 471,827 ms of trusted layout during a
  483,679 ms session.
- The isolated server was stopped; ports 18000, 49341, and 49342 are closed.
  Production remains active at PID 251340 with zero restarts; camera PID 1651
  remains active. No candidate code was promoted.

The physical table remained static during this check. The user had declined a
slow-roll trial, so moving-to-settled and pocketed-ball behavior remain
unverified on a live shot; clean-scan removal semantics are covered by
`tracking_v1::clean_full_layout_replaces_absent_tracks_and_accepts_a_valid_empty_result`.

## Candidate 5 follow-up — 2026-11-18

The next independent review found that uncertain nonempty scans still needed
stronger admission and publication coverage. The candidate now keeps the
latest capture-time high-water for uncertain and `NotVisible` appearance
frames, so a delayed clean scan cannot clear last-good centers after newer
contradictory evidence. `Unknown` coverage is always partial, and missing
optional appearance reference data does not prevent a settled nonempty scan
from establishing useful guidance. A skipped empty or over-age scan publishes
its invalidation before returning.

Partial tracking now protects every persistent last-good track when the
configured active-track capacity is full. Fresh overflow detections are still
published as measured centers with unique session IDs, without evicting the
remembered tracks or extending tracker history past its configured bound. A
later clean full scan reconciles the layout and removes balls that are absent,
whether pocketed or picked up. No pocket-proximity heuristic or confidence
threshold change was added.

The added runtime regression establishes trusted, guidance-valid state from a
synthetic visible cue and verifies that the first wire scene for an uncertain
partial frame contains measured and retained centers with guidance invalid.
Other focused coverage verifies delayed empty-result rejection after
`Unknown → Visible`, nonempty `Unknown` partial retention, missing-reference
recovery, and bounded partial tracking under capacity pressure.

Verification on this source snapshot:

- `make test-rust`: passed, including 433 library tests, 22 live runtime tests,
  and 19 tracker integration tests.
- `make test-rust-camera`: passed, including 445 library tests, all
  camera-feature targets, 22 live runtime tests, and 19 tracker integration
  tests.
- `make lint-rust`, `make format-check`, `make config-check`, and scoped
  `git diff --check`: passed.

This candidate has not yet been rebuilt or staged on native Linux. The isolated
lit-camera browser soak recorded above belongs to the prior candidate; it is
not evidence for this source revision. Production remains untouched. Live
slow-roll, settled reacquisition, pocketing, and pickup transitions still need
physical-table confirmation when the user is ready.

## Candidate 7: focused recovery and staging gate

This pass addresses only defects that could lose known centers, advertise
stale guidance beside new measurements, skip a shot-onset detection, or stop
the live pipeline. A layout replacement now requires positive visibility on
the detector's exact frame; uncertain nonempty scans still publish measured
balls while retaining unmatched last-good centers. The gateway publishes a
scene and its perception state in one update. Due detector work is no longer
cancelled by a motion-generation change, though that generation still blocks
the result from replacing the layout. Startup rejects track/detection limits
that could exceed the public-state ball bound.

The deterministic live-wire tests keep subsequent source frames gated until
they assert the first delivery for the exact partial frame. They cover
pocket-side and interior last-known centers, clean removal after a covered
scan, verified empty-table reconciliation, and shot-onset detection.

Verification on this source snapshot:

- `make test-rust`: passed.
- `make test-rust-camera`: passed, including 448 library tests, all
  camera-feature targets, 26 live runtime tests, and 20 tracker integration
  tests.
- `make lint-rust`, `cargo fmt --manifest-path perception/Cargo.toml --all
  --check`, `make config-check`, and scoped `git diff --check`: passed.

At the review checkpoint, the native Linux build and isolated staging had not
yet been run for this candidate. They have now been completed on the reviewed
source snapshot; current build and static-camera evidence follows below.
Production remains untouched. Physical slow-roll, settled reacquisition,
pocketing, and pickup remain for the user's table trial.

## Candidate 7 native build and isolated staging — 2026-11-19

The independent review of the accepted-scene `layout_basis_frame_id` fix found
no remaining P0/P1 blocker and recommended isolated staging. I synced only the
nine reviewed perception source, test, and contract files into the retained
native Linux source snapshot after verifying its pre-sync backup. Their hashes
match the local source. I did not replace the dirty native tree or modify the
frontend. The native release build completed with
`cargo build --release --locked --features camera-gstreamer-source --bins`.
The manifest records the base commit, source-file hashes, build-log hash,
configuration hashes, and binary hashes at
`/home/jchadwick/build/billiards-deploy-20261114/diagnose-ball-detection-20261118/candidate7-manifest.json`.

The staged config preserves the target camera URL, current model, and copied
calibration data. Its only difference from the exact target config is the
staging-only frame-preview port, `49322` → `49342`. The four native binaries
are in
`/home/jchadwick/build/billiards-deploy-20261114/diagnose-ball-detection-20261118/bin/`;
all four passed `ldd` with no unresolved libraries:

- `billiards-server`: `f5026270da82d09ec5aeee137a4dbf9700bf71d4f3e5e2b57b80eb47f6cd452b`
- `billiards-perception-live`: `ff817005046353dae175d4724f2eed7f233a8938cc700b66bb9fb193f219a9e5`
- `billiards-perception-gstreamer-probe`: `4678dd75bc27b4b422ddbbda2c2909a30cfeca3d7c0a9e4dcf70caa695512e2a`
- `billiards-perception-camera-discovery`: `b421b7283345904ad26c08d3bda7aab0463517b8ec89d6644cf9b229eac2c404`

The candidate is running separately at `http://192.168.1.31:18000/live`, with
scene ingress on loopback port `49341` and frame preview on loopback port
`49342`. The first launch lacked the OpenVINO library path inherited by the
production detector; I stopped that isolated attempt and relaunched with the
same read-only runtime library path. The candidate now loads OpenVINO and uses
the target's camera service at `http://localhost:8001/stream`. Production
processes were not stopped or restarted: the existing gateway and detector
remain PIDs `251346` and `251353`, and camera PID `1651` remains active.

The no-interception browser check used the staged `/live` page and fetched the
actual camera URL, not a response override. The displayed image had
`naturalWidth=3840`, `naturalHeight=2160`, and 12 detector markers aligned to
the visible balls. The live page showed a visible table, usable trusted
layout, 12 measured centers, and valid guidance. The producer publishes
1280×720 detector frames from the 3840×2160 camera stream.

On a static table, 64 seconds of HTTP sampling returned 57/57 nonempty states
with 12/12 balls observed, layout trusted, guidance valid, table visible, and
freshness `fresh`; all 11 readiness checks passed. Source frame IDs advanced
from `5279` to `6551`. A 60-second browser WebSocket observation recorded
1,104 state messages and 103 distinct advancing source frames, each with 12
observed balls and trusted visible layout. Guidance was valid on 977/1,104
messages. A separate 15-second, 200-ms HTTP trace found 73/75 valid-guidance
samples; the two invalid samples kept all 12 centers and trusted layout, and
were caused by cue evidence age just above the configured 250-ms freshness
limit (269 and 283 ms). Guidance recovered automatically. The browser's own
status remained `Receiving`; the extra observer saw abnormal WebSocket closes
(code 1006) and reconnected.

At this pre-promotion checkpoint, `/practice` identified itself as a simulated
sandbox and did not consume live camera data. The real camera and detector
overlay for that trial were available at `/live`; the practice sandbox had not
yet been changed. No ball was moved during that static verification, so motion,
settle recovery, pocketing, and pickup remained physical-table checks. The
stage was isolated and left running for the user's physical trial; production
had not yet been promoted. Candidate 7 was subsequently promoted as recorded
at the top of this note.


## Scoped source commit and provenance

The verbatim native candidate manifest is retained at
`thoughts/ball-detection-candidate7-native-manifest.json`. Its SHA-256 is
`5da8aaa30deee89e01b46858aca9887600194eabf8319663702ff67ed7f6c9d1`. All
nine scoped source hashes in that manifest matched the local files and the
native candidate source snapshot before the selective commit.

The selective commit includes the task-owned scheduler, motion monitor, scene
transport, lifecycle, tracking, and capture-pipeline changes; focused scheduler,
motion, lifecycle, and tracking tests; public-state contract updates; and these
deployment records. It does not stage the entire dirty
`perception/src/capture_runtime.rs` or `perception/src/server.rs`. Those files
also contain pre-existing producer-diagnostics/administration work. The
modified `perception/tests/live_capture_runtime.rs` is also left out as a
whole-file change because its runtime test additions are mixed with other
scheduler/runtime work. The deployed candidate source manifest records the
exact versions used: `capture_runtime.rs` SHA-256
`b7e3034234690d80ea95048c6546406ae973b1baa9b433ec78494bcf6032a335`,
`server.rs` SHA-256
`8f80cfad5d71709b417ec5619adcec2d17bd26ce822f693e089065c013992842`, and
`live_capture_runtime.rs` SHA-256
`a6ee23c9de8fdeb3e63b20756641b7f507536b74f9c5592d39b2756183be82fc`. Their
candidate versions remain in the preserved native build source snapshot and
are represented by the promoted binary hashes above. Other dirty
administration, diagnostics, and calibration files stayed outside this
selective commit.
