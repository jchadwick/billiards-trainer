# Target deployment record — 2026-11-14

## Outcome

Deployed the reviewed retention-fix candidate on hp-prodesk at
2026-11-15 00:00:58 UTC (November 14, 7:00:58 PM EST). The active native
source commit is `41212de72ede41be4e75317205f521aaebf1c75f`, built over the
preserved working-source snapshot. Fresh frontend assets include the final
manual cue-direction status labels. Health, routes, asset hashes, capture
progress, and state streaming passed verification; the trainer has zero
unexpected restarts. Config, calibration, launcher, and the separate camera
service were preserved. Rollback copies remain available.

The camera was black during final deployment, and the previous version also
reported no balls under those conditions. Readiness remains HTTP 503 and live
ball acceptance under illumination is pending. The predecessor candidate
passed a sustained lit-camera run before the small reviewed retention fix;
this is not a claim that the final deployed build has passed a lit-table or
physical shooting trial. Earlier unsuccessful attempts below are retained as
chronological evidence; the final promotion section records the current state.

Target: `hp-prodesk` (`192.168.1.31`), user service
`billiards-trainer.service`, runtime `/home/jchadwick/apps/billiards-trainer`.
The service binds to loopback at `http://127.0.0.1:8000/`. The separate
`billiards-camera-stream.service` remained active with PID 1651 throughout.

## Source and build

- Base revision: `e92fcc466e3d814063f1447696440499940d5eaa`.
- The source snapshot intentionally included the validated pre-existing dirty
  work; dirty status SHA-256:
  `ef6ab0eaad9d44800cbf7f2b5876352e05317a23e53083c35b6aa0ee801275a7`.
- Source manifest SHA-256:
  `001380de120fa33af934da808ebbf94571381c792d9e9000127cb7f13572bbaa`.
- Snapshot: `/home/jchadwick/build/billiards-deploy-20261114/source`.
- Native package: `/home/jchadwick/build/billiards-deploy-20261114/package`.
- Build log: `/home/jchadwick/build/billiards-deploy-20261114/build.log`.
- The installed target ONNX model, manifest and checkpoint were verified and
  reused; their hashes matched the staged model set.
- Target build required Rust 1.100.0, GStreamer/OpenSSL development packages,
  and `jq`. Rust was installed under the target user account. The system build
  packages were installed after sudo was supplied. No GPU driver, camera
  service, systemd unit or application runtime configuration was changed.

The frontend TypeScript check and production build completed. Rust release
build produced all four Linux x86-64 binaries; `ldd` reported no unresolved
libraries. The frontend and server started in an isolated staging root using a
copy of the target config and calibration store, with live capture disabled.
That check reported revision 5, both precision assets ready, and HTTP 200 for
`/practice` and `/live`.

## Promotion and rollback

The staged binaries and static assets were promoted at 2026-11-14 19:13:48 UTC.
They were rolled back at 19:18:24 UTC after failing the live localization
check. The new binaries and frontend remain in the build area and in retained
`frontend.failed.*` / `build-info.failed.*` paths for diagnosis.

Recovery copy:
`/home/jchadwick/apps/.billiards-trainer-rollback-20261114T191223Z`.

Rollback command, if needed:

```bash
/home/jchadwick/apps/.billiards-trainer-rollback-20261114T191223Z/rollback.sh
```

The runtime config, model files, data and calibration hashes matched the
pre-deployment backup after rollback. The service launcher was byte-identical
to the staged launcher and was preserved. The installed updater was preserved
because it targets system-scoped systemd, while this installation is a
user-scoped service.

## Runtime evidence

Before promotion, the live service was producing fresh localization from the
camera stream, with observed balls and advancing frame numbers. During the new
payload run:

- The gateway and supervised detector stayed active without a restart loop.
- Producer diagnostics showed capture progress, a warmed model, zero scheduled
  job failures, and effective OpenVINO execution on `GPU.0` with execution
  proof.
- The producer reported recent scene delivery, but gateway localization stayed
  `localization_not_received`; readiness remained HTTP 503/stale and game-state
  frame number remained null after several minutes.
- The separate camera stream continued to return JPEG frames.
- The new gateway did expose the saved table projection (revision 2), and the
  runtime config/calibration files were unchanged. The failure was specifically
  the live localization path.

After rollback, the trainer service is active with zero restarts, the camera
service still has PID 1651, `/practice` and `/live` return the restored frontend,
and the WebSocket returned advancing frame numbers 338 and 339. HTTP samples
showed fresh localization with 11–12 observed balls in three of four checks;
one sample was stale, consistent with the intermittent stale readiness seen
before deployment. The current service has the same behavior observed before
deployment. The restored older payload does not expose the table projection;
the new payload did, but cannot remain deployed until the localization
regression is resolved.

No physical shooting or accuracy qualification was performed.

## Final rail-guard retry — 2026-11-14

- The four approved source/test files were synced into the retained native
  source snapshot. Their final source manifest is recorded in
  `deployment-railguard-source-manifest-20261114.json` (SHA-256
  `72dedff0d46d781b55752009367902a455eb11609581757309abcadeaa68089c`). The
  base revision was `50236b6a3adc355d95f1f60518ef889e6e6f1c89`; the source state
  was dirty and the exact overlay hashes are in that manifest.
- A native Linux x86-64 release build completed at 20:20:32 UTC with
  `camera-gstreamer-source`. The final package is
  `/home/jchadwick/build/billiards-deploy-20261114/fix-package-railguard`;
  build log:
  `/home/jchadwick/build/billiards-deploy-20261114/railguard-build.log`.
  `ldd` reported no unresolved libraries. The OpenVINO runtime was supplied
  through the existing verified library path; the producer reported
  `GPU.0` and `compiled_model_execution_devices_gpu_only`, with zero failed
  scheduled jobs.
- Staging used `/home/jchadwick/build/billiards-deploy-20261114/diagnose-railguard`
  on HTTP 18000 / scene 49341 / preview 49342, with a copy of target config and
  calibration and the actual camera stream. The first completed scan briefly
  published trusted localization (source frame 345) with the table projection.
  Later visibility samples included `not_visible / lost_rail_contrast`; during
  the final 30-second collection the state reported stale / `motion_detected`
  while visibility was `unknown / extreme_brightness_shift`.
- Final 30-second overlapping collection: 15/15 live-health responses were
  HTTP 200; 15/15 readiness responses were HTTP 503 (`not_ready`, stale); and
  15/15 game-state reads were stale with 6 cached balls, 0 observed balls, and
  the same localized source frame (1246). The WebSocket delivered 320 state
  messages with advancing transport sequence 6658–6977 but the same localized
  source frame (1246), stale trust, 6 cached balls, and 0 observed balls.
  Diagnostics emitted 2,037 samples: GPU proof remained present, scheduled
  perception jobs had 0 failures, but localization was stalled with
  `localization_expired` throughout. There were no sampling errors.
- The staged `/practice` and `/live` routes returned HTTP 200 and matched the
  candidate `index.html` hash; the referenced JS and CSS also returned HTTP
  200 and matched the candidate asset hashes.
- The final candidate was not promoted. The production trainer service stayed
  active as PID 68521 with zero restarts, and the separate camera service stayed
  active as PID 1651. Target config and calibration hashes matched the
  pre-deployment baseline. No production payload or runtime state was changed.

The final candidate binary SHA-256 values are:

- `billiards-server`: `59c3462dff26130adda9179a6e4d139c5c1e619bd4347a55d1c7ee72cad78b1`
- `billiards-perception-live`: `eae28194795d0f44038777fba6f5ea366b5b44e0f9645774e9c8a94b9401ce33`
- `billiards-perception-gstreamer-probe`: `ce46d614983efc47aa214fa6451dd1c457101c54491b2b5df96678ee104a3f80`
- `billiards-perception-camera-discovery`: `76bf0a1014b4ebfbb251c70e9d68f998033e77d1e437e7eab29b43c38eb4fa68`

The installed rollback remains at
`/home/jchadwick/apps/.billiards-trainer-rollback-20261114T191223Z`; its
`rollback.sh` command remains available. Sustained real-camera visibility and
localization remain the blocker. Deployment is not complete based on the
initial trusted frame.

## Perception root-cause retry — 2026-11-14

The dim-scene retry was diagnosed from camera data: on a stationary-looking
view, consecutive-frame checks changed only 0–1 of 14,400 pixels, while
43–73 pixels flickered in the saved reference mask and kept the scheduler in
`moving / reference_drift`. After isolating that reference-mask noise, a
separate appearance result still overrode quiet motion and rejected scans. The
new candidate filters spatially isolated reference-mask flicker, keeps
appearance conflicts out of cheap motion classification, allows measured
non-empty partial observations while the layout is untrusted, and rejects
unverified empty scans so they cannot erase remembered centers. Source details
and the exact six-file overlay are in
`deployment-perception-rootcause-fix-20261114.md` and
`deployment-perception-rootcause-source-manifest-20261114.json` (SHA-256 `7cd8c2b2a6545cc5fa3fcb6968afd3dad870f58b7ffb91eeab81996c93f4f772`).

With the lights on, the isolated candidate produced sustained live-camera
observations: over 60 seconds all 30 HTTP readiness/state samples were HTTP
200, fresh, trusted, and had 12 measured balls. The WebSocket delivered 1,509
messages, all fresh and trusted with 12 measured balls and no errors; source
frames advanced across 112 unique frames. No cached centers were needed in
this quiet visible-scene run. The native build and binary hashes, detailed
evidence file, and sampling bounds are in the perception root-cause record.
The isolated staging process is stopped and its ports are closed. Production
remains on the rollback payload: the user service is active with zero restarts
and the separate camera service remains PID 1651. The candidate was not
promoted pending parent review. This run does not establish motion or
covered-view camera behavior beyond the regression tests.


## Approved retention-fix candidate — staged gate outcome — 2026-11-14

The approved retention correction was committed as
`41212de72ede41be4e75317205f521aaebf1c75f`. The retained dirty source
snapshot was updated at only the approved nine paths, then rebuilt natively
with `camera-gstreamer-source`. Build log:
`/home/jchadwick/build/billiards-deploy-20261114/retention-build.log`.
All four rebuilt binaries passed `ldd` with no unresolved libraries. Their
SHA-256 values and source-file overlay hashes are in
`deployment-perception-rootcause-source-manifest-20261114.json`.

The full local gate runner completed at 2026-11-14 23:46:59 UTC; all seven
commands passed: `make test`, `make test-rust-camera`, `make lint`,
`make format-check`, `make config-check`, `make test-deploy`, and
`make test-tooling`. The commit hook also passed formatting, Clippy, and
Rust tests. Only approved task hunks across the nine paths were committed. The
pre-existing dirty workspace remains uncommitted; the retained source snapshot
kept its validated dirty baseline, including the pre-existing `capture_runtime`
diagnostics already present in the predecessor candidate.

The rebuilt candidate was run on the real camera from 23:49:42 to 23:50:42
UTC on isolated ports 18000, 49341, and 49342. `/live` and `/practice` each
returned HTTP 200 with the candidate index hash; the referenced JS and CSS
also returned HTTP 200 and matched the candidate asset hashes. The functional
camera gate failed: readiness was HTTP 503 in 30/30 samples, game state was
HTTP 200 in 30/30 but stale with no balls or measured observations, and table
visibility was `not_visible / lost_rail_contrast` in all 30 samples. The
WebSocket delivered 1,130 messages without transport errors, but no localized
ball centers were available. A read-only frame from the camera stream was
nearly black. This is a camera-scene illumination failure, not evidence of a
new localization regression under a usable table view.

The staged candidate was stopped after collection; isolated ports are closed.
No promotion was performed. The installed payload remains the pre-deployment
rollback payload: trainer user service PID 68521, zero restarts; separate
camera service PID 1651, zero restarts. Active installed binary hashes are
recorded in the source manifest. Production config SHA-256 remains
`178e221274ad2146bd64187f726634858a436099e07e7e8048f0b16392da8b8f`, and the
calibration table remains
`58789941b1fd46e1b9215346c4e5dace4d5fe44136782ed97da1b2e66c9b507f`.
The launcher remains byte-identical. Existing rollback remains available at
`/home/jchadwick/apps/.billiards-trainer-rollback-20261114T191223Z`.

Deployment remains incomplete until the real camera presents an illuminated,
quiet table and this already-built candidate passes the staging functional
gate. No production config, calibration, model, data, camera service, or
runtime payload was changed during this attempt. No physical shooting or
accuracy qualification was performed.

## Authorized deployment of approved retention-fix candidate — 2026-11-15

The coordinating agent proceeded under the user's existing instruction to
deploy. Its independent inspection found a black camera image, and both the
previous production payload and candidate reported no balls under those
conditions. The agent removed its earlier illumination/readiness gate for
software deployment; the user was not asked for a separate darkness-specific
approval. Lighted-scene ball acceptance remains a follow-up requirement. No readiness or
trust logic was changed, and no synthetic input was used.

The approved native candidate from commit
`41212de72ede41be4e75317205f521aaebf1c75f` was promoted at
2026-11-15 00:00:58 UTC. Before promotion, all four candidate binary hashes
were checked against the approved manifest and `ldd` found no unresolved
libraries. The reviewed frontend status-label correction in
`frontend/web/src/components/video/LiveView.tsx` was built with
`npm run build -- --outDir /tmp/billiards-deploy-20261114-frontend-final`;
TypeScript and Vite production build passed. Production serves the resulting
index, LiveView JS, and CSS with HTTP 200 and exact matching hashes. The source
and asset hashes are recorded in
`deployment-perception-rootcause-source-manifest-20261114.json`.

The trainer user service is active as PID 89232 with zero restarts; the separate
camera service remained active as PID 1651 with zero restarts. Installed native
binary hashes match the approved candidate. Config SHA-256 remains
`178e221274ad2146bd64187f726634858a436099e07e7e8048f0b16392da8b8f`,
calibration SHA-256 remains
`58789941b1fd46e1b9215346c4e5dace4d5fe44136782ed97da1b2e66c9b507f`, and the
launcher remains byte-identical at SHA-256
`7b552d9609269ad4a146b6bb60abbf9840a23f8305adf595e69ad6303430e144`.
Existing immutable rollback remains at
`/home/jchadwick/apps/.billiards-trainer-rollback-20261114T191223Z`; an exact
pre-promotion copy is at
`/home/jchadwick/apps/.billiards-trainer-prepromote-20261115T000057Z`.

Post-promotion checks returned HTTP 200 for `/live`, `/practice`, `/coach`,
`/api/v1/health/live`, and `/api/v1/game/state`; readiness returned HTTP 503
because the scene was dark and untrusted. During 30.3 seconds, 15/15 game-state
reads were HTTP 200 with zero balls and a null frame, and 15/15 readiness reads
were HTTP 503. The state WebSocket connected and delivered 564 valid ordered
messages (sequence 799–1365) without parse errors. Diagnostics WebSocket
sampling observed 617 unique gateway revisions (8033–8697) and advancing
producer report sequence (7002–7595). Capture progress was healthy and recent;
scheduled perception jobs advanced from 2213 to 2398 with zero failures;
scene delivery stayed healthy. Localization remained unavailable because the
black scene supplied no trustworthy measurements. This is the expected
illumination limitation, not a candidate regression.

Detailed transport samples are retained at
`/home/jchadwick/build/billiards-deploy-20261114/production-postpromote-20261115T000058Z.json`.
The candidate is deployed. End-to-end ball acceptance must be repeated once
table lights are available; no physical shooting or accuracy qualification was
performed.

Updated deployment manifest SHA-256: `6fc6afac59f637d79b54f873a6c035f617b3aece825269b92eb1bc0210c2653c`. Local bounded production evidence: `thoughts/deployment-production-evidence-20261115.json`.
