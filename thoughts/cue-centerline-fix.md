# Cue centerline fix

## Root cause

The cue estimator was fitting a robust line to sparse high-gradient pixels and
returning that line directly. In the clean live frame, the stronger side of the
wood shaft dominated the edge sample set, so the overlay followed one boundary
of the cue. The frontend maps source-image coordinates to the displayed video
with the same scale and letterbox offset; the error was in the estimated line,
not the render projection.

The user-supplied screenshot has a yellow annotation and pink markers, so it was
used only to understand the report. No line fitting or pixel measurements were
taken from that annotated image.

## Implementation

`perception/src/cue_geometry.rs` now uses the edge fit for orientation and span,
then samples short cross-sections in the current RGB frame. It pairs the two
side transitions only when their width and interior/exterior contrast agree,
fits the midpoints, checks robust taper width percentiles, and returns that
centerline. Short spans reduce the section count according to a validated
spacing setting and can use multiple longitudinal profiles in one paired
cross-section. Bilinear sampling and valid in-frame exterior samples handle
subpixel boundaries and cues entering near a frame edge.

All new quality and work thresholds are `CueGeometryConfig` values with
defaults and validation. Histogram size, section count, profile samples, peak
count, point visits, and temporary storage remain bounded. The estimator is
fresh per frame and adds no temporal state.

Changed files:

- `perception/src/cue_geometry.rs`: paired-side centerline fit, profile sampling,
  small-frame handling, and configuration limits/defaults.
- `perception/tests/cue_geometry.rs`: contrast-reversal, taper, occlusion,
  competing parallel texture, no-held-line, config, and 16×16 cue regressions.
- `perception/src/cue_worker.rs`: test helper work and temporary-byte budgets
  raised for the bounded profile pass.

No frontend, diagnostics, practice, ball-localization, camera service, or
exposure settings were changed.

## Preserved baseline

Before editing, the workspace state and cue-only diff were saved at:

- `/tmp/billiards-cue-baseline.7LWA9R/status.txt`
- `/tmp/billiards-cue-baseline.7LWA9R/tracked.diff`
- `/tmp/billiards-cue-baseline.7LWA9R/cue_geometry.diff`
- `/tmp/billiards-cue-baseline.7LWA9R/HEAD`
- `/tmp/billiards-cue-baseline.7LWA9R/cue_measure.before`

The baseline cue binary SHA-256 is
`c2a8724bab407d41f79f529fdcdc6e3bbeda6aabeada7cfb3e0af1905af76f96`.
The pre-edit working-tree cue source reconstructed from the saved diff exactly
matched the target's cached deployment source
(`f21f4dbc048a4edbb153f9ff371096cf78845d09e7ad02f64ae18dff802f879c`), so the
Linux candidate could be built by replacing only the staged cue geometry
source. The saved task-only patch is
`/tmp/billiards-cue-baseline.7LWA9R/cue_geometry.ours.diff` (SHA-256
`1691eed702dfcf7f9842e9ab4e2354b7c257da203ab2d6cad69c101e6b3ff03b`). The
test/worker patch is
`/tmp/billiards-cue-baseline.7LWA9R/cue-tests.ours.diff`.

## Clean live-frame measurement

Three unannotated 3840×2160 JPEG snapshots were fetched from the live camera at
`http://127.0.0.1:8001/snapshot` over SSH. The cue and balls stayed placed; JPEG
hashes differ between snapshots because the stream is live.

| UTC capture | Raw JPEG SHA-256 | Old mean center offset | New mean center offset | Raw sections |
| --- | --- | ---: | ---: | ---: |
| 2026-11-19 05:28:00 | `/tmp/billiards-cue-sequence-1.jpg` — `de457859042803d4ea79094b8e4cf5d9c5a7f7343d622aa822dcd893adc32935` | 9.090 px | 0.313 px | 10/12 |
| 2026-11-19 05:28:01 | `/tmp/billiards-cue-sequence-2.jpg` — `98926e387a2105333ee6db1675f08e593c92861b4a2e1ce07d7f95c9602ba042` | 8.934 px | 0.177 px | 9/12 |
| 2026-11-19 05:28:01 | `/tmp/billiards-cue-sequence-3.jpg` — `86f71f3f2fa7ea83f0a32ae167da4030b163352223ccd4f5e7fbf9f326ba6128` | 9.076 px | 0.541 px | 10/12 |

The independent raw-profile measurement finds the visible wood-shaft band in
cross-sections near the shaft's tip-side half, then measures each estimator
line's perpendicular distance to those raw band midpoints. Across 29 usable
profiles, the old estimator averaged 9.03 px from center and the new estimator
0.34 px (96.2% lower). This is a same-camera pixel measurement, not a separate
ground-truth annotation study. Every new estimate reported `observation_age_ms`
0; no previous cue was reused.

## Verification

- `cargo test --manifest-path perception/Cargo.toml -- --test-threads=1` —
  passed the full perception crate suite.
- `cargo test --manifest-path perception/Cargo.toml --test cue_geometry` —
  13 passed, including edge-contrast reversal, taper, partial occlusion,
  parallel texture, thumbnail resolution, and immediate cue disappearance.
- `cargo test --manifest-path perception/Cargo.toml --test live_capture_runtime
  -- --test-threads=1` — 26 passed.
- `cargo clippy --manifest-path perception/Cargo.toml --all-targets -- -D
  warnings` — passed.
- `rustfmt --edition 2021 --check perception/src/cue_geometry.rs
  perception/src/cue_worker.rs perception/tests/cue_geometry.rs` — passed.

## Linux candidate

The host is reachable and the deployed hashes before staging were:

- `billiards-server`: `f5026270da82d09ec5aeee137a4dbf9700bf71d4f3e5e2b57b80eb47f6cd452b`
- `billiards-perception-live`: `ff817005046353dae175d4724f2eed7f233a8938cc700b66bb9fb193f219a9e5`
- `billiards-perception-gstreamer-probe`: `4678dd75bc27b4b422ddbbda2c2909a30cfeca3d7c0a9e4dcf70caa695512e2a`

An isolated build source and baseline artifacts were created at
`/home/jchadwick/build/billiards-cue-centerline-20261119`. The staged cue
source SHA-256 is
`7ec4d67d30056012d7ea437584527a5cbcea28c793e2124b2f1b41b8ebe3a3b2`. The
release build used the existing Cargo target cache at
`/home/jchadwick/build/billiards-deploy-20261114/cargo-target`, then copied
outputs to `artifacts/candidate/`:

- `billiards-server`: `9e0054f826e5dc083f325d06bc2d5e37796c39d7cfa04adfb671e8724203aaf8`
- `billiards-perception-live`: `e962e6c0cfecea3344848d96f66aa6f9d274b9469655213d7c0c5b4e304ea992`
- `billiards-perception-gstreamer-probe`: `0e1632cad0b75c6ff07051292dbc0fe69c93aa651510a3ac7be1aad0cfc5c2f9`

After the build, the deployed app files still matched their original hashes
above. The first candidate binaries were x86-64 ELF files and their shared
libraries resolved, but the live binary had been built without the required
`camera-gstreamer-source` feature. The first promotion attempt exposed this at
runtime and was automatically rolled back. The corrected build and production
outcome are recorded below.

## Production promotion and live check

At 2026-11-19 06:52:11 UTC, the first promotion used the feature-disabled live
binary. The worker exited with `HTTP-MJPEG capture was not compiled`; the
guarded deploy restored the original native payload and restarted the user
service at 06:52:52 UTC. The camera process stayed at PID 1651. This was an
intentional rollback after an artifact/runtime mismatch, not a cue-geometry
regression.

The corrected Linux live binary was rebuilt from the same staged source with
`--features camera-gstreamer-source`. Its `--help` output included
`gstreamer-http-mjpeg`, its ELF dependencies resolved, and its SHA-256 is
`9225eb50b7b75669cefde89c8fd4d5ad3c7ccc31264247a296e45604a8cb80fe`. The
`billiards-server` candidate remained
`9e0054f826e5dc083f325d06bc2d5e37796c39d7cfa04adfb671e8724203aaf8`; the
existing capture helper was left in place at
`4678dd75bc27b4b422ddbbda2c2909a30cfeca3d7c0a9e4dcf70caa695512e2a`.

The corrected native payload was promoted at 2026-11-19 06:57:34 UTC. The
current `billiards-trainer.service` is active with MainPID 319838. The camera
process remains PID 1651. `/api/v1/health/live` and `/live` return HTTP 200;
`/api/v1/health/ready` returns HTTP 503 with `freshness=stale` and
`localization_available=false`. The camera `/snapshot` endpoint returns HTTP
200. No configuration, calibration, model, frontend, launcher, or service-unit
files were promoted. The captured hashes for `config.json`, table calibration,
ONNX/checkpoint models, optics/ball-center calibration, and the user service
unit still match their pre-deploy values.

The live page at `http://192.168.1.31/live` loaded the real camera stream from
`http://192.168.1.31:8001/stream` and received advancing production WebSocket
state messages: sequence 7291–7298 and perception event sequence 7393–7400.
All eight messages contained zero balls, a null cue result, and table
visibility `not_visible`; cue evidence age remained 76–198 ms, so no previous
cue line was held after a negative observation. The UI reports
`lost_rail_contrast`. A raw 3840×2160 camera snapshot captured at 2026-11-19
07:02 UTC was entirely black (SHA-256
`8ed716c788ed934f7a5ead8b30a3234041008f883ac2431206b5337ec1f1891f`). This
explains the current no-cue/no-ball state and means a centered overlay could
not be visually confirmed in the current production frame. The earlier three
clean live images and their 29 raw shaft cross-sections remain the available
centerline evidence; their 0.34 px mean offset is not a guarantee for a future
frame or this black live image.

The browser preview opened the production page. Automated visual snapshots
timed out, so the current UI state was checked through the page and its actual
WebSocket; the camera snapshot itself was inspected. The production health and
stream checks above are from the promoted service, not a fake producer reply.

## Task-only source integration

The local cue file already contained a separate dirty proposal-line seed before
this task. To keep that work out of the task commit, the committed centerline
change is merged onto HEAD without that seed. The deployed source snapshot
retained the pre-existing seed and has source SHA-256
`7ec4d67d30056012d7ea437584527a5cbcea28c793e2124b2f1b41b8ebe3a3b2`. Its
task-specific integration changes pass `rgb8` through both `fit_region` call
sites and its signature, and reserve
`profile_visit_budget(width, height, config)` inside the seed's point-visit
budget. The proposal-seed implementation itself is not part of the task-only
commit.

The immutable rollback payload is
`/home/jchadwick/build/billiards-cue-centerline-20261119/rollback/production-20261119T065121Z`
(read-only directory and files). Its `SHA256SUMS` matches the three original
production native binaries listed above. The only service restarts were the
authorized first promotion, its automatic rollback, and the corrected
promotion; there were zero unexpected restarts.
