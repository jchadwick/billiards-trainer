# R10 scheduler and corner-ball availability

Date: 2026-10-29

## Scope

This is a saved-image application-pipeline regression report. It does not use
the camera. The source JPEG is a 720p table image upscaled to 3840 × 2160 and
served repeatedly as a 5 FPS MJPEG source. The Intel GPU package decodes those
frames; the Coral device is not used for inference. These runs measure pipeline
continuity and localization availability on one static scene, not independent
physical accuracy.

## Corner proposal and refiner result

The live R10 output repeatedly detected the far upper-left pocket ball at
confidence `0.8676758`, well above the configured detector threshold `0.3`.
The detector confidence threshold was not changed. The missing table position
was caused by the refiner rejecting the center as `unstable_center`.

The coarse live box reconstructed from the published center and half-extents is
`[250.875, 270.75, 367.875, 371.25]` source pixels. With the production center
spread limit `1.5 px`, six seed fits qualified, their greatest pairwise spread
was `1.5241 px`, and the full reported spread was `1.6741 px`; the center was
rejected. At `1.6 px`, seven seed fits agreed within `1.5241 px`, producing
center `(309.2295, 321.3342)`. At `1.75 px`, a single initial fit passed without
the seed consensus, so `1.6 px` was the smallest tested change that accepted
the proposal while retaining the complete multi-seed check.

The accepted saved-replay state mapped that center to provisional table
coordinates `(-1114.10, -527.69) mm` from the playing-surface center. That map
was fitted from the existing nine user-measured points. Its measurement
uncertainty is unknown; this coordinate is not independent accuracy evidence.

### Paired threshold comparison

Only `maximum_center_spread_px` was changed between the paired runs; detector
confidence, proposals, image inputs, and all other refiner settings were held
fixed. The reproducible summary is `paired-center-spread-audit-r10.json`.

| Paired corpus | 1.5 px | 1.6 px | Center or status changes |
|---|---:|---:|---|
| 15 saved Intel-GPU detector proposals | 15 accepted; 9/9 referenced | 15 accepted; 9/9 referenced | none |
| 18 saved proposal-seed variants | 16 accepted | 16 accepted | none |
| 297 synthetic known-center cases | 285 accepted | 285 accepted | none; maximum center delta 0 px |
| Exact reconstructed far-corner live proposal | rejected as `unstable_center` | accepted after 7 agreeing seed fits | this proposal only |

The synthetic set had no neighboring-ball errors; its maximum accepted center
error was 0.112 px and p95 was 0.058 px at either setting. This validates the
paired implementation behavior on those synthetic inputs, not real-world
optical accuracy. The 18 proposal variants have no independent physical truth.
The exact far-corner rescue therefore supports 1.6 px as a provisional
availability setting without evidence of a collateral change in these saved
corpora; it does not establish a physical center-error bound. The detector
confidence threshold remained 0.3 throughout, and lowering it would not have
changed this case because its confidence was already 0.8677.

## Target replay results

Selected replay settings: 3840 × 2160 processing, 5 FPS saved MJPEG, Coral
input shape unchanged at `[1,1280,1280,3]`, OpenVINO Intel GPU inference,
VA-API JPEG decode, 15 Hz motion scheduling ceiling, and 1 Hz ball
detection/refinement while aiming. The source stream, not the scheduler, limited
effective motion work to the delivered frame cadence.

| Metric | Result |
|---|---:|
| Replay duration | 60.0 s |
| Captured source frames / cadence | 304 / 5.002 FPS |
| Motion checks / effective cadence | 304 / 5.002 FPS |
| Detector calls | 57 / 0.938 calls/s |
| Cue observations | 171 |
| Detector, cue, overload, obsolete-result failures | 0 |
| Public state samples | 600 |
| Samples with 15/15 precise positions | 600 |
| Samples with fresh state | 600 |
| Period below all 15 balls | none |

At 5 Hz scheduling against the same nominal 5 FPS source, only 176 of 304
motion opportunities ran in 60.8 s. At 6 Hz, a 20.8 s replay produced 89
motion checks over 104 frames (4.28 Hz). At 15 Hz scheduling, all 304 captured
frames received a motion check. Scheduling above the source cadence therefore
absorbs frame-clock jitter; it does not request additional camera frames.

The runtime now makes one scheduling decision per captured frame. Phase and
evidence updates after motion or asynchronous cue completion refresh scheduler
state without consuming a second frame's cadence deadline. Timeouts likewise
advance evidence expiry without consuming work slots. The added regression test
covers the cue-completion boundary immediately after a 200 ms cadence deadline.

## Proven and still unproven

Proven on this fixture:

- A detector-confidence adjustment cannot fix this missing coordinate: the
  corner object already arrives at confidence `0.8676758` against threshold
  `0.3`.
- A `1.6 px` center-consistency limit accepts the exact reconstructed live box
  with multi-seed agreement and yields 15/15 fresh saved-scene positions for a
  continuous 60-second replay.
- The paired comparison changes only that far-corner proposal among the
  available saved and synthetic cases; all other case outcomes and centers are
  identical at 1.5 and 1.6 px. This supports 1.6 as a provisional availability
  adjustment, not as accuracy qualification.
- A 15 Hz motion scheduling ceiling lets a 5 FPS source deliver 5 motion
  checks per second despite host/capture jitter, with no worker overload in the
  replay.
- The prior replay bundle can be left active and restored after qualification;
  all application tests used a separate qualification directory.

Not proven:

- The source is an upscaled 720p image, not a native 4K camera frame.
- Repeating one static scene does not validate transitions, motion blur,
  exposure changes, all pocket occlusions, or general camera cadence.
- The nine-point ball-center mapping has held-out errors around 8.8 mm RMS and
  11.1 mm maximum in the current diagnostic audit; user measurement uncertainty
  is unknown. The requested millimeter accuracy target is not established.
- The far-corner accepted center has no independent physical reference.
- Cue geometry, shot invalidation, and shot coaching were not exercised by this
  no-cue static scene.
- No production deployment or camera stream change was made.

## Reproduction artifacts

- Source JPEG SHA-256:
  `93bbee0684828a40b7aa5b7ec2acb97ff00299454a6ac6f7d1a3569d5b11f7c6`
- VA-API decoded RGB SHA-256:
  `27072950b21179742d9fe84bad5610ff7a7daac1eab9c15d3da76a20bbea18c6`
- Qualification binary SHA-256:
  `83e8cff02c0557d95656d008a7abde137549b39baca9d1bc56c456c3a7cd9931`
- The 60-second report is `target-saved-live-r10-spread16-motion15-60s-observer.json`.
- The final public state is `target-saved-live-r10-spread16-motion15-last-state.json`.
- The exact coarse-box refiner experiment is preserved in
  `offline_exact_live_box_r10.rs` and its output in
  `exact-live-corner-box-r10.json`.
- The saved-only qualification script and per-frame logs are retained beside
  this report. The machine-specific package and its test calibration remain on
  the target under `billiards-qualification-20261029/` and are not production
  eligible.

The R10 runs record refiner source SHA-256
`7eae8d4dda67be8d710d49e0626ec42dbb5bc45e8d492bd2dea303dda9a0fd22`. That
source snapshot is absent from this branch, available local Git history, and
the archived superseded-run bundle; the current
`perception/src/ball_localization.rs` has a different hash. The R10 reports,
inputs, harnesses, and measured outputs are preserved as historical evidence,
but these results cannot be reproduced byte-for-byte from the current source
tree. The qualification binary is external; only its hash is preserved.
