# Phase-aware perception scheduling

The live Rust perception path keeps camera capture at its configured rate and
schedules analysis work independently. It favors a verified, stationary table
layout and a separately updated cue line over running the full detector on
every frame. During a shot, it stops full-layout and cue work while lightweight
motion checks continue. This milestone does not attempt to track moving-ball
trajectories or identify numbered balls.

## Runtime model

One coordinator owns phase selection, motion checks, cue refinement, tracking,
calibration, and publication. A dedicated worker owns the inference model. The
coordinator can continue processing motion and cue evidence while one detector
call is in flight. The worker accepts no pending-frame queue: a busy detector
causes that scheduled opportunity to be skipped, without queuing or catching
up later. Work is tagged with its source session, configuration and calibration
revisions, and motion generation so a result from an obsolete context cannot
restore guidance.

Capture and analysis rates are independent. The phase schedule controls
lightweight motion assessment, full ball-layout detection, and cue processing;
camera capture remains configured separately in the camera settings. The
initial schedule is:

| Phase | Motion checks | Full-layout detection | Cue processing |
|---|---:|---:|---:|
| Pre-shot, trusted layout | 15 Hz | 1 Hz verification | 5 Hz |
| Aiming, cue visible | 15 Hz | 1 Hz verification | 5 Hz |
| Shot underway | 15 Hz | Off | Off |
| Recovery | 15 Hz | 5 Hz | 5 Hz |

These values live in `perception.rust_live.phase_schedule` in the versioned
configuration defaults. Each phase has independent rates; the scheduler drops
missed slots rather than issuing bursts. Recorded-video replay uses the same
schedule by default and accepts a validated override with
`--phase-schedule-json`.

## Phases and state validity

The coordinator starts in recovery. It requires sustained quiet for 800 ms and
three mutually consistent successful full-table observations before it marks
a layout trusted. A visible, fresh cue on a trusted layout selects aiming. Cue
loss returns the state to pre-shot without discarding the verified balls.

Motion invalidates layout trust and guidance immediately. Confirmed motion
selects the shot phase; occlusion or uncertain whole-table changes invalidate
trust and select recovery without claiming that a shot occurred. Recovery
requires new quiet evidence and a new three-observation layout consensus.
Failed detections and missing frames do not count toward settlement.

The initial evidence limits are 200 ms for motion checks, 250 ms for cue
geometry, and 1.5 seconds between successful full-layout verifications. A
retained layout remains explicitly marked as retained, with its measurement
age visible; cue-only updates and skipped detections do not make the ball
measurements appear newer. If evidence expires, the source fails, or calibration
changes, consumers must remove guidance eligibility. These settings are
configurable with the phase schedule.

Cue endpoints and axis are image geometry. When calibration is compatible and
the fitted shaft crosses the playing-surface boundary, the endpoint on the
table is identified as the tip side and the endpoint outside as the butt side.
Image-based endpoint contrast remains a fallback when the table boundary
cannot orient the fit. Cue geometry is not a ball identity or a cue-ball-aware
coaching result, and should not by itself be presented as a complete shot
recommendation.

## Selecting rates

Use the reviewed recordings first, then repeat on the target camera. Keep model,
resolution, provider, exposure, and confidence thresholds fixed for every
comparison. For each candidate, warm up for 30 seconds and collect three
120-second windows, alternating candidate order. Collect detector and gateway
measurements independently. Use the replay report's phase schedule section for
effective rates, achieved motion/cue/detector rates, intentional detector skips,
trusted-layout duration, and recovery delays. Use `--annotations` together with
`--scene-snapshots-jsonl` to score ball measurements against reviewed labels.
Unlabeled or synthetic replay is useful for plumbing but cannot qualify
localization accuracy. Add `--phase-cue-geometry-jsonl` to retain fitted cue
lines and score their undirected axis against the cue endpoints in those same
reviewed labels.

Start with the defaults above. Change one rate only when its corresponding
acceptance check fails:

| Failure | Next rate experiment |
|---|---|
| Cue line feels delayed or cue evidence age exceeds 250 ms at p95 | Cue: 5 → 10 → 15 → 20 Hz |
| Recovery exceeds 2 seconds at p95 after motion stops | Recovery detection: 5 → 10 → 15 Hz |
| Stationary verification expires too often | Pre-shot/aiming detection: 1 → 2 → 5 Hz, after diagnosing missed detections |
| Motion invalidation is late or slow rolls are missed | Motion checks: 15 → 30 Hz, then inspect motion assessment |
| Cue axis misses its 1° p95 target | Correct the geometry or calibration and keep the rate fixed for comparison |
| A higher rate causes detector waiting or overload | Reject that setting; reduce per-call cost or retain the lower rate |

The provisional engineering gates are: no guidance more than 150 ms after
clearly visible ball movement; no false trusted layout in reviewed motion or
occlusion cases; recovery within 2 seconds at p95 after visible movement stops;
cue evidence age below 250 ms at p95 while trackable; localization recall no
more than one percentage point below the reference; p95 center error no more
than 10% above the reference; and cue-axis error within 1° at p95 on independent
annotations. There must also be no unbounded queue, sustained memory growth, or
shutdown failure. A failed geometry gate calls for an algorithm or calibration
fix, not simply a higher processing rate.

Keep the lowest schedule that passes the gates and feels responsive in an
aiming trial. Then run a 20-minute target-device soak with the live browser
open. Check shutdown and restart, evidence ages, accuracy, total resource use,
and memory. Save the selected phase profile and its evidence alongside the
previous qualified Rust build and configuration so the rate change can be
rolled back.
