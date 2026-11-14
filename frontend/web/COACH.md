# Live center-ball coach

Open **Practice → Live Coach**, or navigate directly to `/coach` on the same app
address you normally use. The live coach shares the existing camera receiver,
calibration, and pushed table state; it does not start another camera stream.

## Try a shot at your table

1. In Configuration, choose the camera and calibrate the four playing-surface
   corners and table dimensions if that has not already been done.
2. Place a cue ball and an object ball for a straight pot. Let the table settle
   until the coach reports a trusted layout.
3. Under **Cue ball**, select the label drawn on the cue ball. Labels identify
   reported tracks, not recognized ball numbers. Select your target and pocket,
   or leave them on automatic suggestion.
4. Bring a level cue behind the cue ball, aimed through its center, without
   English. The detected shaft appears yellow. Keep the cue tip reasonably close
   to the ball and the shaft visible to the camera.
5. Compare the green suggested aim with cyan current cue-ball travel. Pink shows
   the first struck object's initial direction; orange is the intended pot path.
   The alignment message reports the angular difference and shaft offset. An
   aligned result means within the configured geometric tolerance, not a
   guaranteed pot.
6. If automatic cue-tip direction is unknown, check **I confirm the cue tip is
   the end nearest my selected cue ball** only when that statement is true.
   This explicitly supplies the missing tip orientation; it does not bypass
   center-line, confidence, distance, or layout checks. Confirmation clears when
   the cue expires, motion invalidates the layout, the layout revision changes,
   or you change the selected cue ball, camera, or calibration.

If no cue is detected, suggested pot guidance still works. A cue that points
away from the selected ball or misses its center does not produce a measured
trajectory. Moving, stale, or incompatible table evidence pauses predictions;
last-known ball markers may remain as dim context.

## Model scope

This first version estimates straight initial travel from the observed cue axis
and circular physical ball geometry. It finds the first cue-ball collision and
the struck object's initial direction along the collision normal. The object
path ends at its next ball contact or the cushion boundary, accounting for ball
radius. It does not continue a combination or simulate a rebound.

The overlay does not predict speed, stopping positions, pocket capture, throw,
English, or post-contact cue-ball rolling/follow/draw. A center-ball stroke can
develop roll before impact, so a missing post-impact cue path must not be read
as a stop-shot prediction. All guidance depends on calibration and detection
quality. Begin with a simple straight shot to check alignment against your
physical table.

## Configuration

These optional Vite build settings adjust the live aim policy. Invalid or absent
values use the defaults. Rebuild the frontend after changing them.

| Setting | Default | Accepted range | Purpose |
| --- | --- | --- | --- |
| `VITE_COACH_ANGLE_TOLERANCE_DEG` | `2` | `0.1`–`10` | Maximum angular difference for an aligned result; the predicted first ball must also match the suggested target. |
| `VITE_COACH_CENTER_TOLERANCE_RADIUS` | `0.25` | `0.01`–`1` | Maximum lateral shaft offset as a fraction of cue-ball radius. |
| `VITE_COACH_MAX_TIP_GAP_MM` | `600` | `10`–`2000` | Maximum projected forward gap from the tip endpoint to the cue-ball center. |
| `VITE_COACH_MIN_CUE_CONFIDENCE` | `0.5` | `0`–`1` | Minimum producer cue confidence for measured aim. |

## Verification and serving

Run `npm test`, `npm run typecheck`, and `npm run build` from `frontend/web`.
Tests cover alignment and invalid shaft geometry, signed and explicitly
confirmed direction, coordinate projection, collision clipping, and live
freshness/motion/calibration behavior. The production build writes to
`backend/static`; refresh a gateway serving that directory to load it. A bundled
installation needs the rebuilt assets in its own configured frontend directory.
Hardware validation still requires a running camera and gateway at your table.
