# 20 × 14 inch checkerboard capture session

## View 001 — center

The first full-resolution image was captured from the user-provided camera
snapshot endpoint on 2026-10-27 local time. The original 3840 × 2160 JPEG is
`view-001.jpg`; `view-001-annotated.png` shows all 54 detected internal
corners. The detector needed a bright-paper crop because the checkerboard was
small within the complete table image. Corner coordinates in the manifest are
translated back to the original, uncropped 4K frame.

The detected neighboring-corner spacing has a 71.64 px median (66.93 px
minimum). This establishes that the current target is readily detectable in
this 4K frame. It is a centered, nearly front-facing view; it does not
establish an optics model, table mapping, or ball-location accuracy.

## View 002 — upper-left

The second original 3840 × 2160 snapshot is `view-002.jpg`, with its 54-corner
overlay in `view-002-annotated.png`. The target moved to the upper-left region
and remains fully visible. Median neighboring-corner spacing is 75.19 px
(71.73 px minimum). This adds spatial coverage, but the observed board is still
nearly front-facing; it is not counted as a strong tilted view.

The active print scale and flat rigid backing have not yet been independently
verified for this assembled 20 × 14 inch target. Its nominal square pitch is
50.8 mm, but the manifest deliberately leaves both verification fields
`pending`. Do not use this session to fit or activate calibration until those
physical properties are confirmed and the remaining views are collected.

Future captures must keep the camera geometry unchanged, use this same target,
and include distinct board tilts and positions across the image. Repeated
captures without moving the target are not independent views; flat translations
alone are insufficient to establish camera intrinsics. Neither captured view
has been used to fit or activate a calibration.
