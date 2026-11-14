# Public-state contract

`public-state-delivery-v1.schema.json` defines the state sent by the Rust
server over HTTP and WebSocket. The JSON fixtures in this directory exercise
conversion of valid Rust scene snapshots into the public contract, including
observed and cached positions, cue status, calibration, source-clock semantics,
and invalid or unavailable observations. The Rust contract tests are the
source of truth for expected output.

`legacy-policy-v1.json` supplies the versioned canonical image dimensions and
assumed ball radius used by the compatibility renderer. These settings define
the shared rendering plane; they are not camera resolution or physical table
dimensions. When a scene measurement includes detector box half-extents, the
adapter publishes them as `radius_extent` in canonical pixels. Calibrated
`table_position` values use millimeters from the playing-surface center and are
projected from the selected detector/refined center using the matching table
calibration context. When that mapping exists, `table_radius_mm` supplies the
configured fixed physical radius (28.575 mm by default) for table-plane math;
the canonical pixel radius is only for screen rendering.
The coordinate origin is the playing-surface center. X increases from
calibration corner 0 toward corner 1; Y increases from corner 0 toward corner 3.
The raw source center is projected with
`ValidatedTableCalibrationV1::project(ImagePointPx, CalibrationContext)`, which
checks camera profile, calibration/rectification revisions, image space, and
source dimensions. This uses the existing calibration object; no additional
mapping service or contour is required.
`lifecycle-policy-v1.json` supplies
bounded storage defaults. A current layout supported by fresh whole-table
motion evidence remains trusted regardless of the full scan's age; motion or
evidence silence makes it stale while preserving last-known markers.

The adapter preserves a source producer identity, session, frame identity, and
timestamp for every observation. It rejects duplicate or out-of-order
observations, accepts forward gaps from latest-only delivery, and clears producer
state when a new producer epoch starts. Health records do not refresh ball
locations. Stale state preserves last-known centers but does not claim current
detector confidence. Only a clean settled full-table scan replaces the layout,
including removing absent balls or publishing a valid empty table.

`public-state-delivery-v2.schema.json` adds the phase-aware live envelope. A
delivery may carry a raw `source_snapshot` for a new localization, or omit it
for a cue/phase-only update. Both forms carry the complete retained
`current_state` when available and a `perception` object with independent ball,
motion, and cue evidence ages. The gateway rebases those ages to its local
monotonic clock. Phase-only updates never renew localization or per-ball
measurement timestamps. Fresh whole-table quiet evidence keeps a layout
trusted across slow periodic scans; motion-evidence silence makes it stale.
Cue capture age controls whether a cue line remains usable, without invalidating
the ball layout. Frame identifiers remain useful provenance but are not a
production center-marker join requirement.

The v2 perception status also carries `empty_layout_verified`. A raw successful
empty detector result remains in `source_snapshot` for evaluation, and this flag
becomes true after one clean successful full-table scan following quiet dwell.
Motion, occlusion, context changes, or expired whole-table motion evidence clear
the flag. Scan-to-scan consensus and scan-age expiry are not used.

Run the contract tests with:

```sh
make test-rust
```

## Native fitted outlines

A current accepted native-pixel observation may include `outline` on its public
ball: `{ "points": [{ "x": ..., "y": ... }, ...], "scale": [1, 1] }`.
Points use the same canonical coordinate conversion as `position`. The source
scene carries the corresponding optional `measurement.image_outline_px` in
native source pixels, and the resolved measurement retains the exact fitted
`BallOutline`. Its half-extents replace coarse box extents for that observation;
the raw detector box and midpoint remain unchanged.

Contours contain 3–512 finite, nondegenerate vertices bounded by the refiner's
ROI resource extent. Vertices may cross an image edge; consumers clip their
canvas rather than clamping individual vertices. Only an observed ball paired
with the exact source session, frame and timestamp may carry a precise outline.
Lost, predicted and retained observations omit it. The server verifies that
public vertices are the canonical conversion of the acknowledged source contour.
Absence of an accepted contour preserves the existing coarse footprint.

A fitted outline identifies image geometry, not independent physical-position
accuracy. Regulation diameter remains a fixed 57.15 mm, and table output is X/Y.

## Active table projection for phase-aware clients

Phase-aware `perception.table_projection` is emitted only when the current
capture session has a validated active table calibration. It identifies the
actual mapping used for production `table_position` values; static corner
configuration by itself is not evidence that a mapping was activated. The
object includes `calibration_revision`, `rectification_revision`, camera and
source-image context, physical playing-surface dimensions, and the
`pixel_to_table_mm` 3×3 homography. Its direction is source-image pixels to
table-plane millimeters. Ball `table_position` uses the playing-surface-center
origin, X from calibration corner 0 toward 1 and Y from corner 0 toward 3.
The frontend can invert this same homography to project table-space paths into
source-image pixels, then scale from `source_width_px`/`source_height_px` to
the displayed video. Reject the object if its camera, image-space, dimensions,
calibration revision, or rectification revision does not match the active
video context. `table_radius_mm` uses the configured physical ball radius;
it is not a measured contour. The ordinary validated table homography remains
the single coordinate mapping even when precision center refinement is
available, so reported world coordinates and display projection cannot diverge.

At the compatibility HTTP root, `table` carries this exact projection object
and `table_calibration_revision` carries its matching revision, or both are
null when the active mapping is unavailable. Phase-aware state also exposes
the object at `observation_status.perception.table_projection`.
