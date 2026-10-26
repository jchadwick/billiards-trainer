# Project memory

## High-frequency diagnostics must be inert when disabled

Motion-check diagnostics are opt-in through `BILLIARDS_MOTION_DIAGNOSTICS=1`.
The live runtime reads this flag once when it is constructed. When diagnostics
are disabled, the per-frame path must not build diagnostic records, serialize
JSON, allocate diagnostic buffers, or write logs; it should only take the
disabled boolean branch. Keep any future high-rate perception diagnostics
behind the same kind of early gate, and verify that the gate precedes all
formatting and I/O.

## 2026-10-25: User requirements for ball geometry and camera correction

Treat the table as a two-dimensional X/Y grid. All balls have the same fixed
physical diameter and the same height. Z-axis estimation and variable-height
modeling are outside the requested scope.

The required output is accurate ball-center X/Y positions and a consistent
physical ball diameter everywhere on the table, including the far edges.
Position-dependent apparent elongation or diameter in the image must be
addressed through image calibration and correction; it must not be represented
as a change in physical ball dimensions. Retain the current camera.

Evaluate position and diameter accuracy at both the table center and the edges.
These are the user's application requirements and modeling scope, rather than
a claim that all optical effects have the same physical cause.

## 2026-10-26: User-measured playing area

The user measured the playing area inside the bumpers, between opposing
cushion noses, as 45 1/2 inches by 92 inches. The exact metric conversions are
1155.7 mm for the short side and 2336.8 mm for the long side. These are user
measurements of the playing surface, excluding the cushions and outer table.

The snapshot at `http://192.168.1.31:8001/snapshot`, inspected on 2026-10-26,
returned a 1280-by-720 image with the long table axis horizontal. This records
that snapshot's orientation; it does not establish a universal X/Y axis setting.

## 2026-10-26: Confirmed ball size and placement limitations

The user guarantees that all balls have a regulation diameter of 2 1/4 inches:
57.15 mm in diameter and 28.575 mm in radius. Treat this as confirmed; do not
ask the user for the ball diameter again.

The user moved the balls into a grid and requested another snapshot, which was
retrieved. The placements were approximately guided by the table's rail dots.
The dots are references; grid placement alone does not establish independently
measured X/Y coordinates. The first confirmed physical point is recorded below.

The user reported a center-ball measurement of 21 3/4 inches (552.45 mm) and
subsequently confirmed that this runs from the TOP cushion to the nearest ball
surface. This historical vertical measurement is superseded by the direct
on-center correction recorded below.

## 2026-10-26: Center-ball measurements for the fresh grid snapshot

These measurements apply to `thoughts/calibration/table-grid-20261026-114241.jpg`.
The confirmed ball diameter is 2.25 inches, so its radius is 1.125 inches.

The user measured 45 inches from the LEFT cushion to the nearest ball surface.
Adding the radius puts the center 46.125 inches (1171.575 mm) from the left
cushion. Relative to the 46-inch midpoint of the 92-inch playing area, centered
X is +0.125 inches (+3.175 mm), when image-right is positive.

Historically, the user reported the other nearest-surface measurement as 21.75 inches from
the TOP cushion. Adding the radius gives a center distance of 22.875 inches
(581.025 mm) from the top cushion. Centered Y is
+0.125 inches (+3.175 mm), relative to the 22.75-inch midpoint of the 45.5-inch
playing area, when image-down is positive. These historical Y values are
superseded by the direct on-center correction below.

This middle-row center ball is the first confirmed physical measurement point
for the matching snapshot. With the upper-left cushion-nose origin, image-right
X and image-down Y, its initially derived center was (1171.575, 581.025) mm;
the Y component is superseded below. The decimal digits
come from inch conversion, not established measurement precision; physical
measurement uncertainty is unknown. The structured record is
`thoughts/calibration/table-grid-20261026-114241-measurements.json`.
Full calibration remains pending: one physical point is insufficient.

## 2026-10-26: Historical purple-ball discrepancy, now resolved below

The user reports that the far-right purple middle-row ball touches the right
rail and has nearest-surface gaps of 21.75 inches to BOTH top and bottom rails.
Record the right gap as zero. These vertical measurements are inconsistent:
21.75 + 2.25 + 21.75 = 45.75 inches, exceeding the user-provided 45.5-inch
playing-area height by 0.25 inches (6.35 mm). This initially prevented acceptance
of the vertical position. The corrected direct on-center measurement below
supersedes these gaps and resolves the discrepancy without changing dimensions.

Given right-rail contact and the existing 92-inch playing-area length, nominal
center X is 92 - 1.125 = 90.875 inches (2308.225 mm) from the left cushion.
This is derived from the reported contact and table dimensions, not a separate
left-distance measurement. The first center-ball measurement remains preserved.

## 2026-10-26: Latest confirmed on-center correction for both measured balls

The user reconfirmed the inside playing-area height as 45.5 inches and corrected
both the middle ball and far-right purple ball to 22 5/8 inches ON CENTER from
the established top-cushion reference. This is a direct center measurement:
22.625 inches = 574.675 mm; do not add the ball radius. Center-origin Y is
-0.125 inches (-3.175 mm) for both balls with image-down positive.

The middle ball's X remains 46.125 inches (1171.575 mm) from the left cushion.
The right-touching purple ball's X remains 90.875 inches (2308.225 mm), derived
from the table length and radius. The two accepted physical points are therefore
(1171.575, 574.675) mm and (2308.225, 574.675) mm from the upper-left origin.
These supersede all earlier vertical values above; historical raw measurements
remain recorded for provenance. Measurement uncertainty remains unknown.

No new image was taken for this correction; it applies to the current arrangement
associated with `thoughts/calibration/table-grid-20261026-114241.jpg` and its
measurement JSON. Full nonlinear calibration remains pending; two physical
points are insufficient.

## 2026-10-26: Confirmed left yellow ball and rail contact

The user confirms all balls on rails touch the cushion. This statement does not
establish additional X/Y coordinates from apparent proximity or pocket jaws;
corner contacts remain ambiguous without the individual reference measurements.

The far-left yellow middle-row ball touches the left cushion and its center is
22.625 inches from the top cushion. Its upper-left-origin center is therefore
(1.125, 22.625) inches = (28.575, 574.675) mm. Relative to the table midpoint,
its center is (-1139.825, -3.175) mm with image-right X and image-down Y.
The X coordinate is derived from confirmed cushion contact and ball radius.
Measurement uncertainty is unknown. The middle and right purple points remain
unchanged; all three accepted points are on one row, so full nonlinear
calibration remains pending.

The middle-row purple ball second from the left and green ball second from the
right have horizontal and subsequently confirmed vertical center measurements
recorded below.

## 2026-10-26: Interior purple and green horizontal measurements

The user supplied center distances from the left cushion of 22.5 inches
(571.5 mm) for the interior purple ball second from the left, and 68.75 inches
(1746.25 mm) for the interior green ball second from the right. These are direct
center measurements; do not add the radius.

The initial reply provided only these two horizontal numbers; vertical
confirmation was initially pending. The user subsequently explicitly confirmed
that both centers are 22.625 inches (574.675 mm) from the top cushion. Their
centered Y is -3.175 mm with image-down positive, and both are now complete
user-confirmed measurement points. The existing three points remain unchanged.

All five middle-row centers now have confirmed left-origin X distances of
1.125, 22.5, 46.125, 68.75, and 90.875 inches, respectively, with top-origin
Y = 22.625 inches for each. These points remain collinear; full nonlinear
calibration still requires measurements away from this row. Subsequent top-row
measurements are recorded below; pocket-adjacent contacts remain unestablished.

## 2026-10-26: Confirmed top orange and red-striped measurements

The top orange solid ball second from the left has a user-measured center
22.375 inches (568.325 mm) from the left cushion. Confirmed top-cushion contact
puts its center 1.125 inches (28.575 mm) from the top cushion.

The top red-striped ball second from the right has a user-confirmed center
68.75 inches (1746.25 mm) from the left cushion. Its confirmed top-cushion
contact likewise puts center Y at 1.125 inches (28.575 mm). The user's initial
62.75-inch (1593.85 mm) horizontal response was explicitly corrected to 68.75
inches; retain it only as a superseded typo, not as an active measurement.

Both points are complete user-confirmed physical measurements with unknown
uncertainty. All five middle-row points remain unchanged. Seven physical
points are now recorded, but full nonlinear calibration and independent
validation remain incomplete.

## 2026-10-26: Confirmed bottom black and yellow measurements

Asked for the bottom black ball second from the left and bottom yellow ball
second from the right, the user replied "same," confirming their horizontal
center distances match the top pair: 22.375 inches (568.325 mm) for black and
68.75 inches (1746.25 mm) for yellow, measured from the left cushion.

Both touch the bottom cushion, so center Y from the top is 45.5 - 1.125 =
44.375 inches (1127.125 mm), or centered Y = +549.275 mm. Measurement
uncertainty remains unknown. Nine complete physical points are now recorded:
five middle-row, two top-row and two bottom-row points. Earlier points and
their measurement history remain unchanged.

The remaining six balls near corners and middle pockets have no measured X/Y
coordinates. Do not infer their positions from apparent contact with corner
jaws or side-pocket geometry. The subsequent exclusion decision below means
no further measurements of those six balls are required now.

## 2026-10-26: Exclude unmeasured pocket balls; exploratory analysis only

The user cannot accurately measure the six pocket-adjacent balls. Exclude them
from physical ground truth; do not request further pocket-ball measurements
now. The nine measured points support an exploratory fit, with corner and
pocket accuracy explicitly unverified. Full calibration and independent
validation remain incomplete.

Offline source-image contour localization and comparison of a homography with
a homography plus one radial parameter are saved in
`thoughts/calibration/table-grid-feasibility-report.md`, with a reproducible
analysis script, raw results JSON and annotated source image alongside it.
Held-out discrepancies are diagnostic and include unknown physical measurement
uncertainty and shared scale/contact errors; they are not verified accuracy.
Later validation can move one ball to accessible interior corner-quadrant
positions safely away from pockets, without measuring the excluded balls.

## 2026-10-26: Nine checkerboard captures; collection paused

The user verified the printed checkerboard is 10 x 7 inches and mounted flat
on rigid backing, establishing a nominal 25.4 mm square pitch. Nine source
captures and detected corner records are saved under
`thoughts/calibration/checkerboard-session/`; collection is paused and the user
can stop moving the board.

Balls were moved to support the tilted board. Retain the original nine physical
ball measurements only for `thoughts/calibration/table-grid-20261026-114241.jpg`.
Those X/Y values are obsolete for new changed-arrangement images and must not
be reused for their ball positions. Checkerboard corners remain independently
usable for optical calibration.

Preliminary future-view optical validation, with models frozen after eight
captures and tested on view 009, reports about 0.228 px RMS with no added
radial term versus 0.231 px with one radial parameter. This currently provides
no evidence to justify added radial warping. No correction is activated and
no physical ball-position accuracy claim follows from these pixel residuals.

## 2026-10-26: Prefer a printed target over manual round-ball measurements

The user finds accurate measurements of round balls difficult with the available
tools and asked for a better method. Do not demand more pocket-ball or tape-over-
ball measurements. Keep the nine samples as a coarse exploratory baseline;
their residuals cannot establish that the user measured incorrectly.

The recommended next step is an offline lens-calibration experiment using a
rigid, flat printed checkerboard of known square pitch. Move the board through
different image locations and angles while keeping the camera and its live
resolution, crop, focus and zoom mode fixed. The board's placement does not need
to be measured: its known pattern supplies the correspondences. Retain separate
images for validation before choosing or activating a correction.

A verified printable target is saved at
`output/pdf/billiards-camera-checkerboard-letter.pdf`: landscape US Letter,
10 by 7 one-inch squares, with 9 by 6 internal corners. Print at 100% / actual
size, verify that the entire grid measures 10 by 7 inches, and mount it flat on
rigid backing. Disable fit-to-page scaling.

Target calibration isolates optical distortion; it does not independently
validate ball-center extraction or guarantee circular corrected silhouettes.
Later actual-ball validation can use an open-center locator with upright side
stops and fiducial marks of known relation to its mechanical center. The ball
rests on the cloth, and repeatable placement reduces manual center measurement.
Keep some locator positions entirely outside calibration fitting.

The application requirement remains fixed-diameter balls and accurate table
X/Y only. No Z estimation or application output is requested. No complete lens
calibration, production integration or deployment has been implemented.

## 2026-10-26: Current source and limits of optical/ball-shape experiments

Initial read-only checks of `http://192.168.1.31:8001/` and `/state` identified uStreamer
v5.4, with the source online at configured 1280 x 720 resolution and desired /
captured frame rates of 20 fps. The hardware encoder reports `quality: 0`;
do not interpret that field as a low JPEG quality ranking. The receipt time,
source/encoder response fields and provenance are saved in
`thoughts/calibration/camera-source-state.json`, excluding stream client IDs
and keys. Higher native resolution support was unknown at that point and no
capture-setting change had yet been made. The verified 4K change below
supersedes this initial source state; its prior receipt is retained in history.

The nine varied checkerboard views and existing-image cushion straightness
diagnostics do not justify adding the tested radial warps. Frozen positive-k1
correction increases bow on several already fairly straight visible rail
transitions, including independent bottom seams; ambiguous edge extraction is
flagged in `thoughts/calibration/cushion-straightness.md`. These observations
do not prove zero lens distortion or establish physical table/ball accuracy.
Checkerboard reprojection validation is not ground truth for ball centers.

Ball-outline aspect/orientation patterns are consistent with position-dependent
apparent shape, but available center-estimation methods disagree by several
source pixels. No method has validated physical ball-center accuracy. Retain
the nine measured ball positions only for their original associated snapshot,
`thoughts/calibration/table-grid-20261026-114241.jpg`; they are invalid for
later checkerboard views after the balls moved.

The requirement remains one configured 57.15 mm diameter and accurate 2D X/Y
for every ball. The completed offline local image-angle experiment is described
below; it does not request ball-depth/Z estimation or change application outputs.

## 2026-10-26: Local 2D view normalization is a candidate, not validated accuracy

The offline experiment in `thoughts/calibration/ball-local-view-normalization.md`
uses image coordinates and frozen zero-radial intrinsics to construct a local
view homography, fit a circle and map its center back. It estimates no ball
height or depth. Clean left/right segmented contours become more circular:
aspect ratios change from 1.267 to 1.053 and from 1.252 to 1.042 respectively.

`thoughts/calibration/ball-local-view-normalization-crops.png` shows actual
source RGB and locally transformed patches on identical native-pixel canvases,
all enlarged exactly 3x for inspection. There is no per-ball resizing, circular
mask or diameter equalization. This is evidence of apparent-shape improvement,
not independent physical center accuracy.

Nine-point leave-one-position-out center diagnostics are mixed: segmentation
changes from 8.948 to 8.471 mm RMS, while gradient-based fitting changes from
8.008 to 8.460 mm RMS. These values include unknown physical-reference and
contour-selection errors; no robust physical X/Y improvement is demonstrated.
No correction is promoted into production.

The current recommendation is to investigate per-ball 2D view normalization as
a localization candidate, with better outer-contour confidence, native-resolution
ROIs and an independent center reference, before considering deployment.
The tested global radial warps remain unsupported. No additional user
measurements or captures are requested now. Preserve the fixed physical
57.15 mm diameter and X/Y-only output requirement throughout this work.

## 2026-10-26: Authorized native 4K20 stream; live inference remains 720p

The user authorized modifying the remote streaming configuration and emphasized
balancing accuracy with FPS. The actual UC40M UVC device advertises MJPG
3840 x 2160 at 20 and 30 fps. User product reference:
`https://www.amazon.com/dp/B08FHBRKSK`; do not assume the installed lens variant.

On `192.168.1.31` (`hp-prodesk`), only `CAMERA_STREAM_RESOLUTION` changed from
1280x720 to 3840x2160 in
`/home/jchadwick/.config/billiards-camera-stream.env`. All other bytes, including
20 fps and MJPEG format, were preserved. The user-owned stream service was
restarted without sudo or daemon reload. The owner-only `0600` backup, exact
before/after hashes and rollback command are documented in
`thoughts/calibration/stream-4k-change.md`.

Four state samples over 15 seconds confirmed online 4K capture at 20 fps;
V4L2 confirmed MJPG 3840 x 2160 / 20 fps, and a snapshot fully decoded at 4K.
The stream service remained active with no automatic restarts and its consumer
reconnected. Backend readiness, already expired / HTTP 503 before the change,
became fresh / HTTP 200 afterward without a forced backend restart.
The preexisting unsupported hardware-quality-setting warning is nonfatal;
do not interpret reported `quality: 0` as a low JPEG quality ranking.

The backend still explicitly scales incoming images to 1280 x 720 through
its HTTP capture helper. `vision.camera.resolution` is authoritative; null
`perception.capture.output_resolution` inherits that configured size rather
than automatically passing through native decoder dimensions. Full-resolution
inference and selective native ROI refinement are not implemented yet.
The old calibration is bound to 720p; do not apply unchanged 720p camera
intrinsics to new native 4K frames before verifying profile, framing and scale.

Candidate future balance: retain 4K20 source detail, run cheap downscaled
motion analysis at higher cadence, resize detector input, and refine selected
ball ROIs from native pixels. This requires profiling and independent center
validation; it is not an implemented or validated pipeline. Physical diameter
remains 57.15 mm and outputs remain X/Y only, with no ball Z model.
