# Checkerboard capture session

The user confirmed a flat rigid backing and a full printed pattern measuring
10 x 7 inches. Square pitch is 25.4 mm; internal corner layout is 9 x 6.
The camera frame remains 1280 x 720. No calibration is activated.

Capture another already-fetched image:

```sh
/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/capture_checkerboard.py --file /tmp/next-view.jpg --note 'User supplied next placement; describe visible location and tilt'
```

Or fetch the current snapshot directly:

```sh
/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/capture_checkerboard.py --url http://192.168.1.31:8001/snapshot --note 'Describe visible location and tilt'
```

The environment is isolated under `/tmp`; its OpenCV/NumPy dependencies are not
production dependencies. If that temporary environment disappears, recreate it
with Python venv and install `opencv-python-headless` inside it. The utility
uses original-resolution SB corner detection, with classic/subpixel fallback.
No upsampling is used. Corner coordinates and quality metrics are saved with
exact source-image bytes, source and decoded-pixel hashes, import/receipt time,
and an annotated PNG. Camera exposure timestamps are unavailable.

The session appends auto-numbered views under a lock and atomically replaces
its manifest. It rejects exact duplicates, incomplete corner patterns, source
geometry/target changes, and estimated target clipping. White paper border
quality requires visual inspection. Repeated identical board poses with new
JPEG pixels are not exact duplicates and may still be redundant for calibration.

View 001 detects all 54 corners. Minimum adjacent-corner pitch is about 11.54
pixels, median 11.89 pixels; the estimated black/white rectangle remains more
than 313 pixels from the image boundary. The white border is visually visible.
The target is small, and numerical subpixel coordinates are not proof of
physical accuracy. A few clear larger/tilted views improve conditioning.

Next views should vary both location and board tilt, including coverage near
image edges while keeping the entire checkerboard and white border visible.
Flat translations alone do not provide sufficient pose diversity for reliable
lens calibration. These target poses do not introduce variable ball-height or
Z-axis estimation into the application's 2D ball model.

Checks: synthetic complete-board detection, known pixel pitch, cropped-board
rejection, and exact duplicate rejection passed. Reopened first-view JSON and
manifest agree with original dimensions, hashes and 54 finite corner values.

## Ball measurement provenance

By view 006, balls had been moved to prop the tilted board. Their arrangement
is no longer the arrangement in `thoughts/calibration/table-grid-20261026-114241.jpg`.
The nine measured ball X/Y points belong only to that original associated
snapshot. Do not apply those physical coordinates to balls in checkerboard
views or later changed-arrangement images. Moving balls does not invalidate
checkerboard corner correspondences; their geometry comes from the board.

## Collection pause after nine views

Nine captures are saved; the user can pause moving the board. Printed pattern
size remains user-verified at 10 x 7 inches with 25.4 mm square pitch and flat
rigid backing. View 009 passes source/pixel hash and 54-corner metadata checks.

Parent-reported preliminary validation on the new ninth view, using models
frozen after eight views, gives approximately 0.228 px RMS with no added radial
term and 0.231 px with one radial parameter. This does not demonstrate a benefit
from additional radial warping. No correction is activated and these pixel
residuals do not establish physical ball-position accuracy. The original nine
ball ground-truth positions remain retained only for their original snapshot;
they are obsolete for new images after the balls were moved.
