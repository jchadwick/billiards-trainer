# Larger checkerboard capture guide

The existing rigid-backed board is a 10 x 7 inch checkerboard. Its squares
project at roughly 12 pixels in the retained 1280 x 720 images. This larger
target gives the same 9 x 6 internal-corner pattern with four times the square
area and twice the linear corner spacing.

## Print and mount

- Print `output/pdf/ball-localization-checkerboard-20x14in.pdf` at **100% / actual
  size**. Turn off “fit to page” and any automatic scaling.
- The PDF page is 22 x 16 inches. The active black-and-white grid is 20 x 14
  inches: 10 columns by 7 rows, each square 2 inches (50.8 mm), with 9 x 6
  internal corners.
- If printing as a poster across sheets, keep every tile at 100%, align and
  join the printed grid without stretching it, then mount the whole pattern
  flat on rigid backing. A large-format print is simpler if available.
- Verify the active checkerboard is 20 x 14 inches, or verify one square is
  exactly 2 inches. The outer white margins are not part of the measured grid.

## Capture

Keep the camera resolution, crop, zoom, and focus unchanged for the entire
session. Prefer original 3840 x 2160 JPEGs if that capture path is available;
otherwise use one consistent native image size throughout. Save original JPEG
files, not screenshots or resized copies. Keep the entire board and its white
border visible and sharply focused.

Collect 9–12 distinct poses: a near-fronto-parallel view near the image center;
views tilted around each of the board's two axes; and views near each image
corner with varied tilts. Moving the board through the field is useful; flat
translations alone are insufficient. Reserve two or three sharp views from
the calibration fit for validation.

Store these in a new session, separate from the earlier 25.4 mm-pitch board.
For each source JPEG, run:

```sh
/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/capture_checkerboard.py \
  --file /path/to/original-view.jpg \
  --session thoughts/calibration/checkerboard-20x14-session \
  --columns 9 --rows 6 --pitch-mm 50.8 \
  --scale-verified --rigid-backed \
  --note 'Describe image region and board tilt'
```

The capture utility records original image bytes, dimensions, hashes, detected
corner positions, and quality checks. Each accepted image must contain all 54
corners. The new target supplies camera-optics calibration; it does not by
itself supply table X/Y coordinates. The saved rail sights and felt spots
remain the table-specific held-out references.

No jig, round-ball measurements, pocket-ball positions, or direct camera
access by the calibration analyst are required. The user can provide the
saved original JPEGs when ready.
