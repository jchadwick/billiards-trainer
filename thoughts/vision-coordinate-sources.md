# Vision coordinate source and normalization notes

The classic OpenCV ball detector measures circles in its input image, which may
be resized or table-cropped. Its `Ball` outputs now declare `coordinate_space="pixel"`
and the exact detector-input resolution. Extreme-overlap merges retain that
metadata. The production vision normalization path maps pixel positions,
velocity, acceleration, predicted positions, history, and radii back to source
frame pixels; crop restoration also restores the crop origin. Already-canonical
model-adapter outputs remain canonical and are not rescaled a second time.

Cue normalization now covers tip/butt, shaft points, aiming geometry, velocity,
length, and width. The hybrid OpenCV refiner converts canonical balls into the
actual refinement frame before cropping and measuring, then converts refined
geometry back to the ball's declared space. The frame-sized fallback table is
tagged as pixel-space with the captured frame resolution. Table normalization
covers corners, rails, playing-area boundaries, dimensions, and nested pocket
positions, corners, and radii; pockets inherit the containing table's metadata.

Cross-resolution tests compare resized and full-resolution ball geometry and
table/pocket boundaries in canonical coordinates, verify no double scaling,
and cover classic detections, cue geometry, and refinement. The full focused
vision and shared-ball suites pass: 102 passed with `--no-cov`. Ruff, Black,
and isort pass on the touched
vision source and tests. Strict mypy was run on the touched source modules but
still reports 75 existing errors across those larger modules; none points to
the new normalization code. The project config also emits its existing Python
3.9 mypy-version warning while the check is explicitly run for Python 3.11.

Remaining transport boundaries are outside this stage: integration/core must
serialize pixel-space vision data with explicit scale metadata and preserve it
through trajectory/state events; frontend consumers must apply transport scale
before drawing. `thoughts/api-coordinate-contract.md` inventories the current
integration and core serializer callsites.
