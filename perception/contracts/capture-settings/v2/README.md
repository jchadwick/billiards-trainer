# Capture settings request v2

This version keeps the requested and negotiated profiles as raw camera input
modes and requires `requested.output_frame_geometry` for emitted, tightly
packed RGB8 frames. The negotiated profile retains its source pixel format and
rational frame rate; RGB8 is fixed by the Rust frame envelope.

The output geometry is mandatory, non-zero, and independent from input-mode
dimensions. Rust uses it for the v2 RGB8 frame-memory checks, calibration image
dimensions, source-report preflight, and per-frame dimension checks. A source
report that omits it or reports different dimensions is rejected before reads
are admitted. Rectification/image-space identity remains separately reported
in `applied_frame_geometry`.

`capture-settings-default.json` mirrors the effective application defaults:
1920×1080 RGB8 input and emitted geometry. The separate
`capture-settings-transformed-output.json` example exercises a 1920×1080 MJPG
input mode that emits 1280×720 RGB8. The request remains a set of expectations;
only the opened source report can describe what the adapter says it applied.
Neither fixture is proof of physical camera behavior.

V1 remains a strict legacy contract with no output-geometry field. Its emitted
frames must match the negotiated profile dimensions. Under v1, a rectified calibration is not bound to emitted frame dimensions. V2 removes that ambiguity by requiring
calibration dimensions to match emitted geometry in either image space.
