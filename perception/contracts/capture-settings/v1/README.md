# Capture settings request v1

`capture-settings-default.json` is the shared Python/Rust interoperability
fixture. `capture-settings.schema.json` defines the structural request
contract; Rust adds cross-field and domain validation.

The document contains the Python owner's effective **requested** settings. It
does not claim that hardware applied them. The opened Rust
`CaptureSessionReport` carries per-control source readbacks separately, with
`applied(value)`, `unsupported`, or `unknown` states. It never fills an unknown
readback from the request. Required exposure mode and gain, plus a requested
manual exposure value, must be reported as applied and match exactly before
the preflight allows the first read. Optional controls remain inspectable
without blocking startup.

The same source-session report carries the source-reported applied frame
geometry: configuration revision, camera-profile ID, output image space, and
rectification revision. Preflight compares each value with the request before
the first read. The profile ID is a configuration fingerprint, not a physical
camera serial number. These fields are a software contract: a future camera
adapter must populate them from settings actually negotiated/applied by its
backend. Fake-source contract tests do not establish hardware behavior. The
report is not yet transported to Python as a runtime acknowledgement, and
calibration compatibility still has to be bound by the scene pipeline.
Read-completion timestamps remain explicitly distinct from exposure time.

`camera_profile_id` is a deterministic fingerprint of the configured device
selection, backend, video mode, pixel format, image space, and rectification
revision. Exposure, gain, brightness, contrast, saturation, focus, buffering,
and calibration revision are excluded: they do not define the geometric
mapping, and the calibration revision is carried separately. The fingerprint
is useful for binding calibration requests to configuration, but it is not a
serial number or verified physical-device identity. A runtime must still bind
source reports and calibration before publishing physical table coordinates.

`backend: "auto"` accepts any backend reported by the selected provider.
Explicit backend requests use the contract identifiers (`v4l2`, `dshow`,
`gstreamer`, `opencv`, or `kinect2`) and must match the opened source report's
`backend_id` exactly. Providers should report these canonical identifiers.

The 100 ms maximum-frame-age default is an initial configurable freshness
limit. It is not a measured performance claim. The source report now includes
the camera-profile fingerprint, output image space, and rectification revision,
but no selected adapter or Python-visible acknowledgement currently proves
those values against a physical device. The Rust live provider remains disabled
until an adapter can report and apply the complete effective settings and
target hardware is qualified. Fake-source readback tests validate the software
contract only; they do not establish that any physical device honors its
readbacks.

V1 does not carry emitted RGB8 width and height as separate fields. Its legacy
rule is that emitted frame dimensions equal the negotiated profile dimensions;
sources that decode or rectify to different dimensions must use the strict v2
contract in [`../v2/README.md`](../v2/README.md).
