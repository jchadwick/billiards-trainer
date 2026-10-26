# Native 4K streaming configuration change — 2026-10-26

The user authorized modifying the streaming setup on `192.168.1.31` and asked
to balance accuracy with frame rate. The UC40M UVC driver advertises MJPG at
3840 x 2160 with both 20 and 30 fps. Native 3840 x 2160 MJPEG capture is now
verified at the existing 20 fps cadence. No camera replacement was required.
The user supplied [MOKOSE product reference](https://www.amazon.com/dp/B08FHBRKSK);
the exact installed lens variant remains unverified. Actual driver identity is
`UC40M Audio: UC40M Video`, using `/dev/video0`, `uvcvideo`, USB 480 Mbit/s.

## Exact change and rollback

Remote user: `jchadwick`; host: `hp-prodesk` (`192.168.1.31`).
Only `/home/jchadwick/.config/billiards-camera-stream.env` changed:

```diff
-CAMERA_STREAM_RESOLUTION=1280x720
+CAMERA_STREAM_RESOLUTION=3840x2160
```

All other bytes, ordering, newline, MJPEG format and 20 fps setting were
preserved. The existing configuration supplies the values; no production
logic was added. The file remains owner-only mode `0600`. No applicable
ancestor `AGENTS.md` was present; this installed configuration is not a Git
checkout. The original-file hash was rechecked before mutation to detect
concurrent edits.

Owner-only rollback backup:
`/home/jchadwick/.config/billiards-camera-stream.env.before-4k-20261026T182205.357395Z`.

- Original SHA-256: `8d51188846e48ad137b1bc44c414cd1b8b4a45bd9e6784a28d10077dae9bf876`
- Updated SHA-256: `46d1a67d8880ea6cfc31f6d68415cb89456fa2c3c07e7d9a73f5008e30e8bf72`

Restart performed as the same user:

```sh
systemctl --user restart billiards-camera-stream.service
```

The existing unit reads the environment file on each start; no daemon reload
or elevated privilege was required. If needed, restore and restart on that
host as `jchadwick`:

```sh
cp -p /home/jchadwick/.config/billiards-camera-stream.env.before-4k-20261026T182205.357395Z /home/jchadwick/.config/billiards-camera-stream.env
systemctl --user restart billiards-camera-stream.service
```

## Verification

- Four service-state samples spanning 15 seconds reported online 3840 x 2160,
  desired 20 fps and captured 20 fps.
- V4L2 readback confirmed negotiated 3840 x 2160 `MJPG`, 20.000 fps.
- A snapshot fully decoded with OpenCV at 3840 x 2160, about 537 KB JPEG.
- Stream service remained active/running with zero automatic restarts, and a
  stream consumer reconnected.
- Backend readiness was already HTTP 503 / expired before this work. After
  stream reconnection it returned HTTP 200 / fresh / ready with localization
  available. No backend restart was forced.
- The hardware encoder's unsupported-quality-setting log message also existed
  before this change. `quality: 0` is retained verbatim and is not interpreted
  as a low JPEG-quality score. Capturing started normally afterward.
- Final byte comparison verified that only the resolution line changed and
  both configuration and backup remained mode `0600`.

Current filtered source/encoder state and prior 720p receipt are recorded in
`camera-source-state.json`; stream client identifiers and keys are omitted.
Framing comparison is a separate artifact owned by the image-analysis task.

## Capture resolution versus inference resolution

The live backend still uses authoritative `vision.camera.resolution=[1280,720]`.
With `perception.capture.output_resolution=null`, Rust requests that configured
output size. The HTTP-MJPEG capture helper explicitly decodes then scales
through `videoscale` and fixed output caps to 1280 x 720. Thus the streamer now
provides native detail for snapshots/streaming, but full-resolution inference
or ROI refinement has not been implemented.

The existing stored table calibration is bound to 1280 x 720. Old 720p camera
intrinsics must not be applied unchanged to new native 4K frames; verify camera
profile, framing and coordinate scaling before any future calibration use.
The stream-only operation preserves current backend output geometry; it does
not itself establish unchanged optical framing or physical ball accuracy.

A candidate accuracy/frame-rate balance is 4K20 capture, cheap downscaled
motion analysis at higher cadence, a resized detector input and selective
native-resolution ball ROI refinement. This needs profiling and independent
center validation; it is not an implemented architecture. Physical diameter
remains 57.15 mm and application ball output remains 2D X/Y, with no Z model.
This work is a reversible operational configuration change, not an architecture
decision or production localization rollout.
