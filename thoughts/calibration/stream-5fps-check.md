# 4K/5 FPS camera-source check — 2026-10-26 ET

The user authorized modifying the target streaming setup. A temporary,
controlled source-cadence check was run on `192.168.1.31` at 3840 × 2160
MJPEG and 5 FPS.

## Results

- All six `/state` samples reported the source online at 3840 × 2160.
- Every sample reported `desired_fps=5` and `captured_fps=5`.
- Samples spanned about 18 seconds after a 5-second warm-up.
- The streamer was restored to 3840 × 2160 MJPEG at 20 FPS. The service was
  active and `/state` again reported `desired_fps=20`, `captured_fps=20`.
- SHA-256 of the restored configuration matches the temporary-check backup:
  `46d1a67d8880ea6cfc31f6d68415cb89456fa2c3c07e7d9a73f5008e30e8bf72`.
- The owner-only backup remains at
  `/home/jchadwick/.config/billiards-camera-stream.env.before-fps-check-20261027T024650Z`.

This verifies source cadence at 5 FPS. It does not measure processing drops,
measurement freshness, precise-position availability, or the new localization
pipeline end to end. Keep 5 FPS as a candidate; do not select production
application settings from this source-only check.

## Follow-up source status — 2026-10-27 02:55 UTC

The target `/state` endpoint responded with the streamer online at 3840 × 2160,
`desired_fps=20`, and `captured_fps=20`. This is a point-in-time source check;
no localization pipeline was deployed or measured.
