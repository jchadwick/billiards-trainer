# Vision Legacy Cleanup

## Decisions

- `ObjectTracker` is the only public tracker class. No source, test, or runtime caller imported the old `BallTracker` alias.
- Track expiry is expressed only in elapsed seconds. The `vision.tracking.max_age_seconds` default is 1.0 seconds, preserving the former default duration of 30 frames at 30 FPS without retaining frame-rate conversion logic.
- The standalone `VisionConfigurationManager` and its video example had no consumers outside that example. The active configuration path remains `backend.config.Config`; video files use the explicit `video_file_path` setting.
- `VisionModule.calibrate_camera()` duplicated calibration behavior that belongs to the live calibration manager and shared camera service. Its stale calibrator instance was removed while calibration classes and `backend/vision/calibration/manager.py` remain intact.
- `VisionModule.ball_detector` duplicated the active `detector` reference and had no consumers.
- `CueStick.is_aiming` remains temporarily because `tools/video_debugger.py` reads it. The follow-up tools migration can replace that read with the canonical cue state, after which the field and adapter assignments can be removed.

## Verification

- Vision-focused suite: 70 passed with `--no-cov`.
- Ruff, Black, isort, and strict mypy for `backend/vision` passed.
- `config.json` and `backend/config_schema_defaults.json` both parse, and both use `max_age_seconds: 1.0` under `vision.tracking`.
