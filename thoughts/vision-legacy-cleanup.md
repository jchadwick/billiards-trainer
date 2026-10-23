# Vision Legacy Cleanup

> Historical cleanup snapshot. The old `video_debugger.py` consumer has been
> retired; current cue state is the canonical `CueState` model.

## Decisions

- `ObjectTracker` is the only public tracker class. No source, test, or runtime caller imported the old `BallTracker` alias.
- Track expiry is expressed only in elapsed seconds. The `vision.tracking.max_age_seconds` default is 1.0 seconds, preserving the former default duration of 30 frames at 30 FPS without retaining frame-rate conversion logic.
- The standalone `VisionConfigurationManager` and its video example had no consumers outside that example. The active configuration path remains `backend.config.Config`; video files use the explicit `video_file_path` setting.
- `VisionModule.calibrate_camera()` duplicated calibration behavior that belongs to the live calibration manager and shared camera service. Its stale calibrator instance was removed while calibration classes and `backend/vision/calibration/manager.py` remain intact.
- `VisionModule.ball_detector` duplicated the active `detector` reference and had no consumers.
- The historical `CueStick.is_aiming` field was removed after `tools/video_debugger.py` was retired; cue detection uses `CueState`.

## Verification

- Vision-focused suite: 70 passed with `--no-cov`.
- Ruff, Black, isort, and strict mypy for `backend/vision` passed.
- `config.json` and `backend/config_schema_defaults.json` both parse, and both use `max_age_seconds: 1.0` under `vision.tracking`.
