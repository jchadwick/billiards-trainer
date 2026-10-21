"""Regression tests for configuration loading and model boundaries."""

import json
from pathlib import Path

import pytest

from backend.common.models import Ball as VisionBall
from backend.config import Config, config
from backend.core.models import BallState


@pytest.fixture
def isolated_config():
    """Keep the process-wide singleton unchanged for other tests."""
    singleton = Config._instance
    singleton_state = singleton.__dict__.copy()
    class_data = Config._config_data
    class_file = Config._config_file

    singleton.__dict__.clear()
    Config._instance = singleton
    Config._config_data = {}
    Config._config_file = None

    try:
        yield singleton
    finally:
        singleton.__dict__.clear()
        singleton.__dict__.update(singleton_state)
        Config._instance = singleton
        Config._config_data = class_data
        Config._config_file = class_file


def test_set_config_file_uses_and_updates_the_existing_singleton(
    isolated_config, tmp_path: Path
) -> None:
    config_path = tmp_path / "settings.json"
    config_path.write_text('{"vision": {"device_id": 7}}', encoding="utf-8")

    result = Config.set_config_file(config_path=config_path)

    assert result is None
    assert Config() is isolated_config is config
    assert Config._instance is config
    assert isolated_config.get("vision.device_id") == 7
    assert isolated_config._config_file == config_path.resolve()


def test_reload_reads_updated_configuration_from_selected_path(
    isolated_config, tmp_path: Path
) -> None:
    config_path = tmp_path / "settings.json"
    config_path.write_text('{"mode": "practice"}', encoding="utf-8")
    Config.set_config_file(config_path)

    config_path.write_text('{"mode": "training"}', encoding="utf-8")
    isolated_config.reload()

    assert isolated_config.get("mode") == "training"


@pytest.mark.parametrize(
    "contents",
    [None, "{ this is not valid JSON"],
    ids=["missing-file", "malformed-file"],
)
def test_missing_or_malformed_file_falls_back_to_empty_config(
    isolated_config, tmp_path: Path, contents: str | None
) -> None:
    config_path = tmp_path / "settings.json"
    if contents is not None:
        config_path.write_text(contents, encoding="utf-8")

    Config.set_config_file(config_path)

    assert isolated_config.get_all() == {}
    assert isolated_config.get("missing.key", "fallback") == "fallback"


def test_set_persists_to_the_file_selected_by_set_config_file(
    isolated_config, tmp_path: Path
) -> None:
    config_path = tmp_path / "settings.json"
    Config.set_config_file(config_path)

    isolated_config.set("vision.camera.loop_video", False)

    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["vision"] == {"camera": {"loop_video": False}}
    assert persisted["metadata"]["configuration_revision"] == isolated_config.get(
        "metadata.configuration_revision"
    )


def test_legacy_file_gets_a_revision_before_the_next_commit(
    isolated_config, tmp_path: Path
) -> None:
    config_path = tmp_path / "settings.json"
    config_path.write_text('{"vision": {"camera": {"fps": 30}}}', encoding="utf-8")

    Config.set_config_file(config_path)

    assert isolated_config.get("metadata.configuration_revision") == 1
    isolated_config.set("vision.camera.fps", 45)
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["metadata"]["configuration_revision"] == 2


def test_malformed_persisted_revision_fails_closed(
    isolated_config, tmp_path: Path
) -> None:
    config_path = tmp_path / "settings.json"
    config_path.write_text(
        '{"metadata": {"configuration_revision": true}, '
        '"vision": {"camera": {"fps": 30}}}',
        encoding="utf-8",
    )

    Config.set_config_file(config_path)

    assert isolated_config.get_all() == {}


def test_reload_does_not_accept_a_lower_persisted_revision(
    isolated_config, tmp_path: Path
) -> None:
    config_path = tmp_path / "settings.json"
    config_path.write_text(
        '{"metadata": {"configuration_revision": 5}, '
        '"vision": {"camera": {"fps": 40}}}',
        encoding="utf-8",
    )
    Config.set_config_file(config_path)
    config_path.write_text(
        '{"metadata": {"configuration_revision": 4}, '
        '"vision": {"camera": {"fps": 60}}}',
        encoding="utf-8",
    )

    isolated_config.reload()

    assert isolated_config.get("vision.camera.fps") == 40
    assert isolated_config.get("metadata.configuration_revision") == 5


def test_reload_rejects_changed_settings_reusing_the_same_revision(
    isolated_config, tmp_path: Path
) -> None:
    config_path = tmp_path / "settings.json"
    config_path.write_text(
        '{"metadata": {"configuration_revision": 5}, '
        '"vision": {"camera": {"fps": 40}}}',
        encoding="utf-8",
    )
    Config.set_config_file(config_path)
    config_path.write_text(
        '{"metadata": {"configuration_revision": 5}, '
        '"vision": {"camera": {"fps": 60}}}',
        encoding="utf-8",
    )

    isolated_config.reload()

    assert isolated_config.get("vision.camera.fps") == 40
    assert isolated_config.get("metadata.configuration_revision") == 5


def test_detection_ball_and_physical_ball_state_have_distinct_roles() -> None:
    # Vision detections use pixel tuples and detection metadata; core state
    # uses canonical Vector2D values and physical state, so keep them distinct.
    assert VisionBall is not BallState
