"""Behavioral contracts for configuration HTTP routes.

The store below intentionally keeps persistence concerns out of these tests:
the route's normalization, validation decisions, serialization and error
translation are the behaviors under test. Its snapshots start from the real
schema defaults, so route validation sees the same supported sections and
settings as a running application.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import json
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import HTTPException, UploadFile

import backend.api.routes.config as config_routes
from backend.config import Config, ConfigValidationError, load_schema_reference


class _ConfigStore:
    """Small in-memory stand-in for the atomic Config store contract."""

    def __init__(self) -> None:
        self.values = load_schema_reference()
        self.apply_calls: list[dict[str, Any]] = []
        self.replace_calls: list[dict[str, Any]] = []
        self.apply_error: Exception | None = None
        self.replace_error: Exception | None = None

    def get_all(self) -> dict[str, Any]:
        return copy.deepcopy(self.values)

    def get(self, path: str, default: Any = None) -> Any:
        value: Any = self.values
        for component in path.split("."):
            if not isinstance(value, dict) or component not in value:
                return default
            value = value[component]
        return copy.deepcopy(value)

    def apply_updates(self, updates: dict[str, Any]) -> list[str]:
        self.apply_calls.append(copy.deepcopy(updates))
        if self.apply_error:
            raise self.apply_error
        _merge(self.values, updates)
        return _leaf_paths(updates)

    def replace_configuration(self, replacement: dict[str, Any]) -> list[str]:
        self.replace_calls.append(copy.deepcopy(replacement))
        if self.replace_error:
            raise self.replace_error
        self.values = copy.deepcopy(replacement)
        return _leaf_paths(replacement)


def _merge(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _leaf_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        return [
            path
            for key, item in value.items()
            for path in _leaf_paths(item, f"{prefix}.{key}" if prefix else key)
        ]
    return [prefix]


def _as_config(store: _ConfigStore) -> Config:
    """Type the route-facing test double through the production dependency API."""
    return cast("Config", store)


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def _upload(filename: str, payload: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(payload))


def test_get_configuration_returns_full_or_selected_section_and_rejects_unknown() -> (
    None
):
    store = _ConfigStore()
    full = _run(config_routes.get_configuration(None, True, _as_config(store)))
    selected = _run(config_routes.get_configuration("vision", False, _as_config(store)))

    assert full.values == store.get_all()
    assert full.is_valid is True
    assert full.validation_errors == []
    assert selected.values == {"vision": store.get("vision")}
    assert selected.schema_version == "1.0.0"
    with pytest.raises(HTTPException) as error:
        _run(config_routes.get_configuration("not-a-section", True, _as_config(store)))
    assert error.value.status_code == 400


def test_get_configuration_translates_store_failures_to_stable_http_error() -> None:
    class BrokenStore(_ConfigStore):
        def get_all(self) -> dict[str, Any]:
            raise RuntimeError("disk path leaked")

    with pytest.raises(HTTPException) as error:
        _run(config_routes.get_configuration(None, True, _as_config(BrokenStore())))
    assert error.value.status_code == 500
    assert error.value.detail == (
        "Configuration retrieval failed. Unable to retrieve system configuration"
    )


def test_update_normalizes_section_updates_and_reports_restart_requirement() -> None:
    store = _ConfigStore()
    result = _run(
        config_routes.update_configuration(
            {"vision": {"camera": {"fps": 30}}},
            "vision",
            False,
            False,
            _as_config(store),
        )
    )

    assert result.success is True
    assert store.apply_calls == [{"vision": {"camera": {"fps": 30}}}]
    assert "vision.camera.fps" in result.updated_fields
    assert result.restart_required is True
    assert result.rollback_available is True


def test_game_session_settings_are_reported_as_restart_required() -> None:
    store = _ConfigStore()
    result = _run(
        config_routes.update_configuration(
            {"game_session": {"operation_timeout_seconds": 6.0}},
            None,
            False,
            False,
            _as_config(store),
        )
    )

    assert result.success is True
    assert result.updated_fields == ["game_session.operation_timeout_seconds"]
    assert result.restart_required is True


def test_update_section_body_must_be_an_object_and_empty_updates_are_rejected() -> None:
    store = _ConfigStore()
    with pytest.raises(HTTPException) as empty_error:
        _run(
            config_routes.update_configuration(
                {}, None, False, False, _as_config(store)
            )
        )
    assert empty_error.value.status_code == 422

    with pytest.raises(HTTPException) as shape_error:
        _run(
            config_routes.update_configuration(
                {"vision": 1}, "vision", False, False, _as_config(store)
            )
        )
    assert shape_error.value.status_code == 422
    assert shape_error.value.detail == "Section update must be an object"


def test_update_validate_only_returns_errors_without_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ConfigStore()
    validation_error = {
        "field": "vision.camera.fps",
        "message": "must be positive",
        "current_value": 0,
        "expected_type": "number",
    }
    monkeypatch.setattr(
        config_routes,
        "validate_configuration",
        lambda *_args, **_kwargs: ([validation_error], ["camera is slow"]),
    )

    result = _run(
        config_routes.update_configuration(
            {"vision": {"camera": {"fps": 0}}}, None, True, False, _as_config(store)
        )
    )

    assert result.success is False
    assert result.updated_fields == []
    assert result.validation_errors[0].field == "vision.camera.fps"
    assert result.warnings == ["camera is slow"]
    assert store.apply_calls == []


def test_update_rejects_validation_errors_and_unforced_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ConfigStore()
    issue = {
        "field": "vision.camera.fps",
        "message": "out of range",
        "current_value": 0,
        "expected_type": "number",
    }
    monkeypatch.setattr(
        config_routes,
        "validate_configuration",
        lambda *_args, **_kwargs: ([issue], []),
    )
    with pytest.raises(HTTPException) as validation_error:
        _run(
            config_routes.update_configuration(
                {"vision": {"camera": {"fps": 0}}},
                None,
                False,
                False,
                _as_config(store),
            )
        )
    assert validation_error.value.status_code == 422
    assert cast("dict[str, Any]", validation_error.value.detail)["errors"] == [issue]
    assert store.apply_calls == []

    monkeypatch.setattr(
        config_routes,
        "validate_configuration",
        lambda *_args, **_kwargs: ([], ["higher camera rate uses more CPU"]),
    )
    with pytest.raises(HTTPException) as warning_error:
        _run(
            config_routes.update_configuration(
                {"vision": {"camera": {"fps": 61}}},
                None,
                False,
                False,
                _as_config(store),
            )
        )
    assert warning_error.value.status_code == 400
    assert "force_update=true" in warning_error.value.detail
    assert store.apply_calls == []


def test_update_can_force_warnings_and_non_runtime_fields_do_not_require_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ConfigStore()
    monkeypatch.setattr(
        config_routes,
        "validate_configuration",
        lambda *_args, **_kwargs: ([], ["advisory"]),
    )
    result = _run(
        config_routes.update_configuration(
            {"metadata": {"operator": "table-1"}}, None, False, True, _as_config(store)
        )
    )
    assert result.success is True
    assert result.warnings == ["advisory"]
    assert result.restart_required is False


@pytest.mark.parametrize("failure", [ConfigValidationError([]), OSError("disk")])
def test_update_maps_store_failures_to_validation_or_internal_error(
    failure: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _ConfigStore()
    monkeypatch.setattr(
        config_routes, "validate_configuration", lambda *_args, **_kwargs: ([], [])
    )
    store.apply_error = failure
    with pytest.raises(HTTPException) as error:
        _run(
            config_routes.update_configuration(
                {"metadata": {"operator": "table-1"}},
                None,
                False,
                True,
                _as_config(store),
            )
        )
    assert error.value.status_code == (
        422 if isinstance(failure, ConfigValidationError) else 500
    )


def test_reset_requires_confirmation_and_can_reset_selected_sections() -> None:
    store = _ConfigStore()
    with pytest.raises(HTTPException) as confirmation_error:
        _run(config_routes.reset_configuration(False, "all", None, _as_config(store)))
    assert confirmation_error.value.status_code == 400

    before = store.get("vision")
    store.values["vision"]["camera"]["fps"] = 20
    response = _run(
        config_routes.reset_configuration(
            True, "section", ["vision"], _as_config(store)
        )
    )
    assert response["success"] is True
    assert response["data"]["reset_type"] == "section"
    assert response["data"]["sections_reset"] == ["vision"]
    assert store.get("vision") == before
    assert len(store.replace_calls) == 1


def test_reset_rejects_missing_sections_and_unknown_section() -> None:
    store = _ConfigStore()
    with pytest.raises(HTTPException) as missing:
        _run(
            config_routes.reset_configuration(True, "section", None, _as_config(store))
        )
    assert missing.value.status_code == 422
    with pytest.raises(HTTPException) as unknown:
        _run(
            config_routes.reset_configuration(
                True, "section", ["missing"], _as_config(store)
            )
        )
    assert unknown.value.status_code == 400


def test_reset_reports_atomic_store_failure() -> None:
    store = _ConfigStore()
    store.replace_error = OSError("disk unavailable")
    with pytest.raises(HTTPException) as error:
        _run(config_routes.reset_configuration(True, "all", None, _as_config(store)))
    assert error.value.status_code == 500
    assert (
        error.value.detail == "Configuration reset failed. Changes were not persisted."
    )


@pytest.mark.parametrize(
    ("name", "payload", "status"),
    [
        ("settings.toml", b"x", 400),
        ("settings.json", b"{broken", 400),
        ("settings.json", b"[]", 400),
    ],
)
def test_import_rejects_unsupported_or_malformed_documents(
    name: str, payload: bytes, status: int
) -> None:
    with pytest.raises(HTTPException) as error:
        _run(
            config_routes.import_configuration(
                _upload(name, payload),
                "replace",
                False,
                False,
                _as_config(_ConfigStore()),
            )
        )
    assert error.value.status_code == status


def test_import_strips_export_metadata_and_replaces_valid_json() -> None:
    store = _ConfigStore()
    payload = json.dumps(
        {
            "vision": {"camera": {"fps": 30}},
            "_metadata": {"exported_by": "api", "version": "1.0.0"},
        }
    ).encode()
    result = _run(
        config_routes.import_configuration(
            _upload("settings.JSON", payload),
            "replace",
            False,
            False,
            _as_config(store),
        )
    )

    assert result["success"] is True
    assert result["data"]["format"] == ".json"
    assert result["data"]["merge_strategy"] == "replace"
    assert result["data"]["sections_imported"] == ["vision"]
    assert "_metadata" not in store.values
    assert store.values["vision"]["camera"]["fps"] == 30


def test_import_yaml_merge_validate_only_and_warning_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ConfigStore()
    yaml_payload = b"vision:\n  camera:\n    fps: 61\n"
    validation: tuple[list[dict[str, Any]], list[str]] = (
        [],
        ["camera rate may increase CPU usage"],
    )
    monkeypatch.setattr(
        config_routes, "validate_configuration", lambda *_args, **_kwargs: validation
    )
    checked = _run(
        config_routes.import_configuration(
            _upload("settings.yaml", yaml_payload),
            "merge",
            True,
            False,
            _as_config(store),
        )
    )
    assert checked["data"]["is_valid"] is True
    assert checked["data"]["sections_found"] == ["vision"]
    assert store.apply_calls == []

    with pytest.raises(HTTPException) as warning_error:
        _run(
            config_routes.import_configuration(
                _upload("settings.yml", yaml_payload),
                "merge",
                False,
                False,
                _as_config(store),
            )
        )
    assert warning_error.value.status_code == 400
    result = _run(
        config_routes.import_configuration(
            _upload("settings.yml", yaml_payload),
            "merge",
            False,
            True,
            _as_config(store),
        )
    )
    assert result["success"] is True
    assert store.apply_calls[-1] == {"vision": {"camera": {"fps": 61}}}


def test_import_validation_and_store_failures_preserve_error_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ConfigStore()
    invalid = {
        "field": "unknown.key",
        "message": "unknown setting",
        "current_value": 1,
        "expected_type": "known key",
    }
    monkeypatch.setattr(
        config_routes,
        "validate_configuration",
        lambda *_args, **_kwargs: ([invalid], []),
    )
    with pytest.raises(HTTPException) as invalid_error:
        _run(
            config_routes.import_configuration(
                _upload("settings.json", b'{"unknown":{"key":1}}'),
                "replace",
                False,
                False,
                _as_config(store),
            )
        )
    assert invalid_error.value.status_code == 422

    monkeypatch.setattr(
        config_routes, "validate_configuration", lambda *_args, **_kwargs: ([], [])
    )
    store.replace_error = ConfigValidationError([invalid])
    with pytest.raises(HTTPException) as store_error:
        _run(
            config_routes.import_configuration(
                _upload("settings.json", b'{"vision":{"camera":{"fps":30}}}'),
                "replace",
                False,
                False,
                _as_config(store),
            )
        )
    assert store_error.value.status_code == 422


def test_export_filters_sections_computes_checksum_and_omits_optional_metadata() -> (
    None
):
    store = _ConfigStore()
    result = _run(
        config_routes.export_configuration(
            "json", ["vision"], False, False, _as_config(store)
        )
    )
    serialized = json.dumps(result.data, indent=2, default=str)

    assert result.format == "json"
    assert result.data.keys() == {"vision"}
    assert result.size == len(serialized.encode("utf-8"))
    assert result.checksum == hashlib.sha256(serialized.encode()).hexdigest()


def test_export_supports_yaml_metadata_and_rejects_unknown_sections() -> None:
    result = _run(
        config_routes.export_configuration(
            "yaml", None, False, True, _as_config(_ConfigStore())
        )
    )
    assert result.format == "yaml"
    assert result.data["_metadata"]["exported_by"] == "api"
    assert "vision" in result.data["_metadata"]["sections"]
    with pytest.raises(HTTPException) as error:
        _run(
            config_routes.export_configuration(
                "json", ["bogus"], False, True, _as_config(_ConfigStore())
            )
        )
    assert error.value.status_code == 400


def test_download_uses_export_format_and_returns_file_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = tmp_path / "download.json"
    file_response_args: dict[str, Any] = {}

    def capture_file_response(**kwargs: Any) -> dict[str, Any]:
        file_response_args.update(kwargs)
        return kwargs

    def open_download(_path: str, mode: str) -> Any:
        return expected.open(mode)

    monkeypatch.setattr(config_routes, "FileResponse", capture_file_response)
    monkeypatch.setattr(config_routes, "open", open_download, raising=False)
    result = _run(
        config_routes.download_configuration("json", None, _as_config(_ConfigStore()))
    )
    data = json.loads(expected.read_text())
    assert data["_metadata"]["exported_by"] == "api"
    assert result["filename"].startswith("billiards_config_")
    assert result["filename"].endswith(".json")
    assert result["media_type"] == "application/octet-stream"
    assert Path(file_response_args["path"]) == Path("/tmp") / result["filename"]


def test_schema_endpoint_filters_sections_and_maps_unknown_or_internal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    whole = _run(config_routes.get_configuration_schema(None))
    selected = _run(config_routes.get_configuration_schema("vision"))
    assert whole["type"] == "object"
    assert "vision" in whole["properties"]
    assert selected["properties"] == whole["properties"]["vision"]["properties"]
    assert "x-cross-field-rules" in selected

    with pytest.raises(HTTPException) as unknown:
        _run(config_routes.get_configuration_schema("unknown"))
    assert unknown.value.status_code == 400

    monkeypatch.setattr(
        config_routes,
        "build_configuration_schema",
        lambda _section=None: (_ for _ in ()).throw(RuntimeError("schema failure")),
    )
    with pytest.raises(HTTPException) as failed:
        _run(config_routes.get_configuration_schema(None))
    assert failed.value.status_code == 500


def test_playing_area_read_returns_configured_calibration_and_maps_store_error() -> (
    None
):
    store = _ConfigStore()
    store.values["table"]["playing_area_corners"] = [
        {"x": 1, "y": 2, "scale": [1, 1]} for _ in range(4)
    ]
    store.values["table"]["marker_dots"] = []
    result = _run(config_routes.get_playing_area_corners(_as_config(store)))
    assert result["calibrated"] is True
    assert result["count"] == 4

    class BrokenStore(_ConfigStore):
        def get(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("unavailable")

    with pytest.raises(HTTPException) as error:
        _run(config_routes.get_playing_area_corners(_as_config(BrokenStore())))
    assert error.value.status_code == 500


def test_set_playing_area_validates_scale_and_persists_geometry() -> None:
    store = _ConfigStore()
    corners = [
        {
            "x": 10,
            "y": 20,
            "scale": [
                config_routes.CANONICAL_WIDTH / 640,
                config_routes.CANONICAL_HEIGHT / 480,
            ],
        },
        {
            "x": 600,
            "y": 20,
            "scale": [
                config_routes.CANONICAL_WIDTH / 640,
                config_routes.CANONICAL_HEIGHT / 480,
            ],
        },
        {
            "x": 600,
            "y": 440,
            "scale": [
                config_routes.CANONICAL_WIDTH / 640,
                config_routes.CANONICAL_HEIGHT / 480,
            ],
        },
        {
            "x": 10,
            "y": 440,
            "scale": [
                config_routes.CANONICAL_WIDTH / 640,
                config_routes.CANONICAL_HEIGHT / 480,
            ],
        },
    ]

    class Request:
        async def json(self) -> dict[str, Any]:
            return {
                "corners": corners,
                "marker_dots": [{"x": 100, "y": 100, "scale": corners[0]["scale"]}],
                "calibration_resolution_width": 640,
                "calibration_resolution_height": 480,
            }

    result = _run(
        config_routes.set_playing_area_corners(
            cast("Any", Request()), _as_config(store)
        )
    )
    assert result["success"] is True
    assert result["count"] == 4
    assert result["marker_dots_count"] == 1
    assert store.get("table.calibration_resolution_width") == 640


def test_set_playing_area_rejects_malformed_json_and_scale_mismatch() -> None:
    class MalformedRequest:
        async def json(self) -> Any:
            raise json.JSONDecodeError("bad", "{", 0)

    with pytest.raises(HTTPException) as malformed:
        _run(
            config_routes.set_playing_area_corners(
                cast("Any", MalformedRequest()), _as_config(_ConfigStore())
            )
        )
    assert malformed.value.status_code == 400

    class MismatchedRequest:
        async def json(self) -> dict[str, Any]:
            return {
                "corners": [{"x": 10, "y": 20, "scale": [1, 1]} for _ in range(4)],
                "calibration_resolution_width": 640,
                "calibration_resolution_height": 480,
            }

    with pytest.raises(HTTPException) as mismatch:
        _run(
            config_routes.set_playing_area_corners(
                cast("Any", MismatchedRequest()), _as_config(_ConfigStore())
            )
        )
    assert mismatch.value.status_code == 422
