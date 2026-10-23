"""Route contracts for the explicitly selected Rust configuration authority."""

from __future__ import annotations

import asyncio
import copy
import io
from typing import Any, cast

import pytest
from fastapi import HTTPException, UploadFile

import backend.api.routes.config as config_routes
from backend.config import Config, ConfigurationRevisionConflict, load_schema_reference
from backend.perception.rust_config_authority import (
    RustConfigurationAuthorityRejected,
    RustConfigurationAuthorityTimeout,
)


class _RustConfigFacade:
    """Observe route-to-facade calls without emulating Rust validation rules."""

    authority_mode = "rust"

    def __init__(self) -> None:
        self.defaults = load_schema_reference()
        self.values = copy.deepcopy(self.defaults)
        self.validation = {
            "valid": True,
            "validation_errors": [],
            "warnings": [],
            "revision": 1,
        }
        self.validation_calls: list[tuple[dict[str, Any], bool]] = []
        self.apply_calls: list[tuple[dict[str, Any], bool]] = []
        self.replace_calls: list[tuple[dict[str, Any], bool]] = []
        self.reset_calls: list[list[str]] = []
        self.schema_calls: list[str | None] = []
        self.export_calls: list[dict[str, Any]] = []
        self.apply_error: Exception | None = None
        self.replace_error: Exception | None = None
        self.reset_error: Exception | None = None
        self.get_error: Exception | None = None

    def get_defaults(self) -> dict[str, Any]:
        return copy.deepcopy(self.defaults)

    def get_all(self) -> dict[str, Any]:
        if self.get_error is not None:
            raise self.get_error
        return copy.deepcopy(self.values)

    def get(self, path: str, default: Any = None) -> Any:
        value: Any = self.values
        for component in path.split("."):
            if not isinstance(value, dict) or component not in value:
                return default
            value = value[component]
        return copy.deepcopy(value)

    def validate_configuration(
        self, candidate: dict[str, Any], *, merge: bool = True
    ) -> dict[str, Any]:
        self.validation_calls.append((copy.deepcopy(candidate), merge))
        return copy.deepcopy(self.validation)

    def apply_updates(
        self, updates: dict[str, Any], *, force_update: bool = False
    ) -> list[str]:
        self.apply_calls.append((copy.deepcopy(updates), force_update))
        if self.apply_error is not None:
            raise self.apply_error
        _merge(self.values, updates)
        return _leaf_paths(updates)

    def replace_configuration(
        self, replacement: dict[str, Any], *, force_update: bool = False
    ) -> list[str]:
        self.replace_calls.append((copy.deepcopy(replacement), force_update))
        if self.replace_error is not None:
            raise self.replace_error
        self.values = copy.deepcopy(replacement)
        return _leaf_paths(replacement)

    def reset_sections(
        self, sections: list[str], *, force_update: bool = False
    ) -> list[str]:
        self.reset_calls.append(list(sections))
        if self.reset_error is not None:
            raise self.reset_error
        for section in sections:
            self.values[section] = copy.deepcopy(self.defaults[section])
        return list(sections)

    def get_schema(self, section: str | None = None) -> dict[str, Any]:
        self.schema_calls.append(section)
        schema = {
            "type": "object",
            "properties": {
                name: {"type": "object", "properties": {}} for name in self.defaults
            },
        }
        if section is None:
            return schema
        return copy.deepcopy(schema["properties"][section])

    def export_configuration(
        self,
        *,
        sections: list[str] | None = None,
        include_defaults: bool = False,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        self.export_calls.append(
            {
                "sections": sections,
                "include_defaults": include_defaults,
                "include_metadata": include_metadata,
            }
        )
        result = copy.deepcopy(self.values)
        if sections is not None:
            result = {section: result[section] for section in sections}
        if not include_metadata:
            result.pop("metadata", None)
        return result


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


def _as_config(facade: _RustConfigFacade) -> Config:
    return cast("Config", facade)


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def _upload(filename: str, payload: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(payload))


def _rejected(code: str, **fields: Any) -> RustConfigurationAuthorityRejected:
    error = {"code": code, "message": f"simulated {code}"}
    return RustConfigurationAuthorityRejected(
        {"protocol_version": 1, "ok": False, "error": error, **fields}
    )


def _forbid_python_configuration_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Rust mode consulted a Python configuration path")

    monkeypatch.setattr(config_routes, "load_schema_reference", forbidden)
    monkeypatch.setattr(config_routes, "validate_configuration", forbidden)
    monkeypatch.setattr(config_routes, "build_configuration_schema", forbidden)
    monkeypatch.setattr(Config, "_save_config", forbidden)


def test_rust_update_validate_only_uses_authority_and_never_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facade = _RustConfigFacade()
    facade.validation = {
        "valid": False,
        "validation_errors": [
            {
                "field": "vision.camera.fps",
                "message": "out of range",
                "current_value": 0,
                "expected_type": "number",
            }
        ],
        "warnings": ["camera is slow"],
        "revision": 1,
    }
    _forbid_python_configuration_paths(monkeypatch)

    result = _run(
        config_routes.update_configuration(
            {"vision": {"camera": {"fps": 0}}},
            None,
            True,
            False,
            _as_config(facade),
        )
    )

    assert result.success is False
    assert result.validation_errors[0].field == "vision.camera.fps"
    assert result.warnings == ["camera is slow"]
    assert facade.validation_calls == [({"vision": {"camera": {"fps": 0}}}, True)]
    assert facade.apply_calls == []


def test_rust_update_cannot_replace_deployment_executable_paths() -> None:
    facade = _RustConfigFacade()
    updates = {
        "perception": {"rust_live": {"executable_path": "/tmp/untrusted-server-binary"}}
    }

    with pytest.raises(HTTPException) as error:
        _run(
            config_routes.update_configuration(
                updates, None, False, True, _as_config(facade)
            )
        )

    assert error.value.status_code == 422
    assert error.value.detail["fields"] == ["perception.rust_live.executable_path"]
    assert facade.apply_calls == []
    assert facade.validation_calls == []


def test_rust_update_warning_force_contract_and_revision_conflict() -> None:
    facade = _RustConfigFacade()
    facade.validation["warnings"] = ["higher camera rate uses more CPU"]
    updates = {"vision": {"camera": {"fps": 61}}}

    with pytest.raises(HTTPException) as warning:
        _run(
            config_routes.update_configuration(
                updates, None, False, False, _as_config(facade)
            )
        )
    assert warning.value.status_code == 400
    assert warning.value.detail["warnings"] == ["higher camera rate uses more CPU"]
    assert facade.apply_calls == []

    facade.validation["warnings"] = []
    result = _run(
        config_routes.update_configuration(
            updates, None, False, True, _as_config(facade)
        )
    )
    assert result.success is True
    assert facade.apply_calls == [(updates, True)]

    facade.apply_error = ConfigurationRevisionConflict(
        "expected revision 1, current revision is 2", current_revision=2
    )
    with pytest.raises(HTTPException) as conflict:
        _run(
            config_routes.update_configuration(
                {"metadata": {"operator": "table-2"}},
                None,
                False,
                True,
                _as_config(facade),
            )
        )
    assert conflict.value.status_code == 409
    assert conflict.value.detail["current_revision"] == 2


def test_rust_import_uses_replace_or_merge_validation_and_force() -> None:
    facade = _RustConfigFacade()
    payload = b'{"vision":{"camera":{"fps":61}},"_metadata":{"from":"api"}}'
    updates = {"vision": {"camera": {"fps": 61}}}

    checked = _run(
        config_routes.import_configuration(
            _upload("settings.json", payload),
            "replace",
            True,
            False,
            _as_config(facade),
        )
    )
    assert checked["data"]["is_valid"] is True
    assert facade.validation_calls[-1] == (updates, False)
    assert facade.replace_calls == []

    facade.validation["warnings"] = ["higher rate uses more CPU"]
    with pytest.raises(HTTPException) as warning:
        _run(
            config_routes.import_configuration(
                _upload("settings.json", payload),
                "merge",
                False,
                False,
                _as_config(facade),
            )
        )
    assert warning.value.status_code == 400
    assert warning.value.detail["warnings"] == ["higher rate uses more CPU"]
    assert facade.validation_calls[-1] == (updates, True)
    assert facade.apply_calls == []

    result = _run(
        config_routes.import_configuration(
            _upload("settings.json", payload),
            "replace",
            False,
            True,
            _as_config(facade),
        )
    )
    assert result["success"] is True
    assert facade.validation_calls[-1] == (updates, False)
    assert facade.replace_calls[-1] == (updates, True)
    assert facade.apply_calls == []

    facade.validation["warnings"] = []
    merged = _run(
        config_routes.import_configuration(
            _upload("settings.json", payload),
            "merge",
            False,
            False,
            _as_config(facade),
        )
    )
    assert merged["success"] is True
    assert facade.validation_calls[-1] == (updates, True)
    assert facade.apply_calls[-1] == (updates, False)


def test_rust_import_cannot_replace_deployment_executable_paths() -> None:
    facade = _RustConfigFacade()
    payload = (
        b'{"perception":{"rust_live":'
        b'{"capture_helper_executable_path":"/tmp/untrusted-helper"}}}'
    )

    with pytest.raises(HTTPException) as error:
        _run(
            config_routes.import_configuration(
                _upload("settings.json", payload),
                "merge",
                False,
                True,
                _as_config(facade),
            )
        )

    assert error.value.status_code == 422
    assert error.value.detail["fields"] == [
        "perception.rust_live.capture_helper_executable_path"
    ]
    assert facade.apply_calls == []
    assert facade.replace_calls == []


def test_rust_reset_section_and_all_use_authority_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facade = _RustConfigFacade()
    facade.values["vision"]["camera"]["fps"] = 30
    _forbid_python_configuration_paths(monkeypatch)

    section_result = _run(
        config_routes.reset_configuration(
            True, "section", ["vision"], _as_config(facade)
        )
    )
    assert section_result["success"] is True
    assert facade.reset_calls == [["vision"]]
    assert facade.replace_calls == []
    assert facade.values["vision"] == facade.defaults["vision"]

    all_result = _run(
        config_routes.reset_configuration(True, "all", None, _as_config(facade))
    )
    assert all_result["success"] is True
    assert facade.replace_calls[-1] == (facade.defaults, False)


def test_rust_schema_and_export_are_authority_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facade = _RustConfigFacade()
    _forbid_python_configuration_paths(monkeypatch)

    whole_schema = _run(
        config_routes.get_configuration_schema(None, _as_config(facade))
    )
    section_schema = _run(
        config_routes.get_configuration_schema("vision", _as_config(facade))
    )
    exported = _run(
        config_routes.export_configuration(
            "json", ["vision"], True, False, _as_config(facade)
        )
    )

    assert "vision" in whole_schema["properties"]
    assert section_schema["type"] == "object"
    assert facade.schema_calls == [None, "vision"]
    assert facade.export_calls == [
        {"sections": ["vision"], "include_defaults": True, "include_metadata": True}
    ]
    assert "_metadata" not in exported.data
    assert set(exported.data) == {"vision"}


def test_rust_authority_errors_map_to_validation_warning_or_unavailable_status() -> (
    None
):
    facade = _RustConfigFacade()
    facade.apply_error = _rejected(
        "validation_failed",
        validation_errors=[
            {
                "field": "invalid.field",
                "message": "invalid value",
                "current_value": 1,
                "expected_type": "known key",
            }
        ],
    )
    with pytest.raises(HTTPException) as invalid:
        _run(
            config_routes.update_configuration(
                {"metadata": {"operator": "table-2"}},
                None,
                False,
                True,
                _as_config(facade),
            )
        )
    assert invalid.value.status_code == 422
    assert invalid.value.detail["errors"][0]["field"] == "invalid.field"

    facade.apply_error = _rejected(
        "warnings_require_force", warnings=["force required"]
    )
    with pytest.raises(HTTPException) as warning:
        _run(
            config_routes.update_configuration(
                {"metadata": {"operator": "table-3"}},
                None,
                False,
                True,
                _as_config(facade),
            )
        )
    assert warning.value.status_code == 400
    assert warning.value.detail["warnings"] == ["force required"]

    facade.get_error = RustConfigurationAuthorityTimeout("authority timed out")
    with pytest.raises(HTTPException) as unavailable:
        _run(config_routes.get_configuration(None, True, _as_config(facade)))
    assert unavailable.value.status_code == 503
    assert "authority is unavailable" in unavailable.value.detail

    facade.get_error = _rejected("store_error")
    with pytest.raises(HTTPException) as store_failure:
        _run(config_routes.get_configuration(None, True, _as_config(facade)))
    assert store_failure.value.status_code == 503
