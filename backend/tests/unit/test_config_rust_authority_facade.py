"""Contracts for the transitional Python facade over Rust configuration."""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import backend.config as config_module
from backend.config import (
    Config,
    ConfigurationAuthorityConfigurationError,
    ConfigurationRevisionConflict,
)
from backend.perception.rust_config_authority import (
    RustConfigurationAuthorityProcessError,
    RustConfigurationAuthorityProtocolError,
    RustConfigurationAuthorityRejected,
)


class FakeRustConfigClient:
    """In-memory protocol peer that records requests and exposes CAS behavior."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.document: dict[str, Any] = {
            "metadata": {"configuration_revision": 4},
            "table": {"enabled": True},
        }
        self.defaults = {
            "metadata": {"configuration_revision": 1},
            "table": {"enabled": False},
        }
        self.schema = {"properties": {"table": {"type": "object"}}}
        self.fail_snapshot: BaseException | None = None

    def call(self, request: dict[str, Any]) -> SimpleNamespace:
        request = copy.deepcopy(request)
        self.calls.append(request)
        operation = request["operation"]
        if operation == "snapshot":
            if self.fail_snapshot is not None:
                raise self.fail_snapshot
            return self._result(
                operation,
                {
                    "revision": self.document["metadata"]["configuration_revision"],
                    "configuration": self.document,
                },
            )
        if operation == "defaults":
            return self._result(operation, {"configuration": self.defaults})
        if operation == "schema":
            return self._result(operation, {"schema": self.schema})
        if operation == "validate":
            return self._result(
                operation,
                {
                    "revision": self.document["metadata"]["configuration_revision"],
                    "valid": True,
                    "validation_errors": [],
                    "warnings": [],
                },
            )
        if operation == "export":
            exported = copy.deepcopy(self.document)
            sections = request.get("sections")
            if sections is not None:
                exported = {key: exported[key] for key in sections if key in exported}
            if not request.get("include_metadata", True):
                exported.pop("metadata", None)
            return self._result(operation, {"configuration": exported})
        if operation == "capture_settings":
            data = request.get("data")
            assert isinstance(data, dict)
            return self._result(
                operation,
                {
                    "revision": self.document["metadata"]["configuration_revision"],
                    "capture_settings": {
                        "schema_version": data["schema_version"],
                        "configuration_revision": self.document["metadata"][
                            "configuration_revision"
                        ],
                        "requested": {"calibration": None},
                    },
                },
            )
        if operation in {"update", "replace", "reset_section"}:
            current_revision = self.document["metadata"]["configuration_revision"]
            if request.get("expected_revision") != current_revision:
                response = {
                    "protocol_version": 1,
                    "ok": False,
                    "operation": operation,
                    "current_revision": current_revision,
                    "error": {
                        "code": "revision_conflict",
                        "message": "revision changed",
                    },
                }
                raise RustConfigurationAuthorityRejected(response)
            if operation == "replace":
                updated = copy.deepcopy(request["data"])
            elif operation == "reset_section":
                updated = copy.deepcopy(self.document)
                for section in request["sections"]:
                    updated[section] = copy.deepcopy(self.defaults[section])
            else:
                updated = copy.deepcopy(self.document)
                updated.update(copy.deepcopy(request["data"]))
            next_revision = current_revision + 1
            updated.setdefault("metadata", {})["configuration_revision"] = next_revision
            updated_fields = ["table.enabled"]
            self.document = updated
            return self._result(
                operation,
                {
                    "revision": next_revision,
                    "configuration": self.document,
                    "updated_fields": updated_fields,
                    "warnings": [],
                },
            )
        raise AssertionError(f"Unexpected Rust operation: {operation}")

    @staticmethod
    def _result(operation: str, content: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            body={"protocol_version": 1, "ok": True, "operation": operation, **content}
        )


@pytest.fixture
def rust_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Config, FakeRustConfigClient, Path]:
    """Start each test with isolated singleton state and an executable selector."""
    monkeypatch.setattr(Config, "_instance", None)
    monkeypatch.setattr(Config, "_config_data", {})
    monkeypatch.setattr(Config, "_config_file", None)
    monkeypatch.setattr(Config, "_authority_mode", None)
    monkeypatch.setattr(Config, "_rust_client", None)
    monkeypatch.setattr(Config, "_rust_revision", None)
    monkeypatch.setattr(Config, "_rust_defaults", None)
    monkeypatch.setattr(Config, "_rust_schema", None)
    monkeypatch.setenv("BILLIARDS_CONFIG_AUTHORITY", "rust")
    executable = tmp_path / "fake-config-authority"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    monkeypatch.setenv("BILLIARDS_CONFIG_AUTHORITY_EXECUTABLE", str(executable))
    monkeypatch.setenv("BILLIARDS_CONFIG_PATH", str(tmp_path / "active.json"))
    client = FakeRustConfigClient()
    monkeypatch.setattr(
        config_module,
        "RustConfigurationAuthorityClient",
        lambda _executable, _config_path: client,
    )
    instance = Config.bootstrap_from_environment()
    monkeypatch.setattr(config_module, "config", instance)
    return instance, client, Path(os.environ["BILLIARDS_CONFIG_PATH"])


def test_rust_getters_use_detached_snapshot_without_relaunching_or_python_file_io(
    rust_config: tuple[Config, FakeRustConfigClient, Path],
) -> None:
    config, client, selected_path = rust_config
    original = copy.deepcopy(client.document)

    assert config.get("table.enabled") is True
    snapshot = config.get_all()
    snapshot["table"]["enabled"] = False
    assert config.get("table.enabled") is True
    assert config.revision == 4
    assert [call["operation"] for call in client.calls] == ["snapshot"]
    assert not selected_path.exists()
    assert client.document == original


def test_rust_mutations_use_cached_revision_and_publish_only_complete_success(
    rust_config: tuple[Config, FakeRustConfigClient, Path],
) -> None:
    config, client, selected_path = rust_config

    updated = config.apply_updates({"table": {"enabled": False}})
    assert updated == ["table.enabled"]
    assert client.calls[-1]["expected_revision"] == 4
    assert config.get("metadata.configuration_revision") == 5
    assert config.revision == 5
    assert not selected_path.exists()

    client.document["metadata"]["configuration_revision"] = 6
    with pytest.raises(ConfigurationRevisionConflict) as raised:
        config.apply_updates({"table": {"enabled": True}})
    assert raised.value.current_revision == 6
    assert config.revision == 5
    assert config.get("metadata.configuration_revision") == 5

    config.reload()
    assert config.revision == 6
    assert config.get("metadata.configuration_revision") == 6


def test_rust_update_forwards_explicit_force_update_flag(
    rust_config: tuple[Config, FakeRustConfigClient, Path],
) -> None:
    config, client, _ = rust_config

    config.apply_updates({"table": {"enabled": False}}, force_update=True)

    assert client.calls[-1]["operation"] == "update"
    assert client.calls[-1]["force_update"] is True


def test_overlapping_rust_mutations_cannot_regress_the_cached_revision(
    rust_config: tuple[Config, FakeRustConfigClient, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = rust_config
    first_response_ready = threading.Event()
    release_first_response = threading.Event()
    second_lock_probe_done = threading.Event()
    second_finished = threading.Event()
    second_lock_was_available: list[bool] = []
    published_revisions: list[int] = []
    published_lock = threading.Lock()
    second_thread_name = "second-config-writer"

    class ObservableRLock:
        """Record whether the second writer can enter while the first publishes."""

        def __init__(self) -> None:
            self._inner = threading.RLock()

        def __enter__(self) -> ObservableRLock:
            if (
                threading.current_thread().name == second_thread_name
                and not second_lock_probe_done.is_set()
            ):
                acquired = self._inner.acquire(blocking=False)
                second_lock_was_available.append(acquired)
                second_lock_probe_done.set()
                if not acquired:
                    self._inner.acquire()
            else:
                self._inner.acquire()
            return self

        def __exit__(self, *_args: Any) -> None:
            self._inner.release()

    observable_lock = ObservableRLock()
    monkeypatch.setattr(config_module, "_CONFIG_LOCK", observable_lock)
    accept_response = config._accept_rust_mutation

    def gated_accept(response: dict[str, Any]) -> list[str]:
        if threading.current_thread().name == "first-config-writer":
            first_response_ready.set()
            if not release_first_response.wait(timeout=5):
                raise TimeoutError("test did not release the first Rust response")
        result = accept_response(response)
        with published_lock:
            published_revisions.append(config.revision or 0)
        return result

    monkeypatch.setattr(config, "_accept_rust_mutation", gated_accept)
    errors: list[BaseException] = []

    def first_writer() -> None:
        try:
            config.apply_updates({"table": {"enabled": False}})
        except BaseException as error:  # Preserve worker failures for the test thread.
            errors.append(error)

    def second_writer() -> None:
        try:
            # Refresh to the first writer's committed revision before CAS. Without
            # the facade lock, this can publish revision 6 before the first thread
            # later publishes its older revision-5 response.
            config.reload()
            config.apply_updates({"table": {"enabled": True}})
        except BaseException as error:  # Preserve worker failures for the test thread.
            errors.append(error)
        finally:
            second_finished.set()

    first = threading.Thread(target=first_writer, name="first-config-writer")
    second = threading.Thread(target=second_writer, name=second_thread_name)
    first.start()
    assert first_response_ready.wait(timeout=5)
    second.start()
    try:
        assert second_lock_probe_done.wait(timeout=5)
        if second_lock_was_available[0]:
            # This branch lets the intentionally broken implementation unwind
            # cleanly before the assertion below reports the lock regression.
            assert second_finished.wait(timeout=5)
    finally:
        release_first_response.set()
        first.join(timeout=5)
        second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert second_lock_was_available == [False]
    assert published_revisions == [5, 6]
    assert config.revision == 6


def test_rust_facade_routes_schema_validation_defaults_and_export_to_authority(
    rust_config: tuple[Config, FakeRustConfigClient, Path],
) -> None:
    config, client, _ = rust_config

    assert config_module.load_schema_reference() == client.defaults
    assert config_module.get_configuration_schema("table") == {"type": "object"}
    errors, warnings = config_module.validate_configuration(
        {"table": {"enabled": True}}, baseline=config.get_all()
    )
    assert errors == []
    assert warnings == []
    assert config.export_configuration(sections=["table"], include_metadata=False) == {
        "table": {"enabled": True}
    }
    assert [call["operation"] for call in client.calls] == [
        "snapshot",
        "defaults",
        "schema",
        "validate",
        "export",
    ]


def test_rust_capture_settings_are_resolved_by_the_authority_with_current_revision(
    rust_config: tuple[Config, FakeRustConfigClient, Path],
) -> None:
    config, client, _ = rust_config

    capture_settings = config.resolve_capture_settings()

    assert capture_settings == {
        "schema_version": 3,
        "configuration_revision": 4,
        "requested": {"calibration": None},
    }
    assert client.calls[-1] == {
        "protocol_version": 1,
        "operation": "capture_settings",
        "data": {"schema_version": 3},
    }


def test_python_authority_cannot_resolve_rust_live_capture_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Config, "_instance", None)
    monkeypatch.setattr(Config, "_authority_mode", None)
    monkeypatch.setattr(Config, "_rust_client", None)
    monkeypatch.setenv("BILLIARDS_CONFIG_AUTHORITY", "python")
    configuration = Config()

    with pytest.raises(
        ConfigurationAuthorityConfigurationError,
        match="BILLIARDS_CONFIG_AUTHORITY=rust",
    ):
        configuration.resolve_capture_settings()


def test_rust_capture_settings_reject_an_inconsistent_revision(
    rust_config: tuple[Config, FakeRustConfigClient, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, client, _ = rust_config
    result = client._result(
        "capture_settings",
        {
            "revision": 5,
            "capture_settings": {
                "schema_version": 3,
                "configuration_revision": 4,
                "requested": {"calibration": None},
            },
        },
    )
    monkeypatch.setattr(client, "call", lambda _request: result)

    with pytest.raises(RustConfigurationAuthorityProtocolError, match="revision"):
        config.resolve_capture_settings()


def test_missing_rust_config_path_fails_before_client_or_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(Config, "_instance", None)
    monkeypatch.setenv("BILLIARDS_CONFIG_AUTHORITY", "rust")
    monkeypatch.delenv("BILLIARDS_CONFIG_PATH", raising=False)
    monkeypatch.setenv(
        "BILLIARDS_CONFIG_AUTHORITY_EXECUTABLE", str(tmp_path / "unused")
    )

    with pytest.raises(
        ConfigurationAuthorityConfigurationError, match="explicit config path"
    ):
        Config.bootstrap_from_environment()


def test_failed_rust_snapshot_never_exposes_a_python_or_previous_snapshot(
    rust_config: tuple[Config, FakeRustConfigClient, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, client, _ = rust_config
    client.fail_snapshot = RustConfigurationAuthorityProcessError("Rust command failed")
    calls: list[str] = []
    monkeypatch.setattr(
        Config,
        "_load_config",
        lambda *_args, **_kwargs: calls.append("python-read"),
    )

    with pytest.raises(RustConfigurationAuthorityProcessError):
        Config.set_config_file(config._config_file or "unused.json")
    with pytest.raises(
        ConfigurationAuthorityConfigurationError, match="snapshot is not initialized"
    ):
        config.get_all()
    assert config._rust_revision is None
    assert calls == []


def test_rust_selector_requires_an_absolute_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(Config, "_instance", None)
    monkeypatch.setenv("BILLIARDS_CONFIG_AUTHORITY", "rust")
    monkeypatch.setenv("BILLIARDS_CONFIG_AUTHORITY_EXECUTABLE", "relative/config-cli")

    with pytest.raises(ConfigurationAuthorityConfigurationError, match="absolute path"):
        Config.bootstrap_from_environment(tmp_path / "active.json")


def test_failed_rust_selection_cannot_reuse_a_previously_loaded_python_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(Config, "_instance", None)
    monkeypatch.setattr(Config, "_config_data", {})
    monkeypatch.setattr(Config, "_config_file", None)
    monkeypatch.setattr(Config, "_authority_mode", None)
    monkeypatch.setenv("BILLIARDS_CONFIG_AUTHORITY", "python")
    python_path = tmp_path / "python-config.json"
    python_path.write_text(
        json.dumps(
            {
                "metadata": {"configuration_revision": 2},
                "table": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )
    instance = Config()
    Config.set_config_file(python_path)
    assert instance.get("table.enabled") is True

    monkeypatch.setenv("BILLIARDS_CONFIG_AUTHORITY", "rust")
    monkeypatch.setenv("BILLIARDS_CONFIG_AUTHORITY_EXECUTABLE", "relative/authority")
    with pytest.raises(ConfigurationAuthorityConfigurationError, match="absolute path"):
        Config.bootstrap_from_environment(tmp_path / "rust-config.json")

    with pytest.raises(
        ConfigurationAuthorityConfigurationError, match="snapshot is not initialized"
    ):
        instance.get_all()


def test_default_authority_remains_python_when_selector_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Config, "_instance", None)
    monkeypatch.delenv("BILLIARDS_CONFIG_AUTHORITY", raising=False)

    assert Config().authority_mode == "python"
