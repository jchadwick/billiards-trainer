"""HTTP contracts for the retained vision compatibility routes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.api.dependencies as dependencies
import backend.api.routes.vision as vision_route
from backend.vision.detection.yolo_detector import ModelValidationError


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    """Mount only these routers and isolate their shared application state."""
    monkeypatch.setattr(dependencies.app_state, "vision_module", None)
    monkeypatch.setattr(vision_route, "MODELS_DIR", tmp_path / "models")
    app = FastAPI()
    app.include_router(vision_route.router, prefix="/api/v1/vision")
    return TestClient(app)


def _write_model(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test model")
    return path


def test_model_info_reports_empty_components_and_lists_onnx_files(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    models = tmp_path / "available-models"
    _write_model(models / "balls.onnx")
    _write_model(models / "ignored.ONNX")
    monkeypatch.setattr(vision_route, "MODELS_DIR", models)

    response = client.get("/api/v1/vision/model/info")

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "model_info": {},
        "available_models": ["balls.onnx"],
    }


def test_model_info_reads_detector_and_handles_route_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    vision = SimpleNamespace(
        detector=SimpleNamespace(get_model_info=lambda: {"name": "detector"})
    )
    monkeypatch.setattr(dependencies.app_state, "vision_module", vision)
    monkeypatch.setattr(vision_route, "MODELS_DIR", Path("/path/that/does/not/exist"))

    response = client.get("/api/v1/vision/model/info")
    assert response.status_code == 200
    assert response.json()["model_info"] == {"name": "detector"}
    assert response.json()["available_models"] == []

    monkeypatch.setattr(
        dependencies.app_state,
        "vision_module",
        SimpleNamespace(detector=SimpleNamespace()),
    )
    without_detector_info = client.get("/api/v1/vision/model/info")
    assert without_detector_info.status_code == 200
    assert without_detector_info.json()["model_info"] == {}

    def fail_state() -> Any:
        raise RuntimeError("state unavailable")

    monkeypatch.setattr(vision_route, "get_app_state", fail_state)
    failed = client.get("/api/v1/vision/model/info")
    assert failed.status_code == 503
    assert failed.json()["detail"] == "Vision processing failed"


@pytest.mark.parametrize("filename", ["weights.pt", "weights.ONNX"])
def test_upload_rejects_missing_or_unsupported_filename(
    client: TestClient, filename: str
) -> None:
    response = client.post(
        "/api/v1/vision/model/upload",
        files={"file": (filename, b"model", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only .onnx files are supported"


def test_upload_requires_file_part(client: TestClient) -> None:
    response = client.post("/api/v1/vision/model/upload")
    assert response.status_code == 422


def test_upload_saves_valid_model_and_can_skip_optional_inference(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models = tmp_path / "uploaded-models"
    monkeypatch.setattr(vision_route, "MODELS_DIR", models)
    monkeypatch.setattr(
        vision_route.YOLODetector,
        "validate_onnx_model",
        staticmethod(lambda path: {"valid": Path(path).exists()}),
    )
    inference_calls: list[str] = []

    def inference(path: str, device: str = "cpu") -> dict[str, Any]:
        inference_calls.append(f"{Path(path).suffix}:{device}")
        return {"success": True}

    monkeypatch.setattr(
        vision_route.YOLODetector, "test_model_inference", staticmethod(inference)
    )
    response = client.post(
        "/api/v1/vision/model/upload?run_inference_test=false",
        files={"file": ("custom.onnx", b"model bytes", "application/octet-stream")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert body["model_name"].startswith("custom_")
    assert body["model_name"].endswith(".onnx")
    assert Path(body["model_path"]).read_bytes() == b"model bytes"
    assert body["validation_results"] == {"valid": True}
    assert body["inference_test"] is None
    assert inference_calls == []


def test_upload_runs_inference_when_enabled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(vision_route, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(
        vision_route.YOLODetector,
        "validate_onnx_model",
        staticmethod(lambda _path: {"valid": True}),
    )
    monkeypatch.setattr(
        vision_route.YOLODetector,
        "test_model_inference",
        staticmethod(lambda _path, _device: {"success": True, "count": 0}),
    )

    response = client.post(
        "/api/v1/vision/model/upload",
        files={"file": ("custom.onnx", b"model bytes", "application/octet-stream")},
    )

    assert response.status_code == 201
    assert response.json()["inference_test"] == {"success": True, "count": 0}


def test_upload_maps_validation_and_inference_failures_to_client_errors(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(vision_route, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(
        vision_route.YOLODetector,
        "validate_onnx_model",
        staticmethod(
            lambda _path: (_ for _ in ()).throw(ModelValidationError("bad graph"))
        ),
    )
    validation_failed = client.post(
        "/api/v1/vision/model/upload?run_inference_test=false",
        files={"file": ("invalid.onnx", b"bad", "application/octet-stream")},
    )
    assert validation_failed.status_code == 400
    assert validation_failed.json()["detail"] == "Model validation failed: bad graph"

    monkeypatch.setattr(
        vision_route.YOLODetector,
        "validate_onnx_model",
        staticmethod(lambda _path: {"valid": True}),
    )
    monkeypatch.setattr(
        vision_route.YOLODetector,
        "test_model_inference",
        staticmethod(lambda _path, _device: {"success": False, "error": "bad output"}),
    )
    inference_failed = client.post(
        "/api/v1/vision/model/upload",
        files={"file": ("invalid-inference.onnx", b"bad", "application/octet-stream")},
    )
    assert inference_failed.status_code == 400
    assert (
        "Model inference test failed: bad output" in inference_failed.json()["detail"]
    )
    assert list((tmp_path / "models").glob("*.onnx")) == []


def test_upload_maps_unexpected_validation_errors_to_server_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(vision_route, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(
        vision_route.YOLODetector,
        "validate_onnx_model",
        staticmethod(
            lambda _path: (_ for _ in ()).throw(RuntimeError("parser crashed"))
        ),
    )

    response = client.post(
        "/api/v1/vision/model/upload?run_inference_test=false",
        files={"file": ("broken.onnx", b"bad", "application/octet-stream")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to upload model: parser crashed"
    assert list(tmp_path.glob("tmp*.onnx")) == []


def test_upload_cleanup_tolerates_validation_removing_temporary_file(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(vision_route, "MODELS_DIR", tmp_path / "models")

    def validator_removes_file(path: str) -> dict[str, Any]:
        Path(path).unlink()
        raise ModelValidationError("file disappeared during validation")

    monkeypatch.setattr(
        vision_route.YOLODetector,
        "validate_onnx_model",
        staticmethod(validator_removes_file),
    )
    response = client.post(
        "/api/v1/vision/model/upload?run_inference_test=false",
        files={"file": ("weights.onnx", b"model", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "file disappeared during validation" in response.json()["detail"]


def test_upload_handles_temporary_file_and_models_directory_failures(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    models = tmp_path / "models"
    monkeypatch.setattr(vision_route, "MODELS_DIR", models)

    def fail_temporary_file(**_kwargs: Any) -> Any:
        raise OSError("temporary storage unavailable")

    monkeypatch.setattr(
        vision_route.tempfile, "NamedTemporaryFile", fail_temporary_file
    )
    temp_failure = client.post(
        "/api/v1/vision/model/upload",
        files={"file": ("weights.onnx", b"model", "application/octet-stream")},
    )
    assert temp_failure.status_code == 500
    assert temp_failure.json()["detail"] == (
        "Failed to upload model: temporary storage unavailable"
    )

    original_mkdir = Path.mkdir

    def fail_models_directory(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == models:
            raise OSError("models directory unavailable")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_models_directory)
    directory_failure = client.post(
        "/api/v1/vision/model/upload",
        files={"file": ("weights.onnx", b"model", "application/octet-stream")},
    )
    assert directory_failure.status_code == 500
    assert directory_failure.json()["detail"] == (
        "Internal server error: models directory unavailable"
    )


def test_load_model_validates_path_extension_and_components(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    monkeypatch.setattr(vision_route, "MODELS_DIR", models)

    missing = client.post(
        "/api/v1/vision/model/load", params={"model_name": "absent.onnx"}
    )
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Model not found: absent.onnx"

    _write_model(models / "weights.pt")
    unsupported = client.post(
        "/api/v1/vision/model/load", params={"model_name": "weights.pt"}
    )
    assert unsupported.status_code == 400
    assert unsupported.json()["detail"] == "Only .onnx models can be loaded"

    _write_model(models / "weights.onnx")
    unavailable = client.post(
        "/api/v1/vision/model/load", params={"model_name": "weights.onnx"}
    )
    assert unavailable.status_code == 503
    assert unavailable.json()["detail"] == "Vision module not available"

    monkeypatch.setattr(dependencies.app_state, "vision_module", SimpleNamespace())
    no_detector = client.post(
        "/api/v1/vision/model/load", params={"model_name": "weights.onnx"}
    )
    assert no_detector.status_code == 503
    assert no_detector.json()["detail"] == "Detector not available in vision module"


def test_load_model_success_failure_and_unexpected_exception(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _write_model(tmp_path / "models" / "weights.onnx")
    monkeypatch.setattr(vision_route, "MODELS_DIR", model.parent)
    detector = SimpleNamespace(reload_model=lambda _path: True)
    monkeypatch.setattr(
        dependencies.app_state, "vision_module", SimpleNamespace(detector=detector)
    )

    loaded = client.post("/api/v1/vision/model/load", params={"model_name": model.name})
    assert loaded.status_code == 200
    assert loaded.json()["success"] is True
    assert loaded.json()["model_path"] == str(model)

    detector.reload_model = lambda _path: False
    failed = client.post("/api/v1/vision/model/load", params={"model_name": model.name})
    assert failed.status_code == 503
    assert failed.json()["detail"] == "Vision processing failed"

    def broken_reload(_path: str) -> bool:
        raise RuntimeError("runtime failed")

    detector.reload_model = broken_reload
    unexpected = client.post(
        "/api/v1/vision/model/load", params={"model_name": model.name}
    )
    assert unexpected.status_code == 503
    assert unexpected.json()["detail"] == "Vision processing failed"


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [({}, 422), ({"model_path": "relative/model.onnx"}, 400)],
)
def test_reload_request_validation_is_exposed_over_anonymous_http(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, str],
    expected_status: int,
) -> None:
    monkeypatch.setattr(
        dependencies.app_state,
        "vision_module",
        SimpleNamespace(detector=SimpleNamespace(reload_model=lambda _path: True)),
    )
    response = client.post("/api/v1/vision/model/reload", json=payload)
    assert response.status_code == expected_status


def test_reload_rejects_unavailable_detector_and_invalid_model_paths(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    response = client.post(
        "/api/v1/vision/model/reload", json={"model_path": str(tmp_path / "model.onnx")}
    )
    assert response.status_code == 503
    assert (
        response.json()["detail"] == "Vision module is not initialized or not available"
    )

    monkeypatch.setattr(
        dependencies.app_state, "vision_module", SimpleNamespace(detector=None)
    )
    no_detector = client.post(
        "/api/v1/vision/model/reload", json={"model_path": str(tmp_path / "model.onnx")}
    )
    assert no_detector.status_code == 400
    assert (
        no_detector.json()["detail"]
        == "Vision module does not have a YOLO detector configured"
    )

    monkeypatch.setattr(
        dependencies.app_state,
        "vision_module",
        SimpleNamespace(detector=SimpleNamespace()),
    )
    no_reload = client.post(
        "/api/v1/vision/model/reload", json={"model_path": str(tmp_path / "model.onnx")}
    )
    assert no_reload.status_code == 400
    assert "does not support model reloading" in no_reload.json()["detail"]

    detector = SimpleNamespace(reload_model=lambda _path: True)
    monkeypatch.setattr(
        dependencies.app_state, "vision_module", SimpleNamespace(detector=detector)
    )
    missing = client.post(
        "/api/v1/vision/model/reload",
        json={"model_path": str(tmp_path / "missing.onnx")},
    )
    assert missing.status_code == 404
    assert "Model file does not exist" in missing.json()["detail"]

    invalid_format_path = _write_model(tmp_path / "model.txt")
    invalid_format = client.post(
        "/api/v1/vision/model/reload", json={"model_path": str(invalid_format_path)}
    )
    assert invalid_format.status_code == 400
    assert "Model must be one of" in invalid_format.json()["detail"]


def test_reload_success_false_validation_exception_and_runtime_exception(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _write_model(tmp_path / "model.ONNX")
    detector = SimpleNamespace(
        reload_model=lambda _path: True,
        get_model_info=lambda: {"model_path": "previous.onnx"},
    )
    monkeypatch.setattr(
        dependencies.app_state, "vision_module", SimpleNamespace(detector=detector)
    )

    success = client.post(
        "/api/v1/vision/model/reload", json={"model_path": str(model)}
    )
    assert success.status_code == 200
    assert success.json()["success"] is True
    assert success.json()["previous_model"] == "previous.onnx"
    assert success.json()["model_info"] == {"model_path": "previous.onnx"}

    detector.reload_model = lambda _path: False
    failed = client.post("/api/v1/vision/model/reload", json={"model_path": str(model)})
    assert failed.status_code == 503

    def invalid_model(_path: str) -> bool:
        raise ModelValidationError("incompatible graph")

    detector.reload_model = invalid_model
    invalid = client.post(
        "/api/v1/vision/model/reload", json={"model_path": str(model)}
    )
    assert invalid.status_code == 400
    assert invalid.json()["detail"] == "Model validation failed: incompatible graph"

    def unexpected_error(_path: str) -> bool:
        raise RuntimeError("reload crash")

    detector.reload_model = unexpected_error
    unexpected = client.post(
        "/api/v1/vision/model/reload", json={"model_path": str(model)}
    )
    assert unexpected.status_code == 503
    assert unexpected.json()["detail"] == "Vision processing failed"
