"""Producer-side contract tests for the YOLO ONNX exporter.

These tests fake only the expensive Ultralytics export call. They exercise the
real staging, manifest, hashing, overwrite, and rollback behavior without
requiring model weights or an ONNX runtime.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools import export_high_res_model as exporter


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = (
    REPOSITORY_ROOT / "perception/contracts/model-export/v1"
)
SOURCE_FIXTURE = CONTRACT_ROOT / "source-checkpoint.fixture"
EXPORTED_FIXTURE = CONTRACT_ROOT / "exported-model.fixture"
MANIFEST_CONTRACT = json.loads(
    (CONTRACT_ROOT / "manifest-contract.json").read_text(encoding="utf-8")
)


class FakeYolo:
    """Stand-in for Ultralytics that writes a deterministic ONNX fixture."""

    def __init__(self, checkpoint: str, payload: bytes | None = None) -> None:
        self.checkpoint = Path(checkpoint)
        self.payload = payload if payload is not None else EXPORTED_FIXTURE.read_bytes()
        self.names = {1: "ball", 0: "cue"}
        self.task = "detect"
        self.model = SimpleNamespace(stride=SimpleNamespace(tolist=lambda: [8, 16, 32]))
        self.options: dict[str, Any] | None = None

    def export(self, **options: Any) -> str:
        self.options = options
        output = self.checkpoint.with_suffix(".onnx")
        output.write_bytes(self.payload)
        return str(output)


class FailingYolo(FakeYolo):
    def export(self, **options: Any) -> str:
        raise RuntimeError("synthetic exporter failure")


def _source_copy(directory: Path) -> Path:
    source = directory / "billiards-yolov8n.pt"
    shutil.copyfile(SOURCE_FIXTURE, source)
    return source


def _use_fake_toolchain(monkeypatch: pytest.MonkeyPatch, factory: Any) -> None:
    monkeypatch.setattr(exporter, "YOLO", factory)
    monkeypatch.setattr(exporter, "_package_version", lambda _name: "test-version")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_fake_export_matches_shared_rust_manifest_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_copy(tmp_path)
    loaded: list[FakeYolo] = []

    def factory(checkpoint: str) -> FakeYolo:
        model = FakeYolo(checkpoint)
        loaded.append(model)
        return model

    _use_fake_toolchain(monkeypatch, factory)
    artifact = Path(exporter.export_high_res_model(source, tmp_path / "out"))
    manifest_path = Path(f"{artifact}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert artifact.name == MANIFEST_CONTRACT["exported_model"]["path"]
    assert artifact.read_bytes() == EXPORTED_FIXTURE.read_bytes()
    assert manifest["manifest_schema_version"] == MANIFEST_CONTRACT[
        "manifest_schema_version"
    ]
    assert manifest["source_model"]["sha256"] == MANIFEST_CONTRACT[
        "source_model"
    ]["sha256"]
    assert manifest["exported_model"]["sha256"] == MANIFEST_CONTRACT[
        "exported_model"
    ]["sha256"]
    assert manifest["exported_model"]["path"] == MANIFEST_CONTRACT[
        "exported_model"
    ]["path"]
    assert manifest["model"]["class_count"] == MANIFEST_CONTRACT["model"][
        "class_count"
    ]
    assert manifest["model"]["class_names"] == MANIFEST_CONTRACT["model"][
        "class_names"
    ]
    assert manifest["model"]["stride"] == [8, 16, 32]
    assert manifest["export"]["input_size"] == [exporter.DEFAULT_IMAGE_SIZE] * 2
    assert manifest["expected_input_contract"]["color_order"] == "RGB"
    assert manifest["expected_input_contract"]["layout"] == "NCHW"
    assert manifest["output_contract"]["nms"].startswith("not embedded")
    assert source.read_bytes() == SOURCE_FIXTURE.read_bytes()
    assert len(loaded) == 1
    assert loaded[0].checkpoint.parent.name.startswith(".yolo-export-")
    assert loaded[0].options == {
        "format": "onnx",
        "imgsz": exporter.DEFAULT_IMAGE_SIZE,
        "batch": 1,
        "device": "cpu",
        "dynamic": False,
        "nms": False,
        "simplify": False,
    }


def test_existing_output_is_not_loaded_or_changed_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_copy(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    artifact = output / "billiards-yolov8n.onnx"
    manifest = output / "billiards-yolov8n.onnx.manifest.json"
    artifact.write_bytes(b"old model")
    manifest.write_text('{"old":true}\n', encoding="utf-8")

    def unexpected_factory(_checkpoint: str) -> FakeYolo:
        pytest.fail("the model should not load when outputs already exist")

    _use_fake_toolchain(monkeypatch, unexpected_factory)
    with pytest.raises(FileExistsError):
        exporter.export_high_res_model(source, output)

    assert artifact.read_bytes() == b"old model"
    assert manifest.read_text(encoding="utf-8") == '{"old":true}\n'


def test_failed_overwrite_preserves_the_previous_artifact_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_copy(tmp_path)
    output = tmp_path / "out"
    _use_fake_toolchain(monkeypatch, lambda checkpoint: FakeYolo(checkpoint))
    artifact = Path(exporter.export_high_res_model(source, output))
    manifest = Path(f"{artifact}.manifest.json")
    previous_artifact = artifact.read_bytes()
    previous_manifest = manifest.read_bytes()

    _use_fake_toolchain(monkeypatch, lambda checkpoint: FailingYolo(checkpoint))
    with pytest.raises(RuntimeError, match="synthetic exporter failure"):
        exporter.export_high_res_model(source, output, overwrite=True)

    assert artifact.read_bytes() == previous_artifact
    assert manifest.read_bytes() == previous_manifest
    assert not list(output.glob(".yolo-export-previous-*"))


@pytest.mark.parametrize(
    ("suffix", "image_size", "expected_error"),
    [
        (".onnx", 1280, ValueError),
        (".pt", 0, ValueError),
        (".pt", True, ValueError),
    ],
)
def test_invalid_source_type_and_image_size_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    image_size: int,
    expected_error: type[Exception],
) -> None:
    source = tmp_path / f"model{suffix}"
    source.write_bytes(b"checkpoint")
    _use_fake_toolchain(monkeypatch, lambda checkpoint: FakeYolo(checkpoint))

    with pytest.raises(expected_error):
        exporter.export_high_res_model(source, tmp_path / "out", imgsz=image_size)


def test_simplification_requires_the_optional_simplifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_copy(tmp_path)
    _use_fake_toolchain(monkeypatch, lambda checkpoint: FakeYolo(checkpoint))
    monkeypatch.setattr(exporter, "_package_version", lambda name: None if name == "onnxslim" else "test")

    with pytest.raises(ValueError, match="requires the optional onnxslim package"):
        exporter.export_high_res_model(source, tmp_path / "out", simplify=True)

    assert not (tmp_path / "out" / "billiards-yolov8n.onnx").exists()


def test_manifest_hashes_match_exact_output_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source_copy(tmp_path)
    payload = b"alternate deterministic ONNX bytes"
    _use_fake_toolchain(monkeypatch, lambda checkpoint: FakeYolo(checkpoint, payload))

    artifact = Path(exporter.export_high_res_model(source, tmp_path / "out", imgsz=640))
    manifest = json.loads(Path(f"{artifact}.manifest.json").read_text(encoding="utf-8"))

    assert manifest["exported_model"]["sha256"] == _sha256(payload)
    assert manifest["exported_model"]["size_bytes"] == len(payload)
    assert manifest["export"]["input_size"] == [640, 640]
    assert manifest["expected_input_contract"]["shape"] == [1, 3, 640, 640]
