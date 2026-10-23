"""Tests for explicit, provenance-preserving model export behavior."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import export_high_res_model as export_tool


class FakeYOLO:
    """Small controllable export double; no weights or export runtimes required."""

    instances = []

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.names = {0: "cue", 1: "ball"}
        self.task = "detect"
        self.model = SimpleNamespace(stride=[8, 16, 32])
        self.export_calls = []
        self.instances.append(self)

    def export(self, **options):
        self.export_calls.append(options)
        destination = self.model_path.with_suffix(".onnx")
        destination.write_bytes(b"deterministic onnx artifact")
        return str(destination)


def test_export_uses_explicit_source_and_writes_hash_lineage_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "chosen-model.pt"
    source.write_bytes(b"source checkpoint contents")
    output = tmp_path / "exports"
    FakeYOLO.instances.clear()
    monkeypatch.setattr(export_tool, "YOLO", FakeYOLO)

    artifact = Path(
        export_tool.export_high_res_model(
            source,
            output_dir=output,
            imgsz=960,
            simplify=False,
        )
    )
    manifest_path = Path(f"{artifact}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert artifact.parent == output
    assert artifact.exists()
    assert manifest["manifest_schema_version"] == 1
    assert (
        manifest["source_model"]["sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert manifest["exported_model"]["sha256"]
    assert manifest["exported_model"]["path"] == artifact.name
    assert manifest["model"]["class_names"] == [
        {"id": 0, "name": "cue"},
        {"id": 1, "name": "ball"},
    ]
    assert manifest["expected_input_contract"]["shape"] == [1, 3, 960, 960]
    assert manifest["output_contract"]["nms"].startswith("not embedded")
    assert len(manifest["exported_model"]["components"]) == 1

    model = FakeYOLO.instances[0]
    options = model.export_calls[0]
    assert options["format"] == "onnx"
    assert options["device"] == "cpu"
    assert options["nms"] is False
    assert options["dynamic"] is False
    assert options["simplify"] is False
    assert source.read_bytes() == b"source checkpoint contents"
    assert list(output.glob(".yolo-export-*")) == []


def test_export_requires_an_explicit_existing_pt_checkpoint(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        export_tool.export_high_res_model(tmp_path / "missing.pt")

    non_pt = tmp_path / "model.mlpackage"
    non_pt.write_bytes(b"not a PT checkpoint")
    with pytest.raises(ValueError, match="PyTorch .pt"):
        export_tool.export_high_res_model(non_pt)


def test_export_reports_the_opt_in_model_dependency_when_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "chosen-model.pt"
    source.write_bytes(b"source checkpoint contents")

    monkeypatch.setattr(export_tool, "YOLO", None)
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    with pytest.raises(RuntimeError, match="make install-model-tools"):
        export_tool.export_high_res_model(source)


def test_export_refuses_to_overwrite_without_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "chosen-model.pt"
    source.write_bytes(b"checkpoint")
    output = tmp_path / "exports"
    output.mkdir()
    (output / "chosen-model.onnx").write_bytes(b"prior export")
    FakeYOLO.instances.clear()
    monkeypatch.setattr(export_tool, "YOLO", FakeYOLO)

    with pytest.raises(FileExistsError, match="overwrite=True"):
        export_tool.export_high_res_model(source, output_dir=output)

    assert FakeYOLO.instances == []
    assert (output / "chosen-model.onnx").read_bytes() == b"prior export"


def test_export_does_not_silently_install_missing_simplifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "chosen-model.pt"
    source.write_bytes(b"checkpoint")
    FakeYOLO.instances.clear()
    monkeypatch.setattr(export_tool, "YOLO", FakeYOLO)
    monkeypatch.setattr(
        export_tool,
        "_package_version",
        lambda package: None if package == "onnxslim" else "test-version",
    )

    with pytest.raises(ValueError, match="install it explicitly"):
        export_tool.export_high_res_model(
            source, output_dir=tmp_path / "exports", simplify=True
        )

    assert FakeYOLO.instances == []


def test_argument_parser_requires_model_path_and_supports_no_simplify():
    with pytest.raises(SystemExit):
        export_tool.build_argument_parser().parse_args([])

    arguments = export_tool.build_argument_parser().parse_args(["--model", "model.pt"])
    assert arguments.model == Path("model.pt")
    assert arguments.simplify is False

    explicit_simplification = export_tool.build_argument_parser().parse_args(
        ["--model", "model.pt", "--simplify"]
    )
    assert explicit_simplification.simplify is True
    with pytest.raises(SystemExit):
        export_tool.build_argument_parser().parse_args(
            ["--model", "model.pt", "--format", "coreml"]
        )
