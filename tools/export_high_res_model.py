#!/usr/bin/env python3
"""Export an explicitly selected YOLO model and write lineage metadata.

Exports are derivatives of an existing PyTorch checkpoint. The JSON manifest
records provenance and the declared input contract; it does not claim that a
runtime's preprocessing or decoded detections have passed equivalence tests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIRECTORY = "models/exports"
DEFAULT_IMAGE_SIZE = 1280
ONNX_SUFFIX = ".onnx"
YOLO: Any | None = None


def _load_yolo() -> Any:
    """Load the heavy model-export stack only when an export is requested."""
    try:
        from ultralytics import YOLO as yolo_factory
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is required for model export; install it with "
            "'make install-model-tools'."
        ) from error
    return yolo_factory


def _sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks so large checkpoints need not be loaded whole."""
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_digest(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash one exported ONNX file and retain its v1 manifest component row."""
    if not path.is_file():
        raise FileNotFoundError(f"ONNX export did not produce a file: {path}")
    file_hash = _sha256_file(path)
    return file_hash, [
        {"path": path.name, "size_bytes": path.stat().st_size, "sha256": file_hash}
    ]


def _class_names(model: Any) -> list[dict[str, Any]] | None:
    """Return model labels in index order without relying on config templates."""
    names = getattr(model, "names", None)
    if isinstance(names, Mapping):
        try:
            return [
                {"id": int(class_id), "name": str(name)}
                for class_id, name in sorted(
                    names.items(), key=lambda item: int(item[0])
                )
            ]
        except (TypeError, ValueError):
            return None
    if isinstance(names, (list, tuple)):
        return [{"id": index, "name": str(name)} for index, name in enumerate(names)]
    return None


def _package_version(distribution: str) -> str | None:
    """Read an installed package version without importing optional runtimes."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_model_metadata(model: Any) -> dict[str, Any]:
    """Capture labels and task metadata from the loaded checkpoint itself."""
    core_model = getattr(model, "model", None)
    stride = getattr(core_model, "stride", None)
    if stride is not None and hasattr(stride, "tolist"):
        stride = stride.tolist()
    elif isinstance(stride, tuple):
        stride = list(stride)

    return {
        "task": getattr(model, "task", None),
        "class_names": _class_names(model),
        "stride": stride,
    }


def _build_manifest(
    *,
    source_model: Path,
    exported_model: Path,
    source_metadata: dict[str, Any],
    image_size: int,
    simplify: bool,
) -> dict[str, Any]:
    """Build a versioned record that separates facts from unverified assumptions."""
    export_hash, components = _artifact_digest(exported_model)
    class_names = source_metadata.get("class_names")
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_model": {
            "path": str(source_model.resolve()),
            "size_bytes": source_model.stat().st_size,
            "sha256": _sha256_file(source_model),
        },
        "exported_model": {
            "path": exported_model.name,
            "format": "onnx",
            "size_bytes": sum(component["size_bytes"] for component in components),
            "sha256": export_hash,
            "components": components,
        },
        "model": {
            **source_metadata,
            "class_count": len(class_names) if class_names is not None else None,
        },
        "export": {
            "input_size": [image_size, image_size],
            "batch_size": 1,
            "dynamic_shape": False,
            "embedded_nms": False,
            "onnx_simplification": simplify,
        },
        "expected_input_contract": {
            "shape": [1, 3, image_size, image_size],
            "layout": "NCHW",
            "color_order": "RGB",
            "dtype": "float32",
            "normalization": "[0, 1]",
            "resize_and_letterbox": (
                "Runtime adapter responsibility; must be verified against the "
                "reference preprocessing before equivalence is claimed."
            ),
        },
        "output_contract": {
            "nms": "not embedded; runtime decoding and thresholds are separate",
            "tensor_names_and_shapes": "not captured; inspect the exported graph",
        },
        "toolchain": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "export_device": "cpu",
            "ultralytics": _package_version("ultralytics"),
            "torch": _package_version("torch"),
            "onnxruntime": _package_version("onnxruntime"),
        },
    }


def export_high_res_model(
    model_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    imgsz: int = DEFAULT_IMAGE_SIZE,
    simplify: bool = False,
    overwrite: bool = False,
) -> str:
    """Export one explicit PT checkpoint and write a sibling JSON manifest.

    The checkpoint is copied into an output-local staging directory because
    Ultralytics writes beside its input checkpoint. That keeps generated files
    in the requested output directory and leaves the source asset untouched.

    Args:
        model_path: Explicit source ``.pt`` checkpoint.
        output_dir: Destination for the exported file/package and its manifest.
        imgsz: Fixed square model input edge length.
        simplify: Request ONNX graph simplification.
        overwrite: Replace an existing artifact and manifest only after a
            successful export.

    Returns:
        The exported artifact path. Its companion manifest is named
        ``<artifact-name>.manifest.json``.
    """
    source = Path(model_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Source model file does not exist: {source}")
    source = source.resolve()
    if source.suffix.lower() != ".pt":
        raise ValueError(f"Source model must be a PyTorch .pt checkpoint: {source}")
    if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz < 1:
        raise ValueError("imgsz must be a positive integer")

    model_factory = YOLO if YOLO is not None else _load_yolo()

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifact_path = destination / f"{source.stem}{ONNX_SUFFIX}"
    manifest_path = destination / f"{artifact_path.name}.manifest.json"
    if not overwrite and (artifact_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"Export output already exists: {artifact_path} or {manifest_path}; "
            "pass overwrite=True to replace it after a successful export"
        )
    if simplify and _package_version("onnxslim") is None:
        raise ValueError(
            "ONNX simplification requires the optional onnxslim package; "
            "install it explicitly or rerun with simplify=False"
        )

    with tempfile.TemporaryDirectory(
        prefix=".yolo-export-", dir=destination
    ) as staging:
        staged_source = Path(staging) / source.name
        shutil.copy2(source, staged_source)
        model = model_factory(str(staged_source))
        source_metadata = _source_model_metadata(model)

        export_options: dict[str, Any] = {
            "format": "onnx",
            "imgsz": imgsz,
            "batch": 1,
            "device": "cpu",
            "dynamic": False,
            "nms": False,
        }
        export_options["simplify"] = simplify

        exported_path = Path(model.export(**export_options))
        if not exported_path.exists():
            raise FileNotFoundError(
                f"Ultralytics reported an export that does not exist: {exported_path}"
            )
        if exported_path.stat().st_size == 0:
            raise ValueError(f"Ultralytics produced an empty export: {exported_path}")

        manifest = _build_manifest(
            source_model=source,
            exported_model=exported_path,
            source_metadata=source_metadata,
            image_size=imgsz,
            simplify=simplify,
        )
        staged_manifest = Path(staging) / manifest_path.name
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # Delay replacement until export and manifest generation both succeed.
        if overwrite:
            if artifact_path.exists():
                artifact_path.unlink()
            manifest_path.unlink(missing_ok=True)
        shutil.move(str(exported_path), str(artifact_path))
        staged_manifest.replace(manifest_path)

    return str(artifact_path)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the explicit and reproducible exporter command-line interface."""
    parser = argparse.ArgumentParser(
        description="Export a selected YOLO PT checkpoint with a lineage manifest."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Explicit source .pt checkpoint; no implicit 'latest run' selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIRECTORY),
        help=f"Destination for model and manifest (default: {DEFAULT_OUTPUT_DIRECTORY}).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Fixed square export input size (default: {DEFAULT_IMAGE_SIZE}).",
    )
    parser.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request ONNX graph simplification (requires onnxslim; default: disabled).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a previous artifact after the new export succeeds.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the exporter and print the model and manifest paths."""
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        artifact = export_high_res_model(
            model_path=arguments.model,
            output_dir=arguments.output_dir,
            imgsz=arguments.imgsz,
            simplify=arguments.simplify,
            overwrite=arguments.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    print(f"Exported model: {artifact}")
    print(f"Manifest: {artifact}.manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
