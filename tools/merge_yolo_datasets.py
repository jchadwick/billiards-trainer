#!/usr/bin/env python3
"""Merge local YOLO detection datasets without contacting a dataset service.

Each input directory must contain a ``data.yaml`` with a class-name list or
index-to-name mapping and a training image split. Image and label directories
may follow either ``train/images`` or ``images/train`` layout. Class indices
are remapped by class name so datasets with different class orders can be
merged safely.

Example::

    python tools/merge_yolo_datasets.py datasets/source_a datasets/source_b \
        --output-dir datasets/merged_billiards_dataset

The destination is built in a temporary sibling directory and installed only
after every input and label has been validated. Existing destinations are
preserved unless ``--overwrite`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml  # type: ignore[import-untyped]  # PyYAML runtime dependency; stubs are optional.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
SPLIT_ALIASES = {
    "train": ("train",),
    "valid": ("val", "valid", "validation"),
    "test": ("test",),
}


class DatasetMergeError(ValueError):
    """An input dataset is invalid or cannot be merged safely."""


@dataclass(frozen=True)
class LocalYoloDataset:
    """Validated local dataset metadata and resolved image split paths."""

    root: Path
    class_names: tuple[str, ...]
    image_splits: dict[str, Path | None]


def _read_class_names(
    configuration: dict[str, Any], yaml_path: Path
) -> tuple[str, ...]:
    raw_names = configuration.get("names")
    if isinstance(raw_names, dict):
        try:
            indexed_names = {int(index): name for index, name in raw_names.items()}
        except (TypeError, ValueError) as exc:
            raise DatasetMergeError(
                f"{yaml_path}: class-name mapping keys must be integers"
            ) from exc
        if sorted(indexed_names) != list(range(len(indexed_names))):
            raise DatasetMergeError(
                f"{yaml_path}: class-name indices must be contiguous from zero"
            )
        names = tuple(indexed_names[index] for index in range(len(indexed_names)))
    elif isinstance(raw_names, list):
        names = tuple(raw_names)
    else:
        raise DatasetMergeError(
            f"{yaml_path}: 'names' must be a list or integer-keyed mapping"
        )

    if not names or any(
        not isinstance(name, str) or not name.strip() for name in names
    ):
        raise DatasetMergeError(f"{yaml_path}: class names must be non-empty strings")
    if len({name.strip() for name in names}) != len(names):
        raise DatasetMergeError(f"{yaml_path}: class names must be unique")

    declared_count = configuration.get("nc")
    if declared_count is not None and declared_count != len(names):
        raise DatasetMergeError(
            f"{yaml_path}: nc={declared_count!r} does not match {len(names)} class names"
        )
    return tuple(name.strip() for name in names)


def _resolve_dataset_root(configuration: dict[str, Any], yaml_path: Path) -> Path:
    configured_root = configuration.get("path")
    if configured_root is None:
        return yaml_path.parent.resolve()
    if not isinstance(configured_root, str) or not configured_root.strip():
        raise DatasetMergeError(f"{yaml_path}: 'path' must be a non-empty string")
    root = Path(configured_root).expanduser()
    if not root.is_absolute():
        root = yaml_path.parent / root
    return root.resolve()


def _image_directory(
    configuration: dict[str, Any], dataset_root: Path, split: str
) -> Path | None:
    configured_value = next(
        (
            configuration.get(alias)
            for alias in SPLIT_ALIASES[split]
            if alias in configuration
        ),
        None,
    )
    if configured_value is not None:
        values = (
            configured_value
            if isinstance(configured_value, list)
            else [configured_value]
        )
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = dataset_root / candidate
            candidate = candidate.resolve()
            if candidate.is_dir():
                return candidate

    conventional = (
        dataset_root / split / "images",
        dataset_root / ("val" if split == "valid" else split) / "images",
        dataset_root / "images" / ("val" if split == "valid" else split),
    )
    return next(
        (candidate.resolve() for candidate in conventional if candidate.is_dir()), None
    )


def _label_directory(image_directory: Path) -> Path:
    parts = list(image_directory.parts)
    image_indices = [
        index for index, part in enumerate(parts) if part.casefold() == "images"
    ]
    if image_indices:
        parts[image_indices[-1]] = "labels"
        return Path(*parts)
    return image_directory.parent / "labels"


def load_local_dataset(dataset_directory: Path) -> LocalYoloDataset:
    """Load and validate one dataset's manifest and discover its image splits."""
    root = dataset_directory.expanduser().resolve()
    if not root.is_dir():
        raise DatasetMergeError(f"Input dataset directory does not exist: {root}")
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        raise DatasetMergeError(f"Input dataset has no data.yaml: {root}")

    try:
        with yaml_path.open(encoding="utf-8") as manifest:
            configuration = yaml.safe_load(manifest)
    except (OSError, yaml.YAMLError) as exc:
        raise DatasetMergeError(
            f"Cannot read dataset manifest {yaml_path}: {exc}"
        ) from exc
    if not isinstance(configuration, dict):
        raise DatasetMergeError(f"{yaml_path}: manifest must contain a YAML mapping")

    class_names = _read_class_names(configuration, yaml_path)
    dataset_root = _resolve_dataset_root(configuration, yaml_path)
    image_splits = {
        split: _image_directory(configuration, dataset_root, split)
        for split in SPLIT_ALIASES
    }
    if image_splits["train"] is None:
        raise DatasetMergeError(f"{yaml_path}: no training image directory was found")
    return LocalYoloDataset(root, class_names, image_splits)


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )


def _remap_label_file(
    source: Path,
    destination: Path,
    class_remap: dict[int, int],
    *,
    dataset_root: Path,
) -> None:
    if not source.exists():
        # YOLO allows images with no objects; represent those as empty labels.
        destination.write_text("", encoding="utf-8")
        return
    if not source.is_file():
        raise DatasetMergeError(f"Expected a label file, found: {source}")

    remapped_lines: list[str] = []
    try:
        for line_number, raw_line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), 1
        ):
            values = raw_line.split()
            if not values:
                continue
            if len(values) < 5:
                raise DatasetMergeError(
                    f"{source}:{line_number}: a detection label needs a class and four box values"
                )
            try:
                source_class = int(values[0])
                coordinates = [float(value) for value in values[1:5]]
                remaining = [float(value) for value in values[5:]]
            except ValueError as exc:
                raise DatasetMergeError(
                    f"{source}:{line_number}: label values must be numeric"
                ) from exc
            if source_class not in class_remap:
                raise DatasetMergeError(
                    f"{source}:{line_number}: class index {source_class} is outside the manifest"
                )
            if not all(math.isfinite(value) for value in coordinates + remaining):
                raise DatasetMergeError(
                    f"{source}:{line_number}: label values must be finite"
                )
            center_x, center_y, width, height = coordinates
            if not (
                0 <= center_x <= 1
                and 0 <= center_y <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):
                raise DatasetMergeError(
                    f"{source}:{line_number}: box coordinates must be normalized to [0, 1]"
                )
            values[0] = str(class_remap[source_class])
            remapped_lines.append(" ".join(values))
    except OSError as exc:
        raise DatasetMergeError(
            f"Cannot read labels under {dataset_root}: {exc}"
        ) from exc

    destination.write_text(
        "\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8"
    )


def merge_yolo_datasets(
    input_directories: Sequence[Path],
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Merge local YOLO datasets with class-name-aware label remapping.

    The output is assembled away from the destination and atomically renamed
    into place after validation. An existing destination is only replaced when
    ``overwrite`` is true.
    """
    if not input_directories:
        raise DatasetMergeError("Provide at least one input dataset directory")

    datasets = [load_local_dataset(path) for path in input_directories]
    resolved_inputs = {dataset.root for dataset in datasets}
    output = output_directory.expanduser().resolve()
    if any(
        output == root or root in output.parents or output in root.parents
        for root in resolved_inputs
    ):
        raise DatasetMergeError(
            "The output directory must be outside every input dataset"
        )
    if output.exists() and not overwrite:
        raise DatasetMergeError(
            f"Output already exists; pass --overwrite to replace it: {output}"
        )
    if output.exists() and not output.is_dir():
        raise DatasetMergeError(f"Output path exists and is not a directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    class_names = tuple(
        sorted(
            {name for dataset in datasets for name in dataset.class_names},
            key=lambda name: (name.casefold(), name),
        )
    )
    merged_class_ids = {name: index for index, name in enumerate(class_names)}
    temporary_output = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    backup: Path | None = None
    split_counts = dict.fromkeys(SPLIT_ALIASES, 0)
    empty_label_count = 0
    file_counter = 0

    try:
        for split in SPLIT_ALIASES:
            (temporary_output / "images" / split).mkdir(parents=True)
            (temporary_output / "labels" / split).mkdir(parents=True)

        for dataset_index, dataset in enumerate(datasets):
            class_remap = {
                source_index: merged_class_ids[name]
                for source_index, name in enumerate(dataset.class_names)
            }
            for split, source_images in dataset.image_splits.items():
                if source_images is None:
                    continue
                source_labels = _label_directory(source_images)
                for image_path in _image_files(source_images):
                    stem = f"source_{dataset_index:03d}_{file_counter:08d}"
                    image_name = f"{stem}{image_path.suffix.casefold()}"
                    destination_image = temporary_output / "images" / split / image_name
                    destination_label = (
                        temporary_output / "labels" / split / f"{stem}.txt"
                    )
                    label_path = source_labels / f"{image_path.stem}.txt"
                    shutil.copy2(image_path, destination_image)
                    if not label_path.exists():
                        empty_label_count += 1
                    _remap_label_file(
                        label_path,
                        destination_label,
                        class_remap,
                        dataset_root=dataset.root,
                    )
                    split_counts[split] += 1
                    file_counter += 1

        manifest = {
            "path": str(output),
            "train": "images/train",
            "val": "images/valid",
            "test": "images/test",
            "nc": len(class_names),
            "names": list(class_names),
        }
        (temporary_output / "data.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )

        if output.exists():
            backup = Path(
                tempfile.mkdtemp(prefix=f".{output.name}.previous-", dir=output.parent)
            )
            backup.rmdir()
            output.rename(backup)
        try:
            temporary_output.rename(output)
        except OSError:
            if backup is not None:
                backup.rename(output)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)

    except Exception:
        shutil.rmtree(temporary_output, ignore_errors=True)
        raise

    return {
        "output": output,
        "class_names": class_names,
        "split_counts": split_counts,
        "empty_label_count": empty_label_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge local YOLO detection datasets without API keys or remote access."
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        type=Path,
        help="Input directories containing a YOLO data.yaml and image splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "merged_billiards_dataset",
        help="Output dataset directory (default: datasets/merged_billiards_dataset).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory after the new merge succeeds.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = merge_yolo_datasets(
            args.datasets, args.output_dir, overwrite=args.overwrite
        )
    except DatasetMergeError as exc:
        print(f"Dataset merge failed: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Dataset merge failed while writing files: {exc}", file=sys.stderr)
        return 2

    print(f"Merged dataset: {result['output']}")
    print(f"Classes ({len(result['class_names'])}): {', '.join(result['class_names'])}")
    for split, count in result["split_counts"].items():
        print(f"{split}: {count} images")
    if result["empty_label_count"]:
        print(f"Images with empty labels: {result['empty_label_count']}")
    print(f"YOLO manifest: {result['output'] / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
