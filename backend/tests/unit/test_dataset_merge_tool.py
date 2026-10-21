"""Regression tests for the offline YOLO dataset merge utility."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tools.merge_yolo_datasets import DatasetMergeError, merge_yolo_datasets


def _make_dataset(
    root: Path,
    *,
    names: list[str] | dict[int, str],
    split_layout: str,
    images: dict[str, list[tuple[str, str | None]]],
) -> None:
    root.mkdir(parents=True)
    configuration: dict[str, object] = {
        "path": str(root),
        "nc": len(names),
        "names": names,
    }
    for split in images:
        yaml_split = "val" if split == "valid" else split
        configuration[yaml_split] = (
            f"{split}/images" if split_layout == "split-first" else f"images/{split}"
        )

    (root / "data.yaml").write_text(
        yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8"
    )
    for split, records in images.items():
        image_dir = (
            root / split / "images"
            if split_layout == "split-first"
            else root / "images" / split
        )
        label_dir = (
            root / split / "labels"
            if split_layout == "split-first"
            else root / "labels" / split
        )
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for image_name, label in records:
            (image_dir / image_name).write_bytes(b"test image payload")
            if label is not None:
                (label_dir / f"{Path(image_name).stem}.txt").write_text(
                    label, encoding="utf-8"
                )


def test_merge_remaps_classes_by_name_and_keeps_supported_layouts(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _make_dataset(
        first,
        names={0: "cue", 1: "ball"},
        split_layout="split-first",
        images={
            "train": [("cue.jpg", "0 0.5 0.5 0.2 0.2\n"), ("empty.png", None)],
            "valid": [("ball.png", "1 0.4 0.6 0.3 0.2\n")],
        },
    )
    _make_dataset(
        second,
        names=["eight", "cue"],
        split_layout="images-first",
        images={"train": [("eight.jpeg", "0 0.25 0.75 0.1 0.15\n")]},
    )

    result = merge_yolo_datasets([first, second], tmp_path / "merged" / "dataset")
    output = result["output"]

    assert result["class_names"] == ("ball", "cue", "eight")
    assert result["split_counts"] == {"train": 3, "valid": 1, "test": 0}
    assert result["empty_label_count"] == 1
    assert yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))[
        "names"
    ] == [
        "ball",
        "cue",
        "eight",
    ]

    train_labels = sorted((output / "labels" / "train").glob("*.txt"))
    train_contents = [path.read_text(encoding="utf-8") for path in train_labels]
    assert [content.split()[0] if content else "" for content in train_contents] == [
        "1",
        "",
        "2",
    ]
    valid_label = next((output / "labels" / "valid").glob("*.txt"))
    assert valid_label.read_text(encoding="utf-8").startswith("0 ")


def test_malformed_label_fails_without_installing_partial_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _make_dataset(
        source,
        names=["ball"],
        split_layout="split-first",
        images={"train": [("bad.jpg", "3 0.5 0.5 0.2 0.2\n")]},
    )
    output = tmp_path / "merged"

    with pytest.raises(DatasetMergeError, match="outside the manifest"):
        merge_yolo_datasets([source], output)

    assert not output.exists()
    assert not list(tmp_path.glob(".merged.building-*"))


def test_existing_output_is_preserved_unless_overwrite_is_explicit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _make_dataset(
        source,
        names=["ball"],
        split_layout="split-first",
        images={"train": [("ball.jpg", "0 0.5 0.5 0.2 0.2\n")]},
    )
    output = tmp_path / "merged"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(DatasetMergeError, match="pass --overwrite"):
        merge_yolo_datasets([source], output)
    assert sentinel.read_text(encoding="utf-8") == "preserve"

    result = merge_yolo_datasets([source], output, overwrite=True)
    assert result["split_counts"]["train"] == 1
    assert not sentinel.exists()
    assert (output / "data.yaml").is_file()


def test_output_cannot_be_the_source_or_one_of_its_parent_directories(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _make_dataset(
        source,
        names=["ball"],
        split_layout="split-first",
        images={"train": [("ball.jpg", "0 0.5 0.5 0.2 0.2\n")]},
    )

    with pytest.raises(DatasetMergeError, match="outside every input dataset"):
        merge_yolo_datasets([source], source / "merged")
