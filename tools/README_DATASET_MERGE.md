# Merge curated YOLO datasets

This offline helper combines datasets that already contain valid `data.yaml`
manifests and image/label splits. It maps class names across input manifests
and refuses to replace existing output unless `--overwrite` is provided. It
does not create, repair, or validate labels.

Install the optional Python image dependencies first:

```bash
uv sync --extra python-tooling
```

Then merge reviewed datasets:

```bash
uv run python tools/merge_yolo_datasets.py \
  /path/to/dataset-a /path/to/dataset-b \
  --output-dir /path/to/merged-dataset
```

The current runtime model contract uses generic `cue` and `ball` classes only;
numbered-ball classification is outside this workflow. A merged dataset does
not establish model accuracy or qualify a runtime artifact.
