# Offline image and dataset helpers

These tools are optional, off-device helpers. They do not run in the Rust
application and do not establish detector accuracy.

For verified checkpoint export to ONNX/CoreML, see
[`README_MODEL_EXPORT.md`](README_MODEL_EXPORT.md).

## Extract candidate frames

`dataset_creator.py` extracts still images from a saved video for manual review
or annotation. Install the optional Python image dependencies first:

```bash
uv sync --extra python-tooling
```

For example:

```bash
uv run python tools/dataset_creator.py /path/to/session.mp4 \
  --output /path/to/frames \
  --count 300 \
  --strategy uniform
```

The utility can select frames using uniform, scene-change, random, or hybrid
sampling. Its quality and edge-case values are heuristics, not validated labels;
review the images yourself. It uses OpenCV frame-rate and seeking behavior, so
its timestamps are approximate and must not be used as a temporal accuracy or
latency ground truth. Keep the original video and use the reviewed Rust CVAT
import workflow when source-time annotations are needed.

## Merge curated YOLO datasets

`merge_yolo_datasets.py` combines local datasets that already contain valid
`data.yaml` manifests and image/label splits. It maps class names across input
manifests and refuses to replace existing output unless `--overwrite` is
provided. Focused tests cover split handling, class remapping, malformed input,
and output safety.

```bash
uv run python tools/merge_yolo_datasets.py \
  /path/to/dataset-a /path/to/dataset-b \
  --output-dir /path/to/merged-dataset
```

Frame extraction and dataset merging do not create trusted labels, train a
model, export a runtime artifact, or verify Rust inference. The current runtime
model contract uses generic `cue` and `ball` classes only; numbered-ball
classification is outside this workflow.
