# Export a detector checkpoint

The exporter creates a fixed-size ONNX derivative and a sibling manifest from
an explicitly selected YOLO `.pt` checkpoint. It does not retrain the model,
prove output equivalence, or qualify live localization accuracy.

Install the optional model-export stack only when needed. This installs into
the configured Python virtual environment (`.venv` by default):

```bash
make install-model-tools
```

Export from a reviewed checkpoint:

```bash
.venv/bin/python tools/export_high_res_model.py \
  --model /path/to/reviewed-checkpoint.pt \
  --output-dir models/exported \
  --imgsz 640 \
  --format onnx
```

The Rust live runtime consumes ONNX and its generated manifest. CoreML output is
retained only for the Python compatibility detector. `--simplify` is disabled
by default and requires the separately installed `onnxslim` package. Compare
real outputs and use the Rust temporal evaluator before treating an export as
qualified; matching checkpoint lineage alone is insufficient.
