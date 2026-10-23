# Export a detector checkpoint

The exporter creates a fixed-size ONNX derivative and a sibling manifest from
an explicitly selected YOLO `.pt` checkpoint. ONNX is the supported model
artifact for the Rust runtime across macOS and Linux. The exporter does not
retrain the model, prove output equivalence, or qualify live localization
accuracy.

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
  --imgsz 1280 \
  --format onnx
```

Set `perception.rust_live.onnx_path` and
`perception.rust_live.source_checkpoint_path` to the resulting model and source
checkpoint in `config.json`. The Rust live runtime requires the ONNX input size
to match `perception.rust_live.image_size` (1280 by default), and verifies the
generated sibling manifest before loading either CPU or OpenVINO inference.
`--simplify` is disabled by default and requires the separately installed
`onnxslim` package. Compare real outputs and use the Rust temporal evaluator
before treating an export as qualified; matching checkpoint lineage alone is
insufficient.
