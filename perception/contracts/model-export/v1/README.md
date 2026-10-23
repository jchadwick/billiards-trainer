# Model export manifest v1

The maintained Python exporter writes this lineage manifest beside each ONNX
artifact. Rust requires the supported schema version, the exact ONNX filename
and SHA-256, the source-checkpoint SHA-256, and a contiguous ordered class map.
Unknown JSON fields remain informational exporter metadata; changing the
required lineage or class-map semantics requires a new version.

`manifest-contract.json` and its two adjacent fixture files are deliberately
small fake artifacts. They are not valid detector weights and must not be used
for inference. The Python exporter test compares its deterministic fake export
against this producer contract, and the Rust loader test verifies the same
manifest and hashes. These tests check cross-language artifact compatibility,
not export correctness, preprocessing parity, model accuracy, or performance.

Run the producer contract test with:

```bash
.venv/bin/python -m pytest backend/tests/unit/test_model_export_tool.py -q --no-cov
```

Run the Rust consumer contract test with:

```bash
cargo test --locked --manifest-path perception/Cargo.toml --lib manifest_v1_contract_fixture_is_accepted_and_unknown_versions_are_rejected
```
