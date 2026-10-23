# ADR-0092: Use ONNX as the only maintained model export

- Status: accepted
- Date: 2026-10-23
- Deciders: Project maintainers
- Supersedes: ADR-0085's CoreML-export retention disposition only
- Superseded by: None

## Context

The supported Rust inference runtime verifies and loads ONNX artifacts on both
macOS and Linux. Explicit `openvino` selection uses the native OpenVINO Runtime;
`cpu` and `auto` use ONNX Runtime. The model exporter still offered a CoreML
package branch, but no Rust runtime consumes that format. Its only current
consumer was the offline Python compatibility detector, which is not a
deployed application service. Keeping that branch also retained a macOS-only
dependency in the model-export installation group and suggested CoreML package
generation was part of the supported model workflow.

The existing CoreML compatibility implementation, model artifacts, and
performance reports are historical/reference material. This decision concerns
the maintained export tool and its optional authoring dependencies; it does
not rewrite past results or change the Rust inference contract.

## Decision

The maintained model-export tool emits only a fixed-shape, lineage-bound ONNX
artifact. Remove the CoreML format option and the `coremltools` dependency from
the model-export extra. Keep ONNX as the shared model artifact for macOS and
Linux Rust inference, including the direct OpenVINO backend. Retain Python
CoreML detector code only as offline compatibility reference until that
reference is removed; do not describe it as an active model-export workflow.

## Consequences

### Positive

- One portable artifact feeds the supported Rust CPU, auto, and OpenVINO paths.
- Export dependencies and tests cover the artifact the application actually
  loads, rather than an unrelated legacy format branch.

### Negative

- Developers can no longer regenerate a CoreML package with this tool. A future
  production decision to add a CoreML-native Rust backend must bring measured
  macOS benefit, model lineage, and accuracy evidence before introducing a new
  export path.

## Alternatives considered

### Keep CoreML as an optional export format

Rejected because the Rust runtime does not load CoreML packages, the Python
detector is not deployed, and no current user workflow requires producing new
CoreML artifacts. Existing packages and run reports remain available as
historical evidence.

## Verification

- `backend/tests/unit/test_model_export_tool.py` verifies explicit checkpoint
  lineage, ONNX output, bounded replacement, and that the removed CoreML CLI
  option is rejected.
- `uv lock --check` verifies the model-export dependency lock after removing
  `coremltools` from that extra.
- A Rust model-load/replay smoke against the generated ONNX artifact remains
  required when an export is changed; a successful export alone does not
  qualify detector accuracy or live performance.
