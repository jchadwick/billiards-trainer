# ADR-0045: Use the native OpenVINO Runtime backend for OpenVINO selection

- Status: accepted
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: [ADR-0043](0043-runtime-selectable-openvino-for-rust-onnx.md)
- Superseded by: None
- Implementation stage: native backend builds on Apple Silicon and Linux,
  passed direct runtime/model smokes on both platforms, produced matching
  detections in short CPU comparisons, and delivered states through the rebuilt
  API container; labeled accuracy, server-stage telemetry, and sustained
  client freshness remain open.

## Context

The application needs one runtime-selectable OpenVINO path that works on both
the Linux target and macOS development systems. The existing Rust inference
engine already consumes a verified ONNX export and owns frame preprocessing,
YOLO decoding, calibration, and scene publication. The earlier implementation
registered ONNX Runtime's OpenVINO execution provider and passed a short Linux
target A/B, but the pinned ONNX Runtime package has no matching prebuilt
OpenVINO provider for Apple Silicon macOS. A selector that cannot execute on
one of the required platforms is not sufficient.

The recorded Linux A/B in the ADR-0043 report remains historical evidence about
ONNX Runtime's provider implementation only. Its latency and output results do
not qualify this direct OpenVINO implementation. The new backend must use the
same ONNX file, RGB-to-NCHW preprocessing, thresholds, detector output contract,
and scene path, while preserving `cpu` and `auto` behavior. The runtime should
be discovered at application launch rather than required to compile the binary.

The target platform scope is Linux x86_64 and Apple Silicon macOS. Current
[OpenVINO system requirements](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html)
list macOS 12.6+ on ARM64 for CPU inference; Intel macOS is not part of this
decision's supported runtime matrix.

## Decision

Use Intel's `openvino` Rust bindings over the OpenVINO C API as the direct
backend selected by `--provider openvino` and the existing Python replay
configuration. Compile the bindings with runtime linking, require OpenVINO
Runtime 2025.1 or newer at launch, and compile the existing ONNX model for the
CPU device. Prefer the binding's standard runtime discovery; when the official
runtime package exposes only version-suffixed libraries, search the explicitly
configured `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH` and load the newest matching
C API library there. Do not search arbitrary directories, register the ONNX
Runtime OpenVINO execution provider, or fall back to another inference backend.

Keep `cpu` and `auto` on ONNX Runtime. Keep the configured production default
at `cpu`; OpenVINO never activates implicitly and a missing/incompatible
runtime fails startup without falling back. Preserve the shared preprocessing,
output decoder, calibration, tracking, and publication contracts. Reuse one
OpenVINO inference request and its owned input tensor for sequential frame
inference. Map a configured inference thread count to OpenVINO's CPU
`INFERENCE_NUM_THREADS` property.

Report the actual binding and native CPU device in runtime provenance. The
OpenVINO C API path does not provide ONNX Runtime per-node provider assignment,
so its node event counts and retained ORT profiles are explicitly unavailable;
the audit must not imply graph-partition or fallback evidence. The safe tensor
API copies prepared values into native-owned input memory, and this cost stays
inside the inference timing.

The supervisor forwards the platform library search path only when OpenVINO is
selected (`LD_LIBRARY_PATH` on Linux, `DYLD_LIBRARY_PATH` on macOS), preserving
the existing minimal child environment. Other OpenVINO install discovery
variables remain available only if they are explicitly added to the supervisor
contract in a later decision.

## Consequences

### Positive

- The runtime selector works through a direct backend on Linux and macOS,
  without depending on an ONNX Runtime provider package for the selected path.
- CPU/auto and OpenVINO share the same exported model, preprocessing,
  postprocessing, and downstream state behavior.
- Runtime loading is deferred until OpenVINO is explicitly selected, so the
  binary can be built on hosts without OpenVINO installed.

### Negative

- The binary carries both ONNX Runtime for `cpu`/`auto` and the OpenVINO
  bindings, increasing dependency surface and build size.
- OpenVINO Runtime must be installed and discoverable at launch; the binding
  rejects versions older than 2025.1 because of C API ABI changes.
- macOS support here means Apple Silicon CPU inference. The explicit selector
  can appear in an Intel Mac binary, but OpenVINO execution there is unsupported.
- The binding's safe input tensor API copies the preprocessed tensor each
  frame. Removing that copy would require a separately reviewed ownership-safe
  API change.
- Direct OpenVINO execution cannot report ONNX Runtime's graph partitioning or
  per-node fallback; selection proves that the compiled model used CPU, not
  accuracy parity or a speedup.
- The old Linux provider A/B does not qualify the new implementation. Native
  Linux runtime, Apple Silicon runtime, labeled accuracy, and production
  performance remain open until run against this backend.

## Alternatives considered

### Keep ONNX Runtime's OpenVINO execution provider

Rejected because the pinned distribution lacks the macOS provider build. A
custom ONNX Runtime build plus packaging a matching provider library on each
platform would reintroduce the library coupling this runtime choice is meant
to avoid and leave Mac dependent on a custom artifact.

### Add a Python OpenVINO inference path

Rejected because it would create a second production preprocessing and
postprocessing path, making backend comparisons and live state ownership less
consistent with the Rust migration.

### Replace ONNX Runtime for every policy

Rejected because the existing CPU and automatic-device paths are established
and remain useful comparison baselines. This decision changes only the
explicit OpenVINO selection.

## Verification

- `cargo test --locked --manifest-path perception/Cargo.toml` validates shared
  inference and runtime selection behavior on the development Mac.
- Linux `cargo test --locked --release --target x86_64-unknown-linux-gnu`
  validates the target build without requiring OpenVINO libraries at build
  time.
- Both platform inference binaries must list `openvino` in `--help`; an
  explicit OpenVINO invocation with missing runtime libraries must fail clearly
  and must not switch to CPU.
- The macOS runtime smoke must run on Apple Silicon with a supported OpenVINO
  Runtime; it does not qualify Intel macOS execution. The bounded 30-frame
  runtime and CPU comparison are recorded in the
  [Apple Silicon runtime report](../../../thoughts/performance-audits/native-openvino-macos-runtime-20261021/report.md).
- A target Linux smoke using the native backend must load the installed
  OpenVINO runtime, run the model, publish a state, and record native backend
  provenance with unavailable per-node counts. Keep the application-level
  replay bounded; the target is for relational validation, not extended tuning.
- Compare native OpenVINO against ONNX Runtime CPU on the same short recorded
  frames, thresholds, and model; verify detection counts and center differences
  before treating speed measurements as meaningful. Short Linux and Apple
  Silicon comparisons now pass the detection-count check. Labeled temporal
  accuracy and broader Apple Silicon validation remain required before
  production recommendations or claiming fully qualified Mac execution.
- The former ONNX Runtime OpenVINO A/B and API smoke remain linked in ADR-0043
  as historical evidence and must not be cited as native-backend qualification.

The bounded direct-backend test is recorded in the [native OpenVINO report](../../../thoughts/performance-audits/rust-live-trained-model-export-20261021/target-linux-replay-20261021/native-openvino-runtime-20261021/report.md).
On the Linux test machine, a one-frame invocation loaded OpenVINO Runtime
2026.4.1 from its test container and emitted native `openvino-rs` provenance.
A sequential 30-frame CPU/native A/B matched all per-frame detection counts and
all 416 same-class detections to below 0.001 source pixels of center delta; the
single-run throughput and latency changes are directional only. A host
invocation without visible OpenVINO libraries failed with a clear load error
and did not fall back to CPU. A later API deployment smoke used the
runtime-selected native OpenVINO binary in the target API container. It
published 221 states from 300 scheduled frame slots and delivered a short
WebSocket sample to the collector. The measured delivery rate was below the
requested 5 Hz, server-stage timings and comparable cross-host latencies were
unavailable, and the frontend renderer was not qualified. See the
[target replay notes](../../../thoughts/performance-audits/target-rust-replay-smoke-20261021/target-smoke-notes.md).

## Revisit when

- A supported ONNX Runtime OpenVINO provider package becomes available on all
  required platforms and offers materially better accuracy or runtime
  observability without sacrificing Mac execution.
- The application adopts an ownership-safe zero-copy OpenVINO input tensor.
- OpenVINO GPU or NPU inference becomes an explicit product requirement.
