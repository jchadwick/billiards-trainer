# ADR-0043: Runtime-selectable OpenVINO for Rust ONNX inference

- Status: superseded
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: [ADR-0045](0045-use-the-native-openvino-runtime-backend-for-openvino-selection.md)
- Implementation stage: historical ONNX Runtime provider implementation was
  exercised on the Linux target; its A/B results do not qualify the native
  OpenVINO backend selected by ADR-0045.

## Context

The Linux target is an Intel Core i5-8600T. Its first recorded-image replay
used ONNX Runtime's CPU provider and processed 2.64 frames/s, with 320.7 ms
mean model inference and 443.8 ms p95 synthetic-source-arrival-to-inference
completion. The user is installing OpenVINO on the target and asked for the
deployed Rust binary to support it. The existing Rust inference command requires
an explicit provider, while the Python replay configuration currently accepts
only `cpu` and `auto`.

This is an optional inference backend choice. It must preserve CPU-only builds,
keep the normal configured provider at `cpu`, and permit direct A/B runs against
the same ONNX model and recorded frames. Merely registering the provider does
not prove that it received model nodes or improved latency. ONNX Runtime's
OpenVINO provider is shipped as a provider library and requires a compatible
OpenVINO runtime installation and environment on the host; target qualification
must record provider-node assignment and any CPU fallback.

## Decision

Compile the ONNX Runtime OpenVINO bindings in every normal Rust build. Provider
selection remains a runtime choice: the inference binary accepts
`--provider openvino`, and the API's explicit Rust replay configuration accepts
`perception.rust_replay.provider = "openvino"` and passes that exact value to
the child. The default remains `cpu`, and there is no silent fallback when the
requested provider cannot be loaded. A platform must provide an ONNX Runtime
OpenVINO provider library compatible with its ONNX Runtime and OpenVINO runtime.

The initial provider configuration targets OpenVINO's CPU device. It does not
change the ONNX model, preprocessing, thresholds, scene contract, transport,
or calibration behavior. CPU remains an explicit comparison provider. `auto`
remains separate and is not evidence that OpenVINO ran. The audit report must
retain the requested provider and observed per-node provider counts; results
with no OpenVINO node assignment are not reported as accelerated.

The build recipe and target runtime requirements are documented in the
perception README. OpenVINO deployment requires sourcing the vendor runtime
environment and checking the ONNX Runtime/OpenVINO version compatibility, as
described in the [ONNX Runtime OpenVINO provider documentation](https://onnxruntime.ai/docs/reference/execution-providers/OpenVINO-ExecutionProvider.html).
OpenVINO itself supports Apple silicon, but ONNX Runtime's documented provider
build instructions currently cover Windows and Linux; macOS inference therefore
requires a compatible provider `.dylib` and remains unqualified in this app.
When the Python supervisor launches the child, it forwards the platform's
dynamic-library search path (`LD_LIBRARY_PATH` on Linux or
`DYLD_LIBRARY_PATH` on macOS) only for an explicitly selected OpenVINO
provider. CPU and auto launches keep the existing minimal child environment.
This allows the dynamic provider to load runtime libraries initialized by the
deployment environment without passing unrelated application or credential
variables.

## Consequences

### Positive

- The same Rust executable can run the portable CPU baseline or explicitly
  request OpenVINO, so CPU/OpenVINO comparisons keep preprocessing and state
  semantics fixed.
- Every normal platform build exposes the OpenVINO runtime choice while the
  configured behavior remains CPU by default.
- Provider assignment evidence can distinguish a real OpenVINO run from a
  session that performed all work on CPU.

### Negative

- Compiling the OpenVINO binding into every inference binary adds a build
  dependency, while the platform provider library and OpenVINO runtime add
  deployment requirements.
- Runtime startup depends on compatible vendor libraries and environment
  variables on the target; an incompatible install must fail clearly.
- The API deployment must provide a compatible OpenVINO runtime library path to
  the backend process. This is currently qualified in a target-host Docker
  runtime, not through a bare-metal `setupvars.sh` installation.
- OpenVINO may execute only supported graph partitions while the CPU executes
  remaining nodes, so provider selection alone does not imply a speedup.
- Model accuracy, precision changes, device selection beyond CPU, and target
  performance remain unqualified until measured with labels and the installed
  runtime.

## Alternatives considered

### Keep Rust inference CPU-only

Rejected because the target owner is installing OpenVINO and the measured CPU
inference time dominates the current replay. It would prevent a controlled
comparison using the existing Rust inference path.

### Use ONNX Runtime's automatic device policy

Rejected as the OpenVINO experiment selector because auto policy does not
provide an explicit, reproducible request that the report can compare against
CPU. The existing `auto` setting remains available as a separate policy.

### Use the Python OpenVINO package instead

Rejected for this experiment because the model replay and intended live
inference owner are moving to the Rust process. A separate Python inference path
would make the baseline incomparable and preserve an additional production
runtime.

## Verification

- The ordinary macOS and Linux Rust builds must compile and advertise
  `--provider openvino`; provider execution still requires a compatible runtime
  library on the host.
- Rust unit tests verify the selected provider name, and Python tests verify
  configuration acceptance and exact pass-through to the supervised process.
- The backend supervisor test verifies the platform OpenVINO library path is
  forwarded only for the OpenVINO selection and unrelated environment values
  remain isolated.
- The target-host OpenVINO 2026.4.1 container loaded the provider and assigned
  the model node to OpenVINO. A 30-frame paired CPU/OpenVINO replay retained
  identical detection counts and effectively identical centers; the short
  result is recorded in the [target A/B report](../../../thoughts/performance-audits/rust-live-trained-model-export-20261021/target-linux-replay-20261021/openvino-runtime-ab/report.md).
- The supervised API selected OpenVINO, loaded its runtime libraries through
  the provider-specific child environment, and published scenes to an
  unauthenticated WebSocket subscriber through the retained frontend. This
  short integration smoke is recorded in the [application report](../../../thoughts/performance-audits/rust-live-trained-model-export-20261021/target-linux-replay-20261021/openvino-runtime-ab/app-integration/report.md).
- The Apple Silicon Rust build compiles with the OpenVINO binding and exposes
  the runtime selection. Actual macOS provider execution is not yet qualified.
- Missing or incompatible libraries must fail without switching to CPU. The
  host-level install remains to be verified; full-replay client latency and
  accuracy remain unmeasured.
- Compare CPU and OpenVINO on the same recorded inputs, with matched warm-up,
  thresholds, frame policy, and accuracy labels. Report throughput, p50/p95/p99
  latency, source freshness, skipped frames, center error, precision, and recall.
  Do not recommend production selection until accuracy is equivalent for the
  required workload.

## Revisit when

- The target's host-level OpenVINO installation differs from the 2026.4.1
  runtime used in the recorded container experiment.
- A GPU or NPU device is selected; the current implementation explicitly
  targets the OpenVINO CPU device.
- API/frontend end-to-end measurements show a material deployment or
  subprocess-environment problem.
