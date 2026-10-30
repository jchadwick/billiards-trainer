# ADR-0104: Run target inference on the verified Intel GPU

- Status: accepted
- Date: 2026-10-29
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

ADR-0102 originally assigned inference to Coral because Intel GPU inference was
unqualified and its runtime absent. The user subsequently reopened architectural
decisions to prioritize accurate, continuous localization. Actual target tests on
2026-10-29 change those premises: an isolated OpenVINO2026.4.1 environment exposes
CPU and GPU, and executes the verified ONNX export on `GPU.0`.

The same saved image and input geometry give inference p95 69.68 ms on GPU and
178.11 ms on CPU (five measured calls after two warmups), versus 1551.04 ms on the
existing Coral model (twenty calls). The compiled Coral graph assigns 127
operators to EdgeTPU and 132 to CPU. It also misses one of the fifteen saved balls
at the same confidence threshold; GPU detects all fifteen. These are short
hardware experiments, not full-pipeline latency or independent accuracy evidence.

## Decision

Use native OpenVINO with explicit Intel GPU compilation for the selected target
inference path, retaining the verified 1280-square ONNX input and checkpoint/hash
contracts. Fail clearly if the GPU is unavailable; do not select CPU or Coral
implicitly. Verify the compiled execution device, preserving bounded single-flight
ownership and source/context rejection. Do not apply CPU inference-thread controls
to the GPU plugin.

Retain the separately verified Intel VA-API JPEG decoder, native source crops and
2D ball-center mapping. GPU JPEG decode and GPU inference share a device: qualify
them together on saved input before selecting camera cadence or deploying.
Begin target comparison at 4K/5FPS; prefer measured availability and responsiveness
rather than a nominal frame-rate target. Package the ONNX model and matching
manifest for this selected configuration. The separately installed native
OpenVINO runtime and Intel compute driver are deployment dependencies.

This replaces only the Coral-placement portion of ADR-0102. Its native-pixel
ownership, physical table model and exact-frame contracts remain in effect.
Preserve Coral research and rollback artifacts; do not run a parallel legacy
localization pipeline or automatic inference fallback in the deployed process.

## Consequences

### Positive

- Target inference has a measured order-of-magnitude improvement over its partial
  Coral graph while recovering the missing saved-scene proposal.
- The application uses the existing native OpenVINO session and its verified
  model contract rather than introducing a Python inference service.
- Missing GPU support is visible instead of silently changing runtime behavior.

### Negative

- OpenVINO and the Intel compute driver become required target dependencies.
- JPEG decode and inference contend for GPU resources; isolated timing does not
  establish simultaneous throughput, process age or motion response.
- Different model arithmetic changes proposals; localization must be checked with
  actual GPU output, not only source PyTorch or CPU ONNX proposals.

## Alternatives considered

### Retain Coral and lower camera FPS

Camera cadence does not repair a 1.55-second inference path or a missing proposal.
The large verified GPU improvement justifies revisiting the earlier assignment.

### Select OpenVINO CPU

The same target experiment is slower on CPU. CPU remains a research comparison;
it is not a hidden fallback for the selected GPU deployment.

### Introduce automatic backend switching

This obscures operational failures and multiplies unqualified behavior. Ship one
selected target configuration and preserve a separate recoverable release.

## Verification

- `thoughts/calibration/target-gpu-saved-20261029/` preserves exact script,
  OpenVINO version, ONNX hash, reported GPU execution device and raw outputs.
- The sibling Coral saved-input report records normalized coordinate output,
  runtime timings and actual post-NMS proposals; no new camera data was taken.
- Native runtime tests must cover GPU selection, provider identity, unsupported
  thread settings and clear startup errors; strict Clippy and all-target tests.
- Target qualification must verify native Rust GPU execution and saved actual
  GPU proposal refinement, then simultaneous JPEG decoding and inference with
  bounded memory and frame-age reporting. Short Python timings alone do not
  qualify deployment or millimeter accuracy.

## Follow-up — 2026-10-29: saved 5 FPS replay is stable on a repeated static image

The target Rust qualification package used VA-API JPEG decode and explicit
OpenVINO Intel GPU inference with the existing `[1,1280,1280,3]` model input.
In a 60-second replay, a saved 1280 × 720 image enlarged to 3840 × 2160 was
repeatedly supplied at 5 FPS. The source delivered 304 frames (5.002 FPS), all
304 received motion checks, and 600 public-state samples had fresh positions
for all 15 balls, with no detector, cue, overload, or obsolete-result failures.

This supports continuous availability on that static fixture and configuration.
It is not a live-camera, native-4K, moving-ball, changing-light, or independent
physical-accuracy test. It does not select production FPS, qualify shot coaching,
or authorize deployment. Exact inputs, outputs, and limitations are recorded in
`thoughts/calibration/refiner-r10-validation-20261029/r10-scheduler-and-corner-availability.md`.
