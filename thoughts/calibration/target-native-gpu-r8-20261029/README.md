# Native Rust GPU integration — 2026-10-29

No camera capture. The target application was stopped for isolated processing and
restored via a shell trap. Inputs are the saved measured JPEG, its repeated-frame
5FPS clip, and the existing actual detector proposal corpus. The retained
`qualify-native-r8.sh` is the exact target command; `source-hashes.json` identifies
frozen Rust source files built in an amd64 Linux container.

The C API runtime is OpenVINO2026.4.1 from the isolated qualification environment.
The native Rust runtime verifies and reports actual execution on `GPU.0`.
Inference averages69.74ms and provider-probe inference72.15ms. There is no hidden
CPU inference fallback. The source ONNX and checkpoint hashes are in the report.

## This run exposes a budget defect; it is not release qualification

| Target isolated refiner workload | Default150ms batch | Generous diagnostic budget |
| --- | --- | --- |
| Measured15balls | 10accepted everyrun, p95150.02ms | 15accepted everyrun, p95207.05ms |
| Alternate15proposals | 10accepted everyrun, p95150.02ms | 13accepted everyrun, p95197.86ms |
| Native4K3balls | 3accepted everyrun, p9543.75ms | 3accepted everyrun, p9543.96ms |

The full Rust replay uses default budgets and5Hz recovering detection. It yields
ten precise coordinates on every processed call because the batch deadline cuts
work off. It processes17of20sourceframes, replacing3queuedframes, and records7
inference deadline misses. It therefore fails the user's reliability preference.
Inference-completion p95284ms excludes subsequent refinement; it is not total
measurement age. The predecoded source excludes camera/HTTP/JPEG costs.

The evidence supports selecting a larger bounded target refinement budget and
lower ball-job cadence, while retaining a5FPSmotionfeed. It does not justify
widening geometric gates, removing freshness checks or claiming flawless5Hz
ball measurements. The actual GPU proposal geometry also requires separate
refinement validation; the source PyTorch proposal counts cannot stand in for it.

Builder requirements: x86-64glibc symbols through2.39 (matching targetUbuntu2.39),
GStreamer development1.26.2 (targetruntime1.24.2), Rust1.99.1. The provided Dockerfile
extends the preexisting verified Rust/GStreamer image with node/npm/jq for the
standard packaging script. Native binary startup and actualGPUinference succeeded
on target; model binaries remain ignored artifacts rather than source commits.
