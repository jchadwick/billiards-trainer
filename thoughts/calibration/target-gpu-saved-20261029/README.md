# Target Intel GPU inference evidence — 2026-10-29

No camera capture. The application was paused for each inference measurement and
restored afterwards; the old deployment and camera settings were backed up.

A new isolated virtual environment at
`/home/jchadwick/billiards-qualification-20261029/openvino-env` contains
OpenVINO2026.4.1. The same hash-verified source-checkpoint ONNX export and saved
measured1280×720 JPEG are used. Input is RGB `[0,1]`, NCHW1280×1280, with114-valued
letterbox rows (280above/below). The corresponding Coral input is NHWC.

Two warmups followed by five measured synchronous calls:

| Device | p95 inference, including tensor/result copy |
| --- | ---: |
| OpenVINO CPU | 178.11ms |
| OpenVINO Intel UHD630 GPU | 69.68ms |
| Existing compiled Coral model (20 measured calls) | 1551.04ms |

The GPU report explicitly names `GPU.0`. GPU and CPU outputs are finite and shape
`[1,6,33600]`. The GPU model output is in model pixels. The actual Coral output
is normalized0–1; that was verified from the saved tensor and production decoding,
not guessed from model names. Coral's conversion report assigns127operators to
EdgeTPU and132toCPU; it is not a fully accelerated graph. This is evidence for a
new inference placement decision, not proof of end-to-end frame latency.

`qualify_saved_openvino.py` is the exact target experiment; the original ONNX,
input and output hashes/versions are in `openvino-saved-report.json` and the
Coral sibling report. The ONNX export itself remains an ignored model artifact.
`decode_saved_output.rs` applies production Rust class-aware NMS to raw float
output (explicit coordinate scale1forONNX or1280forCoral). Source output arrays
are preserved without modifying their numeric values.

The GPU detects15balls at confidence0.3/IoU0.45; Coral detects14and misses the
interior green second-right middle-row ball. `corpus.json` substitutes those
actual GPU boxes into the frozen measured-scene corpus. Nearest saved labels
identify balls only; they do not establish precise center truth. Every label is
one-to-one and within20sourcepixels of its saved proposal. Refiner evaluation
is separate and must preserve its rejections.

Five repetitions of one saved image are a short hardware smoke benchmark.
They are not a10minute throughput run, independent placements, motion response,
GPU inference plus simultaneous JPEG decode, or proof of physical accuracy.
No live camera/profile/calibration was changed by this measurement.
