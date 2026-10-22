# Rust inference prototype: recorded-image results

**Run ID:** `rust-inference-prototype-20261020`

**Scope:** recorded still-image inference, backend/provider inspection and label comparison. This run does not qualify a camera-to-client pipeline or production deployment.

## Result

The existing two-class checkpoint loads and runs through the Rust ONNX Runtime
prototype. On this Mac, ONNX Runtime's explicit `auto` policy partitioned the
graph between CoreML and CPU and was faster than the CPU-only profile after
initialization. The measured Rust model and fixed-square preprocessing stage
fit within a 33 ms frame interval in the short recorded-image run, but the
complete live path is still unmeasured and the current detector does not meet a
strict 2 px localization target on the available labels. Neither provider is
qualified for production.

The compared PT and ONNX paths used the same 1280 x 1280, fixed-square,
`rect=false` harness configuration, confidence 0.4, IoU 0.45, and a 300
detection cap. The current live Python detector does not specify `imgsz`, so
this is not yet parity with its effective default preprocessing.

## Measurements

Each provider was measured in three unpaced repetitions over the same ten 4K
still images, giving 30 per-image samples per provider. Timings exclude model
initialization and the separate profiling session. `preprocess + inference +
decode/NMS` excludes disk image decode, which is not part of the camera pixel
path. End-to-end file timings include the image file decoder and should not be
read as camera latency.

| Requested policy | Observed node providers | Session initialization | Preprocess | Inference | Decode/NMS | Pixel-to-detection proxy, p50 / p95 / p99 | File-to-detection mean |
|---|---|---:|---:|---:|---:|---:|---:|
| CPU | CPU (214 profiled nodes) | 11.4 ms mean | 9.60 ms mean | 42.42 ms mean; 42.69 ms p50; 46.19 ms p95 | 0.089 ms mean | 52.28 / 55.86 / 57.62 ms | 92.29 ms mean |
| Auto max-efficiency | CoreML (4 nodes) + CPU (5 nodes) | 973.5 ms mean | 9.84 ms mean | 14.85 ms mean; 14.51 ms p50; 16.69 ms p95 | 0.089 ms mean | 24.33 / 27.48 / 31.55 ms | 64.79 ms mean |

The auto policy improved the measured inference stage by about 2.9x relative
to CPU on this Mac and model. Here, the reported inference interval includes
`session.run` and copying the output tensor into Rust-owned memory; it is not a
provider-kernel-only measurement. It is a mixed-provider result, not evidence
for a CoreML-only configuration. Its roughly one-second session startup is a
separate readiness cost. These p95/p99 estimates use only 30 samples over ten
repeated stills and are weak tail estimates. The p99 proxy leaves little
headroom for capture, tracking, calibration, state publication, transport and
client receipt at 30 Hz.

## Compatibility and accuracy evidence

- Python PT and Python ONNX produced 73 detections across ten stills. All 73
  paired within 0.00025 px maximum center deviation; maximum confidence
  difference was 3.58e-7.
- Rust CPU output paired all 73 PT detections within 2 px with no unmatched
  outputs; maximum center difference was 0.033 px. The mixed auto output also
  paired all 73 within 2 px; maximum center difference was 0.453 px.
- Rust preprocessing tensors exactly matched the Ultralytics/OpenCV reference
  on all nine PNGs (4,915,200 float values per image). `grid.jpg` differed at
  132,839 values, with 0.000140 mean and 0.011765 maximum absolute difference,
  consistent with a JPEG decoder-path difference. Rust still produced matching
  detections on this fixture.
- A follow-up aligns decoder semantics with the installed Ultralytics NMS path:
  confidence must be strictly above the threshold, and suppression is applied
  to model-space boxes before source-image clipping. Adversarial tests cover
  both edges. Re-running all ten stills preserved all 73 detections and every
  CPU center exactly; the follow-up JSON and provider trace are retained beside
  the original run files, with source hash and verification details in
  [`decoder-followup-verification.json`](evidence/decoder-followup-verification.json).
- Existing annotations cover eight images and 52 ball centers. The reference
  model produced 55 ball detections. One-to-one center association yielded:
  18 matches within 2 px (32.7% precision, 34.6% recall), 42 within 5 px
  (76.4%, 80.8%), 49 within 10 px (89.1%, 94.2%), and all 52 within 20 px
  (94.5%, 100%). The mean assigned center error was 3.81 px and maximum was
  14.99 px. These are image-plane still labels, not physical table-space or
  temporal ground truth. Unmatched predictions are not assumed to be true
  false positives because empty/missing labels may be incomplete.
- The model class map contains only `cue` and `ball`; it cannot recognize
  numbered balls.

The still-label result is not sufficient evidence for the intended precise
location requirement. Do not rank inference throughput as a production
improvement until the location-accuracy target and a representative labeled
corpus are agreed and passed.

## Ranked next work

1. **Resolve accuracy and label adequacy first (blocking).** Define image-space
   and calibrated millimeter tolerances, add verified annotations across table
   positions, dense/occluded/moving balls and camera conditions, and have a
   second pass adjudicate apparent misses. Report per-ball and per-frame
   precision, recall, error distributions and coverage. The current 2 px score
   is too low to recommend either provider.
2. **Match the actual Python detector preprocessing (blocking for migration
   parity).** Record the effective Ultralytics size, rectangular batching,
   confidence, NMS and output semantics on the retained detector. Compare
   PT-versus-export tensors and detections under that exact configuration,
   including resolution changes. Preserve the current fixed-1280 result as a
   separate harness baseline.
3. **Measure the complete fresh-state path.** Add paced replay and a headless
   receiver that checks source identity, observation age, ordering, complete
   scene content and gaps. Measure capture/read through inference, geometry,
   state update, serialization and receipt; separately compare state-only,
   trajectory-enabled and viewer-enabled runs with the same input. Report
   segmented durations until Rust/Python monotonic clocks are verified
   comparable.
4. **Investigate the 9.6–9.8 ms preprocessing stage.** Profile resize and
   channel/layout conversion before changing them, then test reusable buffers
   and conversion fusion against the same golden tensors. Keep this
   lower-priority than localization quality; the measured CPU inference stage
   dominates CPU mode, while preprocessing is a meaningful share of auto mode.
5. **Qualify provider startup and platform behavior.** Measure cold and warm
   initialization, readiness and restarts; determine whether the mixed CoreML /
   CPU graph is stable and packageable. Test Linux CPU correctness and startup
   before claiming cross-platform support. Treat Linux acceleration as a
   separate hardware-backed qualification.
6. **Close image-decoder parity.** Either match OpenCV JPEG decode behavior or
   characterize whether the observed difference changes ball location on a
   broader JPEG fixture set. Direct raw-RGB comparison on the corpus's one JPEG
   confirms that the first divergence is before resize: 695,791 of 8,294,400
   pixels differ in at least one channel (mean absolute channel difference
   0.06365/255, maximum 3/255). The pixel hashes and exact comparison are in
   [`raw-jpeg-decoder-comparison.json`](evidence/raw-jpeg-decoder-comparison.json).
   This does not show whether positions change. Avoid declaring PNG-only parity
   sufficient for user-supplied camera formats.
7. **Only after the above, test identity and consumer isolation separately.**
   Number recognition needs a separate labeled-data/model decision. Trajectory,
   preview encoding and client fan-out need isolated tests and must not delay
   location publication.

Each follow-up finding should include measured workload, evidence, expected
benefit, accuracy and freshness risks, implementation complexity and a
repeatable validation experiment. These results authorize investigation, not
production optimization or a provider default change.

## Reproducibility and retained evidence

The machine-readable [`result.json`](evidence/result.json) records the run
configuration and aggregate measurements. The `evidence/` directory preserves
the three individual CPU and auto JSON runs, provider traces, Python PT/ONNX
comparison and export manifest. The ONNX graph and preprocessed tensor dumps
are not included; the graph is regenerable from the hashed checkpoint and
manifest, and the tensors are large intermediates. The manifest names the
exporter settings and lineage.

The retained Python equivalence tool was also run against the saved
post-decoder-fix Rust CPU result. It verified the exact checkpoint/class map,
fixed-square settings (1280, confidence 0.4, IoU 0.45, 300 detections), all ten
source-file hashes and complete one-to-one image coverage. It matched 73 of 73
detections with no unmatched outputs; center-error mean/p95/max was
0.00274/0.01587/0.03248 px, and maximum confidence difference was 0.000402.
The [versioned JSON](tool-run/equivalence-20261020T123923.419656Z.json) and
[Markdown summary](tool-run/equivalence-20261020T123923.419656Z.md) preserve
this static output-equivalence result. It is not ground-truth accuracy or
camera-to-client performance evidence.

Input corpus: `backend/vision/test_data`, ten 3840 x 2160 images, directory
SHA-256 `77ba44e1f6b05a2b10d7bb1d3217104de2a7921cad9bdac381b2235a82d86009`.
Checkpoint SHA-256:
`f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`.
ONNX export SHA-256:
`77cbca76dfa71714dfba121193d164492ef9773a0ccc577bc7df16b52c2fb6ca`.

Hardware was an Apple M4 Max with 36 GiB RAM, running macOS 26.6.2. The run
used Rust 1.99.1, `ort` 2.0.0-rc.15 with ONNX Runtime 1.31.x, Python 3.11.14,
PyTorch 2.14.1, Ultralytics 8.4.182 and OpenCV 5.0.0. The captured repository
revision was `e748c0f99e8e8c089633aa282154a05697adb243`; the worktree was dirty.
Its measured-state fingerprint and method are in `evidence/result.json`.
That fingerprint predates this report and the evidence-copy documentation.

The recorded-image runs were unpaced and short: three repetitions of ten still
images, not the standard 30-second warm-up, three 120-second measurements and
20-minute bounded-memory run. They do not measure camera capture, temporal
freshness, calibrated positions, socket delivery, UI rendering, Linux, or
viewer/trajectory interference.
