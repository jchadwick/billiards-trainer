# Target replay smoke and short provider comparison

Captured 2026-10-21. This is a deployment and relational A/B smoke on the
available Linux host, not a qualification run for the final target hardware.

## Environment and workload

- Host: Linux 6.8.0-90-generic, x86_64, Intel Core i5-8600T (6 cores), 16 GiB
  RAM.
- Model: `billiards-yolov8n.onnx`, SHA-256
  `7beb3af7923290a5524fe1744de13bd66f4e43ea8a5048347bc8a12c51a1e4b5`;
  checkpoint SHA-256
  `f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`.
- Input: the same 30 recorded 3840x2160 JPEG frames in all runs. Image-set
  digest: `ba36f58852e024527944453a2756f05a62d3e8e24c891a75e0474d26399862d`.
- Effective detector settings: 1280x1280 model input, confidence 0.4, IoU 0.45,
  three warm-up inferences.
- Rust release binary SHA-256:
  `e61db70d7181cfcc8858e25bd9841f858e92067e4f2308bc4fdd95e739fae248`.
- The runtime selector was `--provider cpu` or `--provider openvino`; both
  selections ran in the same already-built binary. CPU reported ONNX Runtime's
  `CPUExecutionProvider`; OpenVINO reported the native `openvino-rs` CPU
  backend. No build-time feature switch was used.

## Short unpaced A/B

Each provider ran once over the same 30 images. CPU ran first, then OpenVINO.
Treat the results as a relational lead: they are not repeated enough to
separate provider effects from host and run-order variation.

| Measure | ONNX Runtime CPU | Native OpenVINO CPU |
|---|---:|---:|
| Median inference | 234.58 ms | 176.36 ms |
| p95 inference | 247.60 ms | 179.64 ms |
| Mean image decode | 24.61 ms | 47.18 ms |
| Mean preprocessing | 50.63 ms | 65.63 ms |
| Median end-to-end per image | 309.79 ms | 289.68 ms |
| Total time for 30 images | 9.21 s | 8.86 s |
| Detections | 416 | 416 |

On these images, the largest absolute box-coordinate difference between the
providers was 0.0007 source pixels, and the largest confidence difference was
0.00000096. This establishes numerical parity for this corpus only; it does
not establish precision, recall, or position accuracy against labels.

Inference alone improved by about 25% at the median, but the full image path
improved by about 6%. The substantial measured decode and preprocessing times
make these the next stages to investigate. Their difference between sequential
runs also shows why the provider comparison needs repeated, interleaved runs
before attributing the full-path difference to the provider.

Raw reports: [CPU](target-cpu-ab.json) and
[OpenVINO](target-openvino-ab.json).

## Paced delivery smoke

The API replay used the same 30-frame corpus, repeated ten times at a scheduled
5 frames/second (300 scheduled frame slots, 60 seconds). The process completed
with exit code 0. The target health snapshot reported 221 accepted snapshots
and 221 state publications, with zero downstream ingress queue replacements.
That is 3.68 published observations/second over the complete replay, below the
requested 5/second cadence; at least 79 scheduled slots were not represented
by a published state.

The separate five-second client window received 19 complete state messages
(3.80 messages/second), containing 16 unique source observations (3.20/second)
and three cached updates. It recorded six source-sequence gaps and no stream
sequence gaps, duplicates, or out-of-order messages. This short window is
consistent with a freshness/throughput deficit; it does not identify which
stage caused it.

The collector ran on a Mac, while the service ran on the Linux host. The two
monotonic clocks are not comparable, so observation-to-receipt latency remains
unavailable. The Rust-provider API also returned no server performance-audit
status: server-side stage events and resource samples are not wired into this
runtime. There are no temporal labels for freshness, identity continuity, cue
quality, or moving/occluded-ball accuracy.

## Ranked follow-up work

1. **Add opt-in server-side Rust audit events and expose their run status.**
   Record capture/replay schedule, decode, preprocess, inference, NMS, tracking,
   scene assembly/serialization, outbound queue and receipt boundaries, plus
   dropped-frame reasons and bounded CPU/memory samples. This is required to
   locate the 5-fps shortfall rather than infer a cause from totals.
2. **Profile and reduce input decode/preprocessing cost.** The two stages
   together average about 113 ms in the OpenVINO run, compared with about
   177 ms inference. First compare decoder and resize/letterbox alternatives
   using identical tensors/detections and the accuracy gate; do not trade
   localization accuracy for speed without labels.
3. **Repeat the provider A/B with interleaved order and multiple runs.** The
   current 30-image sample suggests native OpenVINO materially accelerates
   inference, with a smaller whole-pipeline gain. Preserve identical model,
   input, warm-up, thread settings, and output semantics.
4. **Add temporal ground truth before recommending detector cadence or
   resolution changes.** Needed annotations include per-frame ball positions
   and identities, cue quality, occlusion, motion, and event timestamps.

No production optimization is recommended from these smoke results alone.
