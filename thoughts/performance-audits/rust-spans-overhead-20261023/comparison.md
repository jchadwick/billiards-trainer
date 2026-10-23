# Current Rust audit instrumentation A/B

## Result

On the development Mac, enabling the current Rust live timing audit stayed
within the 2% instrumentation overhead budget for this workload. Across three
120-second measured windows, unique observed source-state throughput was
15.339/s with audit off and 15.344/s with audit on (+0.04%). Mean per-run p95
read-completion-to-publication age was 78.46 ms off and 77.41 ms on (-1.34%).
Local state-receive interval p95 averaged 46.31 ms off and 46.16 ms on.

All messages were complete and correctly ordered. The audited runs reported no
producer record drops, invalid spans, span-capacity drops, or receiver export
losses. They retained 5,523 state-correlated frame timing samples across the
three repetitions. The p95 values were stable across each arm, but the dirty
worktree fingerprint differs between arms because each audit wrote new output
files inside the repository. The revision, executable, model, capture settings,
effective configuration, and input source were identical; repeat with output
outside the repository before treating this as a fully matched qualification.

## Workload and provenance

- Host: Apple M4 Max, macOS, arm64, 14 logical CPUs.
- Revision: `8b9ec3c62883e36569e68c5d96a961c74105541f`.
- Input: `test_data/demo2_short.mp4`, SHA-256
  `4e09e30c49fed7e134f43f276c629394c209757e0440958a51ab148f12f6e388`, looped
  through the local one-client MJPEG fixture at 30 fps.
- Runtime/provider: Rust live detector, ONNX Runtime CPU, 1280 model input,
  1280×720 RGB capture output.
- Effective configuration SHA-256:
  `457414171eb3ce792a0230f40f05306d7b489f1f338ac3c00aa49feb66963b0b`.
- Capture settings SHA-256:
  `2c473b75646dd0c607e79f76928d5d035abbf92390976d99d042a093f85724c6`.
- Detector model SHA-256:
  `4ceec94901553ff8a22f71d8cca684ec314a9fec8b7f219c410a5df16be4c4fa`.
- Detector executable SHA-256:
  `2663be366a7a66c18af71b3b0ccee2a72fd16e76c5b68a5f48ecc1a6d84a1b60`.
- Both arms used a 30-second warm-up and three 120-second windows. The only
  intended change was enabling `--live-pipeline-timing-audit` in the second arm.

## Measurements that locate the current delay

The on arm reports p95 model execution of 44.89–45.03 ms and p95 total
read-completion-to-scene-queue-submission of 77.47–77.93 ms. The separately
measured read-completion-to-inference-start interval is 32.96–33.25 ms p95.
That pre-inference interval is nearly as large as the model call. The audit
executable used for this A/B reports it as one same-process interval and cannot
show how much was mailbox residence. A follow-up code change now measures
capacity-one capture-mailbox residence separately from the remaining
freshness/admission/scheduling work before inference. Those newer measurements
are not present in these historical A/B artifacts and need a matched rerun.

The remaining named work is small on this workload: preprocessing is about
1.0 ms p95, decode/NMS about 0.10 ms, tracking/state assembly about 0.04 ms,
public-state lifecycle about 0.08 ms, JSON serialization about 0.03 ms, and
queue submission about 0.01 ms. These p95 values are separate distributions
and must not be added together. The current evidence points first to model
execution and the pre-inference delay, not JSON or socket transport.

The API receiver observed about 22.3 state messages/s but only about 15.3
unique observed source updates/s. A requested 30 fps is not an independently
measured sensor arrival rate; the audit does not yet expose a complete capture
source-arrival counter for this fixture. Source sequence gaps and repeated
state updates are recorded separately in the raw run reports.

## Limits and next experiments

This is a development-machine relational result using a transcoded looped video.
It does not qualify a physical webcam, the Linux target, camera exposure age,
numbered-ball recognition, motion accuracy, trajectory workload isolation, or
browser viewer interference. No temporal ground-truth labels were available.

The next useful experiments are:

1. Repeat the audit off/on comparison with the current mailbox timing build and
   write outputs outside the repository, then compare queue residence against
   the remaining pre-inference interval.
2. Compare 30 fps with 20 fps and a lower detector input size on the same paced
   clip, retaining latest-frame behavior and measuring unique observations,
   freshness, and drops. Treat any model-size choice as provisional until
   reviewed temporal labels confirm accuracy during motion and occlusion.
3. Build a shared-frame fan-out fixture and headless-browser run before making
   claims about video viewer interference.
4. Run the separate 20-minute bounded-memory audit and, later, qualify on the
   Linux target with the physical camera and Coral present.

## Raw artifacts

- Audit off: [report](off/20261023-101002-47d4603f/report.md) and
  [summary](off/20261023-101002-47d4603f/summary.json).
- Audit on: [report](on/20261023-101650-d681c4cc/report.md) and
  [summary](on/20261023-101650-d681c4cc/summary.json).
- Validated effective config: [live-config.json](live-config.json).
- The initial stale-config rejection is retained in
  `off/20261023-100854-966edb85/failure.json` and is not part of either arm.
