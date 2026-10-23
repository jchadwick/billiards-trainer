# 20 fps camera-sampling comparison

## Result

For this CPU detector and saved-video stream, a 20 fps output cap looks like a
better freshness/coverage balance than 30 fps. Across three 120-second windows,
20 fps delivered 19.616 fresh unique source-frame updates/s, close to the
configured 20 fps, and skipped an average of 0.39 emitted frames/s. At 30 fps,
the detector delivered 22.108 fresh unique source-frame updates/s and skipped
7.86 emitted frames/s. Both arms retained latest-frame replacement, so the
30 fps result also stayed bounded rather than accumulating an old-frame queue.

At 20 fps, mean p95 host read-completion-to-publication age fell from 77.05 ms
to 67.72 ms (-12.1%). P95 receiver-mailbox residence fell from 32.57 ms to
22.89 ms (-29.7%), while model execution stayed near 45 ms. The local client
received states less often, as expected: state-receive interval p95 rose from
46.16 ms to 67.15 ms. The lower rate therefore provides fresher individual
states at a lower update cadence.

Ball-component observed updates were 15.266/s at 30 fps and 13.366/s at
20 fps. The fraction of unique updates whose ball component was marked
`observed` stayed similar (about 69% at 30 fps and 68% at 20 fps). There are no
reviewed temporal labels, so these counts describe model output coverage only;
they do not establish detection recall or accuracy during motion and occlusion.

## Matched workload and measurements

Both arms used the same looped `test_data/demo2_short.mp4` source, model,
1280×720 RGB output, 1280 model input, CPU provider, confidence and IoU
thresholds, audit settings, and 30-second warm-up plus three 120-second windows.
Only `vision.camera.fps` changed from 30 to 20. The input file SHA-256 is
`4e09e30c49fed7e134f43f276c629394c209757e0440958a51ab148f12f6e388`.

| Measure | 30 fps | 20 fps | Change |
| --- | ---: | ---: | ---: |
| Configured capture output rate | 30 fps | 20 fps | -33.3% |
| Fresh unique source-frame updates | 22.108/s | 19.616/s | -11.3% |
| Ball-observed source updates | 15.266/s | 13.366/s | -12.4% |
| Missing emitted source frames | 7.86/s | 0.39/s | -95.0% |
| Read-completion-to-publication age p95 | 77.05 ms | 67.72 ms | -12.1% |
| Receiver mailbox residence p95 | 32.57 ms | 22.89 ms | -29.7% |
| Model execution p95 | 44.96 ms | 45.47 ms | +1.1% |
| Local state-receive interval p95 | 46.16 ms | 67.15 ms | +45.5% |
| Incomplete / invalid-JSON states | 0 / 0 | 0 / 0 | unchanged |
| WebSocket sequence gaps | 0 | 0 | unchanged |

The 20 fps on-device capture helper is not in this comparison: the test uses a
looped network MJPEG stream and a macOS M4 Max. For this HTTP source, GStreamer
uses `videorate` in drop-only mode before RGB conversion and output scaling, so
it discards excess frames without manufacturing duplicate observations. JPEG
decoding still runs for every incoming frame. The source fixture is paced at
the recording's 30 fps; camera exposure and physical sensor arrival time were
not measured.

## Interpretation

Twenty fps is a strong candidate for the next physical-camera qualification on
this CPU workload. It nearly matches delivered fresh updates to the configured
rate, cuts receiver queue waiting, and substantially reduces emitted frames
that the detector cannot keep up with. It also reduces the available ball
observation cadence from 15.27 to 13.37 updates/s, so temporal accuracy labels
must decide whether that tradeoff preserves useful locations through motion.

The p95 state age remains 67.72 ms, above the README's 50 ms reference budget.
Sampling alone cannot reach that budget while p95 model execution is about
45.5 ms and mailbox wait is about 22.9 ms. Faster inference, a supported
accelerator, or a lower-cost model/input setting would be needed to create more
headroom. Any such change needs matched accuracy evidence.

The network fixture could reduce its rate with a GStreamer drop-only stage.
The current physical-camera path requests a camera input profile directly; a
regular webcam may not expose an exact 20 fps mode. Qualify the camera's
reported modes first. If it cannot negotiate 20 fps, add an explicit bounded
latest-frame rate limiter after the camera's supported native mode and before
RGB conversion, preserving the negotiated source cadence separately from the
sample cadence.

Next, test 20 fps and the nearest supported higher rate on the Linux target's
actual webcam, with the same model and resolution. Record source and sampled
rates, mailbox age, latest-frame replacements, CPU, and state age. Then score
reviewed temporal labels for moving, settling, and occluded balls. Do not treat
these development-Mac results as a production default.

## Provenance

- 30 fps reference revision: `8e2c049f969c40c8a68bef3b5d16f9073a4cb867`.
- 20 fps run revision: `c2f43fb5b052efd9fd20b9a1d1c1a92a6bb23ff7`; intervening
  commit changed only audit documentation and saved results. Detector and
  capture-helper executable hashes match the reference run.
- Effective 20 fps config SHA-256:
  `baeb24f555101f3f91e15b411ef3b14fc103370bea143f0aa803081e9c8ba417`.
- Capture-settings SHA-256:
  `061cf6059155fcb759d8046a3538909b9c8478f63f8ad8e9508849b381ba0793`.
- Detector model SHA-256:
  `4ceec94901553ff8a22f71d8cca684ec314a9fec8b7f219c410a5df16be4c4fa`.
- Both worktrees were clean at run start; no audit queue, span, or export losses
  were reported in the 20 fps arm.

## Raw artifacts

- 30 fps reference: [matched audit report](../rust-mailbox-overhead-20261023/comparison.md)
  and [on-arm summary](../rust-mailbox-overhead-20261023/on/20261023-104439-c05107d2/summary.json).
- 20 fps: [runner report](20fps/20261023-105629-2e773c25/report.md),
  [summary](20fps/20261023-105629-2e773c25/summary.json), and
  [effective configuration](live-config-20fps.json).
