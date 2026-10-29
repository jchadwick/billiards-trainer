# Complete target-GPU localization — revision9, 2026-10-29

Saved inputs only. The target application was paused for isolation and restored
on exit. No source image, camera framing, focus, zoom or deployed configuration
was changed. The GPU model is the hash-verified source ONNX; native OpenVINO
reports `GPU.0`. Source hashes and the exact command/settings are retained.

## Spatial and budget result

All15actual target-GPU proposals pass the revision9refiner, including all9old
physical references. Two pocket observations complete the boundednine-seed
search; each accepted seed must independently pass all pixel gates and all
qualified seed centers must agree within the original0.5sourcepixel tolerance.
No pixel gate was loosened. Production-decoded frozen crops now permanently
regress both failures, alongside known-center, highlight and occlusion cases.

On target, actual GPU proposals through the no-trace production batch:

- Selected600msbatch/200msproposal budgets:15accepted on all12measuredruns;
  refinementp95376.65ms.
- Generous diagnostic budget:15accepted on all12runs; p95380.39ms.

The selectedbounds accommodate measuredwork rather than clipping after10balls.
This does not make600ms a measuredlatency or claim a20msrefinementtarget.

## Native full replay

The measured savedJPEG is repeated at5FPS for4seconds; balljobs are scheduled
at1Hz, with5Hzmotion/cuecadence. All5scheduledcalls publish15precisetablepositions
andoutlines. There are zerooverloadskips, obsoletecompletions or inference deadline
misses. MeanGPUinference70.80ms, preprocessing7.79ms. The CPUrefiner runs in addition.

The video replay instrument processes synchronously and replaces3queuedsource
frames during processing. This is not the production asynchronous capture/worker
loop and is not evidence of zero source-frame loss. Its inference-completion
p95122.24ms excludes later refinement; do not call that totalmeasurementlatency.
It also predecodesvideo and excludes JPEGdecode/HTTPcapture.

The next qualification exercises savedMJPEG through actual IntelJPEGdecoding and
nativeGPUinference simultaneously in the liveworker/application path. Freshness
and expiry must match1Hzmeasurements while motion generation continues to invalidate
obsolete positions. None of these tests independently certify physical2mmerror.
The nine reference uncertainties remain unknown; revision9map trainingRMS4.053mm,
maximum8.067mm, leave-one-outRMS8.791mm. Repeatedframes are not independentplacements.
