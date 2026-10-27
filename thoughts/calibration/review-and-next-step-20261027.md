# Experiment review and next action — 2026-10-27

## Decision

Fix and verify the native-crop estimator's acceptance stability using saved
images before further camera/FPS tuning, calibration collection, or rollout.
The immediate problem is demonstrated in software: identical pixels and boxes
can change from accepted to rejected when only a detection index changes.
Slower capture cannot fix that behavior.

This is work within ADR-0102's selected pipeline. It requires neither a new
production mode nor another architecture decision. Preserve the verified GPU
decoder discovery independently of the estimator's outcome.

## Facts retained from the thread

| Subject | Evidence and practical conclusion |
|---|---|
| Physical model | Table X/Y only, playing area 2336.8 × 1155.7 mm; every ball has diameter 57.15 mm. No variable physical diameter or ball-height estimation. |
| Camera detail | Native 3840 × 2160 MJPEG is verified. The previous application capture geometry was 1280 × 720, so native stream resolution alone did not improve its measurements. |
| Intel GPU | `vaapijpegdec` decoded 100 4K frames in 5.044 seconds, about 19.8 FPS, at roughly 64% of one CPU core. This is a useful decode result, not a whole-application performance result. The tested `vajpegdec` failed negotiation. |
| Inference | Keep the existing Coral model and its [1,1280,1280,3] float32 input. Native detail belongs in crop refinement. |
| FPS | A short 4K/5 FPS source check reported 5 FPS throughout six samples over about 18 seconds. The source was restored to 20 FPS. Full-pipeline stability and the winning FPS remain unmeasured. Start later performance qualification at 4K/5 FPS, consistent with the user's preference for continuous availability. |
| Lens models | Eight-view checkerboard holdout RMS was 0.2148 px for zero added radial correction, 0.2190 for k1, and 0.2209 for k1+k2. Frozen ninth-view scores were 0.2279/0.2307/0.2299 px. Tested radial corrections also increased bow in several rail features. These results do not justify applying those corrections or prove zero lens distortion. |
| Local normalization | It makes several clean edge-ball outlines closer to circular. It does not repair selected stripe, highlight, shadow, or pocket edges, and has not demonstrated improved physical X/Y accuracy. |
| Physical reference evidence | Nine user-confirmed positions belong only to `table-grid-20261026-114241.jpg`. Older homography leave-one-position-out RMS was 8.71 mm; related local-normalization comparisons were about 8.01–8.95 mm. These discrepancies mix image localization, reference uncertainty, spatial coverage, and mapping error. They do not establish that the user's measurements are wrong. |
| Reference limits | The nine positions occupy three rows. Six pocket-adjacent balls have no physical X/Y references. Other saved arrangements cannot inherit these nine coordinates. Known table marks and rail geometry are useful, but their reference definitions must be verified before treating them as ball-center correspondences. |
| User constraints | Use existing images; no positioning jig or new pocket-ball measurements. Prefer a stable lower rate. Desired performance/accuracy numbers guide the work rather than overriding reliability. Useful verified discoveries can be adopted independently. |

## What the current implementation establishes

The branch contains capture, estimator, frame-bound measurement, tracking,
calibration, replay, configuration, and audit work, with previously passing
automated checks. Those checks establish software behavior on their fixtures;
they do not establish real-image accuracy or continuous precise availability.
Production optics and ball-center calibration assets remain unqualified.

The previously recorded 4/9 run used manually seeded 36-pixel boxes. The later
all-ball run used segmentation-box centers and gradient-major-diameter + 4 px
square proposals, accepting 5/9 measured balls and 0/6 pocket balls. The four
measured rejections were all `AmbiguousBoundary`; center-shift failures belonged
to unmeasured pocket cases. Different proposal definitions and detection
indices make 4/9 versus 5/9 unsuitable as a clean improvement comparison.

The proposal-size sweep also changed the detection index, which changes the
RANSAC seed. Its acceptance changes cannot be attributed to crop geometry.

## Controlled identity experiment completed in this review

For each of the 15 saved ball regions, hold source pixels, box, intrinsics,
thresholds, and work limits fixed. Run the same proposal with detection indices
0 through 15. The only intended change is the index. The estimator seeds
RANSAC with `config.seed ^ detection_index`.

| Measured ball | Accepted indices out of 16 |
|---|---:|
| Top orange | 3 |
| Top red stripe | 16 |
| Left yellow | 0 |
| Interior purple | 11 |
| Center | 7 |
| Interior green | 2 |
| Right purple | 5 |
| Bottom black | 3 |
| Bottom yellow | 12 |

Seven of nine measured balls change acceptance solely with index. The maximum
pairwise separation of accepted centers for an individual ball reaches
0.820 source pixels. All six pocket regions reject for every tested index.
The complete experiment was repeated with identical outputs. These are repeated
software trials on one frame, not independent placements or a measured live
availability rate. Generous 400 ms per-proposal and 5000 ms batch budgets
isolate fitting behavior; the test makes no runtime-performance claim.

Exact inputs, hashes, runtime parameters, and every result are saved in
`offline-refiner-identity-check.json`; the research harness is
`offline-refiner-identity-check.rs`.

## Code findings that explain the next action

1. **Ambiguity is evaluated before robust refinement.** The code compares raw
   three-point RANSAC circles; only the selected circle reaches IRLS. Several
   raw hypotheses can represent jitter around the same underlying contour.
   A cluster of three sampled guesses does not establish two genuinely
   different converged boundaries. This is a likely false-rejection mechanism,
   not yet a proven explanation for every rejected ball.
2. **Diagnostics cross iteration boundaries.** Residual, inlier, and center-spread
   fields can survive from an earlier aiming iteration when a later iteration
   fails. Prior claims about a rejected fit having a low final residual/spread
   are therefore unsupported without per-iteration evidence.
3. **The ambiguity and stability gates use different coordinates.** Raw
   alternative-center distances are measured in virtual-patch pixels, while
   final quadrant sensitivity is measured in source-image pixels against the
   same configured threshold. Local magnification varies across the image.
4. **Calibration has a separate contract mismatch.** The production utility
   requires at least 12 correspondences, while existing measured references
   number nine and currently fewer are accepted. A diagnostic nine-point fit
   remains possible; the current production utility cannot emit that asset.
   Do not duplicate points, invent references, or silently bypass its contract.

## Next implementation task

Work on the perception module first, using one replacement estimator:

1. Make results for identical pixels and identical boxes invariant to detection
   index/order. Keep independent random-start stress checks in offline audits;
   changing the seed alone must not be treated as proof of estimator robustness.
2. Record candidate edges, hypotheses, selected inliers, fitted outline, and
   rejection cause per aiming iteration. Ensure each quality value names the
   iteration and fit it describes. Export overlays for the saved ball scenes.
3. Refine a bounded set of strongest candidate solutions, reselect inliers,
   cluster converged solutions, and assess genuinely distinct supported
   boundaries. Compare center differences in source-image coordinates. Keep
   coverage, foreground/background, residual, and center-stability checks;
   do not obtain acceptance by simply widening thresholds or suppressing
   genuine ambiguity.
4. Re-run the frozen identity audit. Then separately test box-center offsets,
   box-size variation, and candidate-start variation while holding the other
   variables fixed. Review every measured-ball failure and accepted outline.
   Use the other saved grid, checkerboard, and 4K scenes as additional contour
   checks, respecting occlusion and each image's calibration provenance. Reserve
   some saved scenes from tuning and report them separately.
5. Verify detection-order invariance, source-coordinate comparisons, diagnostic
   iteration correctness, known-center synthetic contours, real saved crops,
   deadlines, and worker integration. Report unresolved real-image failures.

Once image localization is stable, re-evaluate the nine matching physical
references with the final estimator and report acceptance and leave-one-out
mapping discrepancies. Resolve the nine-versus-twelve calibration contract
explicitly before activating an asset. These references support a provisional
diagnostic map, not an independently established 2 mm accuracy claim.

Only then resume full-pipeline target qualification, initially at 4K/5 FPS,
measuring fresh precise availability as well as latency and dropped work.
Stills cannot establish temporal reliability, motion response, or sustained
memory behavior. No additional user measurements are needed for the immediate
estimator task.

## Reproduction of the identity check

From the repository root, temporarily register the research source as a Cargo
example, run it, then remove that temporary registration:

```sh
mkdir -p perception/examples
cp thoughts/calibration/offline-refiner-identity-check.rs perception/examples/offline_refiner_identity_check.rs
cargo run --manifest-path perception/Cargo.toml --example offline_refiner_identity_check
rm perception/examples/offline_refiner_identity_check.rs
```

Use the recorded source/configuration hashes when comparing a future run;
changed estimator code is an intentional new experiment. This review changed
research records only. It did not deploy code or access the camera.
