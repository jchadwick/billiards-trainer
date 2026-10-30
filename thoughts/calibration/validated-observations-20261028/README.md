# Validated-observation preparation — 2026-10-28

The wrapper reads both saved checkerboard sessions, preserves all raw observations, and exports every source-only neighborhood refinement and local pixel-support diagnostic. All 17 views initially remain review-pending. Neighborhood disagreement is a diagnostic flag, not proof that a corner is inaccurate.

The root agent can review the contact sheets and supply a source-, revision- and configuration-bound record choosing existing visibly supported refinements. No point is inferred from a camera projection. Missing selected image support or invalid grid geometry cannot be overridden. Scoring requires at least eight reviewed points in each alternating parity with distributed coverage. Every excluded corner stays in the observation record.

Contact sheets show all corners from both sessions. Tiny crops are nearest-neighbor enlarged for readable display only; source coordinates, refinement and scores always use native source pixels.

Run the synthetic checks with:

```sh
/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/checkerboard_observations.py --self-test
```

Export both saved sessions with:

```sh
/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/checkerboard_observations.py
```

A complete export with no eligible reviewed views exits 2, preventing accidental scoring continuation. This is expected while the review is pending. Use `--reviews FILE --session SESSION --frozen-models MODELS.json` for explicitly reviewed frozen-model scoring. The model dimensions must match; no implicit scaling or camera fitting occurs. JSON labels intrinsic training reuse from source hashes first and recorded target/session identity otherwise. A same-ID view from another target is never treated as the same source.

Review record structure:

```json
{"reviews":[{"session":"checkerboard-20x14-session","view_id":"view-003","source_sha256":"COPY_FROM_OBSERVATIONS","observation_refiner_revision":1,"configuration_sha256":"COPY_FROM_OBSERVATIONS","reviewer":"SOURCE_PIXEL_REVIEWER","reviewed_at_utc":"REVIEW_TIME","accepted_corners":[{"index":0,"variant_label":"pitch-0.3"}],"notes":"Record visible evidence and excluded corners; do not select from model residuals."}]}
```

The example shows the schema, not an accepted review. The wrapper never supplies its own approval. `verification.json` records checks and reproducibility. These data establish source-pixel support and mathematical consistency only; they do not qualify target flatness, physical dimensions or ball X/Y accuracy. No camera or production configuration was accessed.
