# Preserved native-frame experiment findings

Recorded 2026-10-27. Preserve these findings independently of whether the
ball-localization continuation qualifies. Read the scope column; an observed
hardware capability is not full-application qualification.

## Source and implementation provenance

- Original tracked evidence baseline:
  `e9fb9a0f8402c1d5dd69657b0273d6fbc46b5e20`.
- Reproducible unqualified implementation checkpoint, experiment branch only:
  `e595fe255291dd9e820295743fface33f8eae538`.
- Refiner source SHA-256 at the frozen identity audit:
  `a8a1dc215afa88bacfa31da88434707c5b8b39823adab14e0143ca682f063ef0`.
- The experimental checkpoint contains the original experimental capture
  helper. Main's separately qualified capture change includes additional
  stream-error regression tests and startup cleanup; do not merge the whole
  experimental checkpoint to obtain that helper.
- Actual promotion commits and verification are recorded in
  `proven-findings-promotion-record.json`. The detailed continuation is
  `native-ball-localization-next-steps-plan.md`.

To reproduce the old identity result, use the experiment checkpoint and the
archived harness/data from the preservation commits. The main branch deliberately
does not include the unfinished `ball_localization` module imported by that
harness. Main's copy of the harness is a research record, not a runnable main
production feature. A future corrected estimator must write a new result file.

## Evidence index

| Evidence | Retained artifacts | Supported conclusion / limitation |
|---|---|---|
| Native 4K source | `stream-4k-change.md`, `camera-source-state.json`, `stream-4k-verification/after.jpg` | Driver/source observations and a native JPEG establish capability and short observed cadence; no complete-application performance claim. |
| Intel GPU decode | `gpu-saved-frame-qualification-20261027.json` | New raw command/plugin/caps/exit evidence: saved native JPEG decodes through `vaapijpegdec` to YUY2 and then complete 3840 × 2160 RGB. No live camera input or target configuration change. |
| Earlier decoder throughput | `native-resolution-ball-localization-status.md`, `review-and-next-step-20261027.md` | Preserve the recorded 100 frames in 5.044 seconds/about 64%-core observation. Raw original probe output is unavailable; do not invent it or treat it as full-application throughput. |
| 5 FPS source behavior | `stream-5fps-check.md` | Six samples over about 18 seconds; source-only, restored configuration; no temporal precision-availability measurement. |
| Framing comparison | `stream-4k-framing.{md,json,png}`, retained before/after JPEGs | Near-direct 3× coordinate correspondence; identical intrinsics and millimeter accuracy are unproved. |
| Checkerboard experiment | `checkerboard-session/manifest.json`, nine raw views/corner files, `checkerboard-eight-view-baseline.json`, `checkerboard-frozen-validation-view-009.{md,json}` | Tested radial terms do not improve held-out optical scores; no proof of zero distortion or ball-center X/Y accuracy. |
| Rail diagnostic | `cushion-straightness.{md,json,png}`, `analyze_cushion_straightness.py` | Particular frozen radial candidates increase bow on several extracted features. Weak/ambiguous edges and physical-reference ambiguity are retained. |
| Local normalization | `ball-local-view-normalization.{md,json,py,png}`, native crop montage | Clean off-axis contours often become more circular; edge-selection and physical-center accuracy remain unresolved. |
| Nine physical references | `table-grid-20261026-114241.jpg`, matching measurements JSON, `table-grid-feasibility-*`, `ball-outline-consistency.*` | Diagnostic mapping from nine user-confirmed positions on this image only, with unknown reference uncertainty. Six pocket positions lack physical truth. |
| Demonstrated identity defect | `offline-refiner-identity-check.{json,rs}` | Fixed pixels/boxes/config, indices 0–15: 7/9 measured cases change acceptance. All three repeated outputs match. This is software behavior, not a live availability rate. |
| Table-mark grid | `mark-grid-calibration-audit-20261027/`, `table-feature-registration-audit-20261027/`, and their offline audit scripts | Ten long-rail sights match the expected one-eighth-length lattice at 0.57 px fit RMS and 0.99 px leave-one-out RMS. Rail-to-cloth transfer remains provisional: direct projection misses visible cloth dots by 10.57–11.00 px, and two-marker cross-validation is 2.30–2.48 px. No production metric map is established. |
| Larger checkerboard capture | `checkerboard-20x14in-capture.md`, `checkerboard-20x14-session/`, both checkerboard PDFs, `generate_letter_checkerboard_tiles.py`, updated `capture_checkerboard.py` | A four-page US Letter version joins to a 20 × 14-inch active pattern with 2-inch squares; the 22 × 16-inch single-page file remains for large-format printing. Four user-provided 3840 × 2160 snapshots each produced all 54 corners (71.64 px center, 75.19 px upper-left, 80.02 px tilted right, 77.17 px tilted lower-left median spacing) after bright-paper ROI detection. The last two views show perspective foreshortening and cover different regions/orientations. Print scale and flat rigid backing remain unverified; these captures establish no calibrated lens model or X/Y accuracy. |
| Corrected interpretations | `offline-refiner-nine-position-check.md`, `review-and-next-step-20261027.md` | Different proposal definitions confound 4/9 versus 5/9 comparisons. A low reported metric can belong to an earlier iteration. These corrections supersede stronger earlier interpretations. |

## Not established

- Independent 2 mm p95 or 5 mm maximum ball-center accuracy.
- Full-table native 4K precision, pocket/corner physical accuracy, or a valid
  production ball-center asset for the new estimator.
- A qualified table X/Y map from the candidate rail lattice; the cloth-plane
  bridge and optics still need validation.
- Sustained precise availability, selected operating FPS, motion response,
  complete capture-to-publication latency, or long-run memory behavior.
- That the user's manual measurements were incorrect.
- That synthetic exact recovery, passing code tests, lower training residual,
  or rounder-looking balls establish any of the above.

The production calibrator requires 12 correspondences and existing measured
references number nine. Keep diagnostic mapping distinct from activation until
the contract is explicitly resolved. No jig or new pocket measurements are
needed for the next saved-image estimator task. The known index-sensitivity fix
can proceed without waiting for larger-checkerboard captures; those captures
are only needed for the subsequent optics qualification if the existing
intrinsics prove insufficient.

## Operational and repository boundaries

The evidence and independent GPU capture commit can remain on main while
experimental localization stays on its checkpoint branch. No localization
rollout or production camera/FPS selection is implied. GPU decoder qualification
used saved-frame data. The checkerboard session includes one snapshot fetched
from the user-provided camera endpoint; no camera settings were changed. Keep
machine credentials, stream keys, `.env` backups, transient capture locks, and
generated model/browser artifacts out of preservation commits.
