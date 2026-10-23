# Public state compatibility fixtures

`legacy-policy-v1.json` is the versioned canonical-pixel compatibility policy,
matching `backend/core/constants_4k.py`. The Rust adapter loads and validates it
once, and callers may supply explicitly validated configuration instead.
Changing canonical dimensions requires coordinated consumer/preview changes;
these settings are not camera resolution and do not change calibration units.

The other JSON files pair a complete neutral scene with the **actual Python
adapter output**, captured by `generate_fixtures.py`. Regenerate from the root:

```sh
.venv/bin/python perception/contracts/public-state/generate_fixtures.py
cargo test --manifest-path perception/Cargo.toml --test public_state
```

The development oracle imports `SceneSnapshotStateAdapter` directly, uses no
wall-clock time, and retains every returned field. Rust has no runtime Python
dependency. These fixtures cover positive/negative cue evidence, unknown and
numbered identity, measured/cached/estimated positions, missing image geometry,
calibrated positions, canonical scaling, empty scenes, clock comparability,
and non-observation/non-success suppression. Invalid scenes are exercised in
Rust tests because the native boundary validates the complete typed contract
before attempting conversion.

The Python compatibility suite compares all 35 checked-in expected payloads
against the current adapter, so fixture drift is detected without manually
running the generator. Regenerate only when intentionally changing this public
state contract, then review the JSON changes alongside the Rust contract tests.

`lifecycle-policy-v1.json` supplies the validated defaults for bounded cached
state: 100 ms before positions become stale, 1 second before cached positions
are cleared, and at most 64 balls. Callers can provide explicit validated
settings. `SceneStateLifecycle` requires an explicit producer identity per run,
rejects duplicate/out-of-order observations, accepts forward gaps caused by
latest-only delivery, and uses receiver-local monotonic receipt time. Health
records and annotations do not refresh localization. A new producer epoch
clears cached state and sequence watermarks; the same active identity cannot be
reactivated to reset ordering. Stale state retains positions with detection
confidence removed; expired state contains no old ball coordinates.

The conversion itself remains pure and does not convert cue geometry into a
public cue. As in the Python adapter, cue detection status is retained while
the legacy public cue is absent. Millimeter points bind to the snapshot
calibration revision; scene v1 contains no independent per-point calibration
revision to verify.

All degraded outcomes yield no fresh payload, including association-capacity
pressure. If the application intentionally accepts that condition, its caller
must enforce the exact known reason and normalize a validated copy through an
explicit policy; it must not treat arbitrary degraded/failed scenes as fresh.
