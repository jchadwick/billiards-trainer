"""Development-only differential oracle; run from repository root with .venv/bin/python.

No runtime Rust component imports or executes Python. Expected payloads come
from the actual retained adapter, including its omission and age semantics.
"""

import copy
import json
from dataclasses import asdict
from pathlib import Path

from backend.perception.scene_state_adapter import SceneSnapshotStateAdapter

ROOT = Path(__file__).resolve().parents[3]
BASE = json.loads(
    (ROOT / "perception/contracts/v1/scene-rust-tracked-success.json").read_text()
)
OUT = Path(__file__).parent
cases = {}


def add(name, mutate=lambda s: None):
    scene = copy.deepcopy(BASE)
    mutate(scene)
    payload = SceneSnapshotStateAdapter().adapt(
        scene, table={"calibrated": True}, game_type="eight_ball"
    )
    cases[name] = {
        "scene": scene,
        "table": {"calibrated": True},
        "game_type": "eight_ball",
        "expected": asdict(payload) if payload is not None else None,
    }


add("unknown-observed-cue")
add("empty", lambda s: s.update(balls=[]))
add(
    "clock-age",
    lambda s: s["published_at"].update(domain_id="host-clock-a", ticks="1500000000"),
)
add(
    "clock-before-source",
    lambda s: s["published_at"].update(domain_id="host-clock-a", ticks="999999999"),
)
add("clock-not-comparable", lambda s: s["source"]["timestamp"].update(clock="sensor"))
add("exposure", lambda s: s["source"]["timestamp"].update(stage="exposure"))


def half_resolution(s):
    s["source"].update(width_px=1920, height_px=1080)
    m = s["balls"][0]["measurement"]
    m.update(
        image_width_px=1920,
        image_height_px=1080,
        image_center_px={"x": 941.2, "y": 512.35},
    )
    s["cue"].update(image_width_px=1920, image_height_px=1080)


add("half-resolution", half_resolution)


def calibration(s):
    s["revisions"].update(calibration=12, calibration_state="valid")
    s["balls"][0]["measurement"]["table_center_mm"] = {"x": 18.25, "y": -42.5}


add("calibrated", calibration)
for number in (1, 6, 7, 8, 9, 15):
    add(
        f"number-{number}",
        lambda s, n=number: s["balls"][0].update(
            identity={
                "number": n,
                "confidence": 0.8,
                "evidence": [{"source": "number_classifier", "confidence": 0.8}],
            }
        ),
    )
for status in ("tracked", "occluded", "lost"):

    def cached(s, st=status):
        b = s["balls"][0]
        b["status"] = st
        b["last_observation"]["frame_id"] = "previous-frame"

    add(f"{status}-last-observed", cached)
    for kind in ("filtered", "extrapolated"):

        def estimated(s, st=status, k=kind):
            cached(s, st)
            b = s["balls"][0]
            b["estimate"] = {
                "kind": k,
                "as_of": s["published_at"],
                "image_center_px": {"x": 1888.0, "y": 1022.0},
                "image_space": "raw",
                "image_width_px": 3840,
                "image_height_px": 2160,
                "table_center_mm": None,
                "table_position_uncertainty": {
                    "status": "unavailable",
                    "sigma_x_mm": None,
                    "sigma_y_mm": None,
                },
            }

        add(f"{status}-{kind}", estimated)


def omitted(s):
    s["balls"][0].update(status="lost", measurement=None, estimate=None)


add("omitted", omitted)


def mixed(s):
    b = copy.deepcopy(s["balls"][0])
    b.update(track_id="omitted-track", status="lost", measurement=None, estimate=None)
    s["balls"].append(b)


add("mixed-omitted", mixed)
for update in ("annotation", "health"):
    add(update, lambda s, u=update: s.update(update_kind=u))
for outcome in ("degraded", "failed", "skipped"):
    add(outcome, lambda s, o=outcome: s.update(outcome=o, reason="fixture-reason"))
add(
    "cue-unavailable",
    lambda s: s.update(
        cue={k: ("unavailable" if k == "status" else None) for k in s["cue"]}
    ),
)


def cue_negative(s):
    s["cue"].update(status="not_observed", image_center_px=None, confidence=None)


add("cue-negative", cue_negative)


def estimate_without_measurement(s):
    s["balls"][0].update(status="tracked", measurement=None)
    s["balls"][0]["estimate"] = {
        "kind": "filtered",
        "as_of": s["published_at"],
        "image_center_px": {"x": 25.0, "y": 30.0},
        "image_space": "raw",
        "image_width_px": 3840,
        "image_height_px": 2160,
        "table_center_mm": None,
        "table_position_uncertainty": {
            "status": "unavailable",
            "sigma_x_mm": None,
            "sigma_y_mm": None,
        },
    }


add("estimate-without-measurement", estimate_without_measurement)


def table_only_estimate(s):
    calibration(s)
    s["balls"][0]["status"] = "occluded"
    s["balls"][0]["estimate"] = {
        "kind": "filtered",
        "as_of": s["published_at"],
        "image_center_px": None,
        "image_space": None,
        "image_width_px": None,
        "image_height_px": None,
        "table_center_mm": {"x": 100.0, "y": 200.0},
        "table_position_uncertainty": {
            "status": "unavailable",
            "sigma_x_mm": None,
            "sigma_y_mm": None,
        },
    }


add("table-only-estimate-falls-back", table_only_estimate)


def calibrated_estimate(s):
    table_only_estimate(s)
    s["balls"][0]["estimate"].update(
        image_center_px={"x": 100.0, "y": 200.0},
        image_space="raw",
        image_width_px=3840,
        image_height_px=2160,
    )


add("calibrated-estimate", calibrated_estimate)

for name, value in cases.items():
    (OUT / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
print(f"Generated {len(cases)} differential cases")
