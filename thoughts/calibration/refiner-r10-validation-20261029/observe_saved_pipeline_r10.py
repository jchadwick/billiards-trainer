#!/usr/bin/env python3
"""Measure saved-frame end-to-end availability without requiring full startup."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


STATE_URL = "http://127.0.0.1:18000/api/v1/game/state"


def state() -> dict[str, Any]:
    with urllib.request.urlopen(STATE_URL, timeout=2) as response:
        return json.load(response)


def balls_in(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    balls = snapshot.get("balls", [])
    return list(balls.values()) if isinstance(balls, dict) else balls


def counts(snapshot: dict[str, Any]) -> tuple[int, int]:
    balls = balls_in(snapshot)
    precise = sum(
        ball.get("table_position") is not None and ball.get("is_observed") is True
        for ball in balls
    )
    return len(balls), precise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--startup-minimum", type=int, default=15)
    parser.add_argument("--target-precise", type=int, default=15)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    args = parser.parse_args()
    if args.seconds <= 0 or args.sample_hz <= 0 or args.startup_minimum < 0:
        raise ValueError("duration and sampling rate must be positive; startup minimum nonnegative")
    args.output.mkdir(exist_ok=True, parents=True)

    last: dict[str, Any] = {}
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            last = state()
            if counts(last)[1] >= args.startup_minimum:
                break
        except Exception as error:
            last = {"error": str(error)}
        time.sleep(0.2)
    else:
        (args.output / "startup-state.json").write_text(json.dumps(last, indent=2))
        raise RuntimeError(
            f"startup reached only {counts(last)[1]} precise balls; "
            f"minimum is {args.startup_minimum}"
        )

    (args.output / "first-state.json").write_text(json.dumps(last, indent=2))
    start = time.monotonic()
    next_sample = start
    histogram: dict[str, int] = {}
    fewer_than_target: list[dict[str, float]] = []
    episode_start: float | None = None
    fewer_than_startup_minimum: list[dict[str, float]] = []
    startup_episode_start: float | None = None
    frame_ids: set[str] = set()
    samples = 0
    errors = 0
    freshness: dict[str, int] = {}

    while time.monotonic() - start < args.seconds:
        sample_at = time.monotonic()
        try:
            last = state()
            total, precise = counts(last)
            histogram[str(precise)] = histogram.get(str(precise), 0) + 1
            observation = last.get("observation_status") or {}
            freshness_status = str(observation.get("freshness_status", "missing"))
            freshness[freshness_status] = freshness.get(freshness_status, 0) + 1
            identity = observation.get("observation_identity") or {}
            if identity.get("frame_id") is not None:
                frame_ids.add(str(identity["frame_id"]))
            target_unavailable = precise < args.target_precise
            below_startup_minimum = precise < args.startup_minimum
        except Exception:
            errors += 1
            total = 0
            precise = 0
            target_unavailable = True
            below_startup_minimum = True

        elapsed = sample_at - start
        if target_unavailable and episode_start is None:
            episode_start = elapsed
        if not target_unavailable and episode_start is not None:
            fewer_than_target.append(
                {"start_seconds": episode_start, "duration_seconds": elapsed - episode_start}
            )
            episode_start = None
        if below_startup_minimum and startup_episode_start is None:
            startup_episode_start = elapsed
        if not below_startup_minimum and startup_episode_start is not None:
            fewer_than_startup_minimum.append(
                {
                    "start_seconds": startup_episode_start,
                    "duration_seconds": elapsed - startup_episode_start,
                }
            )
            startup_episode_start = None

        samples += 1
        next_sample += 1.0 / args.sample_hz
        time.sleep(max(0.0, next_sample - time.monotonic()))

    elapsed = time.monotonic() - start
    if episode_start is not None:
        fewer_than_target.append(
            {"start_seconds": episode_start, "duration_seconds": elapsed - episode_start}
        )
    if startup_episode_start is not None:
        fewer_than_startup_minimum.append(
            {
                "start_seconds": startup_episode_start,
                "duration_seconds": elapsed - startup_episode_start,
            }
        )

    report = {
        "scope": "saved-image application pipeline; no camera",
        "elapsed_seconds": elapsed,
        "sample_hz": args.sample_hz,
        "samples": samples,
        "startup_minimum_precise_balls": args.startup_minimum,
        "target_precise_balls": args.target_precise,
        "precise_count_histogram": histogram,
        "freshness_status_histogram": freshness,
        "samples_with_fewer_than_target": sum(
            value for key, value in histogram.items() if int(key) < args.target_precise
        ),
        "fewer_than_target_episodes": fewer_than_target,
        "fewer_than_startup_minimum_episodes": fewer_than_startup_minimum,
        "http_errors": errors,
        "distinct_observation_frame_ids": len(frame_ids),
        "production_eligible_accuracy_evidence": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "last-state.json").write_text(json.dumps(last, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
