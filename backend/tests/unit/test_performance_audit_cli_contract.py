"""CLI and artifact contracts for the repeatable performance audit."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest

from backend.config import config
from backend.performance_audit import AsyncAuditRecorder
from backend.tools import collect_performance_audit as collector


def test_parser_uses_configured_audit_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        config,
        "_config_data",
        {
            "performance_audit": {
                "warmup_seconds": 4,
                "measured_seconds": 19,
                "repetitions": 2,
                "soak_seconds": 70,
                "history_size": 31,
                "event_queue_capacity": 7,
                "event_export_limit": 53,
                "sample_rate": 0.25,
                "heartbeat_interval_seconds": 6,
            }
        },
    )

    args = collector.build_parser().parse_args([])

    assert args.warmup == 4
    assert args.duration == 19
    assert args.repetitions == 2
    assert args.soak_duration == 70
    assert args.history_size == 31
    assert args.queue_capacity == 7
    assert args.event_export_limit == 53
    assert args.sample_rate == 0.25
    assert args.heartbeat_interval == 6


def test_missing_service_config_fails_before_starting_the_collector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_config = tmp_path / "missing.json"
    started = False

    def unexpected_run(_args: Any) -> Path:
        nonlocal started
        started = True
        raise AssertionError("collector must not start without its requested config")

    monkeypatch.setattr(collector, "run_audit", unexpected_run)

    with pytest.raises(FileNotFoundError, match="collector config does not exist"):
        collector.main(["--config", str(missing_config)])

    assert started is False


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--streams", "trajectory"], "must include state"),
        (["--streams", "state,video"], "may include trajectory"),
        (["--warmup", "-0.1"], "warmup must be non-negative"),
        (["--duration", "0"], "duration and repetitions positive"),
        (["--repetitions", "0"], "duration and repetitions positive"),
        (["--history-size", "0"], "duration and repetitions positive"),
        (["--queue-capacity", "0"], "duration and repetitions positive"),
        (["--event-export-limit", "0"], "duration and repetitions positive"),
        (["--sample-rate", "0"], "duration and repetitions positive"),
        (["--sample-rate", "1.01"], "duration and repetitions positive"),
        (["--subscriber-delay-ms", "-1"], "duration and repetitions positive"),
        (["--heartbeat-interval", "0"], "duration and repetitions positive"),
    ],
)
def test_invalid_run_settings_fail_before_allocating_an_audit_directory(
    tmp_path: Path,
    arguments: list[str],
    message: str,
) -> None:
    output_root = tmp_path / "audits"
    args = collector.build_parser().parse_args(
        ["--output-root", str(output_root), *arguments]
    )

    with pytest.raises(ValueError, match=message):
        asyncio.run(collector.run_audit(args))

    assert not output_root.exists()


def test_soak_cli_uses_one_repetition_and_configured_soak_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    async def fake_run_audit(args: Any) -> Path:
        observed["duration"] = args.duration
        observed["repetitions"] = args.repetitions
        return Path("unused")

    monkeypatch.setattr(collector, "run_audit", fake_run_audit)

    assert collector.main(["--soak", "--soak-duration", "37"]) == 0
    assert observed == {"duration": 37, "repetitions": 1}


def test_run_writes_bounded_event_trace_and_metadata_to_versioned_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_root = tmp_path / "audits"
    live_audit_root = tmp_path / "live-audits"
    monkeypatch.setattr(
        config,
        "_config_data",
        {"performance_audit": {"output_directory": str(live_audit_root)}},
    )
    monkeypatch.setattr(
        collector,
        "_project_metadata",
        lambda: {
            "revision": "revision-for-test",
            "dirty_worktree_sha256": "dirty-tree-for-test",
            "effective_configuration": {"perception": {"provider": "test"}},
        },
    )

    async def fake_collect_repetition(*args: Any, **kwargs: Any) -> dict[str, Any]:
        recorder = kwargs["recorder"]
        recorder.record({"event_type": "test_state_receipt", "sequence": 9})
        recorder.record({"event_type": "test_state_receipt", "sequence": 10})
        return {
            "repetition": 1,
            "streams": ["state"],
            "state_messages_per_second": 1.0,
            "unique_observations_per_second": 1.0,
            "source_publication_to_client_receipt_ms": {"p95": 8.0},
            "source_sequence_gaps": 0,
            "source_sequence_restarts": 0,
            "source_out_of_order_messages": 0,
            "measurement_started_monotonic_ns": None,
            "measurement_ended_monotonic_ns": None,
            "monotonic_clocks_comparable": False,
            "server_audit_run_id": None,
            "server_event_export": None,
        }

    monkeypatch.setattr(collector, "collect_repetition", fake_collect_repetition)
    args = collector.build_parser().parse_args(
        [
            "--output-root",
            str(output_root),
            "--warmup",
            "0",
            "--duration",
            "1",
            "--repetitions",
            "1",
            "--queue-capacity",
            "2",
            "--history-size",
            "3",
            "--event-export-limit",
            "1",
            "--quiet",
        ]
    )

    run_directory = asyncio.run(collector.run_audit(args))

    assert run_directory.parent == output_root
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{8}", run_directory.name)
    summary_path = run_directory / "summary.json"
    report_path = run_directory / "report.md"
    assert summary_path.is_file()
    assert report_path.is_file()
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 6
    assert report["audit_run_id"] == run_directory.name
    assert report["metadata"]["revision"] == "revision-for-test"
    assert report["metadata"]["dirty_worktree_sha256"] == "dirty-tree-for-test"
    assert (
        report["metadata"]["effective_configuration"]["perception"]["provider"]
        == "test"
    )
    assert report["repetitions"][0]["collector_settings"] == {
        "history_size": 3,
        "queue_capacity": 2,
        "sample_rate": 1.0,
        "event_export_limit": 1,
        "heartbeat_interval_seconds": 20.0,
    }
    export = report["event_export"]
    assert export["written_events"] == 1
    assert export["export_limit_dropped_events"] == 1
    assert export["queue_capacity"] == 2
    event_files = list((run_directory / "events").glob("*/events.jsonl"))
    assert len(event_files) == 1
    exported_events = [
        json.loads(line)
        for line in event_files[0].read_text(encoding="utf-8").splitlines()
    ]
    assert len(exported_events) == 1
    assert exported_events[0]["sequence"] == 9
    markdown = report_path.read_text(encoding="utf-8")
    assert "# Real-time state performance audit" in markdown
    assert "State publication-to-receipt p95" in markdown
    assert "Accuracy gate: no production candidate is recommendable" in markdown


def test_collector_closes_bounded_exporter_when_a_repetition_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorders: list[AsyncAuditRecorder] = []
    original_recorder = collector.AsyncAuditRecorder

    def track_recorder(*args: Any, **kwargs: Any) -> AsyncAuditRecorder:
        recorder = original_recorder(*args, **kwargs)
        recorders.append(recorder)
        return recorder

    async def failed_repetition(*args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs["recorder"].record({"event_type": "before_failure"})
        raise RuntimeError("simulated subscriber failure")

    monkeypatch.setattr(collector, "AsyncAuditRecorder", track_recorder)
    monkeypatch.setattr(collector, "collect_repetition", failed_repetition)
    args = collector.build_parser().parse_args(
        ["--output-root", str(tmp_path / "audits"), "--quiet"]
    )

    with pytest.raises(RuntimeError, match="simulated subscriber failure"):
        asyncio.run(collector.run_audit(args))

    assert len(recorders) == 1
    status = recorders[0].status()
    assert status["closed"] is True
    assert status["worker_alive"] is False
    assert status["written_events"] == 1
