"""Tests for the headless loopback detection-batch replay receiver."""

import argparse
import asyncio
import copy
import io
import json
import struct
from pathlib import Path
from typing import Any

import pytest

from backend.tools.receive_detection_batches import (
    _loopback_address,
    receive_stream,
    serve_once,
)

FIXTURES = (
    Path(__file__).resolve().parents[3] / "perception/contracts/detection-batch/v1"
)


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _framed(*batches: dict[str, Any]) -> bytes:
    frames: list[bytes] = []
    for batch in batches:
        payload = json.dumps(batch, separators=(",", ":")).encode("utf-8")
        frames.append(struct.pack(">I", len(payload)) + payload)
    return b"".join(frames)


def _receive(
    data: bytes, *, expected: int, maximum_details: int = 10
) -> tuple[dict[str, Any], str]:
    async def run() -> tuple[dict[str, Any], str]:
        reader = asyncio.StreamReader()
        reader.feed_data(data)
        reader.feed_eof()
        output = io.StringIO()
        report = await receive_stream(
            reader,
            expected_batches=expected,
            maximum_message_bytes=64 * 1024,
            read_timeout_seconds=0.1,
            maximum_detail_records=maximum_details,
            jsonl_file=output,
        )
        return report, output.getvalue()

    return asyncio.run(run())


def test_receiver_accepts_ordered_batches_and_records_local_receipts() -> None:
    first = _fixture("detection-batch-success.json")
    second = _fixture("detection-batch-empty-success.json")

    report, jsonl = _receive(_framed(first, second), expected=2)

    assert report["complete"] is True
    assert report["ordered"] is True
    assert report["received_batches"] == 2
    assert report["missing_sequences"] == 0
    assert report["duplicate_sequences"] == 0
    assert report["cross_process_one_way_latency"].startswith("unavailable")
    assert report["receipt_records"][0]["receipt_monotonic_ns"].isdigit()
    assert len(jsonl.splitlines()) == 2
    assert json.loads(jsonl.splitlines()[1])["batch_sequence"] == 2


def test_receiver_reports_duplicate_and_missing_sequence_without_freshness_claims() -> (
    None
):
    first = _fixture("detection-batch-success.json")
    duplicate = copy.deepcopy(first)
    duplicate["source"]["frame_id"] = "frame-1043"

    report, _ = _receive(_framed(first, duplicate), expected=2)

    assert report["complete"] is True
    assert report["ordered"] is False
    assert report["duplicate_sequences"] == 1
    assert report["missing_sequences"] == 1
    assert report["out_of_order_records"] == 1


def test_receiver_counts_sequence_gaps_and_reordered_records() -> None:
    first = _fixture("detection-batch-success.json")
    second = _fixture("detection-batch-empty-success.json")
    third = copy.deepcopy(second)
    third["batch_sequence"] = 3
    third["source"]["frame_id"] = "frame-1044"

    report, _ = _receive(_framed(first, third, second), expected=3)

    assert report["ordered"] is False
    assert report["duplicate_sequences"] == 0
    assert report["missing_sequences"] == 0
    assert report["sequence_gap_events"] == 1
    assert report["sequence_gap_total"] == 1
    assert report["out_of_order_records"] == 1


def test_receiver_rejects_producer_identity_changes_within_a_stream() -> None:
    first = _fixture("detection-batch-success.json")
    second = _fixture("detection-batch-empty-success.json")
    second["engine_instance_id"] = "different-engine"

    report, jsonl = _receive(_framed(first, second), expected=2)

    assert not report["complete"]
    assert report["received_batches"] == 1
    assert report["errors"][0].startswith("invalid_batch:ValueError:")
    assert len(jsonl.splitlines()) == 1


def test_receiver_rejects_oversized_and_truncated_frames() -> None:
    async def receive(data: bytes) -> dict[str, Any]:
        reader = asyncio.StreamReader()
        reader.feed_data(data)
        reader.feed_eof()
        return await receive_stream(
            reader,
            expected_batches=1,
            maximum_message_bytes=16,
            read_timeout_seconds=0.1,
            maximum_detail_records=1,
            jsonl_file=io.StringIO(),
        )

    oversized = asyncio.run(receive(struct.pack(">I", 17)))
    truncated = asyncio.run(receive(struct.pack(">I", 3) + b"{}"))

    assert oversized["errors"] == ["frame_exceeds_configured_maximum"]
    assert truncated["errors"] == ["truncated_frame"]
    assert not oversized["complete"]
    assert not truncated["complete"]


def test_receiver_rejects_schema_invalid_batches() -> None:
    invalid = _fixture("detection-batch-success.json")
    invalid["unexpected"] = True

    report, output = _receive(_framed(invalid), expected=1)

    assert not report["complete"]
    assert report["received_batches"] == 0
    assert report["errors"][0].startswith("invalid_batch:ValidationError:")
    assert output == ""


def test_receiver_caps_detailed_receipt_retention_but_keeps_aggregate_counts() -> None:
    first = _fixture("detection-batch-success.json")
    second = _fixture("detection-batch-empty-success.json")

    report, _ = _receive(_framed(first, second), expected=2, maximum_details=1)

    assert report["received_batches"] == 2
    assert report["receipt_records_retained"] == 1
    assert report["receipt_records_dropped"] == 1
    assert report["sequence_values"] is None


def test_receiver_binds_only_loopback_addresses() -> None:
    assert _loopback_address("127.0.0.1:0") == ("127.0.0.1", 0)
    with pytest.raises(
        argparse.ArgumentTypeError, match="receiver address must be loopback"
    ):
        _loopback_address("192.0.2.1:9000")


def test_single_session_loopback_server_accepts_and_records_a_real_socket_batch() -> (
    None
):
    async def run() -> tuple[dict[str, Any], str, tuple[str, int]]:
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[tuple[str, int]] = loop.create_future()
        output = io.StringIO()
        receiver = asyncio.create_task(
            serve_once(
                ("127.0.0.1", 0),
                expected_batches=1,
                maximum_message_bytes=64 * 1024,
                accept_timeout_seconds=1,
                session_timeout_seconds=1,
                read_timeout_seconds=1,
                maximum_detail_records=10,
                jsonl_file=output,
                ready_callback=lambda address: ready.set_result(address),
            )
        )
        address = await ready
        reader, writer = await asyncio.open_connection(*address)
        del reader
        writer.write(_framed(_fixture("detection-batch-success.json")))
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        report, bound = await receiver
        return report, output.getvalue(), bound

    report, jsonl, bound_address = asyncio.run(run())

    assert report["complete"]
    assert report["peer_address"][0] == "127.0.0.1"
    assert bound_address[0] == "127.0.0.1"
    assert json.loads(jsonl)["batch_sequence"] == 1
