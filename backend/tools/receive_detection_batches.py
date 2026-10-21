#!/usr/bin/env python3
"""Receive and validate replayed detection-batch v1 frames over loopback TCP.

This is a headless Stage 2 replay receiver. It records local monotonic receipt
times and writes the unmodified batch JSON as JSONL so the retained equivalence
tool can score the received messages. It does not infer one-way latency from
the Rust and Python monotonic clocks, whose epochs have not been qualified.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
import math
import os
import platform
import statistics
import struct
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import ValidationError  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    ROOT / "perception/contracts/detection-batch/v1/detection-batch.schema.json"
)


def _loopback_address(value: str) -> tuple[str, int]:
    """Parse a host:port pair and prevent binding outside the local machine."""
    try:
        host, port_text = value.rsplit(":", 1)
        address = ipaddress.ip_address(host.strip("[]"))
        port = int(port_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "address must be an IP loopback host:port"
        ) from exc
    if not address.is_loopback:
        raise argparse.ArgumentTypeError("receiver address must be loopback")
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return str(address), port


def _load_validator(schema_path: Path = SCHEMA_PATH) -> Draft202012Validator:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _percentile(values: list[float], rank: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(rank * len(ordered)) - 1))]


def _validate_batch_identity(
    batch: dict[str, Any], previous: dict[str, Any] | None
) -> None:
    """Reject a stream that changes its producer/session/model mid-connection."""
    identity = {
        "engine_instance_id": batch["engine_instance_id"],
        "source_session_id": batch["source"]["session_id"],
        "model_artifact_sha256": batch["model"]["artifact_sha256"],
        "source_checkpoint_sha256": batch["model"]["source_checkpoint_sha256"],
    }
    if previous is not None and identity != previous:
        raise ValueError(
            "batch engine, source session, or model identity changed in stream"
        )


async def _read_exactly(
    reader: asyncio.StreamReader, length: int, timeout_seconds: float
) -> bytes:
    return await asyncio.wait_for(reader.readexactly(length), timeout_seconds)


async def receive_stream(
    reader: asyncio.StreamReader,
    *,
    expected_batches: int,
    maximum_message_bytes: int,
    read_timeout_seconds: float,
    maximum_detail_records: int,
    jsonl_file: Any,
) -> dict[str, Any]:
    """Read a bounded stream and return receipt/order metrics for one run."""
    validator = _load_validator()
    receipt_records: list[dict[str, Any]] = []
    receipt_times_ns: list[int] = []
    sequences: list[int] = []
    errors: list[str] = []
    stable_identity: dict[str, Any] | None = None
    started_ns = time.monotonic_ns()
    extra_data = False

    for _ in range(expected_batches):
        try:
            header = await _read_exactly(reader, 4, read_timeout_seconds)
            payload_size = struct.unpack(">I", header)[0]
            if payload_size == 0:
                errors.append("zero_length_frame")
                break
            if payload_size > maximum_message_bytes:
                errors.append("frame_exceeds_configured_maximum")
                break
            payload = await _read_exactly(reader, payload_size, read_timeout_seconds)
        except asyncio.TimeoutError:
            errors.append("read_timeout")
            break
        except asyncio.IncompleteReadError as exc:
            errors.append(
                "sender_closed_before_expected_batch_count"
                if not exc.partial
                else "truncated_frame"
            )
            break

        receipt_ns = time.monotonic_ns()
        try:
            batch = json.loads(payload)
            if not isinstance(batch, dict):
                raise ValueError("batch payload must be a JSON object")
            validator.validate(batch)
            _validate_batch_identity(batch, stable_identity)
        except (
            json.JSONDecodeError,
            ValidationError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            errors.append(f"invalid_batch:{type(exc).__name__}:{exc}")
            break

        stable_identity = {
            "engine_instance_id": batch["engine_instance_id"],
            "source_session_id": batch["source"]["session_id"],
            "model_artifact_sha256": batch["model"]["artifact_sha256"],
            "source_checkpoint_sha256": batch["model"]["source_checkpoint_sha256"],
        }
        sequence = batch["batch_sequence"]
        sequences.append(sequence)
        receipt_times_ns.append(receipt_ns)
        jsonl_file.write(payload.decode("utf-8"))
        jsonl_file.write("\n")

        if len(receipt_records) < maximum_detail_records:
            receipt_records.append(
                {
                    "batch_sequence": sequence,
                    "frame_id": batch["source"]["frame_id"],
                    "engine_instance_id": batch["engine_instance_id"],
                    "source_session_id": batch["source"]["session_id"],
                    "outcome": batch["outcome"],
                    "detection_count": len(batch["detections"]),
                    "receipt_monotonic_ns": str(receipt_ns),
                }
            )

    jsonl_file.flush()
    if len(sequences) == expected_batches and not errors:
        try:
            extra_data = bool(
                await asyncio.wait_for(reader.read(1), read_timeout_seconds)
            )
            if extra_data:
                errors.append("unexpected_data_after_expected_batch_count")
        except asyncio.TimeoutError:
            errors.append("sender_close_timeout")

    interarrival_ms = [
        (current - previous) / 1_000_000
        for previous, current in zip(receipt_times_ns, receipt_times_ns[1:])
    ]
    expected_sequences = list(range(1, expected_batches + 1))
    duplicates = len(sequences) - len(set(sequences))
    missing_sequences = len(set(expected_sequences) - set(sequences))
    highest_sequence = 0
    sequence_gap_events = 0
    sequence_gap_total = 0
    out_of_order_records = 0
    unexpected_sequences = 0
    for sequence in sequences:
        if sequence < 1 or sequence > expected_batches:
            unexpected_sequences += 1
        if sequence <= highest_sequence:
            out_of_order_records += 1
        elif sequence > highest_sequence + 1:
            sequence_gap_events += 1
            sequence_gap_total += sequence - highest_sequence - 1
            highest_sequence = sequence
        else:
            highest_sequence = sequence
    first_receipt_ns = receipt_times_ns[0] if receipt_times_ns else None
    last_receipt_ns = receipt_times_ns[-1] if receipt_times_ns else None

    return {
        "schema_version": 1,
        "receiver_run_id": str(uuid.uuid4()),
        "receiver_process_id": os.getpid(),
        "receiver_clock": "python_time.monotonic_ns; local receipt times only",
        "cross_process_one_way_latency": "unavailable_until_clock_epochs_are_verified",
        "expected_batches": expected_batches,
        "received_batches": len(sequences),
        "complete": len(sequences) == expected_batches
        and not errors
        and not extra_data,
        "ordered": sequences == expected_sequences,
        "duplicate_sequences": duplicates,
        "missing_sequences": missing_sequences,
        "sequence_gap_events": sequence_gap_events,
        "sequence_gap_total": sequence_gap_total,
        "out_of_order_records": out_of_order_records,
        "unexpected_sequences": unexpected_sequences,
        "sequence_values": (
            sequences if len(sequences) <= maximum_detail_records else None
        ),
        "first_receipt_monotonic_ns": (
            str(first_receipt_ns) if first_receipt_ns is not None else None
        ),
        "last_receipt_monotonic_ns": (
            str(last_receipt_ns) if last_receipt_ns is not None else None
        ),
        "receipt_window_ms": (
            (last_receipt_ns - first_receipt_ns) / 1_000_000
            if first_receipt_ns is not None and last_receipt_ns is not None
            else None
        ),
        "interarrival_ms": {
            "sample_count": len(interarrival_ms),
            "p50": _percentile(interarrival_ms, 0.50),
            "p95": _percentile(interarrival_ms, 0.95),
            "p99": _percentile(interarrival_ms, 0.99),
            "mean": statistics.fmean(interarrival_ms) if interarrival_ms else None,
        },
        "receipt_record_limit": maximum_detail_records,
        "receipt_records_retained": len(receipt_records),
        "receipt_records_dropped": max(0, len(sequences) - len(receipt_records)),
        "receipt_records": receipt_records,
        "errors": errors,
        "unexpected_data_after_expected_count": extra_data,
        "receiver_elapsed_ms": (time.monotonic_ns() - started_ns) / 1_000_000,
        "producer_identity": stable_identity,
    }


async def serve_once(
    bind_address: tuple[str, int],
    *,
    expected_batches: int,
    maximum_message_bytes: int,
    accept_timeout_seconds: float,
    session_timeout_seconds: float,
    read_timeout_seconds: float,
    maximum_detail_records: int,
    jsonl_file: Any,
    ready_callback: Callable[[tuple[str, int]], None] | None = None,
) -> tuple[dict[str, Any], tuple[str, int]]:
    """Accept a single local producer and collect exactly one replay stream."""
    if not ipaddress.ip_address(bind_address[0]).is_loopback:
        raise ValueError("receiver address must be loopback")
    if not 0 <= bind_address[1] <= 65535:
        raise ValueError("receiver port must be between 0 and 65535")
    loop = asyncio.get_running_loop()
    accepted_future: asyncio.Future[None] = loop.create_future()
    result_future: asyncio.Future[dict[str, Any]] = loop.create_future()
    accepted = False

    async def on_client(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        nonlocal accepted
        if accepted:
            writer.close()
            await writer.wait_closed()
            return
        accepted = True
        if not accepted_future.done():
            accepted_future.set_result(None)
        peer = writer.get_extra_info("peername")
        try:
            result = await receive_stream(
                reader,
                expected_batches=expected_batches,
                maximum_message_bytes=maximum_message_bytes,
                read_timeout_seconds=read_timeout_seconds,
                maximum_detail_records=maximum_detail_records,
                jsonl_file=jsonl_file,
            )
            result["peer_address"] = list(peer[:2]) if peer else None
            if not result_future.done():
                result_future.set_result(result)
        except Exception as exc:
            if not result_future.done():
                result_future.set_exception(exc)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(on_client, *bind_address)
    sockets = server.sockets
    if not sockets:
        server.close()
        await server.wait_closed()
        raise RuntimeError("receiver failed to bind a local socket")
    actual_address = sockets[0].getsockname()[:2]
    bound_address = (str(actual_address[0]), int(actual_address[1]))
    if ready_callback is not None:
        ready_callback(bound_address)
    try:
        async with server:
            await asyncio.wait_for(accepted_future, accept_timeout_seconds)
            result = await asyncio.wait_for(result_future, session_timeout_seconds)
    finally:
        server.close()
        await server.wait_closed()
    return result, bound_address


async def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.expected_batches <= 0:
        raise ValueError("expected batch count must be positive")
    if arguments.expected_batches > arguments.maximum_batches:
        raise ValueError("expected batch count exceeds the configured maximum")
    if not 0 < arguments.maximum_message_bytes <= 0xFFFFFFFF:
        raise ValueError(
            "maximum message size must fit a non-zero unsigned 32-bit length"
        )
    if (
        arguments.accept_timeout_ms <= 0
        or arguments.session_timeout_ms <= 0
        or arguments.read_timeout_ms <= 0
    ):
        raise ValueError("receiver timeouts must be positive")
    if arguments.maximum_detail_records < 0:
        raise ValueError("maximum detail record count cannot be negative")

    _load_validator()

    def print_ready(address: tuple[str, int]) -> None:
        host = f"[{address[0]}]" if ":" in address[0] else address[0]
        print(
            json.dumps({"event": "ready", "address": f"{host}:{address[1]}"}),
            flush=True,
        )

    with arguments.batches_jsonl.open("x", encoding="utf-8") as batch_file:
        report, bound_address = await serve_once(
            arguments.bind_address,
            expected_batches=arguments.expected_batches,
            maximum_message_bytes=arguments.maximum_message_bytes,
            accept_timeout_seconds=arguments.accept_timeout_ms / 1000,
            session_timeout_seconds=arguments.session_timeout_ms / 1000,
            read_timeout_seconds=arguments.read_timeout_ms / 1000,
            maximum_detail_records=arguments.maximum_detail_records,
            jsonl_file=batch_file,
            ready_callback=print_ready,
        )
    report["bind_address"] = f"{bound_address[0]}:{bound_address[1]}"
    report["batches_jsonl_path"] = str(arguments.batches_jsonl)
    report["receiver_settings"] = {
        "expected_batches": arguments.expected_batches,
        "maximum_batches": arguments.maximum_batches,
        "maximum_message_bytes": arguments.maximum_message_bytes,
        "accept_timeout_ms": arguments.accept_timeout_ms,
        "session_timeout_ms": arguments.session_timeout_ms,
        "read_timeout_ms": arguments.read_timeout_ms,
        "maximum_detail_records": arguments.maximum_detail_records,
    }
    report["runtime"] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    digest = hashlib.sha256()
    with arguments.batches_jsonl.open("rb") as batch_file:
        for block in iter(lambda: batch_file.read(1024 * 1024), b""):
            digest.update(block)
    report["batches_jsonl_sha256"] = digest.hexdigest()
    return report


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bind-address",
        type=_loopback_address,
        required=True,
        help="local loopback IP:port; use port 0 to request an available port",
    )
    parser.add_argument("--expected-batches", type=int, required=True)
    parser.add_argument("--maximum-batches", type=int, default=10000)
    parser.add_argument("--maximum-message-bytes", type=int, required=True)
    parser.add_argument("--accept-timeout-ms", type=int, required=True)
    parser.add_argument("--session-timeout-ms", type=int, required=True)
    parser.add_argument("--read-timeout-ms", type=int, required=True)
    parser.add_argument("--maximum-detail-records", type=int, default=10000)
    parser.add_argument("--batches-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    try:
        report = asyncio.run(_run(arguments))
    except (OSError, RuntimeError, ValueError, asyncio.TimeoutError) as exc:
        print(f"detection batch receiver failed: {exc}", file=sys.stderr)
        return 2

    encoded = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output_json is not None:
        with arguments.output_json.open("x", encoding="utf-8") as output_file:
            output_file.write(encoded)
            output_file.write("\n")
    print(encoded)
    return 0 if report["complete"] and report["ordered"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
