"""Public contracts for listing, reading, downloading, and clearing logs."""

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.api.routes.logs as logs_routes


@pytest.fixture
def log_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    """Mount only the logs router and isolate it from the host log directory."""
    monkeypatch.setattr(logs_routes, "get_log_directory", lambda: tmp_path)
    app = FastAPI()
    app.include_router(logs_routes.router)
    return TestClient(app)


def test_list_log_files_returns_newest_first_metadata_and_ignores_other_files(
    log_api: TestClient, tmp_path: Path
) -> None:
    older = tmp_path / "app.log.1"
    newer = tmp_path / "error.log"
    ignored = tmp_path / "notes.txt"
    older.write_text("old", encoding="utf-8")
    newer.write_text("new log", encoding="utf-8")
    ignored.write_text("not a log", encoding="utf-8")
    older.touch()
    newer.touch()
    newer_mtime = older.stat().st_mtime + 10
    import os

    os.utime(newer, (newer_mtime, newer_mtime))

    response = log_api.get("/logs/files")

    assert response.status_code == 200
    files = response.json()["files"]
    assert [entry["name"] for entry in files] == ["error.log", "app.log.1"]
    assert files[0]["path"] == "error.log"
    assert files[0]["size"] == len("new log")
    assert files[0]["modified"] == newer_mtime


def test_list_log_files_returns_empty_when_directory_does_not_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing_directory = tmp_path / "missing"
    monkeypatch.setattr(logs_routes, "get_log_directory", lambda: missing_directory)
    app = FastAPI()
    app.include_router(logs_routes.router)

    response = TestClient(app).get("/logs/files")

    assert response.status_code == 200
    assert response.json() == {"files": []}


def test_list_log_files_maps_directory_failures_to_http_500(
    monkeypatch: pytest.MonkeyPatch, log_api: TestClient
) -> None:
    monkeypatch.setattr(
        logs_routes,
        "get_log_directory",
        lambda: (_ for _ in ()).throw(OSError("log directory unavailable")),
    )

    response = log_api.get("/logs/files")

    assert response.status_code == 500
    assert response.json()["detail"] == "log directory unavailable"


def test_view_log_supports_tail_and_offset_pagination(
    log_api: TestClient, tmp_path: Path
) -> None:
    (tmp_path / "app.log").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    tail = log_api.get("/logs/view/app.log", params={"lines": 2})
    page = log_api.get("/logs/view/app.log", params={"offset": 1})
    default_offset = log_api.get("/logs/view/app.log")

    assert tail.status_code == page.status_code == default_offset.status_code == 200
    assert tail.text == "three\nfour\n"
    assert page.text == "two\nthree\nfour\n"
    assert default_offset.text == "one\ntwo\nthree\nfour\n"
    assert tail.headers["content-type"].startswith("text/plain")


def test_view_log_returns_404_for_missing_file_and_403_for_escape(
    log_api: TestClient, tmp_path: Path
) -> None:
    outside = tmp_path.parent / "outside.log"
    outside.write_text("private", encoding="utf-8")
    try:
        missing = log_api.get("/logs/view/absent.log")
        with pytest.raises(logs_routes.HTTPException) as denied:
            asyncio.run(logs_routes.view_log_file("../outside.log"))
    finally:
        outside.unlink(missing_ok=True)

    assert missing.status_code == 404
    assert missing.json()["detail"] == "Log file not found"
    # Direct invocation keeps the traversal input intact; URL routers normalize
    # dot segments before dispatch, which would otherwise test the web server.
    assert denied.value.status_code == 403


def test_view_log_maps_read_errors_to_http_500(
    log_api: TestClient, tmp_path: Path
) -> None:
    # A directory passes the existence check but cannot be opened as a text file.
    (tmp_path / "unreadable.log").mkdir()

    response = log_api.get("/logs/view/unreadable.log")

    assert response.status_code == 500
    assert "unreadable.log" in response.json()["detail"]


def test_download_log_returns_attachment_with_text_payload(
    log_api: TestClient, tmp_path: Path
) -> None:
    (tmp_path / "app.log").write_text("download me\n", encoding="utf-8")

    response = log_api.get("/logs/download/app.log")

    assert response.status_code == 200
    assert response.content == b"download me\n"
    assert response.headers["content-type"].startswith("text/plain")
    assert 'filename="app.log"' in response.headers["content-disposition"]


def test_download_log_reports_missing_and_escaped_paths(
    log_api: TestClient, tmp_path: Path
) -> None:
    outside = tmp_path.parent / "outside-download.log"
    outside.write_text("private", encoding="utf-8")
    try:
        missing = log_api.get("/logs/download/absent.log")
        with pytest.raises(logs_routes.HTTPException) as denied:
            asyncio.run(logs_routes.download_log_file("../outside-download.log"))
    finally:
        outside.unlink(missing_ok=True)

    assert missing.status_code == 404
    assert denied.value.status_code == 403


@pytest.mark.parametrize("log_type", ["app", "error"])
def test_tail_log_selects_expected_file_and_limits_lines(
    log_api: TestClient, tmp_path: Path, log_type: str
) -> None:
    (tmp_path / f"{log_type}.log").write_text("a\nb\nc\n", encoding="utf-8")

    response = log_api.get(f"/logs/tail/{log_type}", params={"lines": 2})

    assert response.status_code == 200
    assert response.text == "b\nc\n"


@pytest.mark.parametrize("lines", [0, 10001])
def test_tail_log_validates_line_limit(log_api: TestClient, lines: int) -> None:
    response = log_api.get("/logs/tail/app", params={"lines": lines})

    assert response.status_code == 422


def test_tail_log_rejects_unsupported_type(log_api: TestClient) -> None:
    response = log_api.get("/logs/tail/debug")

    assert response.status_code == 422


def test_clear_log_truncates_file_and_returns_confirmation(
    log_api: TestClient, tmp_path: Path
) -> None:
    log_file = tmp_path / "error.log"
    log_file.write_text("previous errors\n", encoding="utf-8")

    response = log_api.delete("/logs/clear/error")

    assert response.status_code == 200
    assert response.json() == {"message": "Log file error.log cleared successfully"}
    assert log_file.read_bytes() == b""


def test_clear_log_reports_missing_file_and_rejects_unsupported_type(
    log_api: TestClient,
) -> None:
    missing = log_api.delete("/logs/clear/app")
    unsupported = log_api.delete("/logs/clear/debug")

    assert missing.status_code == 404
    assert unsupported.status_code == 422


def test_clear_log_maps_truncation_failure_to_http_500(
    monkeypatch: pytest.MonkeyPatch, log_api: TestClient, tmp_path: Path
) -> None:
    (tmp_path / "app.log").write_text("contents", encoding="utf-8")
    original_open = open

    def failing_open(
        file: object, mode: str = "r", *args: object, **kwargs: object
    ) -> Any:
        if file == tmp_path / "app.log" and mode == "w":
            raise OSError("read-only log")
        return cast("Any", original_open)(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", failing_open)

    response = log_api.delete("/logs/clear/app")

    assert response.status_code == 500
    assert response.json()["detail"] == "read-only log"
