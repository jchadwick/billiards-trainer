"""Deployment launcher tests for single-worker table ownership."""

import subprocess
import sys
from pathlib import Path

import pytest

from deployment.deploy import (
    build_nginx_configuration,
    build_nginx_websocket_locations,
    build_production_configuration,
    build_systemd_exec_start,
)


@pytest.mark.parametrize("section", ["server", "network"])
def test_production_configuration_rejects_removed_worker_setting(section):
    with pytest.raises(ValueError, match="worker-count settings are unsupported"):
        build_production_configuration({"api": {section: {"workers": 4}}})


def test_generated_production_configuration_uses_single_process_defaults():
    generated = build_production_configuration(
        {
            "api": {
                "server": {"port": 9000},
                "network": {},
                "rate_limiting": {"store": "memory"},
            }
        }
    )

    assert "workers" not in generated["api"]["server"]
    assert "workers" not in generated["api"]["network"]
    assert generated["api"]["server"]["port"] == 8000
    assert generated["api"]["rate_limiting"]["store"] == "memory"
    assert generated["api"]["rate_limiting"]["store_timeout_seconds"] == 2.0
    assert generated["api"]["rate_limiting"]["window_seconds"] == 60
    assert generated["api"]["rate_limiting"]["websocket_connections_per_ip"] == 10


def test_generated_systemd_launcher_uses_real_configured_entry_point():
    target_dir = Path("/opt/billiards-trainer")

    assert build_systemd_exec_start(target_dir) == (
        "/opt/billiards-trainer/bin/billiards-server "
        "--http-bind 0.0.0.0 --http-port 8000 "
        "--scene-bind 127.0.0.1 --scene-port 49321 "
        "--config /opt/billiards-trainer/config/production.json "
        "--project-root /opt/billiards-trainer "
        "--web-root /opt/billiards-trainer/frontend/web/dist --start-live "
        "--live-executable /opt/billiards-trainer/bin/billiards-perception-live "
        "--capture-helper-executable "
        "/opt/billiards-trainer/bin/billiards-perception-gstreamer-probe"
    )


def test_checked_in_service_uses_single_process_and_optional_redis_environment():
    service_file = (
        Path(__file__).resolve().parents[3]
        / "deployment"
        / "systemd"
        / "billiards-trainer.service"
    )
    service = service_file.read_text(encoding="utf-8")

    assert "/opt/billiards-trainer/bin/billiards-server" in service
    assert "--config /opt/billiards-trainer/config/production.json" in service
    assert "EnvironmentFile=-/etc/billiards-trainer.env" in service
    assert "python -m backend.main" not in service
    assert "system_launcher.py" not in service
    assert "--workers 4" not in service


def test_docker_ci_builds_the_repository_root_rust_runtime_image():
    workflow_file = (
        Path(__file__).resolve().parents[3]
        / ".github"
        / "workflows"
        / "docker-build-deploy.yml"
    )
    workflow = workflow_file.read_text(encoding="utf-8")

    assert "context: ." in workflow
    assert "file: ./backend/Dockerfile" in workflow
    assert "target: runtime" in workflow
    assert "./frontend/web/Dockerfile" not in workflow
    assert "billiards-trainer:ci" in workflow


@pytest.mark.parametrize(
    ("module", "rust_commands"),
    [
        ("backend.main", ("make run", "make run-api")),
        ("backend.dev_server", ("make run", "make run-api")),
        ("backend.video", ("make run",)),
    ],
)
def test_python_application_server_launchers_are_retired(module, rust_commands):
    result = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )

    assert result.returncode == 2
    assert "Python" in result.stderr
    assert "retired" in result.stderr
    for command in rust_commands:
        assert command in result.stderr


def test_checked_in_and_generated_nginx_configs_match_websocket_routes():
    nginx_file = (
        Path(__file__).resolve().parents[3]
        / "deployment"
        / "nginx"
        / "billiards-trainer.conf"
    )
    checked_in = nginx_file.read_text(encoding="utf-8")
    locations = build_nginx_websocket_locations()
    generated = build_nginx_configuration(Path("/opt/billiards-trainer"))

    route = "/api/v1/websocket/ws"
    assert f"location = {route} {{" in checked_in
    assert f"location = {route} {{" in generated
    assert f"location = {route} {{" in locations
    assert "location = /api/v1/ws {" not in checked_in
    assert "location = /api/v1/ws {" not in generated
    assert "location = /api/v1/ws {" not in locations
    assert "location /ws" not in checked_in
    assert "location /ws" not in generated
    generated_zone = generated.index(
        "limit_req_zone $binary_remote_addr zone=websocket:10m rate=5r/s;"
    )
    generated_server = generated.index("server {")
    generated_use = generated.index("limit_req zone=websocket")
    assert generated_zone < generated_server < generated_use
