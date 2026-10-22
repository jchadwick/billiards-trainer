"""Supported Rust deployment and launcher contracts."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_checked_in_service_starts_one_rust_process_without_nginx():
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
    assert "nginx" not in service.lower()
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
    assert "billiards-trainer:ci" in workflow


def test_distribution_target_uses_shell_bundle_builder_without_duplicate_frontend_build():
    makefile = Path(__file__).resolve().parents[3] / "Makefile"
    lines = makefile.read_text(encoding="utf-8").splitlines()
    deploy_line = next(line for line in lines if line.startswith("deploy:"))
    deploy_recipe = lines[lines.index(deploy_line) + 1]

    assert "build-frontend" not in deploy_line
    assert "bash scripts/deploy/build-dist.sh" in deploy_recipe


def test_legacy_python_nginx_deployment_paths_are_retired():
    repository = Path(__file__).resolve().parents[3]

    assert not (repository / "deployment" / "deploy.py").exists()
    assert not (repository / "deployment" / "nginx").exists()
    assert not (repository / "frontend" / "web" / "Dockerfile").exists()
    assert not (repository / "frontend" / "web" / "nginx.conf").exists()


@pytest.mark.parametrize(
    ("module", "rust_commands"),
    [
        ("backend.main", ("make run", "make run-api")),
        ("backend.dev_server", ("make run", "make run-api")),
        ("backend.video", ("make run",)),
        ("backend.streaming.enhanced_camera_module", ("make run",)),
    ],
)
def test_python_server_and_camera_launchers_are_retired(module, rust_commands):
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
