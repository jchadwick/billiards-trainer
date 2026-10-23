"""Project-owned deployment helpers stay local and credential-free."""

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DEPLOYMENT_SCRIPTS = (
    PROJECT_ROOT / "frontend/projector/deploy.sh",
    PROJECT_ROOT / "scripts/deploy/build-dist.sh",
    PROJECT_ROOT / "scripts/deploy/update.sh",
)
CURRENT_DEPLOYMENT_GUIDES = (
    PROJECT_ROOT / "docs/DEPLOYMENT_GUIDE.md",
    PROJECT_ROOT / "docs/PROJECTOR_DEPLOYMENT.md",
    PROJECT_ROOT / "docs/docker-deployment.md",
    PROJECT_ROOT / "frontend/projector/INSTALL.md",
    PROJECT_ROOT / "frontend/projector/README.md",
    PROJECT_ROOT / "frontend/projector/SPECS.md",
    PROJECT_ROOT / "frontend/projector/TESTING.md",
)
REMOTE_LOGIN_COMMAND = re.compile(r"(?m)^\s*(?:ssh|scp)\s+")
REMOTE_RSYNC_TARGET = re.compile(r"(?m)^\s*rsync\b[^\n]*\s+\S+@\S+:")


@pytest.mark.parametrize(
    "script_path", PROJECT_DEPLOYMENT_SCRIPTS, ids=lambda p: p.name
)
def test_project_deployment_scripts_do_not_connect_to_remote_hosts(
    script_path: Path,
) -> None:
    script = script_path.read_text(encoding="utf-8")

    assert REMOTE_LOGIN_COMMAND.search(script) is None
    assert REMOTE_RSYNC_TARGET.search(script) is None


@pytest.mark.parametrize("guide_path", CURRENT_DEPLOYMENT_GUIDES, ids=lambda p: p.name)
def test_current_deployment_guides_do_not_instruct_remote_login(
    guide_path: Path,
) -> None:
    guide = guide_path.read_text(encoding="utf-8")

    assert REMOTE_LOGIN_COMMAND.search(guide) is None
    assert REMOTE_RSYNC_TARGET.search(guide) is None
