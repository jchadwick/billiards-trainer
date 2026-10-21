"""Keep repository-controlled CI free of project credentials and publishing."""

from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS_DIRECTORY = REPOSITORY_ROOT / ".github" / "workflows"
WORKFLOW_PATHS = (
    WORKFLOWS_DIRECTORY / "docker-build-deploy.yml",
    WORKFLOWS_DIRECTORY / "test.yml",
    WORKFLOWS_DIRECTORY / "perception.yml",
)
FORBIDDEN_CREDENTIAL_OPERATIONS = (
    "actions/checkout@",
    "docker/login-action",
    "secrets.GITHUB_TOKEN",
    "packages: write",
    "cache-from: type=gha",
    "cache-to: type=gha",
    "actions/upload-artifact",
    "codecov/codecov-action",
)


@pytest.mark.parametrize("workflow_path", WORKFLOW_PATHS, ids=lambda path: path.name)
def test_workflow_uses_public_fetch_and_has_no_credential_operations(
    workflow_path: Path,
) -> None:
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "permissions: {}" in workflow
    assert (
        'git remote add origin "https://github.com/${GITHUB_REPOSITORY}.git"'
        in workflow
    )
    assert "git -c credential.helper= fetch" in workflow

    for forbidden_operation in FORBIDDEN_CREDENTIAL_OPERATIONS:
        assert forbidden_operation not in workflow


def test_container_workflow_builds_locally_without_publishing() -> None:
    workflow = (WORKFLOWS_DIRECTORY / "docker-build-deploy.yml").read_text(
        encoding="utf-8"
    )

    assert "push: false" in workflow
    assert "load: true" in workflow
    assert "ghcr.io" not in workflow
