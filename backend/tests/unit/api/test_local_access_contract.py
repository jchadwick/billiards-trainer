"""The local single-user API deliberately has no login or client credential flow."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.api.models.common import ErrorCode as HttpErrorCode
from backend.api.models.common import StatusCode as HttpStatusCode
from backend.api.utils import validators
from backend.dev_server import create_simple_app


def test_openapi_and_cors_expose_no_client_authentication_contract() -> None:
    for app in (create_app(), create_simple_app()):
        _assert_no_authentication_contract(app)


def test_production_http_route_ignores_credential_looking_headers() -> None:
    """Authentication-looking headers do not gate the anonymous local API."""
    client = TestClient(create_app())
    anonymous = client.get("/api")
    with_credential_like_headers = client.get(
        "/api",
        headers={
            "Authorization": "Bearer deliberately-invalid",
            "X-API-Key": "deliberately-invalid",
            "Cookie": "session=deliberately-invalid",
        },
    )

    assert anonymous.status_code == 200
    assert with_credential_like_headers.status_code == anonymous.status_code
    assert with_credential_like_headers.json() == anonymous.json()


def test_authentication_only_validators_and_statuses_are_absent() -> None:
    """Keep unused username/password and auth-failure contracts retired."""
    assert not hasattr(validators, "USERNAME_VALIDATOR")
    assert not hasattr(validators, "PASSWORD_VALIDATOR")
    assert not any(code.name.startswith("AUTH_") for code in HttpErrorCode)
    assert "UNAUTHORIZED" not in HttpStatusCode.__members__
    assert "FORBIDDEN" not in HttpStatusCode.__members__


def test_application_config_and_ci_have_no_retired_auth_secrets() -> None:
    """Do not restore unused login, API-key, or dashboard-password settings."""
    project_root = Path(__file__).resolve().parents[4]
    retired_names = (
        "JWT_SECRET_KEY",
        "JWT_ALGORITHM",
        "JWT_ACCESS_TOKEN_EXPIRE_MINUTES",
        "API_KEY_HEADER",
        "DEFAULT_API_KEY",
        "DATABASE_URL",
        "GRAFANA_PASSWORD",
    )
    application_templates = (
        project_root / ".env.example",
        project_root / ".env.docker",
        project_root / "config.json",
        project_root / "backend/config_schema_defaults.json",
        project_root / ".github/workflows/docker-build-deploy.yml",
    )

    for template in application_templates:
        contents = template.read_text(encoding="utf-8")
        for retired_name in retired_names:
            assert retired_name not in contents


def _assert_no_authentication_contract(app: FastAPI) -> None:
    schema = app.openapi()
    assert not any("/auth" in path for path in schema["paths"])
    assert "securitySchemes" not in schema.get("components", {})
    for path_item in schema["paths"].values():
        for operation in path_item.values():
            if isinstance(operation, dict):
                assert "security" not in operation

    cors = next(
        middleware
        for middleware in app.user_middleware
        if middleware.cls is CORSMiddleware  # type: ignore[comparison-overlap]
    )
    assert cors.kwargs["allow_credentials"] is False
    assert set(cors.kwargs["expose_headers"]) >= {
        "X-Frame-Preview-Descriptor",
        "X-Frame-Preview-Schema",
    }
