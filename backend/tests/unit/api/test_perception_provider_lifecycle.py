"""The API must reject unavailable providers before starting dependencies."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from backend.api import main
from backend.perception.provider import PerceptionProviderError
from backend.tests.support.compat_api import create_compatibility_app


def test_python_provider_fails_before_starting_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main.config,
        "get",
        lambda key, default=None: {
            "perception.provider": "python",
        }.get(key, default),
    )
    initialize_rate_limiter = AsyncMock(return_value=True)
    monkeypatch.setattr(main.rate_limiter, "initialize", initialize_rate_limiter)

    async def enter_lifespan() -> None:
        async with main.lifespan(create_compatibility_app()):
            pytest.fail("retired Python provider unexpectedly entered lifespan")

    with pytest.raises(PerceptionProviderError, match="must be 'rust'"):
        asyncio.run(enter_lifespan())

    assert initialize_rate_limiter.await_count == 0
    assert main.app_state.is_healthy is False


def test_compatibility_api_has_no_default_asgi_server_instance() -> None:
    """Python API routes are test fixtures, never an implicit server entrypoint."""
    assert not hasattr(main, "app")


def test_compatibility_api_does_not_export_an_asgi_factory() -> None:
    """Python API routing can only be assembled from test support."""
    assert not hasattr(main, "create_app")
