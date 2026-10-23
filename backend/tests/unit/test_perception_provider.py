"""Provider selection must preserve the qualified live path by default."""

from __future__ import annotations

from typing import Any

import pytest

from backend.perception.provider import (
    PerceptionProvider,
    PerceptionProviderError,
    require_api_provider,
)


class ConfigStub:
    """Provide only the setting required by the provider selector."""

    def __init__(
        self,
        provider: Any = None,
        *,
        configured: bool = False,
    ) -> None:
        self._provider = provider
        self._configured = configured

    def get(self, key: str, default: Any = None) -> Any:
        if key == "perception.provider":
            return self._provider if self._configured else default
        raise AssertionError(f"Unexpected configuration key: {key}")


def test_missing_provider_setting_uses_the_rust_live_default() -> None:
    assert require_api_provider(ConfigStub()) is PerceptionProvider.RUST


def test_rust_provider_is_the_only_supported_api_provider() -> None:
    configuration = ConfigStub("rust", configured=True)
    assert require_api_provider(configuration) is PerceptionProvider.RUST


@pytest.mark.parametrize("provider", [None, 12, "automatic", "python"])
def test_invalid_provider_fails_before_startup(provider: Any) -> None:
    configuration = ConfigStub(provider, configured=True)

    with pytest.raises(PerceptionProviderError, match="must be 'rust'"):
        require_api_provider(configuration)
