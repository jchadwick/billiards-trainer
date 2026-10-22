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
        mode: Any = None,
        mode_configured: bool = False,
    ) -> None:
        self._provider = provider
        self._configured = configured
        self._mode = mode
        self._mode_configured = mode_configured

    def get(self, key: str, default: Any = None) -> Any:
        if key == "perception.provider":
            return self._provider if self._configured else default
        if key == "perception.mode":
            return self._mode if self._mode_configured else default
        raise AssertionError(f"Unexpected configuration key: {key}")


def test_missing_provider_setting_uses_the_rust_live_default() -> None:
    assert require_api_provider(ConfigStub()) is PerceptionProvider.RUST


def test_explicit_python_provider_is_supported() -> None:
    configuration = ConfigStub("python", configured=True)

    assert require_api_provider(configuration) is PerceptionProvider.PYTHON


def test_rust_provider_supports_live_and_explicit_recorded_replay_modes() -> None:
    configuration = ConfigStub("rust", configured=True)
    assert require_api_provider(configuration) is PerceptionProvider.RUST

    replay_configuration = ConfigStub(
        "rust", configured=True, mode="recorded_replay", mode_configured=True
    )
    assert require_api_provider(replay_configuration) is PerceptionProvider.RUST


def test_python_provider_rejects_recorded_replay_mode() -> None:
    configuration = ConfigStub(
        "python",
        configured=True,
        mode="recorded_replay",
        mode_configured=True,
    )

    with pytest.raises(PerceptionProviderError, match="supports only"):
        require_api_provider(configuration)


@pytest.mark.parametrize("provider", [None, 12, "automatic"])
def test_invalid_provider_fails_before_startup(provider: Any) -> None:
    configuration = ConfigStub(provider, configured=True)

    with pytest.raises(PerceptionProviderError, match="must be either"):
        require_api_provider(configuration)
