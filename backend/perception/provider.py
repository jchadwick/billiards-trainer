"""Fail-closed selection of the perception runtime used by the API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from backend.config import load_schema_reference


class ConfigurationReader(Protocol):
    """Configuration surface needed to read the selected provider."""

    def get(self, key: str, default: Any = None) -> Any: ...


class PerceptionProvider(str, Enum):
    """Perception engines that may be selected by validated configuration."""

    PYTHON = "python"
    RUST = "rust"


class PerceptionProviderError(RuntimeError):
    """A configured provider cannot safely serve live API observations."""


def require_api_provider(
    configuration: ConfigurationReader,
) -> PerceptionProvider:
    """Resolve the selected mode before starting any perception path.

    Python owns the current live mode. Rust is supervised only in its explicit
    recorded-replay mode; Rust/live remains unsupported until capture ownership
    and platform qualification are complete. No selection silently falls back.
    """
    configuration_defaults = load_schema_reference()
    perception_defaults = configuration_defaults.get("perception")
    if not isinstance(perception_defaults, dict):
        raise PerceptionProviderError(
            "bundled configuration defaults must define perception.provider"
        )
    default_provider = perception_defaults.get("provider")
    if not isinstance(default_provider, str):
        raise PerceptionProviderError(
            "bundled configuration default perception.provider must be a string"
        )

    configured_value = configuration.get("perception.provider", default_provider)
    if not isinstance(configured_value, str):
        raise PerceptionProviderError(
            "perception.provider must be either 'python' or 'rust'"
        )
    try:
        provider = PerceptionProvider(configured_value)
    except ValueError as error:
        raise PerceptionProviderError(
            "perception.provider must be either 'python' or 'rust'"
        ) from error

    default_mode = perception_defaults.get("mode")
    if not isinstance(default_mode, str):
        raise PerceptionProviderError(
            "bundled configuration default perception.mode must be a string"
        )
    configured_mode = configuration.get("perception.mode", default_mode)
    if configured_mode not in {"live", "recorded_replay"}:
        raise PerceptionProviderError(
            "perception.mode must be either 'live' or 'recorded_replay'"
        )
    if provider is PerceptionProvider.PYTHON and configured_mode != "live":
        raise PerceptionProviderError(
            "the Python provider supports only perception.mode='live'"
        )
    if provider is PerceptionProvider.RUST and configured_mode != "recorded_replay":
        raise PerceptionProviderError(
            "the Rust perception provider currently supports only explicit "
            "recorded-input replay. Set perception.mode to 'recorded_replay'; "
            "live camera ownership and provider qualification remain incomplete."
        )
    return provider
