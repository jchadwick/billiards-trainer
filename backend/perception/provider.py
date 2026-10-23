"""Fail-closed selection of the perception runtime used by the API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from backend.config import load_schema_reference


class ConfigurationReader(Protocol):
    """Configuration surface needed to read the selected provider."""

    def get(self, key: str, default: Any = None) -> Any: ...


class PerceptionProvider(str, Enum):
    """The only perception engine accepted by the compatibility API."""

    RUST = "rust"


class PerceptionProviderError(RuntimeError):
    """A configured provider cannot safely serve live API observations."""


def require_api_provider(
    configuration: ConfigurationReader,
) -> PerceptionProvider:
    """Resolve the only supported perception provider before API startup.

    The Python compatibility API may supervise Rust perception for contract
    tests, but it may never start the retired Python perception pipeline.
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
        raise PerceptionProviderError("perception.provider must be 'rust'")
    try:
        provider = PerceptionProvider(configured_value)
    except ValueError as error:
        raise PerceptionProviderError("perception.provider must be 'rust'") from error

    return provider
