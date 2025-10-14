"""Caching utilities for core calculations."""

from typing import Any, Dict, List, Optional

from ...config.manager import ConfigurationModule

# Global configuration instance (lazy loaded)
_config: Optional[ConfigurationModule] = None


def _get_config() -> ConfigurationModule:
    """Get or create configuration instance."""
    global _config
    if _config is None:
        from backend.config import config_manager

        _config = config_manager
    return _config


class CalculationCache:
    """Cache for expensive calculations."""

    def __init__(self, max_size: int | None = None):
        """Initialize calculation cache.

        Args:
            max_size: Maximum number of items to cache. If None, uses config value.
        """
        if max_size is None:
            config = _get_config()
            max_size = config.get("core.utils.cache.default_max_size", default=1000)

        self.max_size = max_size
        self._cache: dict[str, Any] = {}
        self._access_order: list[str] = []

    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Cache a value."""
        if key in self._cache:
            # Update existing
            self._cache[key] = value
            self._access_order.remove(key)
            self._access_order.append(key)
        else:
            # Add new
            if len(self._cache) >= self.max_size:
                # Remove least recently used
                oldest = self._access_order.pop(0)
                del self._cache[oldest]

            self._cache[key] = value
            self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
        self._access_order.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "usage": len(self._cache) / self.max_size if self.max_size > 0 else 0,
        }


# Alias for compatibility
CacheManager = CalculationCache
