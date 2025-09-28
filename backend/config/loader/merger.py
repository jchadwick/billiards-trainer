"""Configuration merger with precedence support.

Handles merging configurations from multiple sources with proper precedence rules,
conflict resolution, and validation.
Implements FR-CFG-003, FR-CFG-005, and FR-CFG-020.
"""

import copy
import logging
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class MergeStrategy(Enum):
    """Configuration merge strategies."""

    OVERRIDE = "override"  # Higher precedence overrides lower
    MERGE_DEEP = "merge_deep"  # Deep merge dictionaries
    MERGE_LIST = "merge_list"  # Merge lists by concatenation
    MERGE_APPEND = "merge_append"  # Append to existing lists
    MERGE_PREPEND = "merge_prepend"  # Prepend to existing lists
    FIRST_WINS = "first_wins"  # First value wins
    LAST_WINS = "last_wins"  # Last value wins


class ConfigSource(Enum):
    """Configuration sources in order of precedence (lowest to highest)."""

    DEFAULT = 1
    FILE = 2
    ENVIRONMENT = 3
    CLI = 4
    RUNTIME = 5


class MergeError(Exception):
    """Exception raised during configuration merging."""

    pass


class ConfigurationMerger:
    """Configuration merger with precedence and strategy support.

    Supports:
    - Multiple merge strategies
    - Source precedence handling
    - Conflict detection and resolution
    - Validation during merge
    - Custom merge handlers
    """

    def __init__(
        self,
        default_strategy: MergeStrategy = MergeStrategy.MERGE_DEEP,
        precedence_order: Optional[list[ConfigSource]] = None,
        custom_handlers: Optional[dict[str, Callable]] = None,
    ):
        """Initialize the configuration merger.

        Args:
            default_strategy: Default merge strategy
            precedence_order: Custom precedence order (None for default)
            custom_handlers: Custom merge handlers for specific keys
        """
        self.default_strategy = default_strategy
        self.precedence_order = precedence_order or [
            ConfigSource.DEFAULT,
            ConfigSource.FILE,
            ConfigSource.ENVIRONMENT,
            ConfigSource.CLI,
            ConfigSource.RUNTIME,
        ]
        self.custom_handlers = custom_handlers or {}
        self._merge_history = []

    def merge_configurations(
        self,
        configs: list[dict[str, Any]],
        sources: Optional[list[ConfigSource]] = None,
        strategies: Optional[dict[str, MergeStrategy]] = None,
    ) -> dict[str, Any]:
        """Merge multiple configurations according to precedence and strategies.

        Args:
            configs: List of configuration dictionaries to merge
            sources: List of source types (must match configs length)
            strategies: Key-specific merge strategies

        Returns:
            Merged configuration dictionary

        Raises:
            MergeError: If merge fails
        """
        if not configs:
            return {}

        if len(configs) == 1:
            return copy.deepcopy(configs[0])

        # Set default sources if not provided
        if sources is None:
            sources = [ConfigSource.FILE] * len(configs)

        if len(sources) != len(configs):
            raise MergeError(
                f"Number of sources ({len(sources)}) must match configs ({len(configs)})"
            )

        # Sort configs by precedence
        config_pairs = list(zip(configs, sources))
        config_pairs.sort(key=lambda x: x[1].value)

        # Start with empty result
        result = {}

        # Merge configurations in precedence order
        for config, source in config_pairs:
            try:
                result = self._merge_single(result, config, source, strategies or {})
                self._merge_history.append(
                    {
                        "source": source,
                        "config_keys": list(config.keys()),
                        "merged_keys": list(result.keys()),
                    }
                )
            except Exception as e:
                raise MergeError(f"Failed to merge config from {source}: {e}")

        logger.info(
            f"Merged {len(configs)} configurations from sources: {[s.name for s in sources]}"
        )
        return result

    def merge_with_inheritance(
        self, configs: list[dict[str, Any]], inheritance_key: str = "inherit"
    ) -> dict[str, Any]:
        """Merge configurations with inheritance support.

        Implements FR-CFG-005: Configuration inheritance and overrides.

        Args:
            configs: List of configuration dictionaries
            inheritance_key: Key that specifies inheritance relationship

        Returns:
            Merged configuration with inheritance applied
        """
        # Build inheritance tree
        inheritance_tree = self._build_inheritance_tree(configs, inheritance_key)

        # Merge following inheritance order
        result = {}
        for config in inheritance_tree:
            result = self._merge_single(result, config, ConfigSource.FILE, {})

        return result

    def merge_two(
        self,
        base: dict[str, Any],
        override: dict[str, Any],
        strategy: Optional[MergeStrategy] = None,
    ) -> dict[str, Any]:
        """Merge two configuration dictionaries.

        Args:
            base: Base configuration
            override: Override configuration
            strategy: Merge strategy to use

        Returns:
            Merged configuration
        """
        strategy = strategy or self.default_strategy
        return self._apply_strategy(base, override, "", strategy)

    def resolve_conflicts(
        self,
        configs: list[dict[str, Any]],
        conflict_resolver: Optional[Callable] = None,
    ) -> dict[str, Any]:
        """Merge configurations with conflict detection and resolution.

        Args:
            configs: List of configurations to merge
            conflict_resolver: Custom conflict resolution function

        Returns:
            Merged configuration with conflicts resolved
        """
        if not configs:
            return {}

        result = copy.deepcopy(configs[0])
        conflicts = []

        for i, config in enumerate(configs[1:], 1):
            config_conflicts = self._detect_conflicts(result, config)
            if config_conflicts:
                conflicts.extend(
                    [
                        (i, key, result.get(key), value)
                        for key, value in config_conflicts.items()
                    ]
                )

                # Resolve conflicts
                if conflict_resolver:
                    for key, value in config_conflicts.items():
                        resolved_value = conflict_resolver(
                            key, result.get(key), value, i
                        )
                        self._set_nested_value(result, key, resolved_value)
                else:
                    # Default: override takes precedence
                    for key, value in config_conflicts.items():
                        self._set_nested_value(result, key, value)

            # Merge non-conflicting values
            result = self._merge_single(result, config, ConfigSource.FILE, {})

        if conflicts:
            logger.warning(f"Resolved {len(conflicts)} configuration conflicts")

        return result

    def validate_merge(
        self, configs: list[dict[str, Any]], validator: Callable[[dict[str, Any]], bool]
    ) -> bool:
        """Validate that a merge would produce a valid configuration.

        Args:
            configs: Configurations to merge
            validator: Validation function

        Returns:
            True if merge would be valid
        """
        try:
            merged = self.merge_configurations(configs)
            return validator(merged)
        except Exception as e:
            logger.error(f"Merge validation failed: {e}")
            return False

    def get_merge_history(self) -> list[dict[str, Any]]:
        """Get the history of merge operations.

        Returns:
            List of merge operation records
        """
        return copy.deepcopy(self._merge_history)

    def clear_history(self):
        """Clear the merge history."""
        self._merge_history.clear()

    def _merge_single(
        self,
        base: dict[str, Any],
        override: dict[str, Any],
        source: ConfigSource,
        strategies: dict[str, MergeStrategy],
    ) -> dict[str, Any]:
        """Merge a single configuration into the base."""
        result = copy.deepcopy(base)

        for key, value in override.items():
            # Check for custom handler
            if key in self.custom_handlers:
                try:
                    result[key] = self.custom_handlers[key](result.get(key), value)
                    continue
                except Exception as e:
                    logger.warning(f"Custom handler failed for {key}: {e}")

            # Use key-specific strategy or default
            strategy = strategies.get(key, self.default_strategy)

            # Apply merge strategy
            if key in result:
                result[key] = self._apply_strategy(result[key], value, key, strategy)
            else:
                result[key] = copy.deepcopy(value)

        return result

    def _apply_strategy(
        self, base_value: Any, override_value: Any, key: str, strategy: MergeStrategy
    ) -> Any:
        """Apply a specific merge strategy."""
        try:
            if strategy == MergeStrategy.OVERRIDE:
                return copy.deepcopy(override_value)

            elif strategy == MergeStrategy.MERGE_DEEP:
                if isinstance(base_value, dict) and isinstance(override_value, dict):
                    return self._deep_merge_dicts(base_value, override_value)
                else:
                    return copy.deepcopy(override_value)

            elif strategy == MergeStrategy.MERGE_LIST:
                if isinstance(base_value, list) and isinstance(override_value, list):
                    return base_value + override_value
                else:
                    return copy.deepcopy(override_value)

            elif strategy == MergeStrategy.MERGE_APPEND:
                if isinstance(base_value, list):
                    if isinstance(override_value, list):
                        return base_value + override_value
                    else:
                        return base_value + [override_value]
                else:
                    return copy.deepcopy(override_value)

            elif strategy == MergeStrategy.MERGE_PREPEND:
                if isinstance(base_value, list):
                    if isinstance(override_value, list):
                        return override_value + base_value
                    else:
                        return [override_value] + base_value
                else:
                    return copy.deepcopy(override_value)

            elif strategy == MergeStrategy.FIRST_WINS:
                return copy.deepcopy(base_value)

            elif strategy == MergeStrategy.LAST_WINS:
                return copy.deepcopy(override_value)

            else:
                raise MergeError(f"Unknown merge strategy: {strategy}")

        except Exception as e:
            logger.warning(f"Strategy {strategy} failed for key {key}: {e}")
            return copy.deepcopy(override_value)

    def _deep_merge_dicts(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge two dictionaries."""
        result = copy.deepcopy(base)

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = copy.deepcopy(value)

        return result

    def _detect_conflicts(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Detect conflicting keys between two configurations."""
        conflicts = {}

        for key, value in override.items():
            if key in base:
                base_value = base[key]
                if base_value != value:
                    # Check if both are dicts (not a conflict if we can merge)
                    if isinstance(base_value, dict) and isinstance(value, dict):
                        nested_conflicts = self._detect_conflicts(base_value, value)
                        if nested_conflicts:
                            conflicts[key] = nested_conflicts
                    else:
                        conflicts[key] = value

        return conflicts

    def _build_inheritance_tree(
        self, configs: list[dict[str, Any]], inheritance_key: str
    ) -> list[dict[str, Any]]:
        """Build inheritance tree from configurations."""
        # This is a simplified implementation
        # In a full implementation, you'd handle complex inheritance chains
        inheritance_order = []

        for config in configs:
            if inheritance_key not in config:
                inheritance_order.append(config)

        # Add configs with inheritance after their parents
        # (simplified - real implementation would handle complex trees)
        for config in configs:
            if inheritance_key in config:
                inheritance_order.append(config)

        return inheritance_order

    def _set_nested_value(self, config: dict[str, Any], key: str, value: Any) -> None:
        """Set a nested value in the configuration dictionary."""
        if "." in key:
            keys = key.split(".")
            current = config

            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            current[keys[-1]] = value
        else:
            config[key] = value
