"""Tests for configuration merger.

Tests all functionality including merge strategies, precedence rules,
conflict resolution, and inheritance support.
"""

from unittest.mock import Mock

import pytest

from backend.config.loader.merger import (
    ConfigSource,
    ConfigurationMerger,
    MergeError,
    MergeStrategy,
)


class TestMergeStrategy:
    """Test cases for MergeStrategy enum."""

    def test_merge_strategy_values(self):
        """Test merge strategy enum values."""
        assert MergeStrategy.OVERRIDE.value == "override"
        assert MergeStrategy.MERGE_DEEP.value == "merge_deep"
        assert MergeStrategy.MERGE_LIST.value == "merge_list"
        assert MergeStrategy.MERGE_APPEND.value == "merge_append"
        assert MergeStrategy.MERGE_PREPEND.value == "merge_prepend"
        assert MergeStrategy.FIRST_WINS.value == "first_wins"
        assert MergeStrategy.LAST_WINS.value == "last_wins"


class TestConfigSource:
    """Test cases for ConfigSource enum."""

    def test_config_source_precedence(self):
        """Test configuration source precedence values."""
        assert ConfigSource.DEFAULT.value == 1
        assert ConfigSource.FILE.value == 2
        assert ConfigSource.ENVIRONMENT.value == 3
        assert ConfigSource.CLI.value == 4
        assert ConfigSource.RUNTIME.value == 5


class TestConfigurationMerger:
    """Test cases for ConfigurationMerger class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.merger = ConfigurationMerger()

    def test_init_with_defaults(self):
        """Test ConfigurationMerger initialization with defaults."""
        assert self.merger.default_strategy == MergeStrategy.MERGE_DEEP
        assert self.merger.precedence_order == [
            ConfigSource.DEFAULT,
            ConfigSource.FILE,
            ConfigSource.ENVIRONMENT,
            ConfigSource.CLI,
            ConfigSource.RUNTIME,
        ]
        assert self.merger.custom_handlers == {}
        assert self.merger._merge_history == []

    def test_init_with_custom_params(self):
        """Test ConfigurationMerger initialization with custom parameters."""
        custom_precedence = [ConfigSource.FILE, ConfigSource.CLI]
        custom_handlers = {"special_key": Mock()}

        merger = ConfigurationMerger(
            default_strategy=MergeStrategy.OVERRIDE,
            precedence_order=custom_precedence,
            custom_handlers=custom_handlers,
        )

        assert merger.default_strategy == MergeStrategy.OVERRIDE
        assert merger.precedence_order == custom_precedence
        assert merger.custom_handlers == custom_handlers

    def test_merge_empty_configs(self):
        """Test merging empty configuration list."""
        result = self.merger.merge_configurations([])
        assert result == {}

    def test_merge_single_config(self):
        """Test merging single configuration."""
        config = {"key": "value"}
        result = self.merger.merge_configurations([config])
        assert result == config
        assert result is not config  # Should be a copy

    def test_merge_two_configs_override_strategy(self):
        """Test merging two configurations with override strategy."""
        base = {"key1": "value1", "key2": "value2"}
        override = {"key2": "new_value2", "key3": "value3"}

        merger = ConfigurationMerger(default_strategy=MergeStrategy.OVERRIDE)
        result = merger.merge_two(base, override)

        assert result["key1"] == "value1"
        assert result["key2"] == "new_value2"
        assert result["key3"] == "value3"

    def test_merge_two_configs_deep_merge_strategy(self):
        """Test merging two configurations with deep merge strategy."""
        base = {
            "app": {"name": "base", "version": "1.0"},
            "database": {"host": "localhost"},
        }
        override = {
            "app": {"name": "override", "debug": True},
            "cache": {"type": "redis"},
        }

        result = self.merger.merge_two(base, override)

        assert result["app"]["name"] == "override"
        assert result["app"]["version"] == "1.0"
        assert result["app"]["debug"] is True
        assert result["database"]["host"] == "localhost"
        assert result["cache"]["type"] == "redis"

    def test_apply_strategy_override(self):
        """Test applying override strategy."""
        result = self.merger._apply_strategy(
            "old", "new", "key", MergeStrategy.OVERRIDE
        )
        assert result == "new"

    def test_apply_strategy_merge_deep(self):
        """Test applying deep merge strategy."""
        base = {"a": 1, "b": {"c": 2}}
        override = {"b": {"d": 3}, "e": 4}

        result = self.merger._apply_strategy(
            base, override, "key", MergeStrategy.MERGE_DEEP
        )

        assert result["a"] == 1
        assert result["b"]["c"] == 2
        assert result["b"]["d"] == 3
        assert result["e"] == 4

    def test_apply_strategy_merge_list(self):
        """Test applying list merge strategy."""
        result = self.merger._apply_strategy(
            [1, 2], [3, 4], "key", MergeStrategy.MERGE_LIST
        )
        assert result == [1, 2, 3, 4]

    def test_apply_strategy_merge_append(self):
        """Test applying append merge strategy."""
        result = self.merger._apply_strategy(
            [1, 2], [3, 4], "key", MergeStrategy.MERGE_APPEND
        )
        assert result == [1, 2, 3, 4]

        result = self.merger._apply_strategy(
            [1, 2], 3, "key", MergeStrategy.MERGE_APPEND
        )
        assert result == [1, 2, 3]

    def test_apply_strategy_merge_prepend(self):
        """Test applying prepend merge strategy."""
        result = self.merger._apply_strategy(
            [1, 2], [3, 4], "key", MergeStrategy.MERGE_PREPEND
        )
        assert result == [3, 4, 1, 2]

        result = self.merger._apply_strategy(
            [1, 2], 3, "key", MergeStrategy.MERGE_PREPEND
        )
        assert result == [3, 1, 2]

    def test_apply_strategy_first_wins(self):
        """Test applying first wins strategy."""
        result = self.merger._apply_strategy(
            "first", "second", "key", MergeStrategy.FIRST_WINS
        )
        assert result == "first"

    def test_apply_strategy_last_wins(self):
        """Test applying last wins strategy."""
        result = self.merger._apply_strategy(
            "first", "second", "key", MergeStrategy.LAST_WINS
        )
        assert result == "second"

    def test_apply_strategy_unknown(self):
        """Test applying unknown strategy."""
        with pytest.raises(MergeError):
            self.merger._apply_strategy("first", "second", "key", "unknown_strategy")

    def test_merge_configurations_with_precedence(self):
        """Test merging configurations with proper precedence."""
        configs = [
            {"key": "file_value", "file_only": True},
            {"key": "env_value", "env_only": True},
            {"key": "cli_value", "cli_only": True},
        ]
        sources = [ConfigSource.FILE, ConfigSource.ENVIRONMENT, ConfigSource.CLI]

        result = self.merger.merge_configurations(configs, sources)

        # CLI should win due to highest precedence
        assert result["key"] == "cli_value"
        assert result["file_only"] is True
        assert result["env_only"] is True
        assert result["cli_only"] is True

    def test_merge_configurations_source_count_mismatch(self):
        """Test merging with mismatched config and source counts."""
        configs = [{"key": "value1"}, {"key": "value2"}]
        sources = [ConfigSource.FILE]  # Only one source for two configs

        with pytest.raises(MergeError):
            self.merger.merge_configurations(configs, sources)

    def test_merge_configurations_with_custom_strategies(self):
        """Test merging with key-specific strategies."""
        configs = [
            {"override_key": "value1", "list_key": [1, 2]},
            {"override_key": "value2", "list_key": [3, 4]},
        ]
        strategies = {
            "override_key": MergeStrategy.FIRST_WINS,
            "list_key": MergeStrategy.MERGE_LIST,
        }

        result = self.merger.merge_configurations(configs, strategies=strategies)

        assert result["override_key"] == "value1"  # First wins
        assert result["list_key"] == [1, 2, 3, 4]  # Lists merged

    def test_detect_conflicts(self):
        """Test conflict detection between configurations."""
        base = {"key1": "value1", "nested": {"key2": "value2"}}
        override = {"key1": "different", "nested": {"key2": "also_different"}}

        conflicts = self.merger._detect_conflicts(base, override)

        assert "key1" in conflicts
        assert conflicts["key1"] == "different"
        assert "nested" in conflicts
        assert conflicts["nested"]["key2"] == "also_different"

    def test_resolve_conflicts_default(self):
        """Test conflict resolution with default resolver."""
        configs = [
            {"key": "value1", "no_conflict": "safe"},
            {"key": "value2", "other_key": "other"},
        ]

        result = self.merger.resolve_conflicts(configs)

        assert result["key"] == "value2"  # Override wins
        assert result["no_conflict"] == "safe"
        assert result["other_key"] == "other"

    def test_resolve_conflicts_custom_resolver(self):
        """Test conflict resolution with custom resolver."""

        def custom_resolver(key, base_value, override_value, config_index):
            if key == "special_key":
                return f"{base_value}+{override_value}"
            return override_value

        configs = [
            {"special_key": "base", "normal_key": "value1"},
            {"special_key": "override", "normal_key": "value2"},
        ]

        result = self.merger.resolve_conflicts(configs, custom_resolver)

        assert result["special_key"] == "base+override"
        assert result["normal_key"] == "value2"

    def test_merge_with_inheritance(self):
        """Test merging configurations with inheritance."""
        configs = [
            {"base": True, "value": "base_value"},
            {"inherit": "parent", "value": "child_value", "child_only": True},
        ]

        result = self.merger.merge_with_inheritance(configs)

        # This is a simplified test - real inheritance would be more complex
        assert "base" in result
        assert "child_only" in result

    def test_validate_merge_success(self):
        """Test merge validation that succeeds."""
        configs = [{"key": "value1"}, {"key": "value2"}]

        def validator(config):
            return "key" in config

        assert self.merger.validate_merge(configs, validator) is True

    def test_validate_merge_failure(self):
        """Test merge validation that fails."""
        configs = [{"key": "value1"}, {"other": "value2"}]

        def validator(config):
            return "required_key" in config

        assert self.merger.validate_merge(configs, validator) is False

    def test_merge_history_tracking(self):
        """Test merge history tracking."""
        configs = [{"key1": "value1"}, {"key2": "value2"}]
        sources = [ConfigSource.FILE, ConfigSource.ENVIRONMENT]

        self.merger.merge_configurations(configs, sources)

        history = self.merger.get_merge_history()
        assert len(history) == 2
        assert history[0]["source"] == ConfigSource.FILE
        assert history[1]["source"] == ConfigSource.ENVIRONMENT

    def test_clear_history(self):
        """Test clearing merge history."""
        configs = [{"key": "value"}]
        self.merger.merge_configurations(configs)

        assert len(self.merger._merge_history) == 1

        self.merger.clear_history()
        assert len(self.merger._merge_history) == 0

    def test_deep_merge_dicts(self):
        """Test deep merging of dictionaries."""
        base = {
            "level1": {"level2": {"key1": "value1", "key2": "value2"}, "other": "value"}
        }
        override = {
            "level1": {
                "level2": {"key2": "new_value2", "key3": "value3"},
                "new_key": "new_value",
            }
        }

        result = self.merger._deep_merge_dicts(base, override)

        assert result["level1"]["level2"]["key1"] == "value1"
        assert result["level1"]["level2"]["key2"] == "new_value2"
        assert result["level1"]["level2"]["key3"] == "value3"
        assert result["level1"]["other"] == "value"
        assert result["level1"]["new_key"] == "new_value"

    def test_set_nested_value(self):
        """Test setting nested values."""
        config = {}

        self.merger._set_nested_value(config, "simple", "value")
        assert config["simple"] == "value"

        self.merger._set_nested_value(config, "nested.key", "nested_value")
        assert config["nested"]["key"] == "nested_value"

        self.merger._set_nested_value(config, "deep.nested.key", "deep_value")
        assert config["deep"]["nested"]["key"] == "deep_value"

    def test_custom_handlers(self):
        """Test custom merge handlers for specific keys."""

        def custom_handler(base_value, override_value):
            if base_value is None:
                return override_value
            return f"{base_value}|{override_value}"

        merger = ConfigurationMerger(custom_handlers={"special_key": custom_handler})

        configs = [
            {"special_key": "base", "normal_key": "value1"},
            {"special_key": "override", "normal_key": "value2"},
        ]

        result = merger.merge_configurations(configs)

        assert result["special_key"] == "base|override"
        assert result["normal_key"] == "value2"


class TestConfigurationMergerIntegration:
    """Integration tests for ConfigurationMerger."""

    def test_complex_multi_source_merge(self):
        """Test complex merge with multiple sources and strategies."""
        # Default configuration
        default_config = {
            "app": {
                "name": "billiards-trainer",
                "version": "1.0.0",
                "debug": False,
                "features": ["vision"],
            },
            "database": {"host": "localhost", "port": 5432, "ssl": False},
        }

        # File configuration
        file_config = {
            "app": {"debug": True, "features": ["projector"]},
            "database": {"host": "file-db-host"},
            "logging": {"level": "INFO"},
        }

        # Environment configuration
        env_config = {
            "app": {"features": ["ai_analysis"]},
            "database": {"host": "env-db-host", "ssl": True},
        }

        # CLI configuration
        cli_config = {"app": {"debug": False}, "database": {"port": 3306}}

        configs = [default_config, file_config, env_config, cli_config]
        sources = [
            ConfigSource.DEFAULT,
            ConfigSource.FILE,
            ConfigSource.ENVIRONMENT,
            ConfigSource.CLI,
        ]

        # Use custom strategies for features (merge lists)
        strategies = {"app.features": MergeStrategy.MERGE_LIST}

        merger = ConfigurationMerger()
        result = merger.merge_configurations(configs, sources, strategies)

        # Check final configuration
        assert result["app"]["name"] == "billiards-trainer"  # from default
        assert result["app"]["version"] == "1.0.0"  # from default
        assert result["app"]["debug"] is False  # CLI overrides all
        # Features should be merged (if strategy is properly applied to nested keys)

        assert result["database"]["host"] == "env-db-host"  # env overrides file
        assert result["database"]["port"] == 3306  # CLI overrides default
        assert result["database"]["ssl"] is True  # from env

        assert result["logging"]["level"] == "INFO"  # from file
