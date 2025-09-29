"""Tests for configuration differ."""

import pytest
from pathlib import Path
from datetime import datetime
import json
import tempfile

from backend.config.utils.differ import ConfigDiffer, ChangeType, ConfigChange, DiffSummary


class TestConfigDiffer:
    """Test cases for ConfigDiffer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.differ = ConfigDiffer()
        self.differ_with_unchanged = ConfigDiffer(ignore_unchanged=False)

    def test_init(self):
        """Test ConfigDiffer initialization."""
        assert self.differ.ignore_unchanged
        assert self.differ.max_depth == 10
        assert self.differ._changes == []

    def test_basic_diff_no_changes(self):
        """Test diff with identical configurations."""
        config1 = {"name": "test", "value": 42}
        config2 = {"name": "test", "value": 42}

        result = self.differ.diff(config1, config2)

        assert result['summary'].total_changes == 0
        assert result['summary'].added_count == 0
        assert result['summary'].removed_count == 0
        assert result['summary'].modified_count == 0
        assert len(result['changes']) == 0

    def test_basic_diff_with_changes(self):
        """Test diff with basic changes."""
        config1 = {"name": "test", "value": 42}
        config2 = {"name": "test", "value": 100, "new_field": "added"}

        result = self.differ.diff(config1, config2)

        assert result['summary'].total_changes == 2
        assert result['summary'].added_count == 1
        assert result['summary'].modified_count == 1

        # Check changes
        changes = result['changes']
        assert len(changes) == 2

        # Find specific changes
        modified_change = next(c for c in changes if c.change_type == ChangeType.MODIFIED)
        added_change = next(c for c in changes if c.change_type == ChangeType.ADDED)

        assert modified_change.path == "value"
        assert modified_change.old_value == 42
        assert modified_change.new_value == 100

        assert added_change.path == "new_field"
        assert added_change.new_value == "added"

    def test_removed_fields(self):
        """Test diff with removed fields."""
        config1 = {"name": "test", "value": 42, "old_field": "removed"}
        config2 = {"name": "test", "value": 42}

        result = self.differ.diff(config1, config2)

        assert result['summary'].total_changes == 1
        assert result['summary'].removed_count == 1

        removed_change = result['changes'][0]
        assert removed_change.change_type == ChangeType.REMOVED
        assert removed_change.path == "old_field"
        assert removed_change.old_value == "removed"

    def test_nested_dict_changes(self):
        """Test diff with nested dictionary changes."""
        config1 = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "settings": {
                    "timeout": 30,
                    "pool_size": 10
                }
            }
        }

        config2 = {
            "database": {
                "host": "production.db",
                "port": 5432,
                "settings": {
                    "timeout": 60,
                    "pool_size": 10,
                    "ssl": True
                }
            }
        }

        result = self.differ.diff(config1, config2)

        assert result['summary'].total_changes >= 2  # host and timeout changed, ssl added

        # Check that nested paths are properly constructed
        changes = result['changes']
        paths = [change.path for change in changes]

        assert "database.host" in paths
        assert "database.settings.timeout" in paths
        assert "database.settings.ssl" in paths

    def test_different_data_types(self):
        """Test diff with different data types."""
        config1 = {
            "string_field": "hello",
            "int_field": 42,
            "float_field": 3.14,
            "bool_field": True,
            "list_field": [1, 2, 3],
            "dict_field": {"nested": "value"}
        }

        config2 = {
            "string_field": "world",
            "int_field": 100,
            "float_field": 2.71,
            "bool_field": False,
            "list_field": [4, 5, 6],
            "dict_field": {"nested": "changed"}
        }

        result = self.differ.diff(config1, config2)

        assert result['summary'].total_changes == 6

        # All should be modifications
        for change in result['changes']:
            assert change.change_type == ChangeType.MODIFIED

    def test_list_comparison(self):
        """Test comparison of list values."""
        config1 = {"items": [1, 2, 3]}
        config2 = {"items": [1, 2, 4]}  # Different last element

        result = self.differ.diff(config1, config2)

        assert result['summary'].total_changes == 1
        assert result['changes'][0].change_type == ChangeType.MODIFIED
        assert result['changes'][0].path == "items"

    def test_none_values(self):
        """Test handling of None values."""
        config1 = {"field1": None, "field2": "value"}
        config2 = {"field1": "now_has_value", "field2": None}

        result = self.differ.diff(config1, config2)

        assert result['summary'].total_changes == 2

        for change in result['changes']:
            assert change.change_type == ChangeType.MODIFIED

    def test_path_objects(self):
        """Test handling of Path objects."""
        config1 = {"path": Path("/old/path")}
        config2 = {"path": Path("/new/path")}

        result = self.differ.diff(config1, config2)

        assert result['summary'].total_changes == 1
        assert result['changes'][0].change_type == ChangeType.MODIFIED

    def test_floating_point_precision(self):
        """Test floating point precision handling."""
        config1 = {"value": 1.0000000001}
        config2 = {"value": 1.0000000002}

        result = self.differ.diff(config1, config2)

        # Should be considered equal due to floating point precision
        assert result['summary'].total_changes == 0

    def test_max_depth_limit(self):
        """Test maximum depth limitation."""
        # Create deeply nested structure
        deep_config1 = {"level": {}}
        deep_config2 = {"level": {}}

        current1 = deep_config1["level"]
        current2 = deep_config2["level"]

        for i in range(15):  # Exceed max_depth of 10
            current1[f"level_{i}"] = {}
            current2[f"level_{i}"] = {}
            current1 = current1[f"level_{i}"]
            current2 = current2[f"level_{i}"]

        # Add difference at the deepest level
        current1["deep_field"] = "old_value"
        current2["deep_field"] = "new_value"

        result = self.differ.diff(deep_config1, deep_config2)

        # Should handle gracefully without crashing
        assert isinstance(result, dict)
        assert 'summary' in result

    def test_ignore_unchanged_setting(self):
        """Test ignore_unchanged setting."""
        config1 = {"unchanged": "same", "changed": "old"}
        config2 = {"unchanged": "same", "changed": "new"}

        # With ignore_unchanged=True (default)
        result1 = self.differ.diff(config1, config2)
        assert 'unchanged' not in result1['details']['unchanged']

        # With ignore_unchanged=False
        result2 = self.differ_with_unchanged.diff(config1, config2)
        assert 'unchanged' in result2['details']['unchanged']

    def test_get_changes_by_type(self):
        """Test filtering changes by type."""
        config1 = {"old_field": "value", "modified_field": "old"}
        config2 = {"modified_field": "new", "new_field": "added"}

        result = self.differ.diff(config1, config2)

        added_changes = self.differ.get_changes_by_type(ChangeType.ADDED)
        removed_changes = self.differ.get_changes_by_type(ChangeType.REMOVED)
        modified_changes = self.differ.get_changes_by_type(ChangeType.MODIFIED)

        assert len(added_changes) == 1
        assert len(removed_changes) == 1
        assert len(modified_changes) == 1

        assert added_changes[0].path == "new_field"
        assert removed_changes[0].path == "old_field"
        assert modified_changes[0].path == "modified_field"

    def test_get_changes_by_path(self):
        """Test filtering changes by path pattern."""
        config1 = {
            "database": {"host": "old", "port": 5432},
            "cache": {"host": "old", "timeout": 30}
        }
        config2 = {
            "database": {"host": "new", "port": 5432},
            "cache": {"host": "new", "timeout": 60}
        }

        result = self.differ.diff(config1, config2)

        # Find all host changes
        host_changes = self.differ.get_changes_by_path("*.host")
        assert len(host_changes) == 2

        # Find database changes
        db_changes = self.differ.get_changes_by_path("database.*")
        assert len(db_changes) == 1

    def test_format_diff_report(self):
        """Test diff report formatting."""
        config1 = {"name": "old", "value": 100}
        config2 = {"name": "new", "count": 200}

        result = self.differ.diff(config1, config2)
        report = self.differ.format_diff_report(result)

        assert "Configuration Diff Report" in report
        assert "Added: 1" in report
        assert "Removed: 1" in report
        assert "Modified: 1" in report
        assert "+ count" in report
        assert "- value" in report
        assert "~ name" in report

    def test_format_diff_report_with_unchanged(self):
        """Test diff report formatting including unchanged values."""
        config1 = {"unchanged": "same", "changed": "old"}
        config2 = {"unchanged": "same", "changed": "new"}

        result = self.differ_with_unchanged.diff(config1, config2)
        report = self.differ_with_unchanged.format_diff_report(result, include_unchanged=True)

        assert "= unchanged" in report
        assert "~ changed" in report

    def test_export_diff_json(self):
        """Test exporting diff to JSON file."""
        config1 = {"name": "test", "value": 42}
        config2 = {"name": "test", "value": 100}

        result = self.differ.diff(config1, config2)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)

        try:
            self.differ.export_diff(result, output_path, format="json")
            assert output_path.exists()

            # Verify content
            with output_path.open('r') as f:
                exported_data = json.load(f)

            assert 'summary' in exported_data
            assert 'changes' in exported_data
            assert exported_data['summary']['total_changes'] == 1

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_export_diff_txt(self):
        """Test exporting diff to text file."""
        config1 = {"name": "test", "value": 42}
        config2 = {"name": "test", "value": 100}

        result = self.differ.diff(config1, config2)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_path = Path(f.name)

        try:
            self.differ.export_diff(result, output_path, format="txt")
            assert output_path.exists()

            # Verify content
            with output_path.open('r') as f:
                content = f.read()

            assert "Configuration Diff Report" in content
            assert "~ name" in content or "Modified: 1" in content

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_compare_versions(self):
        """Test comparing multiple configuration versions."""
        version1 = ("v1.0", {"setting": "old", "feature": False})
        version2 = ("v1.1", {"setting": "updated", "feature": False, "new_option": True})
        version3 = ("v1.2", {"setting": "latest", "feature": True, "new_option": True})

        versions = [version1, version2, version3]
        comparisons = self.differ.compare_versions(versions)

        assert "v1.0 -> v1.1" in comparisons
        assert "v1.1 -> v1.2" in comparisons

        # Check first comparison
        first_diff = comparisons["v1.0 -> v1.1"]
        assert first_diff['summary'].total_changes >= 1  # setting changed, new_option added

        # Check second comparison
        second_diff = comparisons["v1.1 -> v1.2"]
        assert second_diff['summary'].total_changes >= 2  # setting and feature changed

    def test_compare_versions_insufficient_data(self):
        """Test compare_versions with insufficient data."""
        with pytest.raises(ValueError, match="At least 2 versions required"):
            self.differ.compare_versions([("v1.0", {})])

    def test_invalid_input_types(self):
        """Test handling of invalid input types."""
        with pytest.raises(ValueError, match="must be dictionaries"):
            self.differ.diff("not_a_dict", {})

        with pytest.raises(ValueError, match="must be dictionaries"):
            self.differ.diff({}, "not_a_dict")

    def test_change_timestamp(self):
        """Test that changes have timestamps."""
        config1 = {"field": "old"}
        config2 = {"field": "new"}

        result = self.differ.diff(config1, config2)

        for change in result['changes']:
            assert isinstance(change.timestamp, datetime)

    def test_diff_summary_calculations(self):
        """Test diff summary calculations."""
        summary = DiffSummary(added_count=2, removed_count=1, modified_count=3, unchanged_count=5)

        assert summary.total_changes == 6  # added + removed + modified
        assert summary.unchanged_count == 5

    def test_config_change_dataclass(self):
        """Test ConfigChange dataclass functionality."""
        change = ConfigChange(
            path="test.field",
            change_type=ChangeType.MODIFIED,
            old_value="old",
            new_value="new"
        )

        assert change.path == "test.field"
        assert change.change_type == ChangeType.MODIFIED
        assert change.old_value == "old"
        assert change.new_value == "new"
        assert isinstance(change.timestamp, datetime)

    def test_value_equality_edge_cases(self):
        """Test value equality checking edge cases."""
        # Test recursive structures (should not crash)
        recursive1 = {"key": None}
        recursive1["key"] = recursive1

        recursive2 = {"key": None}
        recursive2["key"] = recursive2

        # Should handle gracefully
        result = self.differ._values_equal(recursive1, recursive2)
        assert isinstance(result, bool)

        # Test with sets
        set1 = {1, 2, 3}
        set2 = {3, 2, 1}  # Same elements, different order
        assert self.differ._values_equal(set1, set2)

        set3 = {1, 2, 4}
        assert not self.differ._values_equal(set1, set3)

    def test_format_value_edge_cases(self):
        """Test value formatting edge cases."""
        # Very long string
        long_string = "a" * 200
        formatted = self.differ._format_value(long_string)
        assert len(formatted) <= 103  # 100 + "..."

        # Large list
        large_list = list(range(100))
        formatted = self.differ._format_value(large_list)
        assert "items:" in formatted

        # Large dict
        large_dict = {f"key_{i}": i for i in range(100)}
        formatted = self.differ._format_value(large_dict)
        assert "keys:" in formatted

        # Unprintable object
        class UnprintableClass:
            def __str__(self):
                raise Exception("Cannot print")

        unprintable = UnprintableClass()
        formatted = self.differ._format_value(unprintable)
        assert "unprintable" in formatted.lower()

    def test_serialization_edge_cases(self):
        """Test serialization of complex objects."""
        # Create a diff result with various object types
        config1 = {"path": Path("/test"), "date": datetime.now()}
        config2 = {"path": Path("/new"), "date": datetime.now()}

        result = self.differ.diff(config1, config2)
        serializable = self.differ._make_serializable(result)

        # Should not contain any non-serializable objects
        json.dumps(serializable)  # Should not raise exception