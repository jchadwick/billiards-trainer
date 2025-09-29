"""Configuration diff utilities."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """Types of configuration changes."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class ConfigChange:
    """Represents a single configuration change."""

    path: str
    change_type: ChangeType
    old_value: Any = None
    new_value: Any = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class DiffSummary:
    """Summary of configuration differences."""

    added_count: int = 0
    removed_count: int = 0
    modified_count: int = 0
    unchanged_count: int = 0
    total_changes: int = 0

    def __post_init__(self):
        self.total_changes = self.added_count + self.removed_count + self.modified_count


class ConfigDiffer:
    """Configuration difference calculator with nested support and change tracking."""

    def __init__(self, ignore_unchanged: bool = True, max_depth: int = 10):
        """Initialize configuration differ.

        Args:
            ignore_unchanged: If True, don't include unchanged values in diff
            max_depth: Maximum depth for nested comparison to prevent infinite recursion
        """
        self.ignore_unchanged = ignore_unchanged
        self.max_depth = max_depth
        self._changes: list[ConfigChange] = []

    def diff(self, old_config: dict, new_config: dict, context_path: str = "") -> dict:
        """Calculate differences between two configuration dictionaries.

        Args:
            old_config: Original configuration dictionary
            new_config: Updated configuration dictionary
            context_path: Path context for nested calls (internal use)

        Returns:
            Dictionary containing diff information with structure:
            {
                'summary': DiffSummary,
                'changes': List[ConfigChange],
                'details': {
                    'added': dict,
                    'removed': dict,
                    'modified': dict,
                    'unchanged': dict (if ignore_unchanged=False)
                }
            }
        """
        try:
            self._changes.clear()

            if not isinstance(old_config, dict) or not isinstance(new_config, dict):
                raise ValueError("Both configurations must be dictionaries")

            # Calculate differences
            diff_details = self._calculate_diff(
                old_config, new_config, context_path, depth=0
            )

            # Generate summary
            summary = self._generate_summary()

            result = {
                "summary": summary,
                "changes": self._changes.copy(),
                "details": diff_details,
                "timestamp": datetime.now(),
                "old_config_keys": set(old_config.keys()),
                "new_config_keys": set(new_config.keys()),
            }

            logger.info(
                f"Configuration diff completed: {summary.total_changes} total changes"
            )
            return result

        except Exception as e:
            logger.error(f"Error calculating configuration diff: {str(e)}")
            raise

    def _calculate_diff(
        self, old_config: dict, new_config: dict, context_path: str = "", depth: int = 0
    ) -> dict:
        """Calculate detailed differences between configurations.

        Args:
            old_config: Original configuration
            new_config: Updated configuration
            context_path: Current path context
            depth: Current recursion depth

        Returns:
            Dictionary with categorized differences
        """
        if depth > self.max_depth:
            logger.warning(
                f"Maximum diff depth ({self.max_depth}) reached at path: {context_path}"
            )
            return {"added": {}, "removed": {}, "modified": {}, "unchanged": {}}

        diff_details = {"added": {}, "removed": {}, "modified": {}, "unchanged": {}}

        # Get all unique keys from both configurations
        old_keys = set(old_config.keys())
        new_keys = set(new_config.keys())
        all_keys = old_keys | new_keys

        for key in all_keys:
            current_path = f"{context_path}.{key}" if context_path else key

            if key in old_keys and key in new_keys:
                # Key exists in both configurations
                old_value = old_config[key]
                new_value = new_config[key]

                if self._values_equal(old_value, new_value):
                    # Values are the same
                    if not self.ignore_unchanged:
                        diff_details["unchanged"][key] = new_value
                        self._add_change(
                            current_path, ChangeType.UNCHANGED, old_value, new_value
                        )

                elif isinstance(old_value, dict) and isinstance(new_value, dict):
                    # Both values are dictionaries - recurse
                    nested_diff = self._calculate_diff(
                        old_value, new_value, current_path, depth + 1
                    )

                    # Only include in modified if there are actual changes
                    if (
                        nested_diff["added"]
                        or nested_diff["removed"]
                        or nested_diff["modified"]
                        or (not self.ignore_unchanged and nested_diff["unchanged"])
                    ):
                        diff_details["modified"][key] = {
                            "old_value": old_value,
                            "new_value": new_value,
                            "nested_changes": nested_diff,
                        }
                        self._add_change(
                            current_path, ChangeType.MODIFIED, old_value, new_value
                        )

                else:
                    # Values are different and not both dictionaries
                    diff_details["modified"][key] = {
                        "old_value": old_value,
                        "new_value": new_value,
                    }
                    self._add_change(
                        current_path, ChangeType.MODIFIED, old_value, new_value
                    )

            elif key in new_keys:
                # Key was added
                new_value = new_config[key]
                diff_details["added"][key] = new_value
                self._add_change(current_path, ChangeType.ADDED, None, new_value)

            elif key in old_keys:
                # Key was removed
                old_value = old_config[key]
                diff_details["removed"][key] = old_value
                self._add_change(current_path, ChangeType.REMOVED, old_value, None)

        return diff_details

    def _values_equal(self, value1: Any, value2: Any) -> bool:
        """Check if two values are equal, handling special cases.

        Args:
            value1: First value
            value2: Second value

        Returns:
            True if values are considered equal
        """
        try:
            # Handle None values
            if value1 is None and value2 is None:
                return True
            if value1 is None or value2 is None:
                return False

            # Handle different types
            if type(value1) != type(value2):
                # Try to handle numeric comparisons
                if isinstance(value1, (int, float)) and isinstance(
                    value2, (int, float)
                ):
                    return (
                        abs(value1 - value2) < 1e-9
                    )  # Handle floating point precision
                return False

            # Handle dictionaries recursively (shallow comparison for this method)
            if isinstance(value1, dict) and isinstance(value2, dict):
                if set(value1.keys()) != set(value2.keys()):
                    return False
                # For shallow comparison, check if all values are equal
                for key in value1:
                    if not self._values_equal(value1[key], value2[key]):
                        return False
                return True

            # Handle lists
            if isinstance(value1, list) and isinstance(value2, list):
                if len(value1) != len(value2):
                    return False
                return all(self._values_equal(v1, v2) for v1, v2 in zip(value1, value2))

            # Handle sets
            if isinstance(value1, set) and isinstance(value2, set):
                return value1 == value2

            # Handle Path objects
            if isinstance(value1, Path) and isinstance(value2, Path):
                return str(value1) == str(value2)

            # Default comparison
            return value1 == value2

        except Exception as e:
            logger.warning(f"Error comparing values: {str(e)}")
            return False

    def _add_change(
        self, path: str, change_type: ChangeType, old_value: Any, new_value: Any
    ) -> None:
        """Add a change to the changes list.

        Args:
            path: Configuration path
            change_type: Type of change
            old_value: Original value
            new_value: New value
        """
        change = ConfigChange(
            path=path, change_type=change_type, old_value=old_value, new_value=new_value
        )
        self._changes.append(change)

    def _generate_summary(self) -> DiffSummary:
        """Generate a summary of all changes.

        Returns:
            DiffSummary object with change counts
        """
        summary = DiffSummary()

        for change in self._changes:
            if change.change_type == ChangeType.ADDED:
                summary.added_count += 1
            elif change.change_type == ChangeType.REMOVED:
                summary.removed_count += 1
            elif change.change_type == ChangeType.MODIFIED:
                summary.modified_count += 1
            elif change.change_type == ChangeType.UNCHANGED:
                summary.unchanged_count += 1

        summary.total_changes = (
            summary.added_count + summary.removed_count + summary.modified_count
        )
        return summary

    def get_changes_by_type(self, change_type: ChangeType) -> list[ConfigChange]:
        """Get all changes of a specific type.

        Args:
            change_type: Type of changes to filter

        Returns:
            List of changes of the specified type
        """
        return [change for change in self._changes if change.change_type == change_type]

    def get_changes_by_path(self, path_pattern: str) -> list[ConfigChange]:
        """Get all changes matching a path pattern.

        Args:
            path_pattern: Path pattern to match (supports wildcards)

        Returns:
            List of changes matching the pattern
        """
        import fnmatch

        return [
            change
            for change in self._changes
            if fnmatch.fnmatch(change.path, path_pattern)
        ]

    def format_diff_report(
        self, diff_result: dict, include_unchanged: bool = False
    ) -> str:
        """Format diff result as a human-readable report.

        Args:
            diff_result: Result from diff() method
            include_unchanged: Whether to include unchanged values

        Returns:
            Formatted report string
        """
        try:
            summary = diff_result["summary"]
            changes = diff_result["changes"]

            report_lines = []
            report_lines.append("Configuration Diff Report")
            report_lines.append("=" * 40)
            report_lines.append(f"Timestamp: {diff_result['timestamp']}")
            report_lines.append("")

            # Summary
            report_lines.append("Summary:")
            report_lines.append(f"  Added: {summary.added_count}")
            report_lines.append(f"  Removed: {summary.removed_count}")
            report_lines.append(f"  Modified: {summary.modified_count}")
            if include_unchanged:
                report_lines.append(f"  Unchanged: {summary.unchanged_count}")
            report_lines.append(f"  Total Changes: {summary.total_changes}")
            report_lines.append("")

            # Detailed changes
            if changes:
                report_lines.append("Detailed Changes:")
                report_lines.append("-" * 20)

                for change in changes:
                    if (
                        change.change_type == ChangeType.UNCHANGED
                        and not include_unchanged
                    ):
                        continue

                    symbol = {
                        ChangeType.ADDED: "+",
                        ChangeType.REMOVED: "-",
                        ChangeType.MODIFIED: "~",
                        ChangeType.UNCHANGED: "=",
                    }.get(change.change_type, "?")

                    report_lines.append(f"{symbol} {change.path}")

                    if change.change_type == ChangeType.ADDED:
                        report_lines.append(
                            f"    Added: {self._format_value(change.new_value)}"
                        )
                    elif change.change_type == ChangeType.REMOVED:
                        report_lines.append(
                            f"    Removed: {self._format_value(change.old_value)}"
                        )
                    elif change.change_type == ChangeType.MODIFIED:
                        report_lines.append(
                            f"    Old: {self._format_value(change.old_value)}"
                        )
                        report_lines.append(
                            f"    New: {self._format_value(change.new_value)}"
                        )
                    elif (
                        change.change_type == ChangeType.UNCHANGED and include_unchanged
                    ):
                        report_lines.append(
                            f"    Value: {self._format_value(change.new_value)}"
                        )

                    report_lines.append("")

            return "\n".join(report_lines)

        except Exception as e:
            logger.error(f"Error formatting diff report: {str(e)}")
            return f"Error generating report: {str(e)}"

    def _format_value(self, value: Any, max_length: int = 100) -> str:
        """Format a value for display in reports.

        Args:
            value: Value to format
            max_length: Maximum length of formatted string

        Returns:
            Formatted string representation
        """
        try:
            if value is None:
                return "None"

            if isinstance(value, (str, int, float, bool)):
                result = str(value)
            elif isinstance(value, (list, tuple)):
                if len(value) <= 3:
                    result = str(value)
                else:
                    result = f"[{len(value)} items: {value[0]}, {value[1]}, ...]"
            elif isinstance(value, dict):
                if len(value) <= 2:
                    result = str(value)
                else:
                    keys = list(value.keys())[:2]
                    result = f"{{{len(value)} keys: {keys[0]}, {keys[1]}, ...}}"
            elif isinstance(value, Path):
                result = str(value)
            else:
                result = f"{type(value).__name__}({str(value)})"

            # Truncate if too long
            if len(result) > max_length:
                result = result[: max_length - 3] + "..."

            return result

        except Exception:
            return f"{type(value).__name__}(unprintable)"

    def export_diff(
        self, diff_result: dict, output_path: Path, format: str = "json"
    ) -> None:
        """Export diff result to file.

        Args:
            diff_result: Result from diff() method
            output_path: Path to save the diff
            format: Output format ('json', 'yaml', 'txt')
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == "json":
                # Convert datetime and other non-serializable objects
                serializable_result = self._make_serializable(diff_result)
                with output_path.open("w") as f:
                    json.dump(serializable_result, f, indent=2)

            elif format.lower() == "txt":
                report = self.format_diff_report(diff_result, include_unchanged=True)
                with output_path.open("w") as f:
                    f.write(report)

            elif format.lower() == "yaml":
                try:
                    import yaml

                    serializable_result = self._make_serializable(diff_result)
                    with output_path.open("w") as f:
                        yaml.dump(serializable_result, f, default_flow_style=False)
                except ImportError:
                    logger.warning("PyYAML not available, falling back to JSON")
                    self.export_diff(
                        diff_result, output_path.with_suffix(".json"), "json"
                    )

            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Diff exported to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting diff: {str(e)}")
            raise

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, ConfigChange):
            return {
                "path": obj.path,
                "change_type": obj.change_type.value,
                "old_value": self._make_serializable(obj.old_value),
                "new_value": self._make_serializable(obj.new_value),
                "timestamp": obj.timestamp.isoformat() if obj.timestamp else None,
            }
        elif isinstance(obj, DiffSummary):
            return {
                "added_count": obj.added_count,
                "removed_count": obj.removed_count,
                "modified_count": obj.modified_count,
                "unchanged_count": obj.unchanged_count,
                "total_changes": obj.total_changes,
            }
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj

    def compare_versions(self, config_versions: list[tuple[str, dict]]) -> dict:
        """Compare multiple configuration versions.

        Args:
            config_versions: List of (version_name, config_dict) tuples

        Returns:
            Dictionary with version comparisons
        """
        if len(config_versions) < 2:
            raise ValueError("At least 2 versions required for comparison")

        comparisons = {}

        for i in range(len(config_versions) - 1):
            version1_name, version1_config = config_versions[i]
            version2_name, version2_config = config_versions[i + 1]

            comparison_key = f"{version1_name} -> {version2_name}"
            comparisons[comparison_key] = self.diff(version1_config, version2_config)

        return comparisons
