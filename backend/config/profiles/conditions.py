"""Configuration profile auto-activation conditions.

Provides condition evaluation for automatic profile selection based on:
- System hardware characteristics
- Environment variables and settings
- Time-based conditions
- User-defined custom conditions
- Nested logical operators (AND, OR, NOT)
"""

import logging
import operator
import re
import time
from datetime import datetime
from typing import Any, Optional, Union


class ProfileConditionsError(Exception):
    """Profile conditions specific errors."""

    pass


class ProfileConditions:
    """Profile auto-activation condition evaluator.

    Evaluates complex conditions for automatic profile selection including:
    - Hardware-based conditions (CPU, memory, GPU)
    - Environment-based conditions (OS, user, variables)
    - Time-based conditions (hour, day, date ranges)
    - Logical operators (AND, OR, NOT)
    - Custom condition extensions
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize conditions evaluator.

        Args:
            logger: Logger instance (creates one if None)
        """
        self.logger = logger or logging.getLogger(__name__)

        # Built-in condition handlers
        self._condition_handlers = {
            "system": self._evaluate_system_condition,
            "time": self._evaluate_time_condition,
            "environment": self._evaluate_environment_condition,
            "hardware": self._evaluate_hardware_condition,
            "user": self._evaluate_user_condition,
            "custom": self._evaluate_custom_condition,
        }

        # Comparison operators
        self._operators = {
            "eq": operator.eq,
            "ne": operator.ne,
            "lt": operator.lt,
            "le": operator.le,
            "gt": operator.gt,
            "ge": operator.ge,
            "in": lambda x, y: x in y,
            "not_in": lambda x, y: x not in y,
            "contains": lambda x, y: y in x,
            "not_contains": lambda x, y: y not in x,
            "startswith": lambda x, y: str(x).startswith(str(y)),
            "endswith": lambda x, y: str(x).endswith(str(y)),
            "regex": lambda x, y: bool(re.search(str(y), str(x))),
            "not_regex": lambda x, y: not bool(re.search(str(y), str(x))),
        }

        # Logical operators
        self._logical_operators = {
            "and": self._evaluate_and,
            "or": self._evaluate_or,
            "not": self._evaluate_not,
        }

    def evaluate_conditions(
        self, conditions: dict[str, Any], context: dict[str, Any]
    ) -> float:
        """Evaluate profile conditions against context.

        Args:
            conditions: Condition definition dictionary
            context: Context data for evaluation

        Returns:
            Score from 0.0 to 1.0 indicating match quality
            (0.0 = no match, 1.0 = perfect match)
        """
        try:
            if not conditions:
                return 0.0

            # Handle logical operators at the top level
            if any(op in conditions for op in self._logical_operators):
                return self._evaluate_logical_conditions(conditions, context)

            # Evaluate individual conditions
            total_score = 0.0
            total_weight = 0.0

            for condition_type, condition_spec in conditions.items():
                if condition_type in self._condition_handlers:
                    # Get weight for this condition (default 1.0)
                    weight = (
                        condition_spec.get("weight", 1.0)
                        if isinstance(condition_spec, dict)
                        else 1.0
                    )

                    # Evaluate the condition
                    score = self._condition_handlers[condition_type](
                        condition_spec, context
                    )

                    total_score += score * weight
                    total_weight += weight

                else:
                    self.logger.warning(f"Unknown condition type: {condition_type}")

            # Return weighted average score
            return total_score / total_weight if total_weight > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Error evaluating conditions: {e}")
            return 0.0

    def _evaluate_logical_conditions(
        self, conditions: dict[str, Any], context: dict[str, Any]
    ) -> float:
        """Evaluate logical operators (AND, OR, NOT).

        Args:
            conditions: Logical condition definition
            context: Context data for evaluation

        Returns:
            Aggregated score from logical evaluation
        """
        for op_name, op_func in self._logical_operators.items():
            if op_name in conditions:
                return op_func(conditions[op_name], context)

        # No logical operators found, treat as implicit AND
        return self.evaluate_conditions(conditions, context)

    def _evaluate_and(
        self, sub_conditions: list[dict[str, Any]], context: dict[str, Any]
    ) -> float:
        """Evaluate AND logical operator.

        Returns the minimum score among all conditions (all must match).
        """
        if not sub_conditions:
            return 1.0

        scores = []
        for condition in sub_conditions:
            score = self.evaluate_conditions(condition, context)
            scores.append(score)

            # Short-circuit: if any condition fails completely, return 0
            if score == 0.0:
                return 0.0

        # Return minimum score (weakest link)
        return min(scores)

    def _evaluate_or(
        self, sub_conditions: list[dict[str, Any]], context: dict[str, Any]
    ) -> float:
        """Evaluate OR logical operator.

        Returns the maximum score among all conditions (any can match).
        """
        if not sub_conditions:
            return 0.0

        scores = []
        for condition in sub_conditions:
            score = self.evaluate_conditions(condition, context)
            scores.append(score)

        # Return maximum score (best match)
        return max(scores)

    def _evaluate_not(
        self, sub_condition: dict[str, Any], context: dict[str, Any]
    ) -> float:
        """Evaluate NOT logical operator.

        Returns inverted score (1.0 - original_score).
        """
        score = self.evaluate_conditions(sub_condition, context)
        return 1.0 - score

    def _evaluate_system_condition(
        self, condition: Union[dict[str, Any], Any], context: dict[str, Any]
    ) -> float:
        """Evaluate system-based conditions.

        Args:
            condition: System condition specification
            context: Context data containing system information

        Returns:
            Match score (0.0 to 1.0)
        """
        try:
            if not isinstance(condition, dict):
                return 0.0

            system_context = context.get("system", {})
            if not system_context:
                return 0.0

            return self._evaluate_nested_conditions(condition, system_context)

        except Exception as e:
            self.logger.error(f"Error evaluating system condition: {e}")
            return 0.0

    def _evaluate_time_condition(
        self, condition: Union[dict[str, Any], Any], context: dict[str, Any]
    ) -> float:
        """Evaluate time-based conditions.

        Args:
            condition: Time condition specification
            context: Context data containing time information

        Returns:
            Match score (0.0 to 1.0)
        """
        try:
            if not isinstance(condition, dict):
                return 0.0

            time_context = context.get("time", {})
            if not time_context:
                # Generate current time context if not provided
                now = datetime.now()
                time_context = {
                    "hour": now.hour,
                    "minute": now.minute,
                    "day_of_week": now.weekday(),
                    "day_of_month": now.day,
                    "month": now.month,
                    "year": now.year,
                    "timestamp": time.time(),
                }

            return self._evaluate_nested_conditions(condition, time_context)

        except Exception as e:
            self.logger.error(f"Error evaluating time condition: {e}")
            return 0.0

    def _evaluate_environment_condition(
        self, condition: Union[dict[str, Any], Any], context: dict[str, Any]
    ) -> float:
        """Evaluate environment-based conditions.

        Args:
            condition: Environment condition specification
            context: Context data containing environment information

        Returns:
            Match score (0.0 to 1.0)
        """
        try:
            if not isinstance(condition, dict):
                return 0.0

            env_context = context.get("environment", {})
            if not env_context:
                return 0.0

            return self._evaluate_nested_conditions(condition, env_context)

        except Exception as e:
            self.logger.error(f"Error evaluating environment condition: {e}")
            return 0.0

    def _evaluate_hardware_condition(
        self, condition: Union[dict[str, Any], Any], context: dict[str, Any]
    ) -> float:
        """Evaluate hardware-based conditions.

        Args:
            condition: Hardware condition specification
            context: Context data containing hardware information

        Returns:
            Match score (0.0 to 1.0)
        """
        try:
            if not isinstance(condition, dict):
                return 0.0

            # Hardware info can be in system context
            hardware_context = context.get("hardware", context.get("system", {}))
            if not hardware_context:
                return 0.0

            return self._evaluate_nested_conditions(condition, hardware_context)

        except Exception as e:
            self.logger.error(f"Error evaluating hardware condition: {e}")
            return 0.0

    def _evaluate_user_condition(
        self, condition: Union[dict[str, Any], Any], context: dict[str, Any]
    ) -> float:
        """Evaluate user-based conditions.

        Args:
            condition: User condition specification
            context: Context data containing user information

        Returns:
            Match score (0.0 to 1.0)
        """
        try:
            if not isinstance(condition, dict):
                return 0.0

            user_context = context.get("user", context.get("environment", {}))
            if not user_context:
                return 0.0

            return self._evaluate_nested_conditions(condition, user_context)

        except Exception as e:
            self.logger.error(f"Error evaluating user condition: {e}")
            return 0.0

    def _evaluate_custom_condition(
        self, condition: Union[dict[str, Any], Any], context: dict[str, Any]
    ) -> float:
        """Evaluate custom conditions.

        Args:
            condition: Custom condition specification
            context: Context data

        Returns:
            Match score (0.0 to 1.0)
        """
        try:
            if not isinstance(condition, dict):
                return 0.0

            # Custom conditions can access full context
            return self._evaluate_nested_conditions(condition, context)

        except Exception as e:
            self.logger.error(f"Error evaluating custom condition: {e}")
            return 0.0

    def _evaluate_nested_conditions(
        self, conditions: dict[str, Any], context: dict[str, Any]
    ) -> float:
        """Evaluate nested condition specifications.

        Args:
            conditions: Nested condition dictionary
            context: Context data for evaluation

        Returns:
            Aggregated match score
        """
        try:
            total_score = 0.0
            total_weight = 0.0

            for field_name, field_condition in conditions.items():
                if field_name == "weight":
                    continue  # Skip weight specification

                # Get context value
                context_value = self._get_nested_value(context, field_name)

                # Evaluate field condition
                if isinstance(field_condition, dict):
                    # Complex condition with operators
                    score = self._evaluate_field_condition(
                        context_value, field_condition
                    )
                else:
                    # Simple equality check
                    score = 1.0 if context_value == field_condition else 0.0

                weight = 1.0  # Default weight
                total_score += score * weight
                total_weight += weight

            return total_score / total_weight if total_weight > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Error evaluating nested conditions: {e}")
            return 0.0

    def _evaluate_field_condition(
        self, context_value: Any, condition: dict[str, Any]
    ) -> float:
        """Evaluate a single field condition with operators.

        Args:
            context_value: Value from context
            condition: Condition specification with operators

        Returns:
            Match score (0.0 to 1.0)
        """
        try:
            # Handle range conditions
            if "min" in condition or "max" in condition:
                return self._evaluate_range_condition(context_value, condition)

            # Handle operator-based conditions
            for op_name, op_func in self._operators.items():
                if op_name in condition:
                    expected_value = condition[op_name]
                    try:
                        result = op_func(context_value, expected_value)
                        return 1.0 if result else 0.0
                    except Exception as e:
                        self.logger.debug(f"Operator {op_name} failed: {e}")
                        return 0.0

            # Handle fuzzy matching for numeric values
            if "target" in condition:
                return self._evaluate_fuzzy_condition(context_value, condition)

            return 0.0

        except Exception as e:
            self.logger.error(f"Error evaluating field condition: {e}")
            return 0.0

    def _evaluate_range_condition(self, value: Any, condition: dict[str, Any]) -> float:
        """Evaluate range-based conditions.

        Args:
            value: Value to check
            condition: Range condition with 'min' and/or 'max'

        Returns:
            Match score based on range inclusion
        """
        try:
            if not isinstance(value, (int, float)):
                return 0.0

            min_val = condition.get("min")
            max_val = condition.get("max")

            if min_val is not None and value < min_val:
                return 0.0
            if max_val is not None and value > max_val:
                return 0.0

            # Value is in range, calculate fuzzy score
            if min_val is not None and max_val is not None:
                # Value is within range, return 1.0
                return 1.0
            elif min_val is not None:
                # Only minimum specified, return score based on distance from minimum
                return min(1.0, (value - min_val + 1) / (value + 1))
            elif max_val is not None:
                # Only maximum specified, return score based on distance from maximum
                return min(1.0, (max_val - value + 1) / (max_val + 1))

            return 1.0

        except Exception as e:
            self.logger.error(f"Error evaluating range condition: {e}")
            return 0.0

    def _evaluate_fuzzy_condition(self, value: Any, condition: dict[str, Any]) -> float:
        """Evaluate fuzzy matching conditions.

        Args:
            value: Value to check
            condition: Fuzzy condition with 'target' and optional 'tolerance'

        Returns:
            Fuzzy match score (0.0 to 1.0)
        """
        try:
            target = condition["target"]
            tolerance = condition.get("tolerance", 0.1)

            if not isinstance(value, (int, float)) or not isinstance(
                target, (int, float)
            ):
                return 1.0 if value == target else 0.0

            # Calculate fuzzy score based on distance from target
            distance = abs(value - target)
            if distance == 0:
                return 1.0

            max_distance = abs(target) * tolerance
            if max_distance == 0:
                max_distance = tolerance

            if distance <= max_distance:
                # Linear decay within tolerance
                return 1.0 - (distance / max_distance)
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error evaluating fuzzy condition: {e}")
            return 0.0

    def _get_nested_value(self, data: dict[str, Any], key_path: str) -> Any:
        """Get value from nested dictionary using dot notation.

        Args:
            data: Dictionary to search
            key_path: Dot-separated key path (e.g., 'system.cpu_count')

        Returns:
            Value at the key path or None if not found
        """
        try:
            keys = key_path.split(".")
            current = data

            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None

            return current

        except Exception:
            return None

    def validate_conditions(self, conditions: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate condition specification for correctness.

        Args:
            conditions: Condition specification to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            if not isinstance(conditions, dict):
                return False, ["Conditions must be a dictionary"]

            # Validate logical operators
            for op_name in self._logical_operators:
                if op_name in conditions:
                    op_value = conditions[op_name]
                    if op_name in ["and", "or"]:
                        if not isinstance(op_value, list):
                            errors.append(
                                f"Logical operator '{op_name}' must have a list value"
                            )
                        else:
                            for i, sub_condition in enumerate(op_value):
                                if not isinstance(sub_condition, dict):
                                    errors.append(
                                        f"Logical operator '{op_name}' item {i} must be a dictionary"
                                    )
                    elif op_name == "not" and not isinstance(op_value, dict):
                        errors.append(
                            f"Logical operator '{op_name}' must have a dictionary value"
                        )

            # Validate condition types
            for condition_type, condition_spec in conditions.items():
                if condition_type in self._logical_operators:
                    continue  # Already validated above

                if condition_type not in self._condition_handlers:
                    errors.append(f"Unknown condition type: {condition_type}")
                    continue

                # Validate condition structure
                if isinstance(condition_spec, dict):
                    self._validate_condition_structure(
                        condition_spec, errors, condition_type
                    )

            return len(errors) == 0, errors

        except Exception as e:
            return False, [f"Validation error: {e}"]

    def _validate_condition_structure(
        self, condition: dict[str, Any], errors: list[str], condition_type: str
    ) -> None:
        """Validate the structure of a condition specification.

        Args:
            condition: Condition dictionary to validate
            errors: List to append errors to
            condition_type: Type of condition being validated
        """
        for field_name, field_condition in condition.items():
            if field_name == "weight":
                if (
                    not isinstance(field_condition, (int, float))
                    or field_condition <= 0
                ):
                    errors.append(
                        f"Weight in {condition_type} condition must be a positive number"
                    )
                continue

            if isinstance(field_condition, dict):
                # Validate operators
                valid_operators = set(self._operators.keys()) | {
                    "min",
                    "max",
                    "target",
                    "tolerance",
                }
                invalid_ops = set(field_condition.keys()) - valid_operators

                if invalid_ops:
                    errors.append(
                        f"Invalid operators in {condition_type}.{field_name}: {invalid_ops}"
                    )

                # Validate range conditions
                if "min" in field_condition and "max" in field_condition:
                    min_val = field_condition["min"]
                    max_val = field_condition["max"]
                    if isinstance(min_val, (int, float)) and isinstance(
                        max_val, (int, float)
                    ):
                        if min_val > max_val:
                            errors.append(
                                f"min value cannot be greater than max value in {condition_type}.{field_name}"
                            )

                # Validate fuzzy conditions
                if "target" in field_condition:
                    if "tolerance" in field_condition:
                        tolerance = field_condition["tolerance"]
                        if not isinstance(tolerance, (int, float)) or tolerance < 0:
                            errors.append(
                                f"tolerance must be a non-negative number in {condition_type}.{field_name}"
                            )

    def register_condition_handler(
        self, condition_type: str, handler_func: callable
    ) -> None:
        """Register a custom condition handler.

        Args:
            condition_type: Name of the condition type
            handler_func: Function to handle the condition evaluation
                         Must have signature: (condition, context) -> float
        """
        self._condition_handlers[condition_type] = handler_func
        self.logger.info(f"Registered custom condition handler: {condition_type}")

    def get_supported_conditions(self) -> list[str]:
        """Get list of supported condition types.

        Returns:
            List of condition type names
        """
        return list(self._condition_handlers.keys())

    def get_supported_operators(self) -> list[str]:
        """Get list of supported comparison operators.

        Returns:
            List of operator names
        """
        return list(self._operators.keys())

    def create_time_range_condition(
        self, start_hour: int, end_hour: int
    ) -> dict[str, Any]:
        """Create a time range condition helper.

        Args:
            start_hour: Start hour (0-23)
            end_hour: End hour (0-23)

        Returns:
            Time condition dictionary
        """
        return {"time": {"hour": {"min": start_hour, "max": end_hour}}}

    def create_hardware_condition(
        self,
        min_memory_gb: Optional[float] = None,
        min_cpu_count: Optional[int] = None,
        gpu_required: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Create a hardware condition helper.

        Args:
            min_memory_gb: Minimum memory in GB
            min_cpu_count: Minimum CPU count
            gpu_required: Whether GPU is required

        Returns:
            Hardware condition dictionary
        """
        condition = {"hardware": {}}

        if min_memory_gb is not None:
            condition["hardware"]["memory_gb"] = {"min": min_memory_gb}

        if min_cpu_count is not None:
            condition["hardware"]["cpu_count"] = {"min": min_cpu_count}

        if gpu_required is not None:
            if gpu_required:
                condition["hardware"]["gpu_count"] = {"min": 1}
            else:
                condition["hardware"]["gpu_count"] = {"eq": 0}

        return condition

    def create_platform_condition(self, platforms: list[str]) -> dict[str, Any]:
        """Create a platform condition helper.

        Args:
            platforms: List of supported platforms (e.g., ['Windows', 'Linux'])

        Returns:
            Platform condition dictionary
        """
        return {"system": {"platform": {"in": platforms}}}
