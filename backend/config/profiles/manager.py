"""Configuration profile management with advanced features.

Provides comprehensive profile management including:
- Profile creation, loading, and deletion
- Profile inheritance and merging
- Active profile tracking
- Conditional profile activation
- Profile validation and metadata management
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..models.schemas import ConfigFormat, ConfigProfileEnhanced
from ..storage.persistence import ConfigPersistence, ConfigPersistenceError
from .conditions import ProfileConditions


class ProfileManagerError(Exception):
    """Profile manager specific errors."""

    pass


class ProfileManager:
    """Advanced configuration profile manager.

    Provides comprehensive profile management including inheritance,
    conditions, active profile tracking, and validation.
    """

    def __init__(
        self,
        persistence: Optional[ConfigPersistence] = None,
        config_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize profile manager.

        Args:
            persistence: ConfigPersistence instance (creates one if None)
            config_dir: Configuration directory (defaults to 'config')
            logger: Logger instance (creates one if None)
        """
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.logger = logger or logging.getLogger(__name__)

        # Initialize persistence if not provided
        if persistence is None:
            self._persistence = ConfigPersistence(
                base_dir=self.config_dir, logger=self.logger
            )
        else:
            self._persistence = persistence

        # Initialize conditions manager
        self._conditions = ProfileConditions()

        # Track active profile
        self._active_profile: Optional[str] = None
        self._loaded_profiles: dict[str, ConfigProfileEnhanced] = {}

        # Load active profile state
        self._load_active_profile_state()

    def _load_active_profile_state(self) -> None:
        """Load the active profile state from disk."""
        try:
            state_file = self.config_dir / ".active_profile"
            if state_file.exists():
                with open(state_file) as f:
                    data = json.load(f)
                    self._active_profile = data.get("active_profile")
                    self.logger.debug(
                        f"Loaded active profile state: {self._active_profile}"
                    )
        except Exception as e:
            self.logger.warning(f"Failed to load active profile state: {e}")

    def _save_active_profile_state(self) -> None:
        """Save the active profile state to disk."""
        try:
            state_file = self.config_dir / ".active_profile"
            state_data = {
                "active_profile": self._active_profile,
                "last_updated": time.time(),
            }
            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save active profile state: {e}")

    def create_profile(
        self,
        name: str,
        config_data: Optional[dict[str, Any]] = None,
        description: Optional[str] = None,
        parent: Optional[str] = None,
        conditions: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> bool:
        """Create new configuration profile.

        Args:
            name: Profile name (must be unique)
            config_data: Configuration data for the profile
            description: Profile description
            parent: Parent profile name for inheritance
            conditions: Auto-activation conditions
            tags: Profile tags for organization

        Returns:
            True if created successfully, False otherwise

        Raises:
            ProfileManagerError: If profile creation fails
        """
        try:
            # Validate profile name
            if not name or not name.strip():
                raise ProfileManagerError("Profile name cannot be empty")

            # Check if profile already exists
            if name in self.list_profiles():
                raise ProfileManagerError(f"Profile '{name}' already exists")

            # Validate parent profile if specified
            if parent and parent not in self.list_profiles():
                raise ProfileManagerError(f"Parent profile '{parent}' not found")

            # Create profile object
            profile = ConfigProfileEnhanced(
                name=name,
                description=description or f"Configuration profile: {name}",
                parent=parent,
                conditions=conditions,
                settings=config_data or {},
                tags=tags or [],
                created=datetime.now(),
                modified=datetime.now(),
            )

            # Save profile
            success = self._persistence.save_profile(name, profile)
            if success:
                self._loaded_profiles[name] = profile
                self.logger.info(f"Created profile: {name}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to create profile '{name}': {e}")
            if isinstance(e, ProfileManagerError):
                raise
            return False

    def load_profile(self, name: str) -> Optional[ConfigProfileEnhanced]:
        """Load configuration profile.

        Args:
            name: Profile name to load

        Returns:
            Loaded profile object or None if not found
        """
        try:
            # Check cache first
            if name in self._loaded_profiles:
                return self._loaded_profiles[name]

            # Load from persistence
            profile = self._persistence.load_profile(name)

            # Cache the loaded profile
            self._loaded_profiles[name] = profile

            self.logger.debug(f"Loaded profile: {name}")
            return profile

        except ConfigPersistenceError as e:
            self.logger.error(f"Failed to load profile '{name}': {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error loading profile '{name}': {e}")
            return None

    def list_profiles(self) -> list[str]:
        """List available profiles.

        Returns:
            List of profile names
        """
        try:
            return self._persistence.list_profiles()
        except Exception as e:
            self.logger.error(f"Failed to list profiles: {e}")
            return []

    def delete_profile(self, name: str) -> bool:
        """Delete a configuration profile.

        Args:
            name: Profile name to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Don't allow deleting the active profile
            if name == self._active_profile:
                raise ProfileManagerError(
                    f"Cannot delete active profile '{name}'. Switch to another profile first."
                )

            # Check for profiles that inherit from this one
            dependent_profiles = self._get_dependent_profiles(name)
            if dependent_profiles:
                raise ProfileManagerError(
                    f"Cannot delete profile '{name}' because it is inherited by: {', '.join(dependent_profiles)}"
                )

            # Delete the profile
            success = self._persistence.delete_profile(name)
            if success:
                # Remove from cache
                self._loaded_profiles.pop(name, None)
                self.logger.info(f"Deleted profile: {name}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to delete profile '{name}': {e}")
            if isinstance(e, ProfileManagerError):
                raise
            return False

    def _get_dependent_profiles(self, parent_name: str) -> list[str]:
        """Get profiles that inherit from the given parent.

        Args:
            parent_name: Name of the parent profile

        Returns:
            List of profile names that inherit from the parent
        """
        dependent = []
        try:
            for profile_name in self.list_profiles():
                profile = self.load_profile(profile_name)
                if profile and profile.parent == parent_name:
                    dependent.append(profile_name)
        except Exception as e:
            self.logger.error(f"Error checking dependent profiles: {e}")

        return dependent

    def switch_profile(self, name: str) -> bool:
        """Switch to a different profile.

        Args:
            name: Profile name to switch to

        Returns:
            True if switched successfully, False otherwise
        """
        try:
            # Validate profile exists
            if name not in self.list_profiles():
                raise ProfileManagerError(f"Profile '{name}' not found")

            # Load profile to validate it
            profile = self.load_profile(name)
            if not profile:
                raise ProfileManagerError(f"Failed to load profile '{name}'")

            # Switch active profile
            old_profile = self._active_profile
            self._active_profile = name
            self._save_active_profile_state()

            self.logger.info(f"Switched profile from '{old_profile}' to '{name}'")
            return True

        except Exception as e:
            self.logger.error(f"Failed to switch to profile '{name}': {e}")
            if isinstance(e, ProfileManagerError):
                raise
            return False

    def get_active_profile(self) -> Optional[str]:
        """Get the name of the currently active profile.

        Returns:
            Active profile name or None if no profile is active
        """
        return self._active_profile

    def get_active_profile_data(self) -> Optional[ConfigProfileEnhanced]:
        """Get the currently active profile data.

        Returns:
            Active profile object or None if no profile is active
        """
        if self._active_profile:
            return self.load_profile(self._active_profile)
        return None

    def merge_profile_settings(
        self, profile_name: str, include_inherited: bool = True
    ) -> dict[str, Any]:
        """Get merged profile settings including inheritance.

        Args:
            profile_name: Profile name
            include_inherited: Whether to include settings from parent profiles

        Returns:
            Merged settings dictionary
        """
        try:
            profile = self.load_profile(profile_name)
            if not profile:
                return {}

            settings = {}

            # Build inheritance chain
            if include_inherited:
                inheritance_chain = self._build_inheritance_chain(profile_name)

                # Apply settings from base to derived (parents first)
                for ancestor_name in reversed(inheritance_chain):
                    ancestor = self.load_profile(ancestor_name)
                    if ancestor:
                        settings = self._deep_merge_dict(settings, ancestor.settings)
            else:
                settings = profile.settings.copy()

            return settings

        except Exception as e:
            self.logger.error(
                f"Failed to merge settings for profile '{profile_name}': {e}"
            )
            return {}

    def _build_inheritance_chain(self, profile_name: str) -> list[str]:
        """Build the complete inheritance chain for a profile.

        Args:
            profile_name: Profile name to build chain for

        Returns:
            List of profile names from root parent to the profile itself

        Raises:
            ProfileManagerError: If circular inheritance is detected
        """
        chain = []
        current = profile_name
        visited = set()

        while current:
            if current in visited:
                raise ProfileManagerError(
                    f"Circular inheritance detected in profile '{profile_name}'"
                )

            visited.add(current)
            chain.append(current)

            profile = self.load_profile(current)
            current = profile.parent if profile else None

        return chain

    def _deep_merge_dict(
        self, base: dict[str, Any], overlay: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge two dictionaries, with overlay values taking precedence.

        Args:
            base: Base dictionary
            overlay: Dictionary to merge on top of base

        Returns:
            Deep merged dictionary
        """
        result = base.copy()

        for key, value in overlay.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge_dict(result[key], value)
            else:
                # Direct assignment for non-dict values or new keys
                result[key] = value

        return result

    def update_profile(
        self,
        name: str,
        settings: Optional[dict[str, Any]] = None,
        description: Optional[str] = None,
        conditions: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> bool:
        """Update an existing profile.

        Args:
            name: Profile name to update
            settings: New settings (merged with existing)
            description: New description
            conditions: New conditions
            tags: New tags

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            profile = self.load_profile(name)
            if not profile:
                raise ProfileManagerError(f"Profile '{name}' not found")

            # Update fields if provided
            if settings is not None:
                profile.settings.update(settings)
            if description is not None:
                profile.description = description
            if conditions is not None:
                profile.conditions = conditions
            if tags is not None:
                profile.tags = tags

            # Update modification timestamp
            profile.modified = datetime.now()

            # Save updated profile
            success = self._persistence.save_profile(name, profile)
            if success:
                self._loaded_profiles[name] = profile
                self.logger.info(f"Updated profile: {name}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to update profile '{name}': {e}")
            if isinstance(e, ProfileManagerError):
                raise
            return False

    def check_profile_conditions(
        self, context: Optional[dict[str, Any]] = None
    ) -> Optional[str]:
        """Check profile conditions and return the best matching profile.

        Args:
            context: Context information for condition evaluation

        Returns:
            Name of the best matching profile or None
        """
        try:
            if context is None:
                context = self._gather_system_context()

            best_profile = None
            best_score = -1

            for profile_name in self.list_profiles():
                profile = self.load_profile(profile_name)
                if profile and profile.conditions:
                    score = self._conditions.evaluate_conditions(
                        profile.conditions, context
                    )
                    if score > best_score:
                        best_score = score
                        best_profile = profile_name

            return best_profile if best_score > 0 else None

        except Exception as e:
            self.logger.error(f"Failed to check profile conditions: {e}")
            return None

    def _gather_system_context(self) -> dict[str, Any]:
        """Gather system context for condition evaluation.

        Returns:
            Dictionary of system context information
        """
        import platform
        from datetime import datetime

        import psutil

        try:
            context = {
                "system": {
                    "platform": platform.system(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "memory_gb": psutil.virtual_memory().total / (1024**3),
                    "cpu_count": psutil.cpu_count(),
                },
                "time": {
                    "hour": datetime.now().hour,
                    "day_of_week": datetime.now().weekday(),
                    "timestamp": time.time(),
                },
                "environment": {
                    "user": platform.node(),
                    "python_version": platform.python_version(),
                },
            }

            # Add GPU information if available
            try:
                import GPUtil

                gpus = GPUtil.getGPUs()
                if gpus:
                    context["system"]["gpu_count"] = len(gpus)
                    context["system"]["gpu_memory_gb"] = (
                        sum(gpu.memoryTotal for gpu in gpus) / 1024
                    )
            except ImportError:
                pass

            return context

        except Exception as e:
            self.logger.error(f"Failed to gather system context: {e}")
            return {}

    def auto_select_profile(self, context: Optional[dict[str, Any]] = None) -> bool:
        """Automatically select and switch to the best matching profile.

        Args:
            context: Context information for condition evaluation

        Returns:
            True if a profile was selected and switched, False otherwise
        """
        try:
            best_profile = self.check_profile_conditions(context)
            if best_profile and best_profile != self._active_profile:
                return self.switch_profile(best_profile)
            return False

        except Exception as e:
            self.logger.error(f"Failed to auto-select profile: {e}")
            return False

    def export_profile(
        self, name: str, export_path: Path, format: ConfigFormat = ConfigFormat.JSON
    ) -> bool:
        """Export profile to file.

        Args:
            name: Profile name to export
            export_path: Path to export to
            format: Export format

        Returns:
            True if exported successfully, False otherwise
        """
        try:
            profile = self.load_profile(name)
            if not profile:
                raise ProfileManagerError(f"Profile '{name}' not found")

            return self._persistence.save_config(
                profile.model_dump(), export_path, format
            )

        except Exception as e:
            self.logger.error(f"Failed to export profile '{name}': {e}")
            if isinstance(e, ProfileManagerError):
                raise
            return False

    def import_profile(
        self, import_path: Path, profile_name: Optional[str] = None
    ) -> bool:
        """Import profile from file.

        Args:
            import_path: Path to import from
            profile_name: Name for imported profile (uses file name if None)

        Returns:
            True if imported successfully, False otherwise
        """
        try:
            if not import_path.exists():
                raise ProfileManagerError(f"Import file not found: {import_path}")

            # Load profile data
            data = self._persistence.load_config(import_path)

            # Determine profile name
            if profile_name is None:
                profile_name = data.get("name", import_path.stem)

            # Create profile from imported data
            if "name" in data and "settings" in data:
                # Full profile format
                profile = ConfigProfileEnhanced(**data)
                profile.name = profile_name  # Override name if specified
            else:
                # Raw settings format
                profile = ConfigProfileEnhanced(
                    name=profile_name,
                    description=f"Imported profile from {import_path.name}",
                    settings=data,
                )

            # Save imported profile
            return self._persistence.save_profile(profile_name, profile)

        except Exception as e:
            self.logger.error(f"Failed to import profile from '{import_path}': {e}")
            if isinstance(e, ProfileManagerError):
                raise
            return False

    def get_profile_metadata(self, name: str) -> Optional[dict[str, Any]]:
        """Get profile metadata without loading full settings.

        Args:
            name: Profile name

        Returns:
            Profile metadata dictionary or None
        """
        try:
            profile = self.load_profile(name)
            if not profile:
                return None

            return {
                "name": profile.name,
                "description": profile.description,
                "parent": profile.parent,
                "created": profile.created.isoformat() if profile.created else None,
                "modified": profile.modified.isoformat() if profile.modified else None,
                "tags": profile.tags,
                "has_conditions": bool(profile.conditions),
                "settings_count": len(profile.settings),
                "is_active": name == self._active_profile,
            }

        except Exception as e:
            self.logger.error(f"Failed to get metadata for profile '{name}': {e}")
            return None

    def validate_profile(self, name: str) -> tuple[bool, list[str]]:
        """Validate a profile for consistency and correctness.

        Args:
            name: Profile name to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            profile = self.load_profile(name)
            if not profile:
                return False, [f"Profile '{name}' not found"]

            # Check inheritance chain
            try:
                self._build_inheritance_chain(name)
            except ProfileManagerError as e:
                errors.append(str(e))

            # Validate conditions if present
            if profile.conditions:
                valid, condition_errors = self._conditions.validate_conditions(
                    profile.conditions
                )
                if not valid:
                    errors.extend(condition_errors)

            # Check parent exists if specified
            if profile.parent and profile.parent not in self.list_profiles():
                errors.append(f"Parent profile '{profile.parent}' not found")

            return len(errors) == 0, errors

        except Exception as e:
            self.logger.error(f"Failed to validate profile '{name}': {e}")
            return False, [f"Validation error: {e}"]

    def get_profiles_by_tag(self, tag: str) -> list[str]:
        """Get profiles that have a specific tag.

        Args:
            tag: Tag to search for

        Returns:
            List of profile names with the tag
        """
        matching_profiles = []

        try:
            for profile_name in self.list_profiles():
                profile = self.load_profile(profile_name)
                if profile and tag in profile.tags:
                    matching_profiles.append(profile_name)

        except Exception as e:
            self.logger.error(f"Failed to search profiles by tag '{tag}': {e}")

        return matching_profiles

    def clear_cache(self) -> None:
        """Clear the loaded profiles cache."""
        self._loaded_profiles.clear()
        self.logger.debug("Cleared profile cache")
