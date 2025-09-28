"""File watching for configuration hot reload."""


class FileWatcher:
    """File watcher for configuration changes."""

    def __init__(self):
        pass

    def watch(self, file_path: str, callback) -> None:
        """Watch file for changes."""
        pass

    def stop_watching(self, file_path: str) -> None:
        """Stop watching file."""
        pass
