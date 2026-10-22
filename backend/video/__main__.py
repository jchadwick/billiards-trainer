"""Retired Python camera-process entry point.

The video IPC and process modules remain available to offline compatibility
tests. The supported application owns capture through its Rust server and
supervised Rust capture helper.
"""

from __future__ import annotations

import sys
from typing import NoReturn


def main() -> NoReturn:
    """Reject the retired Python camera-process command."""
    print(
        "The Python camera process has been retired. "
        "Use 'make run' to start the Rust server and its supervised capture "
        "pipeline.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
