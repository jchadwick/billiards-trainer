"""Retired Python application-server entry point.

The FastAPI modules remain available to offline compatibility tests, but the
supported application server is the Rust ``billiards-server`` binary. Keeping
this small module turns old launch commands into an actionable migration hint
instead of silently starting a second server implementation.
"""

from __future__ import annotations

import sys
from typing import NoReturn


def main() -> NoReturn:
    """Reject the retired Python server command with Rust startup guidance."""
    print(
        "The Python application server has been retired. "
        "Use 'make run' to start the Rust server with live capture, or "
        "'make run-api' to start the Rust gateway without live capture.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
