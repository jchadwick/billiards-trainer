#!/usr/bin/env python3
"""Retired Python development-server command; use the Rust Make targets."""

from __future__ import annotations

import sys
from typing import NoReturn


def main() -> NoReturn:
    """Reject the retired command without importing a Python web runtime."""
    print(
        "The Python development server has been retired. "
        "Use 'make run' for the Rust application or 'make run-api' for the "
        "Rust gateway without live capture.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
