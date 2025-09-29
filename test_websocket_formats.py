#!/usr/bin/env python3
"""
Test script to verify WebSocket message format compatibility.
This script can be run independently to check message formats.
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from backend.api.websocket.format_verification import run_format_verification
    import asyncio

    if __name__ == "__main__":
        # Run the verification
        results = asyncio.run(run_format_verification())

        # Exit with error code if there are issues
        if results['summary']['invalid_formats'] > 0:
            sys.exit(1)
        else:
            print("\n✅ All WebSocket message formats are valid!")
            sys.exit(0)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error running verification: {e}")
    sys.exit(1)
