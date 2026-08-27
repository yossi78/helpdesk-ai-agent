"""Run the mock help-desk service for manual testing (e.g. Postman)."""
from __future__ import annotations

import argparse
import sys
import time

from target.app import start_target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m target")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args(argv)

    server = start_target(host=args.host, port=args.port)
    print(f"Help-desk service running at {server.base_url}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
