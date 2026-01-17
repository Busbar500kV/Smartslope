"""CLI entrypoint for smartslope package."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

from smartslope.simulate import main as simulate_main
from smartslope.detect import main as detect_main


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="smartslope",
        description="Smartslope: radar-based slope deformation detection"
    )
    parser.add_argument(
        "command",
        choices=["simulate", "detect", "pipeline"],
        help="Command to run"
    )
    
    args = parser.parse_args()
    
    if args.command == "simulate":
        simulate_main()
        return 0
    elif args.command == "detect":
        detect_main()
        return 0
    elif args.command == "pipeline":
        # Run both simulate and detect
        print("=== Running simulation ===")
        simulate_main()
        print("\n=== Running detection ===")
        detect_main()
        print("\n=== Pipeline complete ===")
        return 0
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
