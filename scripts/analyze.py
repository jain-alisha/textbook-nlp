"""
Analyze classification results for a textbook.

Usage:
    python scripts/analyze.py --name saxon_course1

Stub — full analysis coming soon.
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze classified results.")
    parser.add_argument("--name", required=True, help="Textbook identifier (e.g. saxon_course1).")
    args = parser.parse_args()

    data_dir = Path("data") / args.name
    classified_path = data_dir / "classified_results.csv"

    if not classified_path.exists():
        print(f"Error: {classified_path} not found. Run classify.py first.")
        return

    print(f"Analysis for {args.name} — not yet implemented.")


if __name__ == "__main__":
    main()
