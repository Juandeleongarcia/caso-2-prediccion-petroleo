"""Derive portable R² and error scores without changing frozen experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oil_forecast.academic import build_academic_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "experiments")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/derived/brent-academic-v1")
    arguments = parser.parse_args()
    result = build_academic_metrics(arguments.source_root, arguments.output)
    print(json.dumps({key: value for key, value in result.items() if key != "manifest"}, indent=2))


if __name__ == "__main__":
    main()
