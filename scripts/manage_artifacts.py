"""Build, verify, install or explicitly fetch frozen artifacts without training."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oil_forecast.artifacts import build_bundle, fetch_bundle, install_bundle, verify_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source", type=Path, default=ROOT / "experiments/cpu-final-evaluation-v1")
    build.add_argument("--processed", type=Path, default=ROOT / "data/processed/2026-11-11-official")
    build.add_argument("--output", type=Path, default=ROOT / "artifacts/local/brent-frozen-v1")
    build.add_argument("--archive", type=Path, default=ROOT / "artifacts/releases/brent-frozen-v1.zip")
    install = commands.add_parser("install")
    install.add_argument("--archive", type=Path, required=True)
    install.add_argument("--destination", type=Path, default=ROOT / "artifacts/local/brent-frozen-v1")
    trust = install.add_mutually_exclusive_group(required=True)
    trust.add_argument("--sha256", help="Trusted SHA256 supplied independently of the archive")
    trust.add_argument("--registry", type=Path, help="Versioned registry containing archive.sha256")
    verify = commands.add_parser("verify")
    verify.add_argument("--directory", type=Path, required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--url", required=True)
    fetch.add_argument("--registry", type=Path, required=True)
    fetch.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_bundle(args.source, args.processed, args.output, args.archive)
    elif args.command == "install":
        expected = args.sha256
        if args.registry is not None:
            registry = json.loads(args.registry.read_text(encoding="utf-8"))
            if registry.get("schema_version") != 1 or registry.get("bundle_id") not in {"brent-frozen-v1", "brent-dashboard-v1"}:
                raise ValueError("Unrecognized trusted artifact registry")
            expected = registry["archive"]["sha256"]
        result = install_bundle(args.archive, args.destination, expected)
    elif args.command == "fetch":
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
        if registry.get("schema_version") != 1 or registry.get("bundle_id") not in {"brent-frozen-v1", "brent-dashboard-v1"}:
            raise ValueError("Unrecognized trusted artifact registry")
        result = fetch_bundle(args.url, args.destination, registry["archive"]["sha256"])
    else:
        result = verify_bundle(args.directory)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
