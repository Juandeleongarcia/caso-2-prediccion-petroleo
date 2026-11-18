"""Export a minimal aggregate-only demonstration without restricted resources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = (
    "deployment/public_app/app.py",
    "deployment/public_app/requirements.txt",
    "deployment/public_app/constraints.txt",
    ".streamlit/config.toml",
    "src/oil_forecast/__init__.py",
    "src/oil_forecast/public_study.py",
    "src/oil_forecast/app.css",
    "artifacts/derived/brent-academic-v1/metrics.csv",
    "artifacts/derived/brent-academic-v1/artifact_manifest.json",
)


def build_public_demo(root: Path | str, output: Path | str) -> dict:
    """Copy an explicit allowlist to a new directory, without changing sources."""
    root, output = Path(root).resolve(), Path(output).absolute()
    if output.exists():
        raise FileExistsError("El destino de la demostración ya existe; no se sobrescribe")
    sources = [root / name for name in PUBLIC_FILES]
    if any(not path.is_file() or path.is_symlink() for path in sources):
        raise ValueError("La lista pública debe contener únicamente archivos locales regulares")
    manifest_bytes = (root / "artifacts/derived/brent-academic-v1/artifact_manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != "dd1073129897e6df6c62a0582b5fbbd61e30eb24709dd915840827f26079ce8f":
        raise ValueError("El manifiesto agregado no coincide con el original público")
    csv_bytes = (root / "artifacts/derived/brent-academic-v1/metrics.csv").read_bytes()
    expected = json.loads(manifest_bytes)["files"]["metrics.csv"]
    if len(csv_bytes) != expected["size"] or hashlib.sha256(csv_bytes).hexdigest() != expected["sha256"]:
        raise ValueError("Las métricas agregadas no coinciden con el original público")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".public-export-", dir=output.parent) as temporary:
        stage = Path(temporary) / "public"
        stage.mkdir()
        for name, source in zip(PUBLIC_FILES, sources):
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        if output.exists():
            raise FileExistsError("El destino ya existe; no se sobrescribe")
        stage.rename(output)
    files = {name: {"bytes": (output / name).stat().st_size,
                    "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()}
             for name in PUBLIC_FILES}
    return {"files": files, "scope": "aggregate_results_only", "neural_inference": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_public_demo(ROOT, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
