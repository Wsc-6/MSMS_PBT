from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import load_config


NOTEBOOKS = {
    "prepare": "01_",
    "split": "02_",
    "audit-spectra": "03_",
    "train-baselines": "04_",
    "train-simple-spectrum": "05_",
    "train-transformer": "06_",
}


def _find_notebook(root: Path, prefix: str) -> Path:
    matches = []
    search_roots = [root]
    if root.name == "02_代码与配置" or (root.parent / "01_分析Notebooks").is_dir():
        search_roots.append(root.parent)
    for search_root in search_roots:
        for directory in ("建模分析", "01_分析Notebooks"):
            matches.extend((search_root / directory).glob(f"{prefix}*.ipynb"))
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one notebook matching {prefix}*.ipynb, found {matches}")
    return matches[0]


def _execute_notebook(root: Path, notebook: Path, timeout: int) -> None:
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(notebook),
    ]
    environment = os.environ.copy()
    local_jupyter = root / ".jupyter"
    previous = environment.get("JUPYTER_PATH", "")
    environment["JUPYTER_PATH"] = str(local_jupyter) + (os.pathsep + previous if previous else "")
    environment["IPYTHONDIR"] = str(root / ".jupyter" / "ipython")
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    subprocess.run(command, cwd=root, env=environment, check=True)


def _validate(root: Path) -> None:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    subprocess.run(command, cwd=root, check=True)


def _sirius_status() -> None:
    executable = shutil.which("sirius")
    payload = {"available": bool(executable), "executable": executable}
    if executable:
        completed = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, check=False
        )
        payload["returncode"] = completed.returncode
        payload["version"] = (completed.stdout or completed.stderr).strip()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PBT MS/MS reproducible notebook runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in [*NOTEBOOKS, "run-all", "validate", "sirius-status"]:
        child = subparsers.add_parser(command)
        child.add_argument("--config", default="configs/default.yaml")
        child.add_argument("--timeout", type=int, default=21600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    root = config.root
    if args.command == "validate":
        _validate(root)
    elif args.command == "sirius-status":
        _sirius_status()
    else:
        commands = list(NOTEBOOKS) if args.command == "run-all" else [args.command]
        for command in commands:
            notebook = _find_notebook(root, NOTEBOOKS[command])
            print(f"Executing {notebook.relative_to(root)}", flush=True)
            _execute_notebook(root, notebook, args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
