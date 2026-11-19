"""Plain-Python compatibility runner for environments without uv or poethepoet.

Primary development workflow uses ``uv`` and ``poe`` (see ``pyproject.toml`` and
``README.md``). Use this script only as a fallback when ``uv``/``poe`` are
unavailable.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = {
    "01": ROOT / "Source" / "experiments" / "01_effectiveness.py",
    "02": ROOT / "Source" / "experiments" / "02_kdist.py",
    "03": ROOT / "Source" / "experiments" / "03_runtime.py",
    "04": ROOT / "Source" / "experiments" / "04_limitations.py",
}


def _run(cmd: list[str]) -> int:
    print(f"+ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def cmd_install_source(args: argparse.Namespace) -> int:
    """Install Source dependencies via pip into the active Python environment."""
    rc = _run([sys.executable, "-m", "pip", "install", "-r", "Source/requirements.txt"])
    if rc != 0:
        return rc
    rc = _run([sys.executable, "-m", "pip", "install", "-e", "Source", "--no-deps"])
    if rc == 0 and args.dev:
        rc = _run([sys.executable, "-m", "pip", "install", "pytest>=8.0", "ruff>=0.8"])
    return rc


def cmd_test(_args: argparse.Namespace) -> int:
    return _run([sys.executable, "-m", "pytest"])


def cmd_lint(_args: argparse.Namespace) -> int:
    return _run([sys.executable, "-m", "ruff", "check", "Source", "Manim", "scripts"])


def cmd_format(_args: argparse.Namespace) -> int:
    return _run([sys.executable, "-m", "ruff", "format", "Source", "Manim", "scripts"])


def cmd_exp(args: argparse.Namespace) -> int:
    targets = list(EXPERIMENTS.values()) if args.id == "all" else [EXPERIMENTS[args.id]]
    for script in targets:
        cmd = [sys.executable, str(script)]
        if args.config:
            cmd.extend(["--config", args.config])
        rc = _run(cmd)
        if rc != 0:
            return rc
    return 0


def cmd_report(_args: argparse.Namespace) -> int:
    typst = shutil.which("typst")
    if typst is None:
        print("Error: `typst` CLI not found on PATH.", file=sys.stderr)
        return 1
    return _run([typst, "compile", "Report/main.typ", "Report/Report.pdf"])


def cmd_clean(_args: argparse.Namespace) -> int:
    for rel in (".pytest_cache", ".ruff_cache", "Manim/media", "Report/Report.pdf"):
        path = ROOT / rel
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plain-Python compatibility runner (prefer `uv` + `poe` for development)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_install = sub.add_parser("install-source", help="pip-install Source/requirements.txt")
    p_install.add_argument("--dev", action="store_true", help="Also install pytest and ruff.")
    p_install.set_defaults(func=cmd_install_source)

    for name, fn, help_text in (
        ("test", cmd_test, "Run pytest via python -m pytest"),
        ("lint", cmd_lint, "Run ruff check via python -m ruff"),
        ("format", cmd_format, "Run ruff format via python -m ruff"),
        ("report", cmd_report, "Compile Report/main.typ to Report/Report.pdf"),
        ("clean", cmd_clean, "Remove local caches, Manim media, and compiled PDF"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=fn)

    p_exp = sub.add_parser("exp", help="Run experiment script(s)")
    p_exp.add_argument("id", choices=[*EXPERIMENTS.keys(), "all"], help="Experiment ID or 'all'")
    p_exp.add_argument("--config", default=None, help="Optional YAML config override")
    p_exp.set_defaults(func=cmd_exp)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
