from __future__ import annotations

import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REPO_DIR = WORKSPACE_ROOT / "GPT-SoVITS"


def run(args: list[str]) -> None:
    print("+ " + " ".join(args))
    subprocess.run(args, cwd=REPO_DIR, check=True)


def main() -> None:
    if not REPO_DIR.exists():
        raise SystemExit(
            "GPT-SoVITS submodule is missing. Run `pixi run submodule-init` first."
        )

    extra_req = REPO_DIR / "extra-req.txt"
    requirements = REPO_DIR / "requirements.txt"
    if extra_req.exists():
        run([sys.executable, "-m", "pip", "install", "-r", str(extra_req), "--no-deps"])
    if requirements.exists():
        run([sys.executable, "-m", "pip", "install", "-r", str(requirements)])

    print("GPT-SoVITS Python dependencies installed.")


if __name__ == "__main__":
    main()
