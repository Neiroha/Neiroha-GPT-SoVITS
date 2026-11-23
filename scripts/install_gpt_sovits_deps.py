from __future__ import annotations

import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REPO_DIR = WORKSPACE_ROOT / "GPT-SoVITS"
WINDOWS_ONNXRUNTIME = (
    # Upstream's metadata-only wheel selects the CUDA 12 provider for Faster-Whisper and G2PW.
    "onnxruntime @ https://huggingface.co/lj1995/GPT-SoVITS-windows-package/"
    "resolve/c8b61d4f627d1f02e74c1ba158a4844051d2289f/wheels/cu128/onnxruntime-1.19.2-py3-none-any.whl"
    "#sha256=ccec3c1e1f34ab0e0cfbc6d16b65c8482e1d26cb4d9645445f733341a56127c2"
    " ; sys_platform == 'win32' and platform_machine == 'AMD64'"
)


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
        run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements),
            "resampy>=0.4.2,<0.5", "setuptools>=80,<81",
            "fast-langdetect>=0.3.1,<1", "x-transformers>=1.42,<2",
            WINDOWS_ONNXRUNTIME,
        ])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "fastapi[standard]>=0.115.2,<0.116",
            "starlette>=0.37.2,<0.41",
            "huggingface-hub>=0.26,<0.36",
            "hf-xet>=1.1,<2",
        ]
    )

    print("GPT-SoVITS Python dependencies installed.")


if __name__ == "__main__":
    main()
