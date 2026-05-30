from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAME = "Neiroha-GPT-SoVITS-Portable"
DEFAULT_VOLUME = "1900MB"

EXCLUDE_DIR_NAMES = {
    ".git",
    ".idea",
    ".vscode",
    ".dart_tool",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "task-cache-v0",
}

EXCLUDE_FILE_PATTERNS = {
    "*.pyc",
    "*.pyo",
    "*.tmp",
    "*.log",
}

EXCLUDE_RELATIVE_PATHS = {
    Path("GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base"),
    Path("GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
    Path("GPT-SoVITS/GPT_SoVITS/pretrained_models/g2pw-chinese"),
    Path("GPT-SoVITS/GPT_SoVITS/pretrained_models/s1v3.ckpt"),
    Path("GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro"),
}

COPY_PATHS = [
    ".pixi/envs/default",
    "app",
    "configs",
    "docs",
    "GPT-SoVITS",
    "models",
    "profiles",
    "runtime/voices",
    "scripts",
    ".gitmodules",
    "pixi.lock",
    "pixi.toml",
    "README.md",
    "README_zh.md",
    "start_portable.bat",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Windows portable release staging tree.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Top-level package directory name.")
    parser.add_argument("--out-dir", default=".codex-temp/portable", help="Disposable output root.")
    parser.add_argument("--copy", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--hardlink", action="store_true", help="Hard-link files where possible instead of copying.")
    parser.add_argument("--archive", action="store_true", help="Create a split 7z archive with Bandizip if available.")
    parser.add_argument(
        "--volume",
        default=DEFAULT_VOLUME,
        help="Archive volume size for Bandizip. The default keeps each split part under 2 GB.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=5,
        choices=range(0, 10),
        metavar="0-9",
        help="Bandizip compression level. 0 stores files without compression.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove previous portable output before staging.")
    return parser.parse_args()


def ensure_under_root(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    parent_resolved = parent.resolve()
    if resolved != parent_resolved and parent_resolved not in resolved.parents:
        raise SystemExit(f"Refusing to operate outside {parent_resolved}: {resolved}")
    return resolved


def should_exclude(path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        relative = None
    if relative is not None:
        for excluded in EXCLUDE_RELATIVE_PATHS:
            if relative == excluded or excluded in relative.parents:
                return True
    if path.name in EXCLUDE_DIR_NAMES:
        return True
    if path.is_file() and any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDE_FILE_PATTERNS):
        return True
    parts = set(path.parts)
    if "runtime" in parts and any(name in parts for name in {"cache", "logs", "outputs", "temp"}):
        return True
    if "modelscope" in parts and "runtime" in parts and "cache" in parts:
        return True
    return False


def ignore_names(directory: str, names: list[str]) -> set[str]:
    base = Path(directory)
    ignored = set()
    for name in names:
        if should_exclude(base / name):
            ignored.add(name)
    return ignored


def link_or_copy(src: str, dst: str) -> str:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def copy_entry(source: Path, destination: Path, *, hardlink: bool) -> None:
    if not source.exists():
        print(f"[skip] missing {source.relative_to(ROOT)}")
        return
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            ignore=ignore_names,
            copy_function=link_or_copy if hardlink else shutil.copy2,
            dirs_exist_ok=True,
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if hardlink:
            link_or_copy(str(source), str(destination))
        else:
            shutil.copy2(source, destination)


def write_portable_readme(stage: Path) -> None:
    text = """# Neiroha GPT-SoVITS Portable

This package is intended to run from the unpacked directory on Windows.

## Start

```bat
start_portable.bat
```

or:

```bat
start_portable.bat api
start_portable.bat admin
start_portable.bat serve
```

The launcher uses the bundled `.pixi\\envs\\default\\python.exe` and keeps cache,
logs, generated audio, and temp files under `runtime\\`.

## Default URLs

- FastAPI: `http://127.0.0.1:9880`
- Admin: `http://127.0.0.1:7860`

## Notes

- Windows portable release.
- NVIDIA driver compatible with the bundled Torch/CUDA build is required for GPU inference.
- Model files included in this package determine whether it can run fully offline.
"""
    (stage / "PORTABLE_README.md").write_text(text, encoding="utf-8")


def prepare_runtime_dirs(stage: Path) -> None:
    for rel in ["runtime/cache", "runtime/logs", "runtime/outputs", "runtime/temp"]:
        path = stage / rel
        path.mkdir(parents=True, exist_ok=True)
        (path / ".gitkeep").write_text("", encoding="utf-8")


def find_bz() -> str | None:
    for candidate in [
        shutil.which("bz"),
        r"D:\Programs\Bandizip\Bandizip\bz.exe",
        r"C:\Program Files\Bandizip\bz.exe",
    ]:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def run_archive(stage: Path, archive_dir: Path, *, volume: str, compression_level: int) -> Path:
    bz = find_bz()
    if not bz:
        raise SystemExit("Bandizip bz.exe not found; rerun without --archive or install Bandizip.")
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"{stage.name}.7z"
    command = [
        bz,
        "c",
        "-fmt:7z",
        f"-l:{compression_level}",
        f"-v:{volume}",
        str(archive),
        stage.name,
    ]
    print("[archive]", " ".join(command))
    subprocess.run(command, cwd=stage.parent, check=True)
    test_target = Path(str(archive) + ".001")
    if not test_target.exists():
        test_target = archive
    print("[test]", test_target)
    subprocess.run([bz, "t", str(test_target)], check=True)
    return archive


def main() -> int:
    args = parse_args()
    out_root = ensure_under_root(ROOT / args.out_dir, ROOT / ".codex-temp")
    stage_root = out_root / "stage"
    archive_root = out_root / "archive"
    stage = stage_root / args.name

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    stage.mkdir(parents=True, exist_ok=True)

    hardlink = bool(args.hardlink and not args.copy)
    for rel in COPY_PATHS:
        copy_entry(ROOT / rel, stage / rel, hardlink=hardlink)
    prepare_runtime_dirs(stage)
    write_portable_readme(stage)

    print(f"[stage] {stage}")
    if args.archive:
        archive = run_archive(stage, archive_root, volume=args.volume, compression_level=args.compression_level)
        print(f"[archive] {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
