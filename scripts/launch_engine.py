from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        try:
            from scripts import toml_compat as tomllib
        except ModuleNotFoundError:
            import toml_compat as tomllib

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SERVER_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "server.toml"
LAUNCHER_PATH = WORKSPACE_ROOT / "scripts" / "launch_gpt_sovits.py"

SURFACE_TO_MODE = {
    "api": "api",
    "admin": "admin",
    "both": "api-admin",
}


def load_server_config() -> dict[str, Any]:
    if not SERVER_CONFIG_PATH.exists():
        return {}
    with SERVER_CONFIG_PATH.open("rb") as file:
        return tomllib.load(file)


def build_command(*, surface: str = "", preload_model: bool = False, extra_args: list[str] | None = None) -> list[str]:
    config = load_server_config()
    startup = config.get("startup", {}) if isinstance(config.get("startup"), dict) else {}
    selected_surface = (surface or str(startup.get("surface") or "both")).strip().lower()
    if selected_surface not in SURFACE_TO_MODE:
        raise SystemExit("startup.surface must be one of: api, admin, both")

    command = [
        sys.executable,
        "-B",
        str(LAUNCHER_PATH),
        "--mode",
        SURFACE_TO_MODE[selected_surface],
    ]
    if preload_model or bool(startup.get("preload_model", False)):
        command.append("--preload-model")
    command.extend(extra_args or [])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Config-driven Neiroha GPT-SoVITS launcher.")
    parser.add_argument("--surface", choices=sorted(SURFACE_TO_MODE), default="")
    parser.add_argument("--preload-model", action="store_true")
    args, passthrough = parser.parse_known_args()

    command = build_command(
        surface=args.surface,
        preload_model=args.preload_model,
        extra_args=passthrough,
    )
    return subprocess.run(command, cwd=WORKSPACE_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
