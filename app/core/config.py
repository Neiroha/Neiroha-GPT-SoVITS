"""Configuration helpers for the Neiroha GPT-SoVITS backend."""

from pathlib import Path
from typing import Any

from scripts.launch_gpt_sovits import SERVER_CONFIG_PATH

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        from scripts import toml_compat as tomllib


def load_server_config(path: Path = SERVER_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as file:
        return tomllib.load(file)
