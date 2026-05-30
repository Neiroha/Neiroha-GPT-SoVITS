from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only used by older local Pythons.
    tomllib = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]


def load_toml(path: Path) -> dict[str, Any]:
    if tomllib is not None:
        with path.open("rb") as file:
            return tomllib.load(file)

    result: dict[str, Any] = {}
    current = result
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = result.setdefault(line.strip("[]"), {})
            continue
        if "=" not in line:
            continue
        key, raw_value = [part.strip() for part in line.split("=", 1)]
        if raw_value in {"true", "false"}:
            value: Any = raw_value == "true"
        elif raw_value.startswith("["):
            value = [item.strip().strip('"') for item in raw_value.strip("[]").split(",") if item.strip()]
        else:
            value = raw_value.strip('"')
        current[key] = value
    return result


class RegistryConfigTest(unittest.TestCase):
    def test_server_config_has_startup_contract(self) -> None:
        config = load_toml(ROOT / "configs" / "server.toml")
        self.assertEqual(config["startup"]["surface"], "both")
        self.assertIn("preload_model", config["startup"])
        self.assertEqual(config["startup"]["default_model_preset"], "v2proplus-clone")
        self.assertIn("api_key", config["security"])

    def test_default_voice_set_points_to_runtime_voice(self) -> None:
        voice_set = load_toml(ROOT / "configs" / "voice-sets" / "default.toml")
        voices = voice_set.get("voices", [])
        self.assertIn("genshin-keqing", voices)
        self.assertTrue((ROOT / "runtime" / "voices" / "genshin-keqing" / "voice.toml").is_file())


if __name__ == "__main__":
    unittest.main()
