"""API dependency helpers.

The current implementation delegates runtime construction to the compatibility
launcher while the backend is being split into smaller modules.
"""

from pathlib import Path

from scripts.launch_gpt_sovits import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_PROFILE_PATH,
    DEFAULT_REPO_DIR,
    GPTSoVITSRuntime,
    VoiceRegistry,
)


def build_runtime(
    *,
    repo_dir: Path = DEFAULT_REPO_DIR,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> GPTSoVITSRuntime:
    return GPTSoVITSRuntime(repo_dir=repo_dir, config_path=config_path)


def build_registry(
    *,
    repo_dir: Path = DEFAULT_REPO_DIR,
    profiles_path: Path = DEFAULT_PROFILE_PATH,
) -> VoiceRegistry:
    return VoiceRegistry(profiles_path, repo_dir=repo_dir)

