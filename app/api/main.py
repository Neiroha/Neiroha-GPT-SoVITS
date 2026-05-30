"""ASGI entrypoint for the Neiroha GPT-SoVITS backend."""

from pathlib import Path

from scripts.launch_gpt_sovits import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_PROFILE_PATH,
    DEFAULT_REPO_DIR,
    GPTSoVITSRuntime,
    VoiceRegistry,
    create_api_app,
    materialize_model_preset_config,
)


def build_app(
    *,
    repo_dir: Path = DEFAULT_REPO_DIR,
    config_path: Path = DEFAULT_CONFIG_PATH,
    profiles_path: Path = DEFAULT_PROFILE_PATH,
):
    runtime = GPTSoVITSRuntime(repo_dir=repo_dir, config_path=config_path)
    registry = VoiceRegistry(profiles_path, repo_dir=repo_dir)
    active_preset = registry.get_model_preset(registry.active_model_preset_id())
    if config_path == DEFAULT_CONFIG_PATH:
        runtime.config_path = Path(active_preset.config_path).resolve()
    materialize_model_preset_config(active_preset, runtime.config_path)
    return create_api_app(
        runtime,
        registry,
        default_voice_id=registry.default_voice_id(),
    )


app = build_app()
