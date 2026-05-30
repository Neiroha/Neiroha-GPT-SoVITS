"""Runtime adapter export."""

from scripts.launch_gpt_sovits import GPTSoVITSRuntime

EngineRuntime = GPTSoVITSRuntime

__all__ = ["EngineRuntime", "GPTSoVITSRuntime"]

