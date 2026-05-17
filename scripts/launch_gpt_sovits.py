from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import io
import json
import logging
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterable, Optional, Union

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_DIR = WORKSPACE_ROOT / "GPT-SoVITS"
CONFIG_TEMPLATE_PATH = DEFAULT_REPO_DIR / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
CONFIG_ROOT = WORKSPACE_ROOT / "configs"
SERVER_CONFIG_PATH = CONFIG_ROOT / "server.toml"
UI_CONFIG_PATH = CONFIG_ROOT / "ui.toml"
MODEL_PRESETS_DIR = CONFIG_ROOT / "model-presets"
VOICE_SETS_DIR = CONFIG_ROOT / "voice-sets"
DEFAULT_PROFILE_PATH = WORKSPACE_ROOT / "profiles" / "voices.json"
MODELS_ROOT = WORKSPACE_ROOT / "models"
PRETRAINED_MODELS_DIR = MODELS_ROOT / "pretrained" / "GPT-SoVITS" / "GPT_SoVITS" / "pretrained_models"
DEFAULT_CLONE_GPT_WEIGHTS = PRETRAINED_MODELS_DIR / "s1v3.ckpt"
DEFAULT_CLONE_SOVITS_WEIGHTS = PRETRAINED_MODELS_DIR / "v2Pro" / "s2Gv2ProPlus.pth"
DEFAULT_MODEL_PRESET_ID = "v2proplus-clone"
DEFAULT_VOICE_SET_ID = "default"
DEFAULT_SAMPLE_VOICE_ID = "genshin-keqing"
CLONE_REFERENCE_MIN_SECONDS = 3.05
CLONE_REFERENCE_MAX_SECONDS = 9.95
DEFAULT_DOWNLOAD_SOURCE = "modelscope"
DEFAULT_DEMO_REPO_ID = "UnlimitedBurst/GPT-SoVITS"
DEFAULT_DEMO_SPEAKERS = "派蒙,刻晴,可莉"
DEFAULT_EXTENDED_DEMO_SPEAKERS = "派蒙,刻晴,可莉,胡桃,甘雨,雷电将军,纳西妲,神里绫华,八重神子,钟离"
DEFAULT_SHARED_REPO_ID = "AI-Hobbyist/GPT-SoVits-V2-models"
DEFAULT_SHARED_PRESETS = "genshin-en,genshin-ja,wuthering-cn"
DEFAULT_SHARED_REFERENCE_REPO_ID = "AquaV/genshin-voices-separated"
DEFAULT_SHARED_REFERENCE_CHARACTERS = "Furina,Keqing,Klee,Zhongli,Nahida"
DEFAULT_SHARED_REFERENCE_LANGUAGES = "English(US),Japanese"
RUNTIME_ROOT = WORKSPACE_ROOT / "runtime"
RUNTIME_CACHE_ROOT = RUNTIME_ROOT / "cache"
RUNTIME_LOG_ROOT = RUNTIME_ROOT / "logs"
RUNTIME_STATE_ROOT = RUNTIME_ROOT / "state"
RUNTIME_VOICES_ROOT = RUNTIME_ROOT / "voices"
TEMP_ROOT = RUNTIME_ROOT / "temp"
UPLOAD_ROOT = TEMP_ROOT / "uploads"
OUTPUT_ROOT = RUNTIME_ROOT / "outputs"
LOCAL_REFERENCE_ROOT = MODELS_ROOT / "reference-audio" / "local"
DEFAULT_CONFIG_PATH = RUNTIME_CACHE_ROOT / "tts_infer.yaml"
RUNTIME_EVENT_LOG_PATH = RUNTIME_LOG_ROOT / "backend.log"
RUNTIME_DEBUG_LOG_PATH = RUNTIME_LOG_ROOT / "api-debug.log"

for path in (
    RUNTIME_ROOT,
    RUNTIME_CACHE_ROOT,
    RUNTIME_LOG_ROOT,
    RUNTIME_STATE_ROOT,
    RUNTIME_VOICES_ROOT,
    TEMP_ROOT,
    UPLOAD_ROOT,
    OUTPUT_ROOT,
    LOCAL_REFERENCE_ROOT,
):
    path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TMPDIR", str(TEMP_ROOT))
os.environ.setdefault("TEMP", str(TEMP_ROOT))
os.environ.setdefault("TMP", str(TEMP_ROOT))
os.environ.setdefault("GRADIO_TEMP_DIR", str(TEMP_ROOT / "gradio"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LOGGER = logging.getLogger("neiroha.gpt_sovits")
OPENAI_MODEL_ALIAS = DEFAULT_VOICE_SET_ID
LEGACY_OPENAI_MODEL_ALIASES = {"gpt-sovits", "tts-1", "tts-1-hd"}
SUPPORTED_OPENAI_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "pcm", "ogg", "raw"}
CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "ogg": "audio/ogg",
    "raw": "application/octet-stream",
}


class RuntimeEventLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.RLock()

    def reset_for_launch(self) -> None:
        previous_path = self.path.with_name(f"{self.path.stem}.previous{self.path.suffix}")
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size > 0:
                previous_path.unlink(missing_ok=True)
                try:
                    self.path.replace(previous_path)
                except OSError:
                    previous_path.write_text(
                        self.path.read_text(encoding="utf-8", errors="replace"),
                        encoding="utf-8",
                    )
            self.path.write_text("", encoding="utf-8")

    def append(self, event: str, **fields: Any) -> None:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        details = " ".join(
            f"{key}={self._format_value(value)}"
            for key, value in fields.items()
            if value is not None and self._format_value(value) != ""
        )
        line = f"{timestamp} | {event}"
        if details:
            line = f"{line} | {details}"
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

    def tail(self, limit: int = 120, *, newest_first: bool = True) -> str:
        if not self.path.exists():
            return f"No log file yet: {self.path}"
        with self.lock:
            lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = lines[-max(limit, 1) :]
        if newest_first:
            tail = list(reversed(tail))
        return "\n".join(tail) or f"No log entries yet: {self.path}"

    def _format_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        text = str(value).replace("\r", " ").replace("\n", " ").strip()
        if " " in text:
            return json.dumps(text, ensure_ascii=False)
        return text


RUNTIME_EVENTS = RuntimeEventLog(RUNTIME_EVENT_LOG_PATH)


@contextlib.contextmanager
def runtime_output_scope(debug_enabled: bool):
    target = RUNTIME_DEBUG_LOG_PATH if debug_enabled else Path(os.devnull)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if debug_enabled else "w"
    with target.open(mode, encoding="utf-8", errors="replace") as stream:
        if debug_enabled:
            stream.write(f"\n--- {dt.datetime.now().isoformat(timespec='seconds')} ---\n")
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            yield


def strip_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if value is None:
        value = ""
    return json.dumps(str(value), ensure_ascii=False)


def write_toml_mapping(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    nested: list[tuple[str, dict[str, Any]]] = []
    for key, value in payload.items():
        if isinstance(value, dict):
            nested.append((key, value))
        else:
            lines.append(f"{key} = {toml_value(value)}")
    for table, values in nested:
        lines.append("")
        lines.append(f"[{table}]")
        if values:
            for key, value in values.items():
                lines.append(f"{key} = {toml_value(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def read_mapping_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".toml":
        with path.open("rb") as file:
            payload = tomllib.load(file)
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain an object/table: {path}")
    return payload


def load_ui_config(registry: "VoiceRegistry") -> dict[str, Any]:
    config: dict[str, Any] = {}
    ui_config = registry.server_config().get("ui", {})
    if isinstance(ui_config, dict):
        config.update(ui_config)
    if UI_CONFIG_PATH.exists():
        config.update(read_mapping_file(UI_CONFIG_PATH))
    return config


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = strip_text(value)
        if text:
            return text
    return ""


def model_dump(model: BaseModel) -> dict[str, Any]:
    if isinstance(model, dict):
        return dict(model)
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def json_response(payload: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=payload, status_code=status_code)


def openai_error(
    message: str,
    *,
    status_code: int = 400,
    error_type: str = "invalid_request_error",
) -> JSONResponse:
    return json_response(
        {"error": {"message": message, "type": error_type}},
        status_code=status_code,
    )


def resolve_existing_file(
    raw_path: Optional[str],
    *,
    repo_dir: Path,
    field_name: str,
    required: bool = False,
) -> Optional[str]:
    path_text = strip_text(raw_path)
    if not path_text:
        if required:
            raise FileNotFoundError(f"{field_name} is required.")
        return None

    candidate = Path(path_text).expanduser()
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates = [
            WORKSPACE_ROOT / candidate,
            repo_dir / candidate,
            Path.cwd() / candidate,
        ]

    for item in candidates:
        if item.exists() and item.is_file():
            return str(item.resolve())

    raise FileNotFoundError(f"{field_name} does not exist: {raw_path}")


def resolve_optional_path(raw_path: Optional[str], *, repo_dir: Path) -> Optional[str]:
    path_text = strip_text(raw_path)
    if not path_text:
        return None
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    workspace_candidate = WORKSPACE_ROOT / candidate
    if candidate.parts and candidate.parts[0] in {"configs", "models", "profiles", "runtime"}:
        return str(workspace_candidate.resolve())
    if workspace_candidate.exists():
        return str(workspace_candidate.resolve())
    repo_candidate = repo_dir / candidate
    return str(repo_candidate.resolve())


def profile_path_text(path: Path) -> str:
    resolved = path.resolve()
    with contextlib.suppress(ValueError):
        return resolved.relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    return str(resolved)


def extract_voice_id(voice_value: Any) -> str:
    if isinstance(voice_value, dict):
        return first_non_empty(voice_value.get("id"), voice_value.get("name"))
    return strip_text(voice_value)


def require_supported_format(response_format: str) -> str:
    fmt = strip_text(response_format).lower() or "wav"
    if fmt not in SUPPORTED_OPENAI_FORMATS:
        raise ValueError(
            f"response_format must be one of: {', '.join(sorted(SUPPORTED_OPENAI_FORMATS))}"
        )
    return fmt


def safe_filename_part(value: Any, fallback: str = "speech") -> str:
    text = strip_text(value) or fallback
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text)
    text = re.sub(r"\s+", "_", text).strip("._ ")
    return text or fallback


def safe_ascii_filename_part(value: Any, fallback: str = "speech") -> str:
    raw = strip_text(value) or fallback
    text = safe_filename_part(raw, fallback=fallback)
    ascii_text = text.encode("ascii", errors="ignore").decode("ascii")
    ascii_text = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_text)
    ascii_text = re.sub(r"_+", "_", ascii_text).strip("._-")
    if ascii_text:
        return ascii_text[:80]
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{fallback}_{digest}"


def write_runtime_output(content: bytes, speaker: Any, response_format: str) -> Path:
    fmt = require_supported_format(response_format)
    suffix = "raw" if fmt in {"pcm", "raw"} else fmt
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    speaker_name = safe_ascii_filename_part(speaker)
    path = OUTPUT_ROOT / f"{speaker_name}_{timestamp}.{suffix}"
    counter = 1
    while path.exists():
        path = OUTPUT_ROOT / f"{speaker_name}_{timestamp}_{counter}.{suffix}"
        counter += 1
    path.write_bytes(content)
    return path


def header_safe_path(path: Path) -> str:
    text = str(path)
    try:
        text.encode("latin-1")
        return text
    except UnicodeEncodeError:
        return profile_path_text(path)


def ensure_default_config(config_path: Path) -> Path:
    config_path = config_path.resolve()
    if config_path.exists():
        return config_path

    config_path.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_TEMPLATE_PATH.exists():
        shutil.copyfile(CONFIG_TEMPLATE_PATH, config_path)
    else:
        legacy_template = CONFIG_ROOT / "tts_infer.yaml"
        if legacy_template.exists():
            shutil.copyfile(legacy_template, config_path)
        else:
            raise FileNotFoundError(f"Default TTS config template is missing: {CONFIG_TEMPLATE_PATH}")
    return config_path


def audio_to_int16(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        return data
    if np.issubdtype(data.dtype, np.floating):
        clipped = np.clip(data, -1.0, 1.0)
        return (clipped * 32767).astype("<i2")
    return data.astype("<i2")


def pack_soundfile(data: np.ndarray, rate: int, fmt: str) -> bytes:
    buffer = io.BytesIO()
    sf_format = {"wav": "WAV", "flac": "FLAC", "ogg": "OGG"}.get(fmt)
    if sf_format is None:
        raise ValueError(f"Unsupported soundfile format: {fmt}")
    sf.write(buffer, data, rate, format=sf_format)
    return buffer.getvalue()


def pack_ffmpeg(data: np.ndarray, rate: int, fmt: str) -> bytes:
    pcm = audio_to_int16(data).tobytes()
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-vn",
    ]
    if fmt == "mp3":
        command += ["-f", "mp3", "pipe:1"]
    elif fmt == "aac":
        command += ["-c:a", "aac", "-b:a", "192k", "-f", "adts", "pipe:1"]
    elif fmt == "opus":
        command += ["-c:a", "libopus", "-b:a", "64k", "-f", "ogg", "pipe:1"]
    else:
        raise ValueError(f"Unsupported ffmpeg format: {fmt}")

    process = subprocess.run(
        command,
        input=pcm,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed to encode {fmt}: {detail}")
    return process.stdout


def pack_audio(data: np.ndarray, rate: int, response_format: str) -> bytes:
    fmt = require_supported_format(response_format)
    if fmt in {"pcm", "raw"}:
        return audio_to_int16(data).tobytes()
    if fmt in {"wav", "flac", "ogg"}:
        return pack_soundfile(data, rate, fmt)
    return pack_ffmpeg(data, rate, fmt)


def wave_header_chunk(sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"")
    return buffer.getvalue()


@dataclass
class ModelPreset:
    id: str
    name: str
    engine: str = "gpt-sovits"
    config_path: str = str(DEFAULT_CONFIG_PATH)
    gpt_weights_path: str = str(DEFAULT_CLONE_GPT_WEIGHTS)
    sovits_weights_path: str = str(DEFAULT_CLONE_SOVITS_WEIGHTS)

    @classmethod
    def from_toml(cls, payload: dict[str, Any], *, repo_dir: Path) -> "ModelPreset":
        preset_id = strip_text(payload.get("id")) or DEFAULT_MODEL_PRESET_ID
        gpt_sovits = payload.get("gpt_sovits") if isinstance(payload.get("gpt_sovits"), dict) else {}
        return cls(
            id=preset_id,
            name=first_non_empty(payload.get("name"), preset_id),
            engine=strip_text(payload.get("engine")) or "gpt-sovits",
            config_path=resolve_optional_path(gpt_sovits.get("config_path"), repo_dir=repo_dir)
            or str(DEFAULT_CONFIG_PATH),
            gpt_weights_path=resolve_optional_path(gpt_sovits.get("gpt_weights_path"), repo_dir=repo_dir)
            or str(DEFAULT_CLONE_GPT_WEIGHTS),
            sovits_weights_path=resolve_optional_path(gpt_sovits.get("sovits_weights_path"), repo_dir=repo_dir)
            or str(DEFAULT_CLONE_SOVITS_WEIGHTS),
        )

    def to_native_model(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "gpt_sovits.model_preset",
            "type": "clone",
            "name": self.name,
            "engine": self.engine,
            "config_path": self.config_path,
            "gpt_weights_path": self.gpt_weights_path,
            "sovits_weights_path": self.sovits_weights_path,
            "requires_reference_audio": True,
        }


@dataclass
class VoiceSet:
    id: str
    name: str
    description: str = ""
    voices: list[str] | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "VoiceSet":
        set_id = first_non_empty(payload.get("id"), payload.get("name"), DEFAULT_VOICE_SET_ID)
        voices = payload.get("voices") if isinstance(payload.get("voices"), list) else []
        return cls(
            id=set_id,
            name=first_non_empty(payload.get("name"), set_id),
            description=strip_text(payload.get("description")),
            voices=[strip_text(item) for item in voices if strip_text(item)],
        )

    def to_openai_model(self, voice_count: int) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "model",
            "owned_by": "neiroha",
            "name": self.name,
            "description": self.description,
            "voice_count": voice_count,
        }


@dataclass
class VoiceProfile:
    id: str
    name: str
    ref_audio_path: str = ""
    prompt_text: str = ""
    prompt_lang: str = "zh"
    text_lang: str = "zh"
    aux_ref_audio_paths: list[str] | None = None
    gpt_weights_path: str = ""
    sovits_weights_path: str = ""
    description: str = ""
    model_id: str = ""
    model_name: str = ""
    model_type: str = "prompt_clone"
    voice_set_id: str = DEFAULT_VOICE_SET_ID
    voice_set_name: str = "Default"
    model_preset: str = DEFAULT_MODEL_PRESET_ID
    instruction: str = ""
    speed: float = 1.0
    engine_options: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], *, repo_dir: Path) -> "VoiceProfile":
        profile_id = first_non_empty(payload.get("id"), payload.get("name"))
        if not profile_id:
            raise ValueError("Voice profile requires an id or name.")
        ref_audio_path = resolve_optional_path(
            first_non_empty(payload.get("ref_audio_path"), payload.get("reference_audio")),
            repo_dir=repo_dir,
        ) or ""
        aux_paths = payload.get("aux_ref_audio_paths") or []
        resolved_aux = [
            resolve_optional_path(path, repo_dir=repo_dir) or ""
            for path in aux_paths
            if strip_text(path)
        ]
        return cls(
            id=profile_id,
            name=first_non_empty(payload.get("name"), profile_id),
            ref_audio_path=ref_audio_path,
            prompt_text=strip_text(payload.get("prompt_text")),
            prompt_lang=strip_text(payload.get("prompt_lang")) or "zh",
            text_lang=strip_text(payload.get("text_lang")) or "zh",
            aux_ref_audio_paths=resolved_aux,
            gpt_weights_path=resolve_optional_path(payload.get("gpt_weights_path"), repo_dir=repo_dir)
            or "",
            sovits_weights_path=resolve_optional_path(payload.get("sovits_weights_path"), repo_dir=repo_dir)
            or "",
            description=strip_text(payload.get("description")),
            model_id=strip_text(payload.get("model_id")),
            model_name=strip_text(payload.get("model_name")),
            model_type=strip_text(payload.get("model_type") or payload.get("mode")) or "prompt_clone",
            voice_set_id=strip_text(payload.get("voice_set_id") or payload.get("model")) or DEFAULT_VOICE_SET_ID,
            voice_set_name=strip_text(payload.get("voice_set_name")) or "Default",
            model_preset=strip_text(payload.get("model_preset")) or DEFAULT_MODEL_PRESET_ID,
            instruction=strip_text(payload.get("instruction")),
            speed=float(payload.get("speed") or 1.0),
            engine_options=payload.get("engine_options") if isinstance(payload.get("engine_options"), dict) else {},
        )

    def to_openai_voice(self) -> dict[str, Any]:
        model = self.voice_set_id or self.model_id or DEFAULT_VOICE_SET_ID
        task_mode = self.model_type or "prompt_clone"
        return {
            "id": self.id,
            "voice_id": self.id,
            "name": self.name,
            "object": "voice",
            "description": self.description,
            "model": model,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "task_mode": task_mode,
            "mode": task_mode,
            "model_preset": self.model_preset,
            "text_lang": self.text_lang,
            "prompt_lang": self.prompt_lang,
        }

    def to_speaker(self) -> dict[str, str]:
        return {"name": self.name, "voice_id": self.id}

    def to_native_voice(self, *, model_id: str = "") -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "gpt_sovits.voice",
            "name": self.name,
            "description": self.description,
            "model_id": model_id or self.model_id,
            "model_type": self.model_type,
            "voice_set_id": self.voice_set_id,
            "model_preset": self.model_preset,
            "prompt_lang": self.prompt_lang,
            "text_lang": self.text_lang,
            "has_reference_audio": bool(self.ref_audio_path),
            "has_gpt_weights": bool(self.gpt_weights_path),
            "has_sovits_weights": bool(self.sovits_weights_path),
        }


class VoiceRegistry:
    def __init__(
        self,
        profile_path: Path,
        *,
        repo_dir: Path,
        model_presets_dir: Path = MODEL_PRESETS_DIR,
        voice_sets_dir: Path = VOICE_SETS_DIR,
        runtime_voices_dir: Path = RUNTIME_VOICES_ROOT,
        server_config_path: Path = SERVER_CONFIG_PATH,
        active_state_path: Path = RUNTIME_STATE_ROOT / "active.toml",
    ) -> None:
        self.profile_path = profile_path
        self.repo_dir = repo_dir
        self.model_presets_dir = model_presets_dir
        self.voice_sets_dir = voice_sets_dir
        self.runtime_voices_dir = runtime_voices_dir
        self.server_config_path = server_config_path
        self.active_state_path = active_state_path

    def server_config(self) -> dict[str, Any]:
        if not self.server_config_path.exists():
            return {}
        with self.server_config_path.open("rb") as file:
            return tomllib.load(file)

    def active_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        runtime_config = self.server_config().get("runtime", {})
        if isinstance(runtime_config, dict):
            state.update(runtime_config)
        candidates = [self.active_state_path]
        if self.active_state_path.suffix.lower() == ".toml":
            candidates.append(self.active_state_path.with_suffix(".json"))
        for path in candidates:
            if not path.exists():
                continue
            data = read_mapping_file(path)
            if isinstance(data, dict):
                state.update(data)
            break
        return state

    def active_model_preset_id(self) -> str:
        return strip_text(self.active_state().get("active_model_preset")) or DEFAULT_MODEL_PRESET_ID

    def active_voice_set_id(self) -> str:
        return strip_text(self.active_state().get("active_voice_set")) or DEFAULT_VOICE_SET_ID

    def default_voice_id(self) -> str:
        return strip_text(self.active_state().get("default_voice")) or DEFAULT_SAMPLE_VOICE_ID

    def _uses_layered_config(self) -> bool:
        return self.voice_sets_dir.exists() or self.runtime_voices_dir.exists() or self.model_presets_dir.exists()

    def _read_payload(self) -> list[dict[str, Any]]:
        if not self.profile_path.exists():
            return []
        data = json.loads(self.profile_path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("voices", [])
        if not isinstance(data, list):
            raise ValueError(f"Voice profile file must contain a list or voices object: {self.profile_path}")
        return data

    def list_model_presets(self) -> list[ModelPreset]:
        presets: list[ModelPreset] = []
        if self.model_presets_dir.exists():
            for path in sorted(self.model_presets_dir.glob("*.toml")):
                with path.open("rb") as file:
                    payload = tomllib.load(file)
                presets.append(ModelPreset.from_toml(payload, repo_dir=self.repo_dir))
        if not presets:
            presets.append(
                ModelPreset(
                    id=DEFAULT_MODEL_PRESET_ID,
                    name="GPT-SoVITS v2ProPlus Clone",
                    config_path=str(DEFAULT_CONFIG_PATH),
                    gpt_weights_path=str(DEFAULT_CLONE_GPT_WEIGHTS),
                    sovits_weights_path=str(DEFAULT_CLONE_SOVITS_WEIGHTS),
                )
            )
        return presets

    def get_model_preset(self, preset_id: str) -> ModelPreset:
        preset_id = strip_text(preset_id) or self.active_model_preset_id()
        for preset in self.list_model_presets():
            if preset.id == preset_id:
                return preset
        raise ValueError(f"Unknown model preset: {preset_id}")

    def list_voice_sets(self) -> list[VoiceSet]:
        voice_sets: list[VoiceSet] = []
        if self.voice_sets_dir.exists():
            seen: set[str] = set()
            for pattern in ("*.toml", "*.json"):
                for path in sorted(self.voice_sets_dir.glob(pattern)):
                    payload = read_mapping_file(path)
                    voice_set = VoiceSet.from_mapping(payload)
                    if voice_set.id in seen:
                        continue
                    seen.add(voice_set.id)
                    voice_sets.append(voice_set)
        if not voice_sets:
            profiles = self._read_payload()
            voices = [strip_text(item.get("id") or item.get("name")) for item in profiles if isinstance(item, dict)]
            voice_sets.append(
                VoiceSet(
                    id=DEFAULT_VOICE_SET_ID,
                    name="Default",
                    description="Legacy profiles/voices.json voice set.",
                    voices=[item for item in voices if item],
                )
            )
        return voice_sets

    def get_voice_set(self, model_id: str = "") -> Optional[VoiceSet]:
        target = self.normalize_voice_set_id(model_id)
        for voice_set in self.list_voice_sets():
            if target in {voice_set.id, voice_set.name}:
                return voice_set
        return None

    def normalize_voice_set_id(self, model_id: str = "") -> str:
        model_id = strip_text(model_id)
        if not model_id or model_id in LEGACY_OPENAI_MODEL_ALIASES:
            return self.active_voice_set_id()
        return model_id

    def has_voice_set(self, model_id: str = "") -> bool:
        return self.get_voice_set(model_id) is not None

    def _read_voice_profile(self, voice_id: str, voice_set: VoiceSet) -> Optional[VoiceProfile]:
        voice_dir = self.runtime_voices_dir / voice_id
        voice_path = voice_dir / "voice.toml"
        if not voice_path.exists():
            voice_path = voice_dir / "voice.json"
        if not voice_path.exists():
            return None
        payload = read_mapping_file(voice_path)
        preset_id = strip_text(payload.get("model_preset")) or self.active_model_preset_id()
        preset = self.get_model_preset(preset_id)
        payload = {
            **payload,
            "voice_set_id": voice_set.id,
            "voice_set_name": voice_set.name,
            "model_id": voice_set.id,
            "model_name": voice_set.name,
            "gpt_weights_path": payload.get("gpt_weights_path") or preset.gpt_weights_path,
            "sovits_weights_path": payload.get("sovits_weights_path") or preset.sovits_weights_path,
            "model_preset": preset.id,
        }
        return VoiceProfile.from_mapping(payload, repo_dir=self.repo_dir)

    def list_profiles(self, model_id: str = "") -> list[VoiceProfile]:
        if self._uses_layered_config():
            profiles: list[VoiceProfile] = []
            voice_sets = self.list_voice_sets()
            target_set_id = self.normalize_voice_set_id(model_id) if strip_text(model_id) else ""
            for voice_set in voice_sets:
                if target_set_id and voice_set.id != target_set_id:
                    continue
                for voice_id in voice_set.voices or []:
                    profile = self._read_voice_profile(voice_id, voice_set)
                    if profile is not None:
                        profiles.append(profile)
            return profiles

        profiles = []
        for payload in self._read_payload():
            if not isinstance(payload, dict):
                continue
            profiles.append(VoiceProfile.from_mapping(payload, repo_dir=self.repo_dir))
        return profiles

    def get(self, voice_id: str, *, model_id: str = "") -> Optional[VoiceProfile]:
        voice_id = strip_text(voice_id)
        if not voice_id:
            return None
        for profile in self.list_profiles(model_id):
            if voice_id in {profile.id, profile.name}:
                return profile
        return None

    def first(self, model_id: str = "") -> Optional[VoiceProfile]:
        profiles = self.list_profiles(model_id)
        return profiles[0] if profiles else None


class TTSRequest(BaseModel):
    text: Optional[str] = None
    text_lang: Optional[str] = None
    ref_audio_path: Optional[str] = None
    aux_ref_audio_paths: Optional[list[str]] = None
    prompt_lang: Optional[str] = None
    prompt_text: str = ""
    top_k: int = 15
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: Union[bool, int] = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16


class OpenAISpeechRequest(BaseModel):
    model: str = OPENAI_MODEL_ALIAS
    input: Optional[str] = Field(default=None, description="Text to synthesize.")
    text: Optional[str] = Field(default=None, description="Alias for input.")
    voice: str | dict[str, Any] | None = "default"
    instructions: str = ""
    response_format: str = "mp3"
    speed: float = 1.0
    stream_format: str = "audio"

    # Local GPT-SoVITS extensions. These let OpenAI-compatible clients pass
    # reference audio without relying on a saved voice profile.
    text_lang: Optional[str] = None
    prompt_lang: Optional[str] = None
    prompt_text: str = ""
    ref_audio_path: Optional[str] = None
    reference_audio: Optional[str] = None
    reference_audio_path: Optional[str] = None
    aux_ref_audio_paths: Optional[list[str]] = None
    gpt_weights_path: Optional[str] = None
    sovits_weights_path: Optional[str] = None
    text_split_method: str = "cut5"
    batch_size: int = 1
    top_k: int = 15
    top_p: float = 1
    temperature: float = 1
    seed: int = -1
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False


class CloneSpeechRequest(BaseModel):
    model: str = OPENAI_MODEL_ALIAS
    input: Optional[str] = Field(default=None, description="Text to synthesize.")
    text: Optional[str] = Field(default=None, description="Alias for input.")
    speaker: str = "clone"
    response_format: str = "mp3"
    speed: float = 1.0
    text_lang: str = "zh"
    prompt_lang: str = "zh"
    prompt_text: str = ""
    ref_audio_path: Optional[str] = None
    reference_audio: Optional[str] = None
    reference_audio_path: Optional[str] = None
    aux_ref_audio_paths: Optional[list[str]] = None
    gpt_weights_path: Optional[str] = None
    sovits_weights_path: Optional[str] = None
    text_split_method: str = "cut5"
    batch_size: int = 1
    top_k: int = 15
    top_p: float = 1
    temperature: float = 1
    seed: int = -1
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False


class LoadRequest(BaseModel):
    config_path: Optional[str] = None
    gpt_weights_path: Optional[str] = None
    sovits_weights_path: Optional[str] = None


class WeightsRequest(BaseModel):
    weights_path: str


class ReferAudioRequest(BaseModel):
    refer_audio_path: str


class ControlRequest(BaseModel):
    command: str


class GPTSoVITSRuntime:
    def __init__(
        self,
        *,
        repo_dir: Path,
        config_path: Path,
        device: str = "config",
        is_half: Optional[bool] = None,
        debug_runtime_output: bool = False,
    ) -> None:
        self.repo_dir = repo_dir.resolve()
        self.config_path = config_path.resolve()
        self.device_override = device
        self.is_half_override = is_half
        self.debug_runtime_output = debug_runtime_output
        self.lock = threading.RLock()
        self.tts_config = None
        self.tts_pipeline = None
        self.cut_method_names: list[str] = []
        self.current_gpt_weights_path = ""
        self.current_sovits_weights_path = ""
        self._imports_ready = False

    def _prepare_imports(self) -> None:
        if self._imports_ready:
            return
        if not self.repo_dir.exists():
            raise FileNotFoundError(
                f"GPT-SoVITS submodule is missing: {self.repo_dir}. "
                "Run `git submodule update --init --recursive` first."
            )
        for path in (self.repo_dir, self.repo_dir / "GPT_SoVITS"):
            path_text = str(path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)
        os.chdir(self.repo_dir)
        self._imports_ready = True

    def _build_config(self, config_path: Optional[str] = None):
        self._prepare_imports()
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config

        selected_config = resolve_optional_path(config_path, repo_dir=self.repo_dir) or str(self.config_path)
        selected_config = str(ensure_default_config(Path(selected_config)))
        tts_config = TTS_Config(selected_config)

        if self.device_override and self.device_override != "config":
            import torch

            if self.device_override == "auto":
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                resolved_device = self.device_override
            tts_config.device = torch.device(resolved_device) if resolved_device == "cpu" else resolved_device

        if self.is_half_override is not None:
            tts_config.is_half = bool(self.is_half_override)

        if str(tts_config.device) == "cpu" and tts_config.is_half:
            tts_config.is_half = False

        tts_config.update_configs()
        return tts_config

    def load(self, config_path: Optional[str] = None) -> dict[str, Any]:
        with self.lock:
            self._prepare_imports()
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS
            from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names

            self.tts_config = self._build_config(config_path)
            started_at = time.perf_counter()
            RUNTIME_EVENTS.append("model_load_start", config_path=str(self.tts_config.configs_path))
            with runtime_output_scope(self.debug_runtime_output):
                self.tts_pipeline = TTS(self.tts_config)
            self.cut_method_names = get_method_names()
            self.current_gpt_weights_path = str(self.tts_config.t2s_weights_path)
            self.current_sovits_weights_path = str(self.tts_config.vits_weights_path)
            status = self.status()
            RUNTIME_EVENTS.append(
                "model_load_complete",
                elapsed_seconds=round(time.perf_counter() - started_at, 3),
                config_path=str(self.tts_config.configs_path),
                gpt_weights_path=self.current_gpt_weights_path,
                sovits_weights_path=self.current_sovits_weights_path,
                device=status.get("device"),
                is_half=status.get("is_half"),
            )
            return status

    def unload(self) -> dict[str, Any]:
        with self.lock:
            self.tts_pipeline = None
            self.tts_config = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                LOGGER.debug("Unable to empty CUDA cache after unload.", exc_info=True)
            RUNTIME_EVENTS.append("model_unload")
            return self.status()

    def reload(self, config_path: Optional[str] = None) -> dict[str, Any]:
        with self.lock:
            self.unload()
            return self.load(config_path)

    def get_or_load(self):
        with self.lock:
            if self.tts_pipeline is None:
                self.load()
            return self.tts_pipeline

    def set_gpt_weights(self, weights_path: str) -> dict[str, Any]:
        resolved = resolve_existing_file(weights_path, repo_dir=self.repo_dir, field_name="gpt_weights_path", required=True)
        with self.lock:
            started_at = time.perf_counter()
            RUNTIME_EVENTS.append("gpt_weights_load_start", weights_path=resolved)
            pipeline = self.get_or_load()
            with runtime_output_scope(self.debug_runtime_output):
                pipeline.init_t2s_weights(resolved)
            self.current_gpt_weights_path = resolved
            status = self.status()
            RUNTIME_EVENTS.append(
                "gpt_weights_load_complete",
                elapsed_seconds=round(time.perf_counter() - started_at, 3),
                weights_path=resolved,
            )
            return status

    def set_sovits_weights(self, weights_path: str) -> dict[str, Any]:
        resolved = resolve_existing_file(
            weights_path,
            repo_dir=self.repo_dir,
            field_name="sovits_weights_path",
            required=True,
        )
        with self.lock:
            started_at = time.perf_counter()
            RUNTIME_EVENTS.append("sovits_weights_load_start", weights_path=resolved)
            pipeline = self.get_or_load()
            with runtime_output_scope(self.debug_runtime_output):
                pipeline.init_vits_weights(resolved)
            self.current_sovits_weights_path = resolved
            status = self.status()
            RUNTIME_EVENTS.append(
                "sovits_weights_load_complete",
                elapsed_seconds=round(time.perf_counter() - started_at, 3),
                weights_path=resolved,
            )
            return status

    def set_refer_audio(self, refer_audio_path: str) -> dict[str, Any]:
        resolved = resolve_existing_file(
            refer_audio_path,
            repo_dir=self.repo_dir,
            field_name="refer_audio_path",
            required=True,
        )
        with self.lock:
            pipeline = self.get_or_load()
            with runtime_output_scope(self.debug_runtime_output):
                pipeline.set_ref_audio(resolved)
            return {"message": "success", "refer_audio_path": resolved}

    def apply_profile_weights(self, profile: Optional[VoiceProfile]) -> None:
        if profile is None:
            return
        if profile.gpt_weights_path and profile.gpt_weights_path != self.current_gpt_weights_path:
            self.set_gpt_weights(profile.gpt_weights_path)
        if profile.sovits_weights_path and profile.sovits_weights_path != self.current_sovits_weights_path:
            self.set_sovits_weights(profile.sovits_weights_path)

    def validate_request(self, request: dict[str, Any]) -> dict[str, Any]:
        text = strip_text(request.get("text"))
        if not text:
            raise ValueError("text is required.")
        request["text"] = text

        text_lang = strip_text(request.get("text_lang")).lower() or "zh"
        prompt_lang = strip_text(request.get("prompt_lang")).lower() or text_lang
        request["text_lang"] = text_lang
        request["prompt_lang"] = prompt_lang

        ref_audio_path = resolve_existing_file(
            request.get("ref_audio_path"),
            repo_dir=self.repo_dir,
            field_name="ref_audio_path",
            required=True,
        )
        request["ref_audio_path"] = ref_audio_path

        aux_paths = []
        for aux_path in request.get("aux_ref_audio_paths") or []:
            aux_paths.append(
                resolve_existing_file(
                    aux_path,
                    repo_dir=self.repo_dir,
                    field_name="aux_ref_audio_paths",
                    required=True,
                )
            )
        request["aux_ref_audio_paths"] = aux_paths

        media_type = require_supported_format(strip_text(request.get("media_type")) or "wav")
        request["media_type"] = media_type

        config = self.tts_config or self._build_config()
        languages = getattr(config, "languages", [])
        if languages:
            if text_lang not in languages:
                raise ValueError(f"text_lang '{text_lang}' is not supported. Supported: {', '.join(languages)}")
            if prompt_lang not in languages:
                raise ValueError(f"prompt_lang '{prompt_lang}' is not supported. Supported: {', '.join(languages)}")

        if self.cut_method_names and request.get("text_split_method") not in self.cut_method_names:
            raise ValueError(
                f"text_split_method '{request.get('text_split_method')}' is not supported. "
                f"Supported: {', '.join(self.cut_method_names)}"
            )

        return request

    def normalize_streaming_mode(self, request: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        streaming_mode = request.get("streaming_mode", False)
        return_fragment = request.get("return_fragment", False)
        fixed_length_chunk = False

        if isinstance(streaming_mode, bool):
            pass
        elif streaming_mode == 0:
            streaming_mode = False
            return_fragment = False
        elif streaming_mode == 1:
            streaming_mode = False
            return_fragment = True
        elif streaming_mode == 2:
            streaming_mode = True
        elif streaming_mode == 3:
            streaming_mode = True
            fixed_length_chunk = True
        else:
            raise ValueError("streaming_mode must be 0, 1, 2, 3, true, or false.")

        request["streaming_mode"] = streaming_mode
        request["return_fragment"] = return_fragment
        request["fixed_length_chunk"] = fixed_length_chunk
        return request, bool(streaming_mode or return_fragment)

    def synthesize_once(self, request: dict[str, Any]) -> tuple[int, np.ndarray]:
        with self.lock:
            pipeline = self.get_or_load()
            request = self.validate_request(request)
            request, is_streaming = self.normalize_streaming_mode(request)
            if is_streaming:
                request["streaming_mode"] = False
                request["return_fragment"] = False
                request["fixed_length_chunk"] = False
            with runtime_output_scope(self.debug_runtime_output):
                generator = pipeline.run(request)
                return next(generator)

    def synthesize_stream(self, request: dict[str, Any]) -> Generator[tuple[int, np.ndarray], None, None]:
        with self.lock:
            pipeline = self.get_or_load()
            request = self.validate_request(request)
            request, _ = self.normalize_streaming_mode(request)
            with runtime_output_scope(self.debug_runtime_output):
                for item in pipeline.run(request):
                    yield item

    def status(self) -> dict[str, Any]:
        config = self.tts_config
        return {
            "provider": "gpt-sovits",
            "repo_dir": str(self.repo_dir),
            "config_path": str(self.config_path),
            "loaded": self.tts_pipeline is not None,
            "device": str(getattr(config, "device", self.device_override)),
            "is_half": bool(getattr(config, "is_half", self.is_half_override or False)),
            "version": getattr(config, "version", None),
            "languages": list(getattr(config, "languages", [])),
            "cut_methods": self.cut_method_names,
            "gpt_weights_path": self.current_gpt_weights_path,
            "sovits_weights_path": self.current_sovits_weights_path,
        }


def native_request_from_query(
    *,
    text: Optional[str],
    text_lang: Optional[str],
    ref_audio_path: Optional[str],
    prompt_lang: Optional[str],
    prompt_text: str,
    aux_ref_audio_paths: Optional[list[str]],
    top_k: int,
    top_p: float,
    temperature: float,
    text_split_method: str,
    batch_size: int,
    batch_threshold: float,
    split_bucket: bool,
    speed_factor: float,
    fragment_interval: float,
    seed: int,
    media_type: str,
    streaming_mode: Union[bool, int],
    parallel_infer: bool,
    repetition_penalty: float,
    sample_steps: int,
    super_sampling: bool,
    overlap_length: int,
    min_chunk_length: int,
) -> dict[str, Any]:
    return {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": batch_size,
        "batch_threshold": batch_threshold,
        "speed_factor": speed_factor,
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty,
        "sample_steps": sample_steps,
        "super_sampling": super_sampling,
        "overlap_length": overlap_length,
        "min_chunk_length": min_chunk_length,
    }


def request_with_profile(
    payload: OpenAISpeechRequest,
    *,
    runtime: GPTSoVITSRuntime,
    registry: VoiceRegistry,
    default_voice_id: str = "",
) -> tuple[dict[str, Any], Optional[VoiceProfile]]:
    data = model_dump(payload)
    model_id = strip_text(data.get("model"))
    voice_id = extract_voice_id(data.get("voice"))
    if voice_id in {"", "default"} and strip_text(default_voice_id):
        voice_id = strip_text(default_voice_id)
    profile = registry.get(voice_id, model_id=model_id)
    if profile is None and voice_id in {"", "default"}:
        profile = registry.first(model_id)

    text = first_non_empty(data.get("input"), data.get("text"))
    response_format = require_supported_format(data.get("response_format") or "mp3")
    ref_audio = first_non_empty(
        data.get("ref_audio_path"),
        data.get("reference_audio"),
        data.get("reference_audio_path"),
        profile.ref_audio_path if profile else "",
    )
    prompt_text = first_non_empty(data.get("prompt_text"), profile.prompt_text if profile else "")
    text_lang = first_non_empty(data.get("text_lang"), profile.text_lang if profile else "zh")
    prompt_lang = first_non_empty(data.get("prompt_lang"), profile.prompt_lang if profile else text_lang)
    aux_ref_audio_paths = data.get("aux_ref_audio_paths") or (profile.aux_ref_audio_paths if profile else [])

    request = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
        "top_k": data.get("top_k", 15),
        "top_p": data.get("top_p", 1),
        "temperature": data.get("temperature", 1),
        "text_split_method": data.get("text_split_method", "cut5"),
        "batch_size": data.get("batch_size", 1),
        "batch_threshold": 0.75,
        "speed_factor": data.get("speed") or (profile.speed if profile else 1.0),
        "fragment_interval": 0.3,
        "seed": data.get("seed", -1),
        "media_type": response_format,
        "streaming_mode": False,
        "parallel_infer": True,
        "repetition_penalty": data.get("repetition_penalty", 1.35),
        "sample_steps": data.get("sample_steps", 32),
        "super_sampling": data.get("super_sampling", False),
        "overlap_length": 2,
        "min_chunk_length": 16,
    }

    gpt_weights_path = first_non_empty(data.get("gpt_weights_path"), profile.gpt_weights_path if profile else "")
    sovits_weights_path = first_non_empty(
        data.get("sovits_weights_path"),
        profile.sovits_weights_path if profile else "",
    )
    if gpt_weights_path or sovits_weights_path:
        profile = profile or VoiceProfile(id=voice_id or "request", name=voice_id or "request")
        profile.gpt_weights_path = resolve_optional_path(gpt_weights_path, repo_dir=runtime.repo_dir) or ""
        profile.sovits_weights_path = resolve_optional_path(sovits_weights_path, repo_dir=runtime.repo_dir) or ""

    return request, profile


def request_with_clone(
    payload: CloneSpeechRequest,
    *,
    runtime: GPTSoVITSRuntime,
) -> tuple[dict[str, Any], VoiceProfile]:
    data = model_dump(payload)
    text = first_non_empty(data.get("input"), data.get("text"))
    response_format = require_supported_format(data.get("response_format") or "mp3")
    ref_audio = first_non_empty(
        data.get("ref_audio_path"),
        data.get("reference_audio"),
        data.get("reference_audio_path"),
    )
    prompt_text = strip_text(data.get("prompt_text"))
    if not ref_audio:
        raise ValueError("ref_audio_path or reference_audio is required for clone synthesis.")
    if not prompt_text:
        raise ValueError("prompt_text is required and must match the reference audio.")

    request = {
        "text": text,
        "text_lang": strip_text(data.get("text_lang")) or "zh",
        "ref_audio_path": ref_audio,
        "aux_ref_audio_paths": data.get("aux_ref_audio_paths") or [],
        "prompt_text": prompt_text,
        "prompt_lang": strip_text(data.get("prompt_lang")) or strip_text(data.get("text_lang")) or "zh",
        "top_k": data.get("top_k", 15),
        "top_p": data.get("top_p", 1),
        "temperature": data.get("temperature", 1),
        "text_split_method": data.get("text_split_method", "cut5"),
        "batch_size": data.get("batch_size", 1),
        "batch_threshold": 0.75,
        "speed_factor": data.get("speed", 1.0),
        "fragment_interval": 0.3,
        "seed": data.get("seed", -1),
        "media_type": response_format,
        "streaming_mode": False,
        "parallel_infer": True,
        "repetition_penalty": data.get("repetition_penalty", 1.35),
        "sample_steps": data.get("sample_steps", 32),
        "super_sampling": data.get("super_sampling", False),
        "overlap_length": 2,
        "min_chunk_length": 16,
    }

    gpt_weights_path = first_non_empty(data.get("gpt_weights_path"), str(DEFAULT_CLONE_GPT_WEIGHTS))
    sovits_weights_path = first_non_empty(data.get("sovits_weights_path"), str(DEFAULT_CLONE_SOVITS_WEIGHTS))
    speaker = first_non_empty(data.get("speaker"), "clone")
    profile = VoiceProfile(
        id="clone-v2proplus",
        name=speaker,
        gpt_weights_path=resolve_optional_path(gpt_weights_path, repo_dir=runtime.repo_dir) or "",
        sovits_weights_path=resolve_optional_path(sovits_weights_path, repo_dir=runtime.repo_dir) or "",
        description="v2ProPlus reference-audio clone mode.",
    )
    return request, profile


def native_model_catalog(registry: VoiceRegistry) -> list[dict[str, Any]]:
    return [preset.to_native_model() for preset in registry.list_model_presets()]


def audio_response(
    sample_rate: int,
    audio_data: np.ndarray,
    response_format: str,
    *,
    speaker: Any = "",
    metrics: Optional[dict[str, float]] = None,
) -> Response:
    fmt = require_supported_format(response_format)
    content = pack_audio(audio_data, sample_rate, fmt)
    output_path = write_runtime_output(content, speaker or "speech", fmt)
    headers = {
        "Content-Disposition": f'inline; filename="{output_path.name}"',
        "X-Neiroha-Output-Path": header_safe_path(output_path),
    }
    if metrics:
        headers.update(
            {
                "X-Neiroha-Audio-Seconds": f"{metrics.get('audio_seconds', 0.0):.6f}",
                "X-Neiroha-Elapsed-Seconds": f"{metrics.get('elapsed_seconds', 0.0):.6f}",
                "X-Neiroha-RTF": f"{metrics.get('rtf', 0.0):.6f}",
            }
        )
    return Response(
        content=content,
        media_type=CONTENT_TYPES[fmt],
        headers=headers,
    )


def streaming_audio_response(
    chunks: Iterable[tuple[int, np.ndarray]],
    *,
    response_format: str,
) -> StreamingResponse:
    fmt = require_supported_format(response_format)

    def generator() -> Generator[bytes, None, None]:
        wav_header_sent = False
        for sample_rate, chunk in chunks:
            if fmt == "wav":
                if not wav_header_sent:
                    yield wave_header_chunk(sample_rate=sample_rate)
                    wav_header_sent = True
                yield audio_to_int16(chunk).tobytes()
            else:
                yield pack_audio(chunk, sample_rate, fmt)

    return StreamingResponse(generator(), media_type=CONTENT_TYPES[fmt])


def log_rtf(
    terminal_enabled: bool,
    *,
    mode: str,
    speaker: str,
    sample_rate: int,
    audio_data: np.ndarray,
    started_at: float,
) -> dict[str, float]:
    elapsed = max(time.perf_counter() - started_at, 0.0)
    duration = len(audio_data) / sample_rate if sample_rate else 0.0
    rtf = elapsed / duration if duration > 0 else 0.0
    metrics = {"audio_seconds": duration, "elapsed_seconds": elapsed, "rtf": rtf}
    RUNTIME_EVENTS.append(
        "synthesis_complete",
        mode=mode,
        speaker=speaker,
        audio_seconds=round(duration, 3),
        elapsed_seconds=round(elapsed, 3),
        rtf=round(rtf, 3),
    )
    if terminal_enabled:
        LOGGER.info(
            "TTS performance mode=%s speaker=%s audio=%.3fs elapsed=%.3fs rtf=%.3f",
            mode,
            speaker,
            duration,
            elapsed,
            rtf,
        )
    return metrics


def save_upload(upload: Optional[UploadFile], *, prefix: str) -> Optional[str]:
    if upload is None or not upload.filename:
        return None
    suffix = Path(upload.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_ROOT, prefix=f"{prefix}_", suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        return tmp.name


def normalize_clone_reference_audio(ref_audio_path: str, *, repo_dir: Path) -> tuple[str, Optional[str]]:
    resolved = resolve_existing_file(ref_audio_path, repo_dir=repo_dir, field_name="ref_audio_path", required=True)
    audio, sample_rate = sf.read(resolved, always_2d=True, dtype="float32")
    if sample_rate <= 0 or len(audio) <= 0:
        raise ValueError(f"Reference audio is empty or unreadable: {resolved}")

    original_seconds = len(audio) / sample_rate
    normalized = audio
    if original_seconds < CLONE_REFERENCE_MIN_SECONDS:
        target_frames = int(np.ceil(CLONE_REFERENCE_MIN_SECONDS * sample_rate))
        pad_frames = max(target_frames - len(audio), 0)
        normalized = np.pad(audio, ((0, pad_frames), (0, 0)), mode="constant")
    elif original_seconds > CLONE_REFERENCE_MAX_SECONDS:
        target_frames = int(np.floor(CLONE_REFERENCE_MAX_SECONDS * sample_rate))
        normalized = audio[:target_frames]
    else:
        return resolved, None

    with tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_ROOT, prefix="clone_ref_norm_", suffix=".wav") as tmp:
        normalized_path = tmp.name
    sf.write(normalized_path, normalized, sample_rate)
    normalized_seconds = len(normalized) / sample_rate
    RUNTIME_EVENTS.append(
        "clone_reference_normalized",
        original_seconds=round(original_seconds, 3),
        normalized_seconds=round(normalized_seconds, 3),
        path=normalized_path,
    )
    return normalized_path, normalized_path


def cleanup_temp_files(paths: Iterable[Optional[str]]) -> None:
    for path in paths:
        if not path:
            continue
        with contextlib.suppress(OSError):
            Path(path).unlink(missing_ok=True)


def create_api_app(
    runtime: GPTSoVITSRuntime,
    registry: VoiceRegistry,
    *,
    default_voice_id: str = "",
    rtf_log: bool = False,
    admin_url: str = "",
) -> FastAPI:
    app = FastAPI(
        title="Neiroha GPT-SoVITS Launcher",
        version="0.1.0",
        description="Local GPT-SoVITS wrapper with native and OpenAI-compatible TTS routes.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        return {
            "message": "Neiroha GPT-SoVITS launcher is running.",
            "health": "/health",
            "openai_speech": "/v1/audio/speech",
            "native_clone": "/gpt-sovits/clone",
            "native_clone_upload": "/gpt-sovits/clone/upload",
            "capabilities": "/gpt-sovits/capabilities",
            "native_speech": "/tts",
            "admin": admin_url or "/admin",
        }

    @app.get("/health")
    @app.get("/api/health", include_in_schema=False)
    def health():
        return {
            "status": "ok",
            "active_voice_set": registry.active_voice_set_id(),
            "active_model_preset": registry.active_model_preset_id(),
            "default_voice": strip_text(default_voice_id) or registry.default_voice_id(),
            "admin_url": admin_url,
            **runtime.status(),
        }

    @app.get("/v1/models")
    def list_models():
        voice_sets = registry.list_voice_sets()
        return {
            "object": "list",
            "data": [
                voice_set.to_openai_model(len(registry.list_profiles(voice_set.id)))
                for voice_set in voice_sets
            ],
        }

    @app.get("/v1/audio/voices")
    @app.get("/v1/audio/speakers", include_in_schema=False)
    def list_voices():
        voices = [profile.to_openai_voice() for profile in registry.list_profiles()]
        if not voices:
            voices = [
                {
                    "id": "default",
                    "voice_id": "default",
                    "name": "default",
                    "object": "voice",
                    "model": registry.active_voice_set_id(),
                    "task_mode": "prompt_clone",
                    "description": "Configure runtime/voices/<voice-id>/voice.toml or pass ref_audio_path in the request.",
                }
            ]
        compact = [
            {
                "voice_id": item["voice_id"],
                "name": item["name"],
                "model": item.get("model", registry.active_voice_set_id()),
            }
            for item in voices
        ]
        return {"object": "list", "data": voices, "voices": compact}

    @app.get("/gpt-sovits/models")
    def list_native_models():
        return {"object": "list", "data": native_model_catalog(registry)}

    @app.get("/gpt-sovits/voices")
    def list_native_voices(model_id: Optional[str] = None):
        profiles = registry.list_profiles()
        data = []
        for profile in profiles:
            current_model_id = first_non_empty(profile.model_preset, DEFAULT_MODEL_PRESET_ID)
            if model_id and model_id != current_model_id:
                continue
            data.append(profile.to_native_voice(model_id=current_model_id))
        return {"object": "list", "data": data}

    @app.get("/gpt-sovits/capabilities")
    def capabilities():
        return {
            "object": "gpt_sovits.capabilities",
            "speech_backends": {
                "main_inference": {
                    "backend": "pytorch",
                    "onnx_supported": False,
                    "note": "This launcher uses GPT-SoVITS' PyTorch TTS runtime. Upstream contains export tooling, but ONNX is not a drop-in inference backend here.",
                },
                "text_frontend": {
                    "backend": "mixed",
                    "onnxruntime_used_by": ["g2pw"],
                },
            },
            "routes": {
                "openai_speech": "/v1/audio/speech",
                "trained_models": "/gpt-sovits/models",
                "trained_voices": "/gpt-sovits/voices",
                "clone_speech": "/gpt-sovits/clone",
                "clone_upload": "/gpt-sovits/clone/upload",
            },
            "clone_reference_audio": {
                "upstream_required_seconds": [3.0, 10.0],
                "auto_normalize_without_upstream_patch": True,
                "short_audio": "pad trailing silence",
                "long_audio": "trim from start",
                "normalized_seconds": [CLONE_REFERENCE_MIN_SECONDS, CLONE_REFERENCE_MAX_SECONDS],
            },
            "logging": {
                "runtime_log": "/gpt-sovits/logs",
                "log_path": str(RUNTIME_EVENT_LOG_PATH),
                "raw_runtime_output_default": "suppressed",
                "raw_runtime_output_debug_log": str(RUNTIME_DEBUG_LOG_PATH),
            },
            "admin_url": admin_url,
        }

    @app.get("/speakers")
    def speakers():
        profiles = registry.list_profiles()
        if not profiles:
            return [{"name": "default", "voice_id": "default"}]
        return [profile.to_speaker() for profile in profiles]

    @app.get("/gpt-sovits/meta")
    def meta():
        return {
            **runtime.status(),
            "profiles_path": str(registry.profile_path),
            "server_config_path": str(registry.server_config_path),
            "voice_sets_dir": str(registry.voice_sets_dir),
            "model_presets_dir": str(registry.model_presets_dir),
            "runtime_voices_dir": str(registry.runtime_voices_dir),
            "active_voice_set": registry.active_voice_set_id(),
            "active_model_preset": registry.active_model_preset_id(),
            "default_voice": strip_text(default_voice_id) or registry.default_voice_id(),
            "admin_url": admin_url,
            "voice_sets": [
                item.to_openai_model(len(registry.list_profiles(item.id)))
                for item in registry.list_voice_sets()
            ],
            "voices": [profile.to_openai_voice() for profile in registry.list_profiles()],
            "native_models": native_model_catalog(registry),
            "routes": {
                "native_tts": "/tts",
                "openai_speech": "/v1/audio/speech",
                "native_models": "/gpt-sovits/models",
                "native_voices": "/gpt-sovits/voices",
                "native_clone": "/gpt-sovits/clone",
                "native_clone_upload": "/gpt-sovits/clone/upload",
                "capabilities": "/gpt-sovits/capabilities",
                "models": "/v1/models",
                "voices": "/v1/audio/voices",
                "logs": "/gpt-sovits/logs",
                "load": "/gpt-sovits/load",
                "unload": "/gpt-sovits/unload",
                "set_gpt_weights": "/set_gpt_weights",
                "set_sovits_weights": "/set_sovits_weights",
            },
        }

    @app.get("/gpt-sovits/events")
    def runtime_events(limit: int = Query(80, ge=1, le=500)):
        return {
            "object": "log",
            "path": str(RUNTIME_EVENT_LOG_PATH),
            "data": RUNTIME_EVENTS.tail(limit),
        }

    @app.get("/gpt-sovits/logs")
    def runtime_logs(limit: int = Query(120, ge=1, le=1000)):
        return Response(
            content=RUNTIME_EVENTS.tail(limit),
            media_type="text/plain; charset=utf-8",
        )

    @app.post("/v1/audio/speech")
    def openai_audio_speech(payload: OpenAISpeechRequest):
        if payload.model and payload.model not in LEGACY_OPENAI_MODEL_ALIASES and not registry.has_voice_set(payload.model):
            return openai_error(
                f"Unknown voice set model='{payload.model}'. Use GET /v1/models.",
                status_code=400,
            )
        if payload.stream_format not in {"", "audio"}:
            return openai_error("Only stream_format='audio' is supported by this local launcher.")

        try:
            started_at = time.perf_counter()
            request, profile = request_with_profile(
                payload,
                runtime=runtime,
                registry=registry,
                default_voice_id=default_voice_id,
            )
            runtime.apply_profile_weights(profile)
            sample_rate, audio_data = runtime.synthesize_once(request)
            speaker = profile.id if profile else extract_voice_id(payload.voice) or default_voice_id or "default"
            metrics = log_rtf(
                rtf_log,
                mode=profile.model_type if profile else "prompt_clone",
                speaker=speaker,
                sample_rate=sample_rate,
                audio_data=audio_data,
                started_at=started_at,
            )
            return audio_response(sample_rate, audio_data, request["media_type"], speaker=speaker, metrics=metrics)
        except FileNotFoundError as exc:
            return openai_error(str(exc), status_code=404)
        except (ValueError, RuntimeError) as exc:
            return openai_error(str(exc), status_code=400)
        except Exception as exc:
            LOGGER.exception("OpenAI-compatible synthesis failed.")
            return openai_error(str(exc), status_code=500, error_type="server_error")

    @app.post("/gpt-sovits/clone")
    def clone_audio_speech(payload: CloneSpeechRequest):
        if payload.model and payload.model not in {OPENAI_MODEL_ALIAS, *LEGACY_OPENAI_MODEL_ALIASES}:
            return openai_error(
                f"This clone route serves '{OPENAI_MODEL_ALIAS}'. Received model='{payload.model}'.",
                status_code=400,
            )

        normalized_ref = None
        try:
            started_at = time.perf_counter()
            request, profile = request_with_clone(payload, runtime=runtime)
            request["ref_audio_path"], normalized_ref = normalize_clone_reference_audio(
                request["ref_audio_path"],
                repo_dir=runtime.repo_dir,
            )
            runtime.apply_profile_weights(profile)
            sample_rate, audio_data = runtime.synthesize_once(request)
            metrics = log_rtf(
                rtf_log,
                mode="clone",
                speaker=payload.speaker or "clone",
                sample_rate=sample_rate,
                audio_data=audio_data,
                started_at=started_at,
            )
            return audio_response(
                sample_rate,
                audio_data,
                request["media_type"],
                speaker=payload.speaker or "clone",
                metrics=metrics,
            )
        except FileNotFoundError as exc:
            return openai_error(str(exc), status_code=404)
        except (ValueError, RuntimeError, OSError) as exc:
            return openai_error(str(exc), status_code=400)
        except Exception as exc:
            LOGGER.exception("Clone synthesis failed.")
            return openai_error(str(exc), status_code=500, error_type="server_error")
        finally:
            cleanup_temp_files([normalized_ref])

    @app.get("/tts")
    def tts_get(
        text: Optional[str] = None,
        text_lang: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        aux_ref_audio_paths: Optional[list[str]] = Query(None),
        prompt_lang: Optional[str] = None,
        prompt_text: str = "",
        top_k: int = 15,
        top_p: float = 1,
        temperature: float = 1,
        text_split_method: str = "cut5",
        batch_size: int = 1,
        batch_threshold: float = 0.75,
        split_bucket: bool = True,
        speed_factor: float = 1.0,
        fragment_interval: float = 0.3,
        seed: int = -1,
        media_type: str = "wav",
        streaming_mode: Union[bool, int] = False,
        parallel_infer: bool = True,
        repetition_penalty: float = 1.35,
        sample_steps: int = 32,
        super_sampling: bool = False,
        overlap_length: int = 2,
        min_chunk_length: int = 16,
    ):
        request = native_request_from_query(
            text=text,
            text_lang=text_lang,
            ref_audio_path=ref_audio_path,
            aux_ref_audio_paths=aux_ref_audio_paths,
            prompt_lang=prompt_lang,
            prompt_text=prompt_text,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            text_split_method=text_split_method,
            batch_size=batch_size,
            batch_threshold=batch_threshold,
            split_bucket=split_bucket,
            speed_factor=speed_factor,
            fragment_interval=fragment_interval,
            seed=seed,
            media_type=media_type,
            streaming_mode=streaming_mode,
            parallel_infer=parallel_infer,
            repetition_penalty=repetition_penalty,
            sample_steps=sample_steps,
            super_sampling=super_sampling,
            overlap_length=overlap_length,
            min_chunk_length=min_chunk_length,
        )
        return native_tts_response(request)

    @app.post("/tts")
    def tts_post(payload: TTSRequest):
        return native_tts_response(model_dump(payload))

    def native_tts_response(request: dict[str, Any]):
        try:
            request["media_type"] = require_supported_format(request.get("media_type") or "wav")
            normalized, is_streaming = runtime.normalize_streaming_mode(dict(request))
            if is_streaming:
                return streaming_audio_response(
                    runtime.synthesize_stream(normalized),
                    response_format=normalized["media_type"],
                )
            started_at = time.perf_counter()
            sample_rate, audio_data = runtime.synthesize_once(request)
            metrics = log_rtf(
                rtf_log,
                mode="native",
                speaker="native",
                sample_rate=sample_rate,
                audio_data=audio_data,
                started_at=started_at,
            )
            return audio_response(sample_rate, audio_data, request["media_type"], speaker="native", metrics=metrics)
        except FileNotFoundError as exc:
            return json_response({"message": str(exc)}, status_code=404)
        except (ValueError, RuntimeError) as exc:
            return json_response({"message": str(exc)}, status_code=400)
        except Exception as exc:
            LOGGER.exception("Native synthesis failed.")
            return json_response({"message": "tts failed", "Exception": str(exc)}, status_code=500)

    @app.post("/gpt-sovits/speech/upload")
    def speech_upload(
        text: str = Form(...),
        text_lang: str = Form("zh"),
        prompt_text: str = Form(""),
        prompt_lang: str = Form("zh"),
        response_format: str = Form("wav"),
        speed: float = Form(1.0),
        ref_audio_path: Optional[str] = Form(None),
        ref_audio: Optional[UploadFile] = File(None),
    ):
        temp_ref = None
        try:
            temp_ref = save_upload(ref_audio, prefix="ref")
            request = {
                "text": text,
                "text_lang": text_lang,
                "ref_audio_path": temp_ref or ref_audio_path,
                "prompt_text": prompt_text,
                "prompt_lang": prompt_lang,
                "speed_factor": speed,
                "media_type": response_format,
                "streaming_mode": False,
                "text_split_method": "cut5",
                "batch_size": 1,
            }
            started_at = time.perf_counter()
            sample_rate, audio_data = runtime.synthesize_once(request)
            metrics = log_rtf(
                rtf_log,
                mode="upload",
                speaker="upload",
                sample_rate=sample_rate,
                audio_data=audio_data,
                started_at=started_at,
            )
            return audio_response(sample_rate, audio_data, response_format, speaker="upload", metrics=metrics)
        finally:
            cleanup_temp_files([temp_ref])

    @app.post("/gpt-sovits/clone/upload")
    def clone_upload(
        text: str = Form(...),
        text_lang: str = Form("zh"),
        prompt_text: str = Form(...),
        prompt_lang: str = Form("zh"),
        response_format: str = Form("wav"),
        speed: float = Form(1.0),
        speaker: str = Form("clone"),
        gpt_weights_path: Optional[str] = Form(None),
        sovits_weights_path: Optional[str] = Form(None),
        ref_audio_path: Optional[str] = Form(None),
        ref_audio: Optional[UploadFile] = File(None),
    ):
        temp_ref = None
        normalized_ref = None
        try:
            temp_ref = save_upload(ref_audio, prefix="clone_ref")
            payload = CloneSpeechRequest(
                text=text,
                text_lang=text_lang,
                prompt_text=prompt_text,
                prompt_lang=prompt_lang,
                response_format=response_format,
                speed=speed,
                speaker=speaker,
                ref_audio_path=temp_ref or ref_audio_path,
                gpt_weights_path=gpt_weights_path,
                sovits_weights_path=sovits_weights_path,
            )
            started_at = time.perf_counter()
            request, profile = request_with_clone(payload, runtime=runtime)
            request["ref_audio_path"], normalized_ref = normalize_clone_reference_audio(
                request["ref_audio_path"],
                repo_dir=runtime.repo_dir,
            )
            runtime.apply_profile_weights(profile)
            sample_rate, audio_data = runtime.synthesize_once(request)
            metrics = log_rtf(
                rtf_log,
                mode="clone-upload",
                speaker=payload.speaker,
                sample_rate=sample_rate,
                audio_data=audio_data,
                started_at=started_at,
            )
            return audio_response(sample_rate, audio_data, request["media_type"], speaker=payload.speaker, metrics=metrics)
        except FileNotFoundError as exc:
            return openai_error(str(exc), status_code=404)
        except (ValueError, RuntimeError, OSError) as exc:
            return openai_error(str(exc), status_code=400)
        except Exception as exc:
            LOGGER.exception("Clone upload synthesis failed.")
            return openai_error(str(exc), status_code=500, error_type="server_error")
        finally:
            cleanup_temp_files([temp_ref, normalized_ref])

    @app.post("/gpt-sovits/load")
    def load_model(payload: LoadRequest):
        try:
            status = runtime.load(payload.config_path)
            if payload.gpt_weights_path:
                status = runtime.set_gpt_weights(payload.gpt_weights_path)
            if payload.sovits_weights_path:
                status = runtime.set_sovits_weights(payload.sovits_weights_path)
            return status
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/gpt-sovits/unload")
    def unload_model():
        return runtime.unload()

    @app.post("/gpt-sovits/reload")
    def reload_model(payload: LoadRequest):
        try:
            return runtime.reload(payload.config_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/set_refer_audio")
    def set_refer_audio(refer_audio_path: str):
        try:
            return runtime.set_refer_audio(refer_audio_path)
        except Exception as exc:
            return json_response({"message": "set refer audio failed", "Exception": str(exc)}, status_code=400)

    @app.post("/set_refer_audio")
    def set_refer_audio_post(payload: ReferAudioRequest):
        return set_refer_audio(payload.refer_audio_path)

    @app.get("/set_gpt_weights")
    def set_gpt_weights(weights_path: str):
        try:
            runtime.set_gpt_weights(weights_path)
            return Response("success", media_type="text/plain")
        except Exception as exc:
            return json_response({"message": "change gpt weight failed", "Exception": str(exc)}, status_code=400)

    @app.post("/set_gpt_weights")
    @app.post("/gpt-sovits/set_gpt_weights")
    def set_gpt_weights_post(payload: WeightsRequest):
        return set_gpt_weights(payload.weights_path)

    @app.get("/set_sovits_weights")
    def set_sovits_weights(weights_path: str):
        try:
            runtime.set_sovits_weights(weights_path)
            return Response("success", media_type="text/plain")
        except Exception as exc:
            return json_response({"message": "change sovits weight failed", "Exception": str(exc)}, status_code=400)

    @app.post("/set_sovits_weights")
    @app.post("/gpt-sovits/set_sovits_weights")
    def set_sovits_weights_post(payload: WeightsRequest):
        return set_sovits_weights(payload.weights_path)

    @app.get("/control")
    def control(command: str):
        return handle_control(command, runtime)

    @app.post("/control")
    def control_post(payload: ControlRequest):
        return handle_control(payload.command, runtime)

    return app


def handle_control(command: str, runtime: GPTSoVITSRuntime):
    command = strip_text(command).lower()
    if command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        return Response(status_code=204)
    if command == "restart":
        os.execl(sys.executable, sys.executable, *sys.argv)
    if command == "unload":
        return runtime.unload()
    if command in {"load", "reload"}:
        return runtime.reload() if command == "reload" else runtime.load()
    return json_response({"message": "command must be one of: load, unload, reload, restart, exit"}, status_code=400)


def build_gradio_blocks(runtime: GPTSoVITSRuntime, registry: VoiceRegistry):
    import gradio as gr

    def status_text() -> str:
        return json.dumps(runtime.status(), ensure_ascii=False, indent=2)

    def load_model(config_path: str, gpt_weights_path: str, sovits_weights_path: str) -> str:
        try:
            status = runtime.load(config_path or None)
            if strip_text(gpt_weights_path):
                status = runtime.set_gpt_weights(gpt_weights_path)
            if strip_text(sovits_weights_path):
                status = runtime.set_sovits_weights(sovits_weights_path)
            return json.dumps(status, ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"ERROR: {exc}"

    def unload_model() -> str:
        return json.dumps(runtime.unload(), ensure_ascii=False, indent=2)

    def voice_choices() -> list[str]:
        choices = [profile.id for profile in registry.list_profiles()]
        return choices or ["default"]

    def synthesize_preview(
        text: str,
        voice: str,
        ref_audio_file: Optional[str],
        ref_audio_path: str,
        prompt_text: str,
        text_lang: str,
        prompt_lang: str,
        speed: float,
    ):
        profile = registry.get(voice) or (registry.first() if voice == "default" else None)
        request = {
            "text": text,
            "text_lang": text_lang or (profile.text_lang if profile else "zh"),
            "ref_audio_path": ref_audio_file or ref_audio_path or (profile.ref_audio_path if profile else ""),
            "prompt_text": prompt_text or (profile.prompt_text if profile else ""),
            "prompt_lang": prompt_lang or (profile.prompt_lang if profile else text_lang or "zh"),
            "speed_factor": speed,
            "media_type": "wav",
            "streaming_mode": False,
            "text_split_method": "cut5",
            "batch_size": 1,
        }
        runtime.apply_profile_weights(profile)
        sample_rate, audio_data = runtime.synthesize_once(request)
        return sample_rate, audio_data

    with gr.Blocks(title="Neiroha GPT-SoVITS") as blocks:
        gr.Markdown("# Neiroha GPT-SoVITS")
        with gr.Tab("Status"):
            status_box = gr.Code(value=status_text, language="json", label="Runtime status")
            with gr.Row():
                refresh_btn = gr.Button("Refresh")
                unload_btn = gr.Button("Unload model")
            refresh_btn.click(status_text, outputs=status_box)
            unload_btn.click(unload_model, outputs=status_box)
        with gr.Tab("Load / Weights"):
            config_input = gr.Textbox(value=str(runtime.config_path), label="TTS config path")
            gpt_input = gr.Textbox(label="GPT weights path")
            sovits_input = gr.Textbox(label="SoVITS weights path")
            load_btn = gr.Button("Load / Apply")
            load_btn.click(load_model, inputs=[config_input, gpt_input, sovits_input], outputs=status_box)
        with gr.Tab("Speech"):
            voice_dropdown = gr.Dropdown(choices=voice_choices(), value=voice_choices()[0], label="Voice profile")
            text_input = gr.Textbox(label="Text", lines=4)
            with gr.Row():
                text_lang = gr.Textbox(value="zh", label="Text language")
                prompt_lang = gr.Textbox(value="zh", label="Prompt language")
                speed = gr.Slider(0.25, 4.0, value=1.0, step=0.05, label="Speed")
            ref_audio_file = gr.Audio(type="filepath", label="Reference audio upload")
            ref_audio_path = gr.Textbox(label="Reference audio path")
            prompt_text = gr.Textbox(label="Prompt text", lines=2)
            synth_btn = gr.Button("Synthesize")
            audio_output = gr.Audio(label="Output")
            synth_btn.click(
                synthesize_preview,
                inputs=[
                    text_input,
                    voice_dropdown,
                    ref_audio_file,
                    ref_audio_path,
                    prompt_text,
                    text_lang,
                    prompt_lang,
                    speed,
                ],
                outputs=audio_output,
            )
        with gr.Tab("Profiles"):
            profiles_box = gr.Code(
                value=lambda: json.dumps(
                    [profile.to_openai_voice() for profile in registry.list_profiles()],
                    ensure_ascii=False,
                    indent=2,
                ),
                language="json",
                label=str(registry.profile_path),
            )
            gr.Button("Reload profile view").click(
                lambda: json.dumps(
                    [profile.to_openai_voice() for profile in registry.list_profiles()],
                    ensure_ascii=False,
                    indent=2,
                ),
                outputs=profiles_box,
            )
    return blocks.queue(max_size=8, default_concurrency_limit=1)


class ManagedApiProcess:
    def __init__(
        self,
        *,
        api_host: str,
        api_port: int,
        repo_dir: Path,
        config_path: Path,
        profiles_path: Path,
        device: str,
        is_half: Optional[bool],
        default_voice_id: str,
        log_level: str,
        terminal_rtf_log: bool,
        debug_runtime_output: bool,
    ) -> None:
        self.api_host = api_host
        self.api_port = api_port
        self.repo_dir = repo_dir
        self.config_path = config_path
        self.profiles_path = profiles_path
        self.device = device
        self.is_half = is_half
        self.default_voice_id = default_voice_id
        self.log_level = log_level
        self.terminal_rtf_log = terminal_rtf_log
        self.debug_runtime_output = debug_runtime_output
        self.process: Optional[subprocess.Popen] = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def external_health_ok(self) -> bool:
        health_url = f"http://127.0.0.1:{self.api_port}/health"
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def external_admin_ready(self) -> bool:
        capabilities_url = f"http://127.0.0.1:{self.api_port}/gpt-sovits/capabilities"
        try:
            with urllib.request.urlopen(capabilities_url, timeout=2) as response:
                if response.status != 200:
                    return False
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError, urllib.error.URLError):
            return False
        routes = payload.get("routes", {}) if isinstance(payload, dict) else {}
        return routes.get("clone_upload") == "/gpt-sovits/clone/upload"

    def command(self, default_voice_id: str, preload_model: bool) -> list[str]:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            "api",
            "--repo-dir",
            str(self.repo_dir),
            "--config",
            str(self.config_path),
            "--profiles",
            str(self.profiles_path),
            "--host",
            self.api_host,
            "--port",
            str(self.api_port),
            "--device",
            self.device,
            "--log-level",
            self.log_level,
        ]
        if default_voice_id:
            command += ["--default-voice", default_voice_id]
        if preload_model:
            command.append("--preload-model")
        if self.is_half is True:
            command.append("--half")
        elif self.is_half is False:
            command.append("--no-half")
        if self.terminal_rtf_log:
            command.append("--rtf-log")
        if self.debug_runtime_output:
            command.append("--debug-runtime-output")
        return command

    def start(self, *, default_voice_id: str = "", preload_model: bool = False) -> str:
        if self.is_running():
            return f"Managed API is already running with PID {self.process.pid}."
        if self.external_health_ok():
            if self.external_admin_ready():
                return f"FastAPI is already reachable on port {self.api_port}; using the external process."
            return (
                f"Port {self.api_port} already has a FastAPI process, but it does not expose the current "
                "admin routes such as /gpt-sovits/clone/upload. Stop that old process or switch ports, then start again."
            )
        selected_voice = strip_text(default_voice_id) or strip_text(self.default_voice_id)
        command = self.command(selected_voice, preload_model)
        self.process = subprocess.Popen(command, cwd=WORKSPACE_ROOT)
        time.sleep(1)
        if self.process.poll() is not None:
            return f"API process exited immediately with code {self.process.returncode}."
        voice_label = selected_voice or "default"
        return f"Started API PID {self.process.pid} on port {self.api_port}, default voice={voice_label}."

    def stop(self) -> str:
        if not self.is_running():
            return "Managed API is not running."
        assert self.process is not None
        pid = self.process.pid
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
        return f"Stopped API PID {pid}."

    def status(self) -> str:
        if self.is_running():
            return f"Managed API PID {self.process.pid} is running on port {self.api_port}."
        if self.process is None:
            return "Managed API has not been started from this admin page."
        return f"Managed API exited with code {self.process.returncode}."


class ManagedGradioProcess:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        api_base: str,
        repo_dir: Path,
        config_path: Path,
        profiles_path: Path,
        log_level: str,
        debug_runtime_output: bool,
    ) -> None:
        self.host = host
        self.port = port
        self.api_base = api_base
        self.repo_dir = repo_dir
        self.config_path = config_path
        self.profiles_path = profiles_path
        self.log_level = log_level
        self.debug_runtime_output = debug_runtime_output
        self.stdout_path = RUNTIME_LOG_ROOT / "admin-ui.out.log"
        self.stderr_path = RUNTIME_LOG_ROOT / "admin-ui.err.log"
        self.process: Optional[subprocess.Popen] = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def command(self) -> list[str]:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            "admin-ui",
            "--repo-dir",
            str(self.repo_dir),
            "--config",
            str(self.config_path),
            "--profiles",
            str(self.profiles_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--api-base",
            self.api_base,
            "--log-level",
            self.log_level,
        ]
        if self.debug_runtime_output:
            command.append("--debug-runtime-output")
            command.append("--rtf-log")
        return command

    def start(self) -> str:
        if self.is_running():
            assert self.process is not None
            return f"Admin UI is already running with PID {self.process.pid}."
        self.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with self.stdout_path.open("w", encoding="utf-8") as stdout_file, self.stderr_path.open(
            "w",
            encoding="utf-8",
        ) as stderr_file:
            self.process = subprocess.Popen(
                self.command(),
                cwd=WORKSPACE_ROOT,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )
        time.sleep(1)
        if self.process.poll() is not None:
            return f"Admin UI process exited immediately with code {self.process.returncode}."
        return (
            f"Started admin UI PID {self.process.pid} on port {self.port}, API={self.api_base}. "
            f"Logs: {self.stdout_path}, {self.stderr_path}"
        )

    def stop(self) -> str:
        if not self.is_running():
            return "Admin UI is not running."
        assert self.process is not None
        pid = self.process.pid
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
        return f"Stopped admin UI PID {pid}."


class ManagedDownloadProcess:
    def __init__(self) -> None:
        self.process: Optional[subprocess.Popen] = None
        self.name = ""
        self.stdout_path = RUNTIME_LOG_ROOT / "admin-download.out.log"
        self.stderr_path = RUNTIME_LOG_ROOT / "admin-download.err.log"

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, name: str, args: list[str]) -> str:
        if self.is_running():
            assert self.process is not None
            return f"Download already running: {self.name} (PID {self.process.pid}).\n\n{self.tail()}"
        self.name = name
        command = [sys.executable, "-u", str(WORKSPACE_ROOT / "scripts" / "download_gpt_sovits_assets.py"), *args]
        self.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with self.stdout_path.open("w", encoding="utf-8") as stdout_file, self.stderr_path.open(
            "w",
            encoding="utf-8",
        ) as stderr_file:
            self.process = subprocess.Popen(
                command,
                cwd=WORKSPACE_ROOT,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )
        return self.status()

    def stop(self) -> str:
        if not self.is_running():
            return f"No active download.\n\n{self.tail()}"
        assert self.process is not None
        pid = self.process.pid
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
        return f"Stopped download PID {pid}.\n\n{self.tail()}"

    def status(self) -> str:
        if self.is_running():
            assert self.process is not None
            return f"Download running: {self.name} (PID {self.process.pid}).\n\n{self.tail()}"
        if self.process is None:
            return f"No download has been started from this admin page.\n\n{self.tail()}"
        return f"Download exited with code {self.process.returncode}: {self.name}.\n\n{self.tail()}"

    def tail(self, lines: int = 80) -> str:
        chunks = []
        for label, path in (("stdout", self.stdout_path), ("stderr", self.stderr_path)):
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                chunks.append(f"[{label}] unable to read {path}: {exc}")
                continue
            tail_text = "\n".join(reversed(text.splitlines()[-lines:]))
            if tail_text:
                chunks.append(f"[{label}] {path} (newest first)\n{tail_text}")
        return "\n\n".join(chunks) or "No download logs yet."


def build_gradio_admin_blocks_legacy(
    api_base: str,
    registry: VoiceRegistry,
    *,
    process_manager: Optional[ManagedApiProcess] = None,
):
    import gradio as gr
    import requests

    base = api_base.rstrip("/")
    download_manager = ManagedDownloadProcess()
    ui_config = load_ui_config(registry)
    language = strip_text(os.environ.get("NEIROHA_GPT_SOVITS_UI_LANG") or ui_config.get("default_language") or "zh")
    language = "en" if language.lower().startswith("en") else "zh"
    ui_title = strip_text(ui_config.get("title")) or "Neiroha GPT-SoVITS Admin"
    ui_text = {
        "zh": {
            "api_offline": "### API 离线\n",
            "api_online": "### API 在线\n",
            "api_base": "API Base",
            "refresh_time": "刷新时间",
            "error": "错误",
            "wait_preload": "如果是从 `start_api_admin.bat` 启动，等预加载完成后这里会自动刷新。",
            "loaded": "已加载",
            "not_loaded": "未加载",
            "model_status": "模型状态",
            "device": "设备",
            "half": "半精度",
            "available_model": "可用 model",
            "available_voice": "可用 voice",
            "gpt_weights": "GPT 权重",
            "sovits_weights": "SoVITS 权重",
            "external_process": "API 进程由外层 launcher / start_api_admin.bat 管理；此 Admin 只连接并显示状态。",
            "home": "首页",
            "trial": "试音",
            "clone_config": "克隆配置",
            "model_presets": "Model Presets",
            "download": "下载",
            "logs": "日志",
            "runtime_status": "运行状态",
            "api_process": "API 进程",
            "refresh": "刷新",
            "load_active": "加载当前 preset",
            "unload_model": "卸载模型",
            "preload": "预加载",
            "default_voice": "默认 voice",
            "start_api": "启动 API",
            "stop_api": "停止 API",
            "voice_set_model": "voice set / model",
            "voice": "voice",
            "text": "文本",
            "format": "格式",
            "speed": "速度",
            "generate": "生成",
            "audio_output": "输出音频",
            "metrics": "RTF / 输出",
            "save_to_voice_set": "保存到 voice set",
            "use_model_preset": "使用 model preset",
            "voice_id": "voice id",
            "name": "name",
            "upload_reference": "上传参考音频",
            "reference_path": "或填写参考音频路径",
            "prompt_text": "prompt_text",
            "prompt_lang": "prompt_lang",
            "text_lang": "text_lang",
            "default_speed": "默认速度",
            "trained_weight_override": "可选：覆盖为别人训练过的 GPT/SoVITS 权重",
            "gpt_ckpt": "GPT 权重 .ckpt",
            "sovits_pth": "SoVITS 权重 .pth",
            "save_voice": "保存 voice",
            "save_result": "保存结果",
            "current_preset": "当前底层 preset",
            "load": "加载",
            "unload": "卸载",
            "reload": "重载",
            "preset_status": "Presets / 状态",
            "new_preset": "新增 / 更新训练模型 preset",
            "save_preset": "保存 preset",
            "config_path": "config_path",
            "download_source": "下载源",
            "force_redownload": "强制重新下载",
            "download_base": "下载预训练基座",
            "download_v2pro": "下载 v2ProPlus 克隆基座",
            "download_sample": "下载单个示例参考音频",
            "refresh_log": "刷新日志",
            "stop_download": "停止下载",
            "download_status": "下载状态 / 日志",
            "backend_log": "backend.log（最新在上，自动刷新）",
            "prompt_required": "prompt_text is required.",
            "reference_required": "reference audio is required.",
            "weights_required": "GPT 权重和 SoVITS 权重路径都要填。",
        },
        "en": {
            "api_offline": "### API Offline\n",
            "api_online": "### API Online\n",
            "api_base": "API Base",
            "refresh_time": "Refreshed",
            "error": "Error",
            "wait_preload": "If launched from `start_api_admin.bat`, this panel will refresh automatically after preload finishes.",
            "loaded": "loaded",
            "not_loaded": "not loaded",
            "model_status": "Model",
            "device": "Device",
            "half": "Half",
            "available_model": "Models",
            "available_voice": "Voices",
            "gpt_weights": "GPT weights",
            "sovits_weights": "SoVITS weights",
            "external_process": "The API process is managed by the outer launcher / start_api_admin.bat; this Admin only connects to it.",
            "home": "Home",
            "trial": "Test Voice",
            "clone_config": "Voice Config",
            "model_presets": "Model Presets",
            "download": "Downloads",
            "logs": "Logs",
            "runtime_status": "Runtime Status",
            "api_process": "API Process",
            "refresh": "Refresh",
            "load_active": "Load Active Preset",
            "unload_model": "Unload Model",
            "preload": "Preload",
            "default_voice": "Default voice",
            "start_api": "Start API",
            "stop_api": "Stop API",
            "voice_set_model": "voice set / model",
            "voice": "voice",
            "text": "Text",
            "format": "Format",
            "speed": "Speed",
            "generate": "Generate",
            "audio_output": "Output audio",
            "metrics": "RTF / Output",
            "save_to_voice_set": "Save to voice set",
            "use_model_preset": "Use model preset",
            "voice_id": "voice id",
            "name": "name",
            "upload_reference": "Upload reference audio",
            "reference_path": "Or reference audio path",
            "prompt_text": "prompt_text",
            "prompt_lang": "prompt_lang",
            "text_lang": "text_lang",
            "default_speed": "Default speed",
            "trained_weight_override": "Optional: override with trained GPT/SoVITS weights",
            "gpt_ckpt": "GPT weights .ckpt",
            "sovits_pth": "SoVITS weights .pth",
            "save_voice": "Save voice",
            "save_result": "Save result",
            "current_preset": "Current runtime preset",
            "load": "Load",
            "unload": "Unload",
            "reload": "Reload",
            "preset_status": "Presets / Status",
            "new_preset": "Add / Update Trained Model Preset",
            "save_preset": "Save preset",
            "config_path": "config_path",
            "download_source": "Download source",
            "force_redownload": "Force redownload",
            "download_base": "Download pretrained base",
            "download_v2pro": "Download v2ProPlus clone base",
            "download_sample": "Download single sample reference",
            "refresh_log": "Refresh logs",
            "stop_download": "Stop download",
            "download_status": "Download status / logs",
            "backend_log": "backend.log (newest first, auto-refresh)",
            "prompt_required": "prompt_text is required.",
            "reference_required": "reference audio is required.",
            "weights_required": "GPT and SoVITS weight paths are both required.",
        },
    }[language]

    def t(key: str) -> str:
        return ui_text.get(key, key)

    def api_url(path: str) -> str:
        return f"{base}{path if path.startswith('/') else '/' + path}"

    def status_text() -> str:
        try:
            response = requests.get(api_url("/health"), timeout=10)
            return json.dumps(response.json(), ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"ERROR: {exc}"

    def managed_status_text() -> str:
        if process_manager is None:
            return "This admin page is connected to an external FastAPI process."
        return process_manager.status()

    def start_api(default_voice: str, preload_model: bool) -> tuple[str, str]:
        if process_manager is None:
            return "No managed API process is configured for this admin page.", status_text()
        result = process_manager.start(default_voice_id=default_voice, preload_model=preload_model)
        time.sleep(1)
        return result, status_text()

    def stop_api() -> tuple[str, str]:
        if process_manager is None:
            return "No managed API process is configured for this admin page.", status_text()
        result = process_manager.stop()
        return result, status_text()

    def load_model(config_path: str, gpt_weights_path: str, sovits_weights_path: str) -> str:
        payload = {
            "config_path": strip_text(config_path) or None,
            "gpt_weights_path": strip_text(gpt_weights_path) or None,
            "sovits_weights_path": strip_text(sovits_weights_path) or None,
        }
        try:
            response = requests.post(api_url("/gpt-sovits/load"), json=payload, timeout=120)
            response.raise_for_status()
            return json.dumps(response.json(), ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"ERROR: {exc}"

    def unload_model() -> str:
        try:
            response = requests.post(api_url("/gpt-sovits/unload"), timeout=60)
            response.raise_for_status()
            return json.dumps(response.json(), ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"ERROR: {exc}"

    def set_gpt_weights(weights_path: str) -> str:
        try:
            response = requests.post(
                api_url("/gpt-sovits/set_gpt_weights"),
                json={"weights_path": strip_text(weights_path)},
                timeout=120,
            )
            response.raise_for_status()
            return status_text()
        except Exception as exc:
            return f"ERROR: {exc}"

    def set_sovits_weights(weights_path: str) -> str:
        try:
            response = requests.post(
                api_url("/gpt-sovits/set_sovits_weights"),
                json={"weights_path": strip_text(weights_path)},
                timeout=120,
            )
            response.raise_for_status()
            return status_text()
        except Exception as exc:
            return f"ERROR: {exc}"

    def voice_choices() -> list[str]:
        try:
            response = requests.get(api_url("/v1/audio/voices"), timeout=10)
            response.raise_for_status()
            data = response.json().get("data", [])
            choices = [item.get("id", "default") for item in data if isinstance(item, dict)]
            return choices or ["default"]
        except Exception:
            choices = [profile.id for profile in registry.list_profiles()]
            return choices or ["default"]

    def profiles_text() -> str:
        try:
            response = requests.get(api_url("/v1/audio/voices"), timeout=10)
            response.raise_for_status()
            return json.dumps(response.json(), ensure_ascii=False, indent=2)
        except Exception as exc:
            payload = {
                "object": "list",
                "source": "local_profiles",
                "api_error": str(exc),
                "data": [profile.to_openai_voice() for profile in registry.list_profiles()],
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)

    def local_profiles_payload() -> list[dict[str, Any]]:
        if not registry.profile_path.exists():
            return []
        data = json.loads(registry.profile_path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("voices", [])
        if not isinstance(data, list):
            raise ValueError(f"Voice profile file must contain a list or voices object: {registry.profile_path}")
        return [item for item in data if isinstance(item, dict)]

    def write_local_profiles(profiles: list[dict[str, Any]]) -> None:
        registry.profile_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"voices": profiles}
        registry.profile_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def profile_editor_text() -> str:
        try:
            payload = {"profiles_path": str(registry.profile_path), "voices": local_profiles_payload()}
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"ERROR: {exc}"

    def refresh_voice_dropdowns(selected_voice: str = ""):
        choices = voice_choices()
        selected = selected_voice if selected_voice in choices else (choices[0] if choices else "default")
        return gr.update(choices=choices, value=selected), gr.update(choices=choices, value=selected)

    def resolve_profile_file(raw_path: str, field_name: str) -> Path:
        resolved = resolve_existing_file(raw_path, repo_dir=registry.repo_dir, field_name=field_name, required=True)
        return Path(resolved)

    def save_profile_reference_audio(voice_id: str, ref_audio_file: Optional[str], ref_audio_path: str) -> Path:
        if ref_audio_file:
            source = Path(ref_audio_file)
            suffix = source.suffix or ".wav"
            target_dir = LOCAL_REFERENCE_ROOT / safe_filename_part(voice_id, "voice")
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"reference{suffix}"
            counter = 1
            while target.exists():
                target = target_dir / f"reference_{counter}{suffix}"
                counter += 1
            shutil.copyfile(source, target)
            return target.resolve()
        return resolve_profile_file(ref_audio_path, "ref_audio_path")

    def save_openai_voice_profile(
        voice_id: str,
        name: str,
        model_type: str,
        model_id: str,
        model_name: str,
        description: str,
        ref_audio_file: Optional[str],
        ref_audio_path: str,
        prompt_text: str,
        prompt_lang: str,
        text_lang: str,
        gpt_weights_path: str,
        sovits_weights_path: str,
    ):
        cleaned_id = safe_filename_part(voice_id or name, "custom-voice")
        if not cleaned_id:
            return (
                "ERROR: voice id is required.",
                profile_editor_text(),
                *refresh_voice_dropdowns(),
                native_models_text(),
                native_voices_text(),
            )
        prompt_text = strip_text(prompt_text)
        if not prompt_text:
            return (
                "ERROR: prompt text is required and must match the reference audio.",
                profile_editor_text(),
                *refresh_voice_dropdowns(),
                native_models_text(),
                native_voices_text(),
            )
        try:
            ref_path = save_profile_reference_audio(cleaned_id, ref_audio_file, ref_audio_path)
            gpt_path = resolve_profile_file(gpt_weights_path, "gpt_weights_path")
            sovits_path = resolve_profile_file(sovits_weights_path, "sovits_weights_path")
            profile = {
                "id": cleaned_id,
                "name": strip_text(name) or cleaned_id,
                "description": strip_text(description)
                or f"OpenAI voice profile using {Path(gpt_path).name} and {Path(sovits_path).name}.",
                "model_id": strip_text(model_id) or "custom-openai-voice",
                "model_name": strip_text(model_name) or strip_text(model_id) or "Custom OpenAI Voice",
                "model_type": strip_text(model_type) or "reference-profile",
                "ref_audio_path": profile_path_text(ref_path),
                "prompt_text": prompt_text,
                "prompt_lang": strip_text(prompt_lang) or strip_text(text_lang) or "zh",
                "text_lang": strip_text(text_lang) or strip_text(prompt_lang) or "zh",
                "aux_ref_audio_paths": [],
                "gpt_weights_path": profile_path_text(gpt_path),
                "sovits_weights_path": profile_path_text(sovits_path),
            }
            profiles = local_profiles_payload()
            by_id = {strip_text(item.get("id") or item.get("name")): item for item in profiles}
            by_id[cleaned_id] = profile
            write_local_profiles(list(by_id.values()))
        except Exception as exc:
            return (
                f"ERROR: {exc}",
                profile_editor_text(),
                *refresh_voice_dropdowns(),
                native_models_text(),
                native_voices_text(),
            )

        return (
            f"Saved OpenAI voice profile `{cleaned_id}` to {registry.profile_path}.",
            profile_editor_text(),
            *refresh_voice_dropdowns(cleaned_id),
            native_models_text(),
            native_voices_text(),
        )

    def delete_openai_voice_profile(voice_id: str):
        target_id = strip_text(voice_id)
        if not target_id:
            return (
                "ERROR: voice id is required.",
                profile_editor_text(),
                *refresh_voice_dropdowns(),
                native_models_text(),
                native_voices_text(),
            )
        profiles = local_profiles_payload()
        kept = [item for item in profiles if strip_text(item.get("id") or item.get("name")) != target_id]
        if len(kept) == len(profiles):
            status = f"Voice profile `{target_id}` was not found."
        else:
            write_local_profiles(kept)
            status = f"Deleted OpenAI voice profile `{target_id}` from {registry.profile_path}."
        return (
            status,
            profile_editor_text(),
            *refresh_voice_dropdowns(),
            native_models_text(),
            native_voices_text(),
        )

    def native_models_text() -> str:
        try:
            response = requests.get(api_url("/gpt-sovits/models"), timeout=10)
            response.raise_for_status()
            return json.dumps(response.json(), ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"ERROR: {exc}"

    def native_voices_text() -> str:
        try:
            response = requests.get(api_url("/gpt-sovits/voices"), timeout=10)
            response.raise_for_status()
            return json.dumps(response.json(), ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"ERROR: {exc}"

    def runtime_events_text() -> str:
        try:
            response = requests.get(api_url("/gpt-sovits/events"), params={"limit": 120}, timeout=10)
            response.raise_for_status()
            data = response.json().get("data", [])
        except Exception as exc:
            return f"ERROR: {exc}"
        if not data:
            return "No runtime events yet."
        lines = []
        for item in data:
            if not isinstance(item, dict):
                continue
            event = item.get("event", "event")
            when = item.get("time", "")
            if event == "synthesis_complete":
                lines.append(
                    f"{when} synthesis mode={item.get('mode')} speaker={item.get('speaker')} "
                    f"audio={item.get('audio_seconds')}s elapsed={item.get('elapsed_seconds')}s "
                    f"rtf={item.get('rtf')}"
                )
            elif event.endswith("_complete") and "elapsed_seconds" in item:
                detail = item.get("weights_path") or item.get("config_path") or item.get("sovits_weights_path") or ""
                lines.append(f"{when} {event} elapsed={item.get('elapsed_seconds')}s {detail}")
            elif event == "clone_reference_normalized":
                lines.append(
                    f"{when} clone_reference_normalized "
                    f"{item.get('original_seconds')}s -> {item.get('normalized_seconds')}s"
                )
            else:
                detail = {key: value for key, value in item.items() if key not in {"time", "event"}}
                lines.append(f"{when} {event} {json.dumps(detail, ensure_ascii=False)}")
        return "\n".join(lines[-120:])

    def force_arg(force: bool) -> list[str]:
        return ["--force"] if force else []

    def start_base_assets_download(source: str, force: bool) -> str:
        return download_manager.start(
            "pretrained base assets",
            ["--source", source or DEFAULT_DOWNLOAD_SOURCE, *force_arg(force)],
        )

    def start_v2pro_download(source: str, force: bool) -> str:
        return download_manager.start(
            "v2ProPlus clone base weights",
            [
                "--source",
                source or DEFAULT_DOWNLOAD_SOURCE,
                "--skip-base-assets",
                "--v2pro-plus",
                *force_arg(force),
            ],
        )

    def start_demo_download(source: str, repo_id: str, speakers: str, activate: bool, force: bool) -> str:
        args = [
            "--source",
            source or DEFAULT_DOWNLOAD_SOURCE,
            "--skip-base-assets",
            "--genshin-demo",
            "--genshin-repo-id",
            strip_text(repo_id) or DEFAULT_DEMO_REPO_ID,
            "--genshin-speakers",
            strip_text(speakers) or DEFAULT_DEMO_SPEAKERS,
            *force_arg(force),
        ]
        if activate:
            args.append("--activate-voices")
        return download_manager.start("trained demo voices", args)

    def start_extended_demo_download(source: str, repo_id: str, activate: bool, force: bool) -> str:
        args = [
            "--source",
            source or DEFAULT_DOWNLOAD_SOURCE,
            "--skip-base-assets",
            "--genshin-demo",
            "--genshin-extended-demo",
            "--genshin-repo-id",
            strip_text(repo_id) or DEFAULT_DEMO_REPO_ID,
            *force_arg(force),
        ]
        if activate:
            args.append("--activate-voices")
        return download_manager.start("extended trained demo voices", args)

    def start_shared_multispeaker_download(source: str, repo_id: str, presets: str, force: bool) -> str:
        return download_manager.start(
            "shared multi-speaker weights",
            [
                "--source",
                source or DEFAULT_DOWNLOAD_SOURCE,
                "--skip-base-assets",
                "--shared-multispeaker-demo",
                "--shared-repo-id",
                strip_text(repo_id) or DEFAULT_SHARED_REPO_ID,
                "--shared-presets",
                strip_text(presets) or DEFAULT_SHARED_PRESETS,
                *force_arg(force),
            ],
        )

    def start_shared_reference_download(
        source: str,
        repo_id: str,
        reference_repo_id: str,
        characters: str,
        languages: str,
        activate: bool,
        force: bool,
    ) -> str:
        args = [
            "--source",
            source or DEFAULT_DOWNLOAD_SOURCE,
            "--skip-base-assets",
            "--shared-reference-demo",
            "--shared-repo-id",
            strip_text(repo_id) or DEFAULT_SHARED_REPO_ID,
            "--shared-reference-repo-id",
            strip_text(reference_repo_id) or DEFAULT_SHARED_REFERENCE_REPO_ID,
            "--shared-reference-characters",
            strip_text(characters) or DEFAULT_SHARED_REFERENCE_CHARACTERS,
            "--shared-reference-languages",
            strip_text(languages) or DEFAULT_SHARED_REFERENCE_LANGUAGES,
            *force_arg(force),
        ]
        if activate:
            args.append("--activate-voices")
        return download_manager.start("shared reference voices", args)

    def audio_file_from_response(response: requests.Response, speaker: str) -> str:
        output_header = strip_text(response.headers.get("X-Neiroha-Output-Path"))
        if output_header and Path(output_header).exists():
            return output_header
        output_path = write_runtime_output(response.content, speaker or "preview", "wav")
        return str(output_path)

    def response_metrics_text(response: requests.Response, output_path: str) -> str:
        rtf = response.headers.get("X-Neiroha-RTF")
        audio_seconds = response.headers.get("X-Neiroha-Audio-Seconds")
        elapsed_seconds = response.headers.get("X-Neiroha-Elapsed-Seconds")
        if not rtf:
            return (
                "RTF: unavailable\n"
                "The connected FastAPI process did not return performance headers. "
                "Restart FastAPI from this admin page if it is an older process.\n"
                f"Output: {output_path}"
            )
        return (
            f"RTF: {float(rtf):.3f}\n"
            f"Audio: {float(audio_seconds or 0):.3f}s\n"
            f"Elapsed: {float(elapsed_seconds or 0):.3f}s\n"
            f"Output: {output_path}"
        )

    def http_error_text(exc: requests.HTTPError, path: str) -> str:
        response = exc.response
        if response is not None and response.status_code == 404 and path == "/gpt-sovits/clone/upload":
            return (
                "Clone upload endpoint not found on the connected FastAPI process. "
                "This usually means the configured API port is still serving an older launcher. "
                "Stop that process or change ports, then start FastAPI again from the admin page."
            )
        if response is None:
            return str(exc)
        return f"HTTP {response.status_code}: {response.text}"

    def trained_synthesize_preview(
        text: str,
        voice: str,
        text_lang: str,
        speed: float,
    ):
        payload = {
            "model": OPENAI_MODEL_ALIAS,
            "voice": voice or "default",
            "input": text,
            "response_format": "wav",
            "speed": speed,
            "text_lang": text_lang or None,
        }
        try:
            response = requests.post(api_url("/v1/audio/speech"), json=payload, timeout=300)
            response.raise_for_status()
        except requests.HTTPError as exc:
            return None, http_error_text(exc, "/v1/audio/speech")
        except requests.RequestException as exc:
            return None, f"Request failed: {exc}"
        output_path = audio_file_from_response(response, voice or "preview")
        return output_path, response_metrics_text(response, output_path)

    def clone_synthesize_preview(
        text: str,
        speaker: str,
        ref_audio_file: Optional[str],
        prompt_text: str,
        text_lang: str,
        prompt_lang: str,
        speed: float,
        gpt_weights_path: str,
        sovits_weights_path: str,
    ):
        if not ref_audio_file:
            return None, "请先上传参考音频后再合成克隆音色。"
        try:
            with open(ref_audio_file, "rb") as audio_file:
                files = {"ref_audio": (Path(ref_audio_file).name, audio_file, "audio/wav")}
                data = {
                    "text": text,
                    "speaker": speaker or "clone",
                    "text_lang": text_lang or "zh",
                    "prompt_lang": prompt_lang or text_lang or "zh",
                    "prompt_text": prompt_text or "",
                    "response_format": "wav",
                    "speed": str(speed),
                    "gpt_weights_path": gpt_weights_path or "",
                    "sovits_weights_path": sovits_weights_path or "",
                }
                response = requests.post(api_url("/gpt-sovits/clone/upload"), data=data, files=files, timeout=300)
            response.raise_for_status()
        except requests.HTTPError as exc:
            return None, http_error_text(exc, "/gpt-sovits/clone/upload")
        except requests.RequestException as exc:
            return None, f"Request failed: {exc}"
        output_path = audio_file_from_response(response, speaker or "clone")
        return output_path, response_metrics_text(response, output_path)

    UI_TEXT = {
        "中文": {
            "title": "# Neiroha GPT-SoVITS 管理台",
            "endpoint": f"FastAPI 端点：`{base}`",
            "language": "语言",
            "trained_hint": "使用 `/v1/audio/voices` 中已有的已训练音色。",
            "clone_hint": "上传参考音频并填写对应文本，使用 v2ProPlus 基座权重进行少样本声音克隆；过短会补静音，过长会临时裁剪，不改原仓库。",
            "status": "FastAPI 运行状态",
            "refresh": "刷新",
            "unload": "卸载模型",
            "process": "托管 FastAPI 进程",
            "default_voice": "默认音色",
            "preload": "启动 API 时预加载选中的音色",
            "start_api": "启动 FastAPI",
            "stop_api": "停止 FastAPI",
            "config": "TTS 配置路径",
            "gpt_weights": "GPT 权重路径",
            "sovits_weights": "SoVITS 权重路径",
            "load": "加载 / 应用",
            "set_gpt": "只设置 GPT",
            "set_sovits": "只设置 SoVITS",
            "download_source": "下载源",
            "download_force": "强制重新下载",
            "download_repo": "训练模型仓库",
            "download_speakers": "示例说话人",
            "download_shared_repo": "共享权重仓库",
            "download_shared_presets": "共享权重预设",
            "download_reference_repo": "参考音频数据集",
            "download_reference_characters": "共享参考音频角色",
            "download_reference_languages": "共享参考音频语言",
            "download_activate": "下载后激活 voices.json",
            "download_base": "下载预训练基座",
            "download_v2pro": "下载 v2ProPlus 克隆基座",
            "download_demo": "下载 3 角色 demo",
            "download_extended_demo": "下载扩展多角色 voices",
            "download_shared": "下载共享多说话人权重",
            "download_shared_refs": "下载共享参考音频并生成 voices",
            "download_stop": "停止下载",
            "download_status": "下载状态 / 日志",
            "runtime_events": "运行事件 / RTF",
            "profile_editor": "OpenAI Voice 配置器",
            "profile_editor_hint": "把任意 GPT/SoVITS 权重和参考音频登记成 `/v1/audio/voices` 可见的 voice。",
            "profile_id": "Voice ID",
            "profile_name": "显示名称",
            "profile_model_type": "模型类型",
            "profile_model_id": "模型 ID",
            "profile_model_name": "模型名称",
            "profile_description": "描述",
            "profile_save": "保存 / 更新 voice",
            "profile_delete": "删除 voice",
            "profile_json": "本地 profiles/voices.json",
            "voice_profile": "音色配置",
            "text": "文本",
            "text_language": "文本语言",
            "speed": "速度",
            "trained_button": "合成已训练音色",
            "clone_button": "合成克隆音色",
            "output": "输出",
            "rtf": "性能 / RTF",
            "speaker": "输出说话人名",
            "prompt_language": "参考音频语言",
            "reference_upload": "上传参考音频",
            "reference_path": "参考音频路径",
            "prompt_text": "参考音频对应文本",
            "clone_gpt": "克隆 GPT 权重",
            "clone_sovits": "克隆 SoVITS 权重",
            "profiles": "/v1/audio/voices",
            "native_models": "/gpt-sovits/models",
            "native_voices": "/gpt-sovits/voices",
            "refresh_voices": "刷新音色",
            "refresh_native": "刷新原生模型视图",
        },
        "English": {
            "title": "# Neiroha GPT-SoVITS Admin",
            "endpoint": f"FastAPI endpoint: `{base}`",
            "language": "Language",
            "trained_hint": "Use saved trained voices from `/v1/audio/voices`.",
            "clone_hint": "Upload a reference clip and its transcript, then synthesize with v2ProPlus. Short clips are padded and long clips are trimmed in a temporary file without patching upstream.",
            "status": "FastAPI runtime status",
            "refresh": "Refresh",
            "unload": "Unload model",
            "process": "Managed FastAPI process",
            "default_voice": "Default voice",
            "preload": "Preload selected voice on API start",
            "start_api": "Start FastAPI",
            "stop_api": "Stop FastAPI",
            "config": "TTS config path",
            "gpt_weights": "GPT weights path",
            "sovits_weights": "SoVITS weights path",
            "load": "Load / Apply",
            "set_gpt": "Set GPT only",
            "set_sovits": "Set SoVITS only",
            "download_source": "Download source",
            "download_force": "Force redownload",
            "download_repo": "Trained model repo",
            "download_speakers": "Demo speakers",
            "download_shared_repo": "Shared-weights repo",
            "download_shared_presets": "Shared presets",
            "download_reference_repo": "Reference audio dataset",
            "download_reference_characters": "Shared reference characters",
            "download_reference_languages": "Shared reference languages",
            "download_activate": "Activate voices.json after download",
            "download_base": "Download pretrained base",
            "download_v2pro": "Download v2ProPlus clone base",
            "download_demo": "Download 3-role demo",
            "download_extended_demo": "Download extended voices",
            "download_shared": "Download shared multi-speaker weights",
            "download_shared_refs": "Download shared refs and generate voices",
            "download_stop": "Stop download",
            "download_status": "Download status / logs",
            "runtime_events": "Runtime events / RTF",
            "profile_editor": "OpenAI Voice Profile Builder",
            "profile_editor_hint": "Register any GPT/SoVITS weights plus reference audio as a `/v1/audio/voices` voice.",
            "profile_id": "Voice ID",
            "profile_name": "Display name",
            "profile_model_type": "Model type",
            "profile_model_id": "Model ID",
            "profile_model_name": "Model name",
            "profile_description": "Description",
            "profile_save": "Save / update voice",
            "profile_delete": "Delete voice",
            "profile_json": "Local profiles/voices.json",
            "voice_profile": "Voice profile",
            "text": "Text",
            "text_language": "Text language",
            "speed": "Speed",
            "trained_button": "Synthesize trained voice",
            "clone_button": "Synthesize clone",
            "output": "Output",
            "rtf": "Performance / RTF",
            "speaker": "Output speaker name",
            "prompt_language": "Prompt language",
            "reference_upload": "Reference audio upload",
            "reference_path": "Reference audio path",
            "prompt_text": "Prompt text matching reference audio",
            "clone_gpt": "Clone GPT weights",
            "clone_sovits": "Clone SoVITS weights",
            "profiles": "/v1/audio/voices",
            "native_models": "/gpt-sovits/models",
            "native_voices": "/gpt-sovits/voices",
            "refresh_voices": "Refresh voices",
            "refresh_native": "Refresh native model view",
        },
    }

    def ui_value(language: str, key: str) -> str:
        return UI_TEXT.get(language, UI_TEXT["中文"])[key]

    def ui_copy(language: str) -> tuple[str, str, str, str]:
        return (
            ui_value(language, "title"),
            ui_value(language, "endpoint"),
            ui_value(language, "trained_hint"),
            ui_value(language, "clone_hint"),
        )

    def ui_updates(language: str):
        return (
            ui_value(language, "title"),
            ui_value(language, "endpoint"),
            gr.update(label=ui_value(language, "language")),
            gr.update(label=ui_value(language, "status")),
            gr.update(value=ui_value(language, "refresh")),
            gr.update(value=ui_value(language, "unload")),
            gr.update(label=ui_value(language, "process")),
            gr.update(label=ui_value(language, "default_voice")),
            gr.update(label=ui_value(language, "preload")),
            gr.update(value=ui_value(language, "start_api")),
            gr.update(value=ui_value(language, "stop_api")),
            gr.update(value=ui_value(language, "refresh")),
            gr.update(label=ui_value(language, "config")),
            gr.update(label=ui_value(language, "gpt_weights")),
            gr.update(label=ui_value(language, "sovits_weights")),
            gr.update(value=ui_value(language, "load")),
            gr.update(value=ui_value(language, "set_gpt")),
            gr.update(value=ui_value(language, "set_sovits")),
            gr.update(label=ui_value(language, "download_source")),
            gr.update(label=ui_value(language, "download_force")),
            gr.update(label=ui_value(language, "download_repo")),
            gr.update(label=ui_value(language, "download_speakers")),
            gr.update(label=ui_value(language, "download_shared_repo")),
            gr.update(label=ui_value(language, "download_shared_presets")),
            gr.update(label=ui_value(language, "download_reference_repo")),
            gr.update(label=ui_value(language, "download_reference_characters")),
            gr.update(label=ui_value(language, "download_reference_languages")),
            gr.update(label=ui_value(language, "download_activate")),
            gr.update(value=ui_value(language, "download_base")),
            gr.update(value=ui_value(language, "download_v2pro")),
            gr.update(value=ui_value(language, "download_demo")),
            gr.update(value=ui_value(language, "download_extended_demo")),
            gr.update(value=ui_value(language, "download_shared")),
            gr.update(value=ui_value(language, "download_shared_refs")),
            gr.update(value=ui_value(language, "refresh")),
            gr.update(value=ui_value(language, "download_stop")),
            gr.update(label=ui_value(language, "download_status")),
            gr.update(label=ui_value(language, "runtime_events")),
            ui_value(language, "profile_editor_hint"),
            gr.update(label=ui_value(language, "profile_id")),
            gr.update(label=ui_value(language, "profile_name")),
            gr.update(label=ui_value(language, "profile_model_type")),
            gr.update(label=ui_value(language, "profile_model_id")),
            gr.update(label=ui_value(language, "profile_model_name")),
            gr.update(label=ui_value(language, "profile_description")),
            gr.update(label=ui_value(language, "reference_upload")),
            gr.update(label=ui_value(language, "reference_path")),
            gr.update(label=ui_value(language, "prompt_text")),
            gr.update(label=ui_value(language, "prompt_language")),
            gr.update(label=ui_value(language, "text_language")),
            gr.update(label=ui_value(language, "gpt_weights")),
            gr.update(label=ui_value(language, "sovits_weights")),
            gr.update(value=ui_value(language, "profile_save")),
            gr.update(value=ui_value(language, "profile_delete")),
            gr.update(label=ui_value(language, "profile_json")),
            ui_value(language, "trained_hint"),
            gr.update(label=ui_value(language, "voice_profile")),
            gr.update(label=ui_value(language, "text")),
            gr.update(label=ui_value(language, "text_language")),
            gr.update(label=ui_value(language, "speed")),
            gr.update(value=ui_value(language, "trained_button")),
            gr.update(label=ui_value(language, "output")),
            gr.update(label=ui_value(language, "rtf")),
            ui_value(language, "clone_hint"),
            gr.update(label=ui_value(language, "text")),
            gr.update(label=ui_value(language, "speaker")),
            gr.update(label=ui_value(language, "text_language")),
            gr.update(label=ui_value(language, "prompt_language")),
            gr.update(label=ui_value(language, "speed")),
            gr.update(label=ui_value(language, "reference_upload")),
            gr.update(label=ui_value(language, "prompt_text")),
            gr.update(label=ui_value(language, "clone_gpt")),
            gr.update(label=ui_value(language, "clone_sovits")),
            gr.update(value=ui_value(language, "clone_button")),
            gr.update(label=ui_value(language, "output")),
            gr.update(label=ui_value(language, "rtf")),
            gr.update(label=ui_value(language, "profiles")),
            gr.update(label=ui_value(language, "native_models")),
            gr.update(label=ui_value(language, "native_voices")),
            gr.update(value=ui_value(language, "refresh_voices")),
            gr.update(value=ui_value(language, "refresh_native")),
        )

    with gr.Blocks(title="Neiroha GPT-SoVITS Admin") as blocks:
        title_md = gr.Markdown(ui_copy("中文")[0])
        endpoint_md = gr.Markdown(ui_copy("中文")[1])
        language_radio = gr.Radio(["中文", "English"], value="中文", label=ui_value("中文", "language"))
        with gr.Tab("已训练音色测试"):
            trained_hint = gr.Markdown(ui_copy("中文")[2])
            trained_text_input = gr.Textbox(label=ui_value("中文", "text"), lines=4)
            with gr.Row():
                voice_dropdown = gr.Dropdown(
                    choices=voice_choices(),
                    value=voice_choices()[0],
                    label=ui_value("中文", "voice_profile"),
                )
                trained_text_lang = gr.Textbox(value="zh", label=ui_value("中文", "text_language"))
                trained_speed = gr.Slider(0.25, 4.0, value=1.0, step=0.05, label=ui_value("中文", "speed"))
            trained_synth_btn = gr.Button(ui_value("中文", "trained_button"))
            trained_audio_output = gr.Audio(label=ui_value("中文", "output"), type="filepath")
            trained_metrics_box = gr.Textbox(label=ui_value("中文", "rtf"), lines=4)
            trained_synth_btn.click(
                trained_synthesize_preview,
                inputs=[
                    trained_text_input,
                    voice_dropdown,
                    trained_text_lang,
                    trained_speed,
                ],
                outputs=[trained_audio_output, trained_metrics_box],
            )
        with gr.Tab("声音克隆测试"):
            clone_hint = gr.Markdown(ui_copy("中文")[3])
            clone_text_input = gr.Textbox(label=ui_value("中文", "text"), lines=4)
            clone_ref_audio_file = gr.Audio(type="filepath", label=ui_value("中文", "reference_upload"))
            clone_prompt_text = gr.Textbox(label=ui_value("中文", "prompt_text"), lines=2)
            with gr.Row():
                clone_speaker = gr.Textbox(value="clone", label=ui_value("中文", "speaker"))
                clone_text_lang = gr.Textbox(value="zh", label=ui_value("中文", "text_language"))
                clone_prompt_lang = gr.Textbox(value="zh", label=ui_value("中文", "prompt_language"))
                clone_speed = gr.Slider(0.25, 4.0, value=1.0, step=0.05, label=ui_value("中文", "speed"))
            with gr.Row():
                clone_gpt_input = gr.Textbox(value=str(DEFAULT_CLONE_GPT_WEIGHTS), label=ui_value("中文", "clone_gpt"))
                clone_sovits_input = gr.Textbox(value=str(DEFAULT_CLONE_SOVITS_WEIGHTS), label=ui_value("中文", "clone_sovits"))
            clone_synth_btn = gr.Button(ui_value("中文", "clone_button"))
            clone_audio_output = gr.Audio(label=ui_value("中文", "output"), type="filepath")
            clone_metrics_box = gr.Textbox(label=ui_value("中文", "rtf"), lines=4)
            clone_synth_btn.click(
                clone_synthesize_preview,
                inputs=[
                    clone_text_input,
                    clone_speaker,
                    clone_ref_audio_file,
                    clone_prompt_text,
                    clone_text_lang,
                    clone_prompt_lang,
                    clone_speed,
                    clone_gpt_input,
                    clone_sovits_input,
                ],
                outputs=[clone_audio_output, clone_metrics_box],
            )
        with gr.Tab("API 状态"):
            status_box = gr.Code(value=status_text, language="json", label=ui_value("中文", "status"))
            with gr.Row():
                refresh_btn = gr.Button(ui_value("中文", "refresh"))
                unload_btn = gr.Button(ui_value("中文", "unload"))
            refresh_btn.click(status_text, outputs=status_box)
            unload_btn.click(unload_model, outputs=status_box)
        with gr.Tab("运行事件"):
            runtime_events_box = gr.Textbox(
                value=runtime_events_text,
                label=ui_value("中文", "runtime_events"),
                lines=18,
            )
            runtime_events_refresh_btn = gr.Button(ui_value("中文", "refresh"))
            runtime_events_refresh_btn.click(runtime_events_text, outputs=runtime_events_box)
        with gr.Tab("API 进程"):
            process_box = gr.Textbox(value=managed_status_text, label=ui_value("中文", "process"))
            process_voice = gr.Dropdown(choices=voice_choices(), value=voice_choices()[0], label=ui_value("中文", "default_voice"))
            preload_checkbox = gr.Checkbox(value=False, label=ui_value("中文", "preload"))
            with gr.Row():
                start_api_btn = gr.Button(ui_value("中文", "start_api"))
                stop_api_btn = gr.Button(ui_value("中文", "stop_api"))
                process_refresh_btn = gr.Button(ui_value("中文", "refresh"))
            start_api_btn.click(
                start_api,
                inputs=[process_voice, preload_checkbox],
                outputs=[process_box, status_box],
            )
            stop_api_btn.click(stop_api, outputs=[process_box, status_box])
            process_refresh_btn.click(managed_status_text, outputs=process_box)
        with gr.Tab("加载 / 权重"):
            config_input = gr.Textbox(
                value=str(DEFAULT_CONFIG_PATH),
                label=ui_value("中文", "config"),
            )
            gpt_input = gr.Textbox(
                value=str(PRETRAINED_MODELS_DIR / "s1v3.ckpt"),
                label=ui_value("中文", "gpt_weights"),
            )
            sovits_input = gr.Textbox(
                value=str(PRETRAINED_MODELS_DIR / "v2Pro" / "s2Gv2ProPlus.pth"),
                label=ui_value("中文", "sovits_weights"),
            )
            with gr.Row():
                load_btn = gr.Button(ui_value("中文", "load"))
                gpt_btn = gr.Button(ui_value("中文", "set_gpt"))
                sovits_btn = gr.Button(ui_value("中文", "set_sovits"))
            load_btn.click(load_model, inputs=[config_input, gpt_input, sovits_input], outputs=status_box)
            gpt_btn.click(set_gpt_weights, inputs=gpt_input, outputs=status_box)
            sovits_btn.click(set_sovits_weights, inputs=sovits_input, outputs=status_box)
        with gr.Tab("模型下载"):
            with gr.Row():
                download_source = gr.Dropdown(
                    choices=["modelscope", "hf", "hf-mirror"],
                    value=DEFAULT_DOWNLOAD_SOURCE,
                    label=ui_value("中文", "download_source"),
                )
                download_force = gr.Checkbox(value=False, label=ui_value("中文", "download_force"))
            with gr.Row():
                download_repo = gr.Textbox(value=DEFAULT_DEMO_REPO_ID, label=ui_value("中文", "download_repo"))
                download_speakers = gr.Textbox(value=DEFAULT_DEMO_SPEAKERS, label=ui_value("中文", "download_speakers"))
                download_activate = gr.Checkbox(value=True, label=ui_value("中文", "download_activate"))
            with gr.Row():
                download_shared_repo = gr.Textbox(value=DEFAULT_SHARED_REPO_ID, label=ui_value("中文", "download_shared_repo"))
                download_shared_presets = gr.Textbox(
                    value=DEFAULT_SHARED_PRESETS,
                    label=ui_value("中文", "download_shared_presets"),
                )
            with gr.Row():
                download_reference_repo = gr.Textbox(
                    value=DEFAULT_SHARED_REFERENCE_REPO_ID,
                    label=ui_value("中文", "download_reference_repo"),
                )
                download_reference_characters = gr.Textbox(
                    value=DEFAULT_SHARED_REFERENCE_CHARACTERS,
                    label=ui_value("中文", "download_reference_characters"),
                )
                download_reference_languages = gr.Textbox(
                    value=DEFAULT_SHARED_REFERENCE_LANGUAGES,
                    label=ui_value("中文", "download_reference_languages"),
                )
            with gr.Row():
                download_base_btn = gr.Button(ui_value("中文", "download_base"))
                download_v2pro_btn = gr.Button(ui_value("中文", "download_v2pro"))
                download_demo_btn = gr.Button(ui_value("中文", "download_demo"))
                download_extended_demo_btn = gr.Button(ui_value("中文", "download_extended_demo"))
                download_shared_btn = gr.Button(ui_value("中文", "download_shared"))
                download_shared_refs_btn = gr.Button(ui_value("中文", "download_shared_refs"))
                download_refresh_btn = gr.Button(ui_value("中文", "refresh"))
                download_stop_btn = gr.Button(ui_value("中文", "download_stop"))
            download_status_box = gr.Textbox(
                value=download_manager.status,
                label=ui_value("中文", "download_status"),
                lines=16,
            )
            download_base_btn.click(
                start_base_assets_download,
                inputs=[download_source, download_force],
                outputs=download_status_box,
            )
            download_v2pro_btn.click(
                start_v2pro_download,
                inputs=[download_source, download_force],
                outputs=download_status_box,
            )
            download_demo_btn.click(
                start_demo_download,
                inputs=[download_source, download_repo, download_speakers, download_activate, download_force],
                outputs=download_status_box,
            )
            download_extended_demo_btn.click(
                start_extended_demo_download,
                inputs=[download_source, download_repo, download_activate, download_force],
                outputs=download_status_box,
            )
            download_shared_btn.click(
                start_shared_multispeaker_download,
                inputs=[download_source, download_shared_repo, download_shared_presets, download_force],
                outputs=download_status_box,
            )
            download_shared_refs_btn.click(
                start_shared_reference_download,
                inputs=[
                    download_source,
                    download_shared_repo,
                    download_reference_repo,
                    download_reference_characters,
                    download_reference_languages,
                    download_activate,
                    download_force,
                ],
                outputs=download_status_box,
            )
            download_refresh_btn.click(download_manager.status, outputs=download_status_box)
            download_stop_btn.click(download_manager.stop, outputs=download_status_box)
        with gr.Tab("说话人"):
            profiles_box = gr.Code(value=profiles_text, language="json", label=ui_value("中文", "profiles"))
            voices_refresh_btn = gr.Button(ui_value("中文", "refresh_voices"))
            voices_refresh_btn.click(profiles_text, outputs=profiles_box)
            native_models_box = gr.Code(value=native_models_text, language="json", label=ui_value("中文", "native_models"))
            native_voices_box = gr.Code(value=native_voices_text, language="json", label=ui_value("中文", "native_voices"))
            native_refresh_btn = gr.Button(ui_value("中文", "refresh_native"))
            native_refresh_btn.click(
                lambda: (native_models_text(), native_voices_text()),
                outputs=[native_models_box, native_voices_box],
            )
        with gr.Tab("OpenAI Voice 配置"):
            profile_editor_hint = gr.Markdown(ui_value("中文", "profile_editor_hint"))
            profile_status_box = gr.Textbox(label=ui_value("中文", "profile_editor"), lines=2)
            with gr.Row():
                profile_id_input = gr.Textbox(value="custom-voice", label=ui_value("中文", "profile_id"))
                profile_name_input = gr.Textbox(value="Custom Voice", label=ui_value("中文", "profile_name"))
                profile_model_type = gr.Dropdown(
                    choices=["reference-profile", "trained", "shared-trained"],
                    value="reference-profile",
                    label=ui_value("中文", "profile_model_type"),
                )
            with gr.Row():
                profile_model_id = gr.Textbox(value="custom-openai-voice", label=ui_value("中文", "profile_model_id"))
                profile_model_name = gr.Textbox(value="Custom OpenAI Voice", label=ui_value("中文", "profile_model_name"))
            profile_description = gr.Textbox(label=ui_value("中文", "profile_description"), lines=2)
            profile_ref_audio_file = gr.Audio(type="filepath", label=ui_value("中文", "reference_upload"))
            profile_ref_audio_path = gr.Textbox(label=ui_value("中文", "reference_path"))
            profile_prompt_text = gr.Textbox(label=ui_value("中文", "prompt_text"), lines=2)
            with gr.Row():
                profile_prompt_lang = gr.Textbox(value="zh", label=ui_value("中文", "prompt_language"))
                profile_text_lang = gr.Textbox(value="zh", label=ui_value("中文", "text_language"))
            with gr.Row():
                profile_gpt_weights = gr.Textbox(value=str(DEFAULT_CLONE_GPT_WEIGHTS), label=ui_value("中文", "gpt_weights"))
                profile_sovits_weights = gr.Textbox(value=str(DEFAULT_CLONE_SOVITS_WEIGHTS), label=ui_value("中文", "sovits_weights"))
            with gr.Row():
                profile_save_btn = gr.Button(ui_value("中文", "profile_save"))
                profile_delete_btn = gr.Button(ui_value("中文", "profile_delete"))
            profile_json_box = gr.Code(value=profile_editor_text, language="json", label=ui_value("中文", "profile_json"))
            profile_save_btn.click(
                save_openai_voice_profile,
                inputs=[
                    profile_id_input,
                    profile_name_input,
                    profile_model_type,
                    profile_model_id,
                    profile_model_name,
                    profile_description,
                    profile_ref_audio_file,
                    profile_ref_audio_path,
                    profile_prompt_text,
                    profile_prompt_lang,
                    profile_text_lang,
                    profile_gpt_weights,
                    profile_sovits_weights,
                ],
                outputs=[
                    profile_status_box,
                    profile_json_box,
                    voice_dropdown,
                    process_voice,
                    native_models_box,
                    native_voices_box,
                ],
            )
            profile_delete_btn.click(
                delete_openai_voice_profile,
                inputs=[profile_id_input],
                outputs=[
                    profile_status_box,
                    profile_json_box,
                    voice_dropdown,
                    process_voice,
                    native_models_box,
                    native_voices_box,
                ],
            )

        language_radio.change(
            ui_updates,
            inputs=language_radio,
            outputs=[
                title_md,
                endpoint_md,
                language_radio,
                status_box,
                refresh_btn,
                unload_btn,
                process_box,
                process_voice,
                preload_checkbox,
                start_api_btn,
                stop_api_btn,
                process_refresh_btn,
                config_input,
                gpt_input,
                sovits_input,
                load_btn,
                gpt_btn,
                sovits_btn,
                download_source,
                download_force,
                download_repo,
                download_speakers,
                download_shared_repo,
                download_shared_presets,
                download_reference_repo,
                download_reference_characters,
                download_reference_languages,
                download_activate,
                download_base_btn,
                download_v2pro_btn,
                download_demo_btn,
                download_extended_demo_btn,
                download_shared_btn,
                download_shared_refs_btn,
                download_refresh_btn,
                download_stop_btn,
                download_status_box,
                runtime_events_box,
                profile_editor_hint,
                profile_id_input,
                profile_name_input,
                profile_model_type,
                profile_model_id,
                profile_model_name,
                profile_description,
                profile_ref_audio_file,
                profile_ref_audio_path,
                profile_prompt_text,
                profile_prompt_lang,
                profile_text_lang,
                profile_gpt_weights,
                profile_sovits_weights,
                profile_save_btn,
                profile_delete_btn,
                profile_json_box,
                trained_hint,
                voice_dropdown,
                trained_text_input,
                trained_text_lang,
                trained_speed,
                trained_synth_btn,
                trained_audio_output,
                trained_metrics_box,
                clone_hint,
                clone_text_input,
                clone_speaker,
                clone_text_lang,
                clone_prompt_lang,
                clone_speed,
                clone_ref_audio_file,
                clone_prompt_text,
                clone_gpt_input,
                clone_sovits_input,
                clone_synth_btn,
                clone_audio_output,
                clone_metrics_box,
                profiles_box,
                native_models_box,
                native_voices_box,
                voices_refresh_btn,
                native_refresh_btn,
            ],
        )
    return blocks.queue(max_size=8, default_concurrency_limit=1)


def build_gradio_admin_blocks(
    api_base: str,
    registry: VoiceRegistry,
    *,
    process_manager: Optional[ManagedApiProcess] = None,
):
    import gradio as gr
    import requests

    base = api_base.rstrip("/")
    download_manager = ManagedDownloadProcess()
    ui_config = load_ui_config(registry)
    language = strip_text(os.environ.get("NEIROHA_GPT_SOVITS_UI_LANG") or ui_config.get("default_language") or "zh")
    language = "en" if language.lower().startswith("en") else "zh"
    ui_title = strip_text(ui_config.get("title")) or "Neiroha GPT-SoVITS Admin"
    ui_text = {
        "zh": {
            "api_offline": "### API 离线\n",
            "api_online": "### API 在线\n",
            "api_base": "API Base",
            "refresh_time": "刷新时间",
            "error": "错误",
            "wait_preload": "如果是从 `start_api_admin.bat` 启动，等预加载完成后这里会自动刷新。",
            "loaded": "已加载",
            "not_loaded": "未加载",
            "model_status": "模型状态",
            "device": "设备",
            "half": "半精度",
            "active_preset": "当前 preset",
            "active_voice_set": "当前 voice set",
            "default_voice": "默认 voice",
            "available_model": "可用 model",
            "available_voice": "可用 voice",
            "gpt_weights": "GPT 权重",
            "sovits_weights": "SoVITS 权重",
            "external_process": "API 进程由外层 launcher / start_api_admin.bat 管理；此 Admin 只连接并显示状态。",
            "home": "首页",
            "trial": "试音",
            "clone_config": "克隆配置",
            "model_presets": "Model Presets",
            "download": "下载",
            "logs": "日志",
            "runtime_status": "运行状态",
            "api_process": "API 进程",
            "refresh": "刷新",
            "load_active": "加载当前 preset",
            "unload_model": "卸载模型",
            "preload": "预加载",
            "start_api": "启动 API",
            "stop_api": "停止 API",
            "voice_set_model": "voice set / model",
            "voice": "voice",
            "text": "文本",
            "format": "格式",
            "speed": "速度",
            "generate": "生成",
            "audio_output": "输出音频",
            "metrics": "RTF / 输出",
            "save_to_voice_set": "保存到 voice set",
            "use_model_preset": "使用 model preset",
            "voice_id": "voice id",
            "name": "name",
            "upload_reference": "上传参考音频",
            "reference_path": "或填写参考音频路径",
            "prompt_text": "prompt_text",
            "prompt_lang": "prompt_lang",
            "text_lang": "text_lang",
            "default_speed": "默认速度",
            "trained_weight_override": "可选：覆盖为别人训练过的 GPT/SoVITS 权重",
            "gpt_ckpt": "GPT 权重 .ckpt",
            "sovits_pth": "SoVITS 权重 .pth",
            "save_voice": "保存 voice",
            "save_result": "保存结果",
            "voice_sets": "Voice Sets",
            "current_preset": "当前底层 preset",
            "load": "加载",
            "unload": "卸载",
            "reload": "重载",
            "preset_status": "Presets / 状态",
            "new_preset": "新增 / 更新训练模型 preset",
            "preset_id": "preset id",
            "save_preset": "保存 preset",
            "config_path": "config_path",
            "download_source": "下载源",
            "force_redownload": "强制重新下载",
            "download_base": "下载预训练基座",
            "download_v2pro": "下载 v2ProPlus 克隆基座",
            "download_sample": "下载单个示例参考音频",
            "refresh_log": "刷新日志",
            "stop_download": "停止下载",
            "download_status": "下载状态 / 日志",
            "backend_log": "backend.log（最新在上，自动刷新）",
            "prompt_required": "prompt_text is required.",
            "reference_required": "reference audio is required.",
            "weights_required": "GPT 权重和 SoVITS 权重路径都要填。",
            "saved_model_preset": "已保存模型 preset",
            "saved_voice_profile": "已保存 voice 配置",
            "pretrained_base_assets": "预训练基座资源",
            "v2pro_clone_base": "v2ProPlus 克隆基座",
            "single_sample_reference": "单个示例参考音频",
            "log_endpoint_unavailable": "API 日志接口不可用",
        },
        "en": {
            "api_offline": "### API Offline\n",
            "api_online": "### API Online\n",
            "api_base": "API Base",
            "refresh_time": "Refreshed",
            "error": "Error",
            "wait_preload": "If launched from `start_api_admin.bat`, this panel will refresh automatically after preload finishes.",
            "loaded": "loaded",
            "not_loaded": "not loaded",
            "model_status": "Model",
            "device": "Device",
            "half": "Half",
            "active_preset": "Active preset",
            "active_voice_set": "Voice set",
            "default_voice": "Default voice",
            "available_model": "Models",
            "available_voice": "Voices",
            "gpt_weights": "GPT weights",
            "sovits_weights": "SoVITS weights",
            "external_process": "The API process is managed by the outer launcher / start_api_admin.bat; this Admin only connects to it.",
            "home": "Home",
            "trial": "Test Voice",
            "clone_config": "Voice Config",
            "model_presets": "Model Presets",
            "download": "Downloads",
            "logs": "Logs",
            "runtime_status": "Runtime Status",
            "api_process": "API Process",
            "refresh": "Refresh",
            "load_active": "Load Active Preset",
            "unload_model": "Unload Model",
            "preload": "Preload",
            "start_api": "Start API",
            "stop_api": "Stop API",
            "voice_set_model": "voice set / model",
            "voice": "voice",
            "text": "Text",
            "format": "Format",
            "speed": "Speed",
            "generate": "Generate",
            "audio_output": "Output audio",
            "metrics": "RTF / Output",
            "save_to_voice_set": "Save to voice set",
            "use_model_preset": "Use model preset",
            "voice_id": "voice id",
            "name": "name",
            "upload_reference": "Upload reference audio",
            "reference_path": "Or reference audio path",
            "prompt_text": "prompt_text",
            "prompt_lang": "prompt_lang",
            "text_lang": "text_lang",
            "default_speed": "Default speed",
            "trained_weight_override": "Optional: override with trained GPT/SoVITS weights",
            "gpt_ckpt": "GPT weights .ckpt",
            "sovits_pth": "SoVITS weights .pth",
            "save_voice": "Save voice",
            "save_result": "Save result",
            "voice_sets": "Voice Sets",
            "current_preset": "Current runtime preset",
            "load": "Load",
            "unload": "Unload",
            "reload": "Reload",
            "preset_status": "Presets / Status",
            "new_preset": "Add / Update Trained Model Preset",
            "preset_id": "preset id",
            "save_preset": "Save preset",
            "config_path": "config_path",
            "download_source": "Download source",
            "force_redownload": "Force redownload",
            "download_base": "Download pretrained base",
            "download_v2pro": "Download v2ProPlus clone base",
            "download_sample": "Download single sample reference",
            "refresh_log": "Refresh logs",
            "stop_download": "Stop download",
            "download_status": "Download status / logs",
            "backend_log": "backend.log (newest first, auto-refresh)",
            "prompt_required": "prompt_text is required.",
            "reference_required": "reference audio is required.",
            "weights_required": "GPT and SoVITS weight paths are both required.",
            "saved_model_preset": "Saved model preset",
            "saved_voice_profile": "Saved voice profile",
            "pretrained_base_assets": "pretrained base assets",
            "v2pro_clone_base": "v2ProPlus clone base",
            "single_sample_reference": "single sample reference voice",
            "log_endpoint_unavailable": "API log endpoint unavailable",
        },
    }[language]

    def t(key: str) -> str:
        return ui_text.get(key, key)

    def api_url(path: str) -> str:
        return f"{base}{path if path.startswith('/') else '/' + path}"

    def request_json(method: str, path: str, **kwargs: Any) -> str:
        response = requests.request(method, api_url(path), timeout=20, **kwargs)
        try:
            payload = response.json()
        except Exception:
            payload = response.text
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def request_payload(method: str, path: str, timeout: int = 5, **kwargs: Any) -> Any:
        response = requests.request(method, api_url(path), timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()

    def request_text(method: str, path: str, timeout: int = 5, **kwargs: Any) -> str:
        response = requests.request(method, api_url(path), timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.text

    def home_status() -> str:
        now = dt.datetime.now().strftime("%H:%M:%S")
        try:
            health = request_payload("GET", "/health", timeout=2)
            models = request_payload("GET", "/v1/models", timeout=2).get("data", [])
            voices = request_payload("GET", "/v1/audio/voices", timeout=2).get("data", [])
        except Exception as exc:
            return (
                t("api_offline")
                + f"- {t('api_base')}: `{base}`\n"
                + f"- {t('refresh_time')}: `{now}`\n"
                + f"- {t('error')}: `{exc}`\n"
                + f"- {t('wait_preload')}"
            )

        loaded = t("loaded") if health.get("loaded") else t("not_loaded")
        device = health.get("device") or "-"
        is_half = health.get("is_half")
        active_preset = health.get("active_model_preset") or registry.active_model_preset_id()
        active_set = health.get("active_voice_set") or registry.active_voice_set_id()
        default_voice = health.get("default_voice") or registry.default_voice_id()
        gpt_weights = health.get("gpt_weights_path") or "-"
        sovits_weights = health.get("sovits_weights_path") or "-"
        voice_list = ", ".join(strip_text(item.get("voice_id") or item.get("id")) for item in voices) or "-"
        model_list = ", ".join(strip_text(item.get("id")) for item in models) or "-"
        return (
            t("api_online")
            + f"- {t('api_base')}: `{base}`\n"
            + f"- {t('refresh_time')}: `{now}`\n"
            + f"- {t('model_status')}: `{loaded}`  {t('device')}: `{device}`  {t('half')}: `{is_half}`\n"
            f"- {t('active_preset')}: `{active_preset}`\n"
            f"- {t('active_voice_set')}: `{active_set}`  {t('default_voice')}: `{default_voice}`\n"
            f"- {t('available_model')}: `{model_list}`\n"
            f"- {t('available_voice')}: `{voice_list}`\n"
            f"- {t('gpt_weights')}: `{gpt_weights}`\n"
            f"- {t('sovits_weights')}: `{sovits_weights}`"
        )

    def process_status() -> str:
        now = dt.datetime.now().strftime("%H:%M:%S")
        if process_manager:
            status = process_manager.status()
        else:
            status = t("external_process")
        return f"{status}\n{t('api_base')}: {base}\n{t('refresh_time')}: {now}"

    def home_refresh() -> tuple[str, str]:
        return home_status(), process_status()

    def start_api(default_voice: str, preload: bool) -> tuple[str, str]:
        if process_manager is None:
            return process_status(), home_status()
        return process_manager.start(default_voice, preload), home_status()

    def stop_api() -> tuple[str, str]:
        if process_manager is None:
            return process_status(), home_status()
        return process_manager.stop(), home_status()

    def model_choices() -> list[str]:
        try:
            data = requests.get(api_url("/v1/models"), timeout=2).json().get("data", [])
            choices = [strip_text(item.get("id")) for item in data if strip_text(item.get("id"))]
            return choices or [registry.active_voice_set_id()]
        except Exception:
            return [voice_set.id for voice_set in registry.list_voice_sets()]

    def voice_choices(model_id: str = "") -> list[str]:
        try:
            data = requests.get(api_url("/v1/audio/voices"), timeout=2).json().get("data", [])
            choices = [
                strip_text(item.get("voice_id") or item.get("id"))
                for item in data
                if not model_id or item.get("model") == model_id
            ]
            return choices or [registry.default_voice_id()]
        except Exception:
            return [profile.id for profile in registry.list_profiles(model_id)] or [registry.default_voice_id()]

    def refresh_choices(model_id: str):
        models = model_choices()
        selected_model = model_id if model_id in models else models[0]
        voices = voice_choices(selected_model)
        selected_voice = voices[0] if voices else registry.default_voice_id()
        return gr.update(choices=models, value=selected_model), gr.update(choices=voices, value=selected_voice)

    def refresh_voice_dropdown(model_id: str):
        voices = voice_choices(model_id)
        return gr.update(choices=voices, value=voices[0] if voices else registry.default_voice_id())

    def synthesize_preview(model_id: str, voice_id: str, text: str, response_format: str, speed: float):
        payload = {
            "model": model_id or registry.active_voice_set_id(),
            "voice": voice_id or registry.default_voice_id(),
            "input": text,
            "response_format": response_format or "wav",
            "speed": speed or 1.0,
        }
        response = requests.post(api_url("/v1/audio/speech"), json=payload, timeout=600)
        if response.status_code >= 400:
            return None, response.text
        output_path = response.headers.get("X-Neiroha-Output-Path")
        if not output_path:
            suffix = require_supported_format(response_format or "wav")
            output_path = str(OUTPUT_ROOT / f"admin_preview_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}.{suffix}")
            Path(output_path).write_bytes(response.content)
        elif not Path(output_path).is_absolute():
            output_path = str((WORKSPACE_ROOT / output_path).resolve())
        metrics = {
            "output_path": output_path,
            "audio_seconds": response.headers.get("X-Neiroha-Audio-Seconds", ""),
            "elapsed_seconds": response.headers.get("X-Neiroha-Elapsed-Seconds", ""),
            "rtf": response.headers.get("X-Neiroha-RTF", ""),
        }
        return output_path, json.dumps(metrics, ensure_ascii=False, indent=2)

    def presets_text() -> str:
        return json.dumps(
            {"data": [preset.to_native_model() for preset in registry.list_model_presets()]},
            ensure_ascii=False,
            indent=2,
        )

    def voice_sets_text() -> str:
        return json.dumps(
            {
                "active_voice_set": registry.active_voice_set_id(),
                "default_voice": registry.default_voice_id(),
                "data": [
                    voice_set.to_openai_model(len(registry.list_profiles(voice_set.id)))
                    | {"voices": voice_set.voices or []}
                    for voice_set in registry.list_voice_sets()
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

    def load_preset(preset_id: str) -> str:
        preset = registry.get_model_preset(preset_id)
        return request_json(
            "POST",
            "/gpt-sovits/load",
            json={
                "config_path": preset.config_path,
                "gpt_weights_path": preset.gpt_weights_path,
                "sovits_weights_path": preset.sovits_weights_path,
            },
        )

    def load_active_preset() -> str:
        return load_preset(registry.active_model_preset_id())

    def load_active_and_refresh() -> tuple[str, str]:
        load_active_preset()
        return home_refresh()

    def unload_model() -> str:
        return request_json("POST", "/gpt-sovits/unload")

    def unload_and_refresh() -> tuple[str, str]:
        unload_model()
        return home_refresh()

    def reload_preset(preset_id: str) -> str:
        preset = registry.get_model_preset(preset_id)
        return request_json(
            "POST",
            "/gpt-sovits/reload",
            json={
                "config_path": preset.config_path,
                "gpt_weights_path": preset.gpt_weights_path,
                "sovits_weights_path": preset.sovits_weights_path,
            },
        )

    def write_model_preset(
        preset_id: str,
        preset_name: str,
        config_path: str,
        gpt_weights_path: str,
        sovits_weights_path: str,
    ) -> tuple[str, Any, Any]:
        preset_id = safe_filename_part(preset_id, fallback="trained-preset")
        preset_name = strip_text(preset_name) or preset_id
        config_path = strip_text(config_path) or profile_path_text(DEFAULT_CONFIG_PATH)
        gpt_weights_path = strip_text(gpt_weights_path)
        sovits_weights_path = strip_text(sovits_weights_path)
        if not gpt_weights_path or not sovits_weights_path:
            raise gr.Error(t("weights_required"))

        preset_path = MODEL_PRESETS_DIR / f"{preset_id}.toml"
        preset_path.parent.mkdir(parents=True, exist_ok=True)
        config_path_text = config_path.replace("\\", "/")
        gpt_weights_text = gpt_weights_path.replace("\\", "/")
        sovits_weights_text = sovits_weights_path.replace("\\", "/")
        toml_string = lambda value: json.dumps(str(value), ensure_ascii=False)
        preset_path.write_text(
            "\n".join(
                [
                    "schema_version = 1",
                    f"id = {toml_string(preset_id)}",
                    f"name = {toml_string(preset_name)}",
                    'engine = "gpt-sovits"',
                    "",
                    "[gpt_sovits]",
                    f"config_path = {toml_string(config_path_text)}",
                    f"gpt_weights_path = {toml_string(gpt_weights_text)}",
                    f"sovits_weights_path = {toml_string(sovits_weights_text)}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        choices = [preset.id for preset in registry.list_model_presets()]
        status = f"{t('saved_model_preset')}: {profile_path_text(preset_path)}"
        return status, gr.update(choices=choices, value=preset_id), gr.update(choices=choices, value=preset_id)

    def write_voice_profile(
        voice_set_id: str,
        model_preset_id: str,
        voice_id: str,
        voice_name: str,
        ref_audio_file: Optional[str],
        ref_audio_path: str,
        prompt_text: str,
        prompt_lang: str,
        text_lang: str,
        speed: float,
        gpt_weights_path: str,
        sovits_weights_path: str,
    ) -> tuple[str, str, Any, Any]:
        voice_id = safe_filename_part(voice_id, fallback="local-voice")
        voice_name = strip_text(voice_name) or voice_id
        prompt_text = strip_text(prompt_text)
        if not prompt_text:
            raise gr.Error(t("prompt_required"))
        voice_set_id = strip_text(voice_set_id) or registry.active_voice_set_id()
        model_preset_id = strip_text(model_preset_id) or registry.active_model_preset_id()
        voice_dir = RUNTIME_VOICES_ROOT / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)

        reference_audio = strip_text(ref_audio_path)
        if ref_audio_file:
            source = Path(ref_audio_file)
            suffix = source.suffix or ".wav"
            target = voice_dir / f"reference{suffix}"
            shutil.copyfile(source, target)
            reference_audio = profile_path_text(target)
        if not reference_audio:
            raise gr.Error(t("reference_required"))

        payload = {
            "schema_version": 1,
            "id": voice_id,
            "name": voice_name,
            "mode": "prompt_clone",
            "model_preset": model_preset_id,
            "reference_audio": reference_audio,
            "prompt_audio": "",
            "prompt_text": prompt_text,
            "text_lang": strip_text(text_lang) or "zh",
            "prompt_lang": strip_text(prompt_lang) or "zh",
            "instruction": "",
            "speed": float(speed or 1.0),
            "engine_options": {},
        }
        if strip_text(gpt_weights_path) or strip_text(sovits_weights_path):
            payload["gpt_weights_path"] = strip_text(gpt_weights_path)
            payload["sovits_weights_path"] = strip_text(sovits_weights_path)
        voice_profile_path = voice_dir / "voice.toml"
        write_toml_mapping(voice_profile_path, payload)
        (voice_dir / "voice.json").unlink(missing_ok=True)

        voice_set_path = VOICE_SETS_DIR / f"{voice_set_id}.toml"
        legacy_voice_set_path = VOICE_SETS_DIR / f"{voice_set_id}.json"
        if voice_set_path.exists():
            voice_set_payload = read_mapping_file(voice_set_path)
        elif legacy_voice_set_path.exists():
            voice_set_payload = read_mapping_file(legacy_voice_set_path)
        else:
            voice_set_payload = {
                "schema_version": 1,
                "id": voice_set_id,
                "name": voice_set_id,
                "description": "Local voice set.",
                "voices": [],
            }
        voices = [
            strip_text(item)
            for item in voice_set_payload.get("voices", [])
            if strip_text(item) and strip_text(item) != voice_id
        ]
        voice_set_payload["voices"] = [*voices, voice_id]
        write_toml_mapping(voice_set_path, voice_set_payload)
        legacy_voice_set_path.unlink(missing_ok=True)
        status = (
            f"{t('saved_voice_profile')}: {profile_path_text(voice_profile_path)}\n"
            f"Reference audio: {reference_audio}\n"
            f"Voice set: {profile_path_text(voice_set_path)}"
        )
        models, voices_update = refresh_choices(voice_set_id)
        return status, voice_sets_text(), models, voices_update

    def start_base_assets_download(source: str, force: bool) -> str:
        args = ["--source", source or DEFAULT_DOWNLOAD_SOURCE]
        if force:
            args.append("--force")
        return download_manager.start(t("pretrained_base_assets"), args)

    def start_v2pro_download(source: str, force: bool) -> str:
        args = ["--source", source or DEFAULT_DOWNLOAD_SOURCE, "--skip-base-assets", "--v2pro-plus"]
        if force:
            args.append("--force")
        return download_manager.start(t("v2pro_clone_base"), args)

    def start_sample_voice_download(source: str, force: bool) -> str:
        args = ["--source", source or DEFAULT_DOWNLOAD_SOURCE, "--skip-base-assets", "--sample-reference", "--activate-voices"]
        if force:
            args.append("--force")
        return download_manager.start(t("single_sample_reference"), args)

    def runtime_events_text() -> str:
        try:
            return request_text("GET", "/gpt-sovits/logs?limit=160", timeout=3)
        except Exception as exc:
            return f"{t('log_endpoint_unavailable')}: {exc}\n\n{RUNTIME_EVENTS.tail(160)}"

    preset_ids = [preset.id for preset in registry.list_model_presets()]
    initial_models = model_choices()
    initial_model = initial_models[0] if initial_models else registry.active_voice_set_id()
    initial_voices = voice_choices(initial_model)

    with gr.Blocks(title=ui_title) as blocks:
        gr.Markdown(f"# {ui_title}")
        with gr.Tab(t("home")):
            status_box = gr.Markdown(value=home_status(), label=t("runtime_status"))
            process_box = gr.Textbox(value=process_status(), label=t("api_process"), lines=3)
            with gr.Row():
                refresh_btn = gr.Button(t("refresh"))
                load_active_btn = gr.Button(t("load_active"))
                unload_active_btn = gr.Button(t("unload_model"))
            if process_manager is not None:
                with gr.Row():
                    preload_checkbox = gr.Checkbox(value=False, label=t("preload"))
                    process_default_voice = gr.Dropdown(
                        choices=voice_choices(initial_model),
                        value=registry.default_voice_id(),
                        label=t("default_voice"),
                    )
                    start_api_btn = gr.Button(t("start_api"))
                    stop_api_btn = gr.Button(t("stop_api"))
                start_api_btn.click(start_api, inputs=[process_default_voice, preload_checkbox], outputs=[process_box, status_box])
                stop_api_btn.click(stop_api, outputs=[process_box, status_box])
            refresh_btn.click(home_refresh, outputs=[status_box, process_box])
            load_active_btn.click(load_active_and_refresh, outputs=[status_box, process_box])
            unload_active_btn.click(unload_and_refresh, outputs=[status_box, process_box])
            home_timer = gr.Timer(value=2.0, active=True)
            home_timer.tick(home_refresh, outputs=[status_box, process_box])

        with gr.Tab(t("trial")):
            with gr.Row():
                model_dropdown = gr.Dropdown(choices=initial_models, value=initial_model, label=t("voice_set_model"))
                voice_dropdown = gr.Dropdown(
                    choices=initial_voices,
                    value=initial_voices[0] if initial_voices else registry.default_voice_id(),
                    label=t("voice"),
                )
                refresh_choices_btn = gr.Button(t("refresh"))
            default_preview_text = (
                "Hello, this is a Neiroha GPT-SoVITS voice cloning test."
                if language == "en"
                else "你好，这是 Neiroha GPT-SoVITS 的语音复刻测试。"
            )
            text_input = gr.Textbox(value=default_preview_text, label=t("text"), lines=3)
            with gr.Row():
                format_dropdown = gr.Dropdown(choices=sorted(SUPPORTED_OPENAI_FORMATS), value="wav", label=t("format"))
                speed_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label=t("speed"))
                synth_btn = gr.Button(t("generate"))
            audio_output = gr.Audio(type="filepath", label=t("audio_output"))
            metrics_box = gr.Code(label=t("metrics"), language="json")
            model_dropdown.change(refresh_voice_dropdown, inputs=model_dropdown, outputs=voice_dropdown)
            refresh_choices_btn.click(refresh_choices, inputs=model_dropdown, outputs=[model_dropdown, voice_dropdown])
            synth_btn.click(
                synthesize_preview,
                inputs=[model_dropdown, voice_dropdown, text_input, format_dropdown, speed_slider],
                outputs=[audio_output, metrics_box],
            )

        with gr.Tab(t("clone_config")):
            with gr.Row():
                clone_voice_set = gr.Dropdown(
                    choices=[voice_set.id for voice_set in registry.list_voice_sets()],
                    value=registry.active_voice_set_id(),
                    label=t("save_to_voice_set"),
                )
                clone_model_preset = gr.Dropdown(
                    choices=preset_ids,
                    value=registry.active_model_preset_id() if registry.active_model_preset_id() in preset_ids else preset_ids[0],
                    label=t("use_model_preset"),
                )
                clone_voice_id = gr.Textbox(value="local-voice", label=t("voice_id"))
                clone_voice_name = gr.Textbox(value="Local Voice", label=t("name"))
            clone_ref_file = gr.Audio(type="filepath", label=t("upload_reference"))
            clone_ref_path = gr.Textbox(label=t("reference_path"))
            clone_prompt_text = gr.Textbox(label=t("prompt_text"), lines=2)
            with gr.Row():
                clone_prompt_lang = gr.Textbox(value="zh", label=t("prompt_lang"))
                clone_text_lang = gr.Textbox(value="zh", label=t("text_lang"))
                clone_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label=t("default_speed"))
            with gr.Accordion(t("trained_weight_override"), open=False):
                clone_gpt_weights = gr.Textbox(
                    label=t("gpt_ckpt"),
                    placeholder="models/voices/.../xxx.ckpt",
                )
                clone_sovits_weights = gr.Textbox(
                    label=t("sovits_pth"),
                    placeholder="models/voices/.../xxx.pth",
                )
            save_voice_btn = gr.Button(t("save_voice"))
            save_voice_status = gr.Textbox(label=t("save_result"))
            voice_sets_box = gr.Code(value=voice_sets_text, language="json", label=t("voice_sets"))
            save_voice_btn.click(
                write_voice_profile,
                inputs=[
                    clone_voice_set,
                    clone_model_preset,
                    clone_voice_id,
                    clone_voice_name,
                    clone_ref_file,
                    clone_ref_path,
                    clone_prompt_text,
                    clone_prompt_lang,
                    clone_text_lang,
                    clone_speed,
                    clone_gpt_weights,
                    clone_sovits_weights,
                ],
                outputs=[save_voice_status, voice_sets_box, model_dropdown, voice_dropdown],
            )

        with gr.Tab(t("model_presets")):
            preset_dropdown = gr.Dropdown(
                choices=preset_ids,
                value=registry.active_model_preset_id() if registry.active_model_preset_id() in preset_ids else preset_ids[0],
                label=t("current_preset"),
            )
            with gr.Row():
                load_btn = gr.Button(t("load"))
                unload_btn = gr.Button(t("unload"))
                reload_btn = gr.Button(t("reload"))
            preset_status = gr.Code(value=presets_text, language="json", label=t("preset_status"))
            load_btn.click(load_preset, inputs=preset_dropdown, outputs=preset_status)
            unload_btn.click(unload_model, outputs=preset_status)
            reload_btn.click(reload_preset, inputs=preset_dropdown, outputs=preset_status)
            with gr.Accordion(t("new_preset"), open=False):
                with gr.Row():
                    new_preset_id = gr.Textbox(value="trained-local", label=t("preset_id"))
                    new_preset_name = gr.Textbox(value="Trained Local", label=t("name"))
                new_preset_config = gr.Textbox(
                    value=profile_path_text(DEFAULT_CONFIG_PATH),
                    label=t("config_path"),
                )
                new_preset_gpt = gr.Textbox(label=t("gpt_ckpt"))
                new_preset_sovits = gr.Textbox(label=t("sovits_pth"))
                save_preset_btn = gr.Button(t("save_preset"))
                preset_save_status = gr.Textbox(label=t("save_result"))
                save_preset_btn.click(
                    write_model_preset,
                    inputs=[
                        new_preset_id,
                        new_preset_name,
                        new_preset_config,
                        new_preset_gpt,
                        new_preset_sovits,
                    ],
                    outputs=[preset_save_status, preset_dropdown, clone_model_preset],
                )

        with gr.Tab(t("download")):
            with gr.Row():
                download_source = gr.Dropdown(
                    choices=["modelscope", "hf", "hf-mirror"],
                    value=DEFAULT_DOWNLOAD_SOURCE,
                    label=t("download_source"),
                )
                download_force = gr.Checkbox(value=False, label=t("force_redownload"))
            with gr.Row():
                download_base_btn = gr.Button(t("download_base"))
                download_v2pro_btn = gr.Button(t("download_v2pro"))
                download_sample_btn = gr.Button(t("download_sample"))
                download_refresh_btn = gr.Button(t("refresh_log"))
                download_stop_btn = gr.Button(t("stop_download"))
            download_status_box = gr.Textbox(value=download_manager.status, label=t("download_status"), lines=18)
            download_base_btn.click(start_base_assets_download, inputs=[download_source, download_force], outputs=download_status_box)
            download_v2pro_btn.click(start_v2pro_download, inputs=[download_source, download_force], outputs=download_status_box)
            download_sample_btn.click(start_sample_voice_download, inputs=[download_source, download_force], outputs=download_status_box)
            download_refresh_btn.click(download_manager.status, outputs=download_status_box)
            download_stop_btn.click(download_manager.stop, outputs=download_status_box)

        with gr.Tab(t("logs")):
            events_box = gr.Textbox(value=runtime_events_text(), label=t("backend_log"), lines=28)
            events_refresh_btn = gr.Button(t("refresh"))
            events_refresh_btn.click(runtime_events_text, outputs=events_box)
            log_timer = gr.Timer(value=2.0, active=True)
            log_timer.tick(runtime_events_text, outputs=events_box)

    return blocks.queue(max_size=8, default_concurrency_limit=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch GPT-SoVITS for Neiroha with native and OpenAI-compatible APIs.",
    )
    parser.add_argument(
        "--mode",
        choices=["api", "admin", "api-admin", "api-preload", "api-admin-preload", "admin-ui", "webui", "combined"],
        default="api",
    )
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--device", default="config", help="config, auto, cpu, cuda, or cuda:N")
    parser.add_argument("--half", action="store_true", help="Force half precision.")
    parser.add_argument("--no-half", action="store_true", help="Force full precision.")
    parser.add_argument("--preload-model", action="store_true")
    parser.add_argument("--default-voice", default=os.environ.get("NEIROHA_GPT_SOVITS_DEFAULT_VOICE", ""))
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-host", default=None)
    parser.add_argument("--api-port", type=int, default=None)
    parser.add_argument("--auto-start-api", action="store_true")
    parser.add_argument("--gradio-path", default="/admin")
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--rtf-log",
        action="store_true",
        default=os.environ.get("NEIROHA_GPT_SOVITS_RTF_LOG", "0").lower() in {"1", "true", "yes"},
        help="Also print RTF performance lines to the FastAPI terminal. Headers and admin events are always kept.",
    )
    parser.add_argument(
        "--no-rtf-log",
        action="store_true",
        help="Compatibility flag: keep terminal RTF logs disabled.",
    )
    parser.add_argument(
        "--debug-runtime-output",
        action="store_true",
        default=os.environ.get("NEIROHA_GPT_SOVITS_DEBUG_RUNTIME_OUTPUT", "0").lower() in {"1", "true", "yes"},
        help=f"Capture raw GPT-SoVITS stdout/stderr in {RUNTIME_DEBUG_LOG_PATH} instead of suppressing it.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def mode_settings(mode: str, preload_model: bool) -> tuple[str, bool]:
    if mode == "admin-ui":
        return "admin", preload_model
    if mode == "webui":
        return "api-admin", preload_model
    if mode == "api-preload":
        return "api", True
    if mode == "api-admin-preload":
        return "api-admin", True
    return mode, preload_model


def resolve_api_port(api_base: str, explicit_port: Optional[int]) -> int:
    if explicit_port is not None:
        return explicit_port
    parsed = urllib.parse.urlparse(api_base)
    if parsed.port is not None:
        return parsed.port
    return 9880


def socket_bind_host(host: str) -> str:
    host = strip_text(host)
    if not host:
        return "127.0.0.1"
    return host


def browser_host(host: str) -> str:
    host = socket_bind_host(host)
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def http_url(host: str, port: int) -> str:
    return f"http://{browser_host(host)}:{int(port)}"


def _socket_family(host: str) -> socket.AddressFamily:
    return socket.AF_INET6 if ":" in socket_bind_host(host) else socket.AF_INET


def can_bind_port(host: str, port: int) -> tuple[bool, str]:
    try:
        with socket.socket(_socket_family(host), socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((socket_bind_host(host), int(port)))
        return True, ""
    except OSError as exc:
        return False, str(exc)


def random_bindable_port(host: str) -> int:
    with socket.socket(_socket_family(host), socket.SOCK_STREAM) as sock:
        sock.bind((socket_bind_host(host), 0))
        return int(sock.getsockname()[1])


def resolve_bind_port(host: str, requested_port: int, label: str) -> int:
    requested_port = int(requested_port)
    ok, reason = can_bind_port(host, requested_port)
    if ok:
        return requested_port
    selected_port = random_bindable_port(host)
    message = (
        f"{label} configured port {requested_port} is unavailable on {socket_bind_host(host)}; "
        f"using random port {selected_port}. Reason: {reason}"
    )
    LOGGER.warning(message)
    RUNTIME_EVENTS.append(
        "port_fallback",
        service=label,
        host=socket_bind_host(host),
        requested_port=requested_port,
        selected_port=selected_port,
        reason=reason,
    )
    return selected_port


def local_api_base(api_base: str, api_port: int) -> str:
    parsed = urllib.parse.urlparse(api_base)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"{scheme}://{host}:{api_port}"


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.mode != "admin-ui":
        RUNTIME_EVENTS.reset_for_launch()
    is_half = True if args.half else False if args.no_half else None
    terminal_rtf_log = bool(args.rtf_log and not args.no_rtf_log)

    runtime = GPTSoVITSRuntime(
        repo_dir=args.repo_dir,
        config_path=args.config,
        device=args.device,
        is_half=is_half,
        debug_runtime_output=args.debug_runtime_output,
    )
    registry = VoiceRegistry(args.profiles, repo_dir=args.repo_dir.resolve())
    server_config = registry.server_config()
    api_config = server_config.get("api", {}) if isinstance(server_config.get("api"), dict) else {}
    admin_config = server_config.get("admin", {}) if isinstance(server_config.get("admin"), dict) else {}
    effective_mode, preload_model = mode_settings(args.mode, args.preload_model)

    api_host = (
        args.api_host
        or (args.host if effective_mode in {"api", "combined"} else None)
        or api_config.get("host")
        or "127.0.0.1"
    )
    requested_api_port = (
        args.api_port
        or (args.port if effective_mode in {"api", "combined"} else None)
        or int(api_config.get("port") or 9880)
    )
    admin_host = (
        (args.host if effective_mode in {"admin", "api-admin"} else None)
        or admin_config.get("host")
        or "127.0.0.1"
    )
    requested_admin_port = (
        (args.port if effective_mode in {"admin", "api-admin"} else None)
        or int(admin_config.get("port") or 17860)
    )
    starts_api = effective_mode in {"api", "api-admin", "combined"}
    starts_standalone_admin = effective_mode in {"admin", "api-admin"}
    api_port = (
        resolve_bind_port(api_host, int(requested_api_port), "FastAPI")
        if starts_api
        else int(requested_api_port)
    )
    admin_port = (
        resolve_bind_port(admin_host, int(requested_admin_port), "Gradio Admin")
        if starts_standalone_admin
        else int(requested_admin_port)
    )
    api_base = args.api_base or http_url(api_host, api_port)
    admin_url = http_url(admin_host, admin_port) if starts_standalone_admin else ""
    default_voice = strip_text(args.default_voice) or registry.default_voice_id()
    if args.config == DEFAULT_CONFIG_PATH:
        runtime.config_path = Path(registry.get_model_preset(registry.active_model_preset_id()).config_path).resolve()

    if preload_model and effective_mode != "admin":
        runtime.load()
        default_profile = registry.get(default_voice)
        if default_profile is not None:
            runtime.apply_profile_weights(default_profile)

    RUNTIME_EVENTS.append(
        "launcher_start",
        mode=effective_mode,
        requested_mode=args.mode,
        api_host=api_host,
        api_port=api_port,
        api_url=http_url(api_host, api_port),
        admin_host=admin_host,
        admin_port=admin_port,
        admin_url=admin_url,
        repo_dir=str(args.repo_dir),
        config_path=str(runtime.config_path),
        debug_runtime_output=bool(args.debug_runtime_output),
        terminal_rtf_log=terminal_rtf_log,
    )

    if effective_mode == "admin":
        LOGGER.info("Gradio Admin URL: %s", admin_url)
        LOGGER.info("Admin is connecting to FastAPI API Base: %s", api_base)
        blocks = build_gradio_admin_blocks(api_base, registry, process_manager=None)
        blocks.launch(server_name=admin_host, server_port=admin_port, show_error=True)
        return

    app = create_api_app(
        runtime,
        registry,
        default_voice_id=default_voice,
        rtf_log=terminal_rtf_log,
        admin_url=admin_url,
    )

    if effective_mode == "api-admin":
        ui_api_base = local_api_base(api_base, api_port)
        admin_process = ManagedGradioProcess(
            host=admin_host,
            port=admin_port,
            api_base=ui_api_base,
            repo_dir=args.repo_dir,
            config_path=runtime.config_path,
            profiles_path=args.profiles,
            log_level=args.log_level,
            debug_runtime_output=args.debug_runtime_output,
        )
        RUNTIME_EVENTS.append(
            "admin_child_start",
            api_host=api_host,
            api_port=api_port,
            api_url=http_url(api_host, api_port),
            admin_host=admin_host,
            admin_port=admin_port,
            admin_url=admin_url,
        )
        admin_message = admin_process.start()
        LOGGER.info(admin_message)
        LOGGER.info("FastAPI URL: %s", http_url(api_host, api_port))
        LOGGER.info("Gradio Admin URL: %s", admin_url)
        try:
            uvicorn.run(app, host=api_host, port=api_port, log_level=args.log_level)
        finally:
            RUNTIME_EVENTS.append("admin_child_stop", message=admin_process.stop())
        return

    if effective_mode == "combined":
        import gradio as gr

        mount_path = args.gradio_path if args.gradio_path.startswith("/") else f"/{args.gradio_path}"
        blocks = build_gradio_blocks(runtime, registry)
        app = gr.mount_gradio_app(app, blocks, path=mount_path, show_error=True)

    LOGGER.info("FastAPI URL: %s", http_url(api_host, api_port))
    if admin_url:
        LOGGER.info("Gradio Admin URL: %s", admin_url)
    uvicorn.run(app, host=api_host, port=api_port, log_level=args.log_level)


if __name__ == "__main__":
    main()
