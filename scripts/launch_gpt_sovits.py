from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import io
import json
import logging
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
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
CONFIG_TEMPLATE_PATH = WORKSPACE_ROOT / "configs" / "tts_infer.yaml"
DEFAULT_PROFILE_PATH = WORKSPACE_ROOT / "profiles" / "voices.json"
MODELS_ROOT = WORKSPACE_ROOT / "models"
PRETRAINED_MODELS_DIR = MODELS_ROOT / "pretrained" / "GPT-SoVITS" / "GPT_SoVITS" / "pretrained_models"
RUNTIME_ROOT = WORKSPACE_ROOT / "runtime"
RUNTIME_CACHE_ROOT = RUNTIME_ROOT / "cache"
TEMP_ROOT = RUNTIME_ROOT / "temp"
UPLOAD_ROOT = TEMP_ROOT / "uploads"
OUTPUT_ROOT = RUNTIME_ROOT / "outputs"
DEFAULT_CONFIG_PATH = RUNTIME_CACHE_ROOT / "tts_infer.yaml"

for path in (RUNTIME_ROOT, RUNTIME_CACHE_ROOT, TEMP_ROOT, UPLOAD_ROOT, OUTPUT_ROOT):
    path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TMPDIR", str(TEMP_ROOT))
os.environ.setdefault("TEMP", str(TEMP_ROOT))
os.environ.setdefault("TMP", str(TEMP_ROOT))
os.environ.setdefault("GRADIO_TEMP_DIR", str(TEMP_ROOT / "gradio"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LOGGER = logging.getLogger("neiroha.gpt_sovits")
OPENAI_MODEL_ALIAS = "gpt-sovits"
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


def strip_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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
    if workspace_candidate.exists():
        return str(workspace_candidate.resolve())
    repo_candidate = repo_dir / candidate
    return str(repo_candidate.resolve())


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


def write_runtime_output(content: bytes, speaker: Any, response_format: str) -> Path:
    fmt = require_supported_format(response_format)
    suffix = "raw" if fmt in {"pcm", "raw"} else fmt
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    speaker_name = safe_filename_part(speaker)
    path = OUTPUT_ROOT / f"{speaker_name}_{timestamp}.{suffix}"
    counter = 1
    while path.exists():
        path = OUTPUT_ROOT / f"{speaker_name}_{timestamp}_{counter}.{suffix}"
        counter += 1
    path.write_bytes(content)
    return path


def ensure_default_config(config_path: Path) -> Path:
    config_path = config_path.resolve()
    if config_path.exists():
        return config_path

    config_path.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_TEMPLATE_PATH.exists():
        shutil.copyfile(CONFIG_TEMPLATE_PATH, config_path)
    else:
        fallback = DEFAULT_REPO_DIR / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
        if fallback.exists():
            shutil.copyfile(fallback, config_path)
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

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], *, repo_dir: Path) -> "VoiceProfile":
        profile_id = first_non_empty(payload.get("id"), payload.get("name"))
        if not profile_id:
            raise ValueError("Voice profile requires an id or name.")
        ref_audio_path = resolve_optional_path(payload.get("ref_audio_path"), repo_dir=repo_dir) or ""
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
        )

    def to_openai_voice(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "object": "voice",
            "description": self.description,
        }

    def to_speaker(self) -> dict[str, str]:
        return {"name": self.name, "voice_id": self.id}


class VoiceRegistry:
    def __init__(self, profile_path: Path, *, repo_dir: Path) -> None:
        self.profile_path = profile_path
        self.repo_dir = repo_dir

    def _read_payload(self) -> list[dict[str, Any]]:
        if not self.profile_path.exists():
            return []
        data = json.loads(self.profile_path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("voices", [])
        if not isinstance(data, list):
            raise ValueError(f"Voice profile file must contain a list or voices object: {self.profile_path}")
        return data

    def list_profiles(self) -> list[VoiceProfile]:
        profiles = []
        for payload in self._read_payload():
            if not isinstance(payload, dict):
                continue
            profiles.append(VoiceProfile.from_mapping(payload, repo_dir=self.repo_dir))
        return profiles

    def get(self, voice_id: str) -> Optional[VoiceProfile]:
        voice_id = strip_text(voice_id)
        if not voice_id:
            return None
        for profile in self.list_profiles():
            if voice_id in {profile.id, profile.name}:
                return profile
        return None

    def first(self) -> Optional[VoiceProfile]:
        profiles = self.list_profiles()
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
    ) -> None:
        self.repo_dir = repo_dir.resolve()
        self.config_path = config_path.resolve()
        self.device_override = device
        self.is_half_override = is_half
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
            LOGGER.info("Loading GPT-SoVITS with config=%s", self.tts_config.configs_path)
            self.tts_pipeline = TTS(self.tts_config)
            self.cut_method_names = get_method_names()
            self.current_gpt_weights_path = str(self.tts_config.t2s_weights_path)
            self.current_sovits_weights_path = str(self.tts_config.vits_weights_path)
            return self.status()

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
            pipeline = self.get_or_load()
            pipeline.init_t2s_weights(resolved)
            self.current_gpt_weights_path = resolved
            return self.status()

    def set_sovits_weights(self, weights_path: str) -> dict[str, Any]:
        resolved = resolve_existing_file(
            weights_path,
            repo_dir=self.repo_dir,
            field_name="sovits_weights_path",
            required=True,
        )
        with self.lock:
            pipeline = self.get_or_load()
            pipeline.init_vits_weights(resolved)
            self.current_sovits_weights_path = resolved
            return self.status()

    def set_refer_audio(self, refer_audio_path: str) -> dict[str, Any]:
        resolved = resolve_existing_file(
            refer_audio_path,
            repo_dir=self.repo_dir,
            field_name="refer_audio_path",
            required=True,
        )
        with self.lock:
            pipeline = self.get_or_load()
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
            generator = pipeline.run(request)
            return next(generator)

    def synthesize_stream(self, request: dict[str, Any]) -> Generator[tuple[int, np.ndarray], None, None]:
        with self.lock:
            pipeline = self.get_or_load()
            request = self.validate_request(request)
            request, _ = self.normalize_streaming_mode(request)
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
    voice_id = extract_voice_id(data.get("voice"))
    if voice_id in {"", "default"} and strip_text(default_voice_id):
        voice_id = strip_text(default_voice_id)
    profile = registry.get(voice_id)
    if profile is None and voice_id in {"", "default"}:
        profile = registry.first()

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


def audio_response(
    sample_rate: int,
    audio_data: np.ndarray,
    response_format: str,
    *,
    speaker: Any = "",
) -> Response:
    fmt = require_supported_format(response_format)
    content = pack_audio(audio_data, sample_rate, fmt)
    output_path = write_runtime_output(content, speaker or "speech", fmt)
    return Response(
        content=content,
        media_type=CONTENT_TYPES[fmt],
        headers={
            "Content-Disposition": f'inline; filename="{output_path.name}"',
            "X-Neiroha-Output-Path": str(output_path),
        },
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


def save_upload(upload: Optional[UploadFile], *, prefix: str) -> Optional[str]:
    if upload is None or not upload.filename:
        return None
    suffix = Path(upload.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_ROOT, prefix=f"{prefix}_", suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        return tmp.name


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
            "native_speech": "/tts",
            "admin": "/admin",
        }

    @app.get("/health")
    @app.get("/api/health", include_in_schema=False)
    def health():
        return {
            "status": "ok",
            "default_voice": strip_text(default_voice_id) or "default",
            **runtime.status(),
        }

    @app.get("/v1/models")
    def list_models():
        status = runtime.status()
        return {
            "object": "list",
            "data": [
                {
                    "id": OPENAI_MODEL_ALIAS,
                    "object": "model",
                    "owned_by": "local",
                    "root_model": "RVC-Boss/GPT-SoVITS",
                    "loaded": status["loaded"],
                    "version": status["version"],
                }
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
                    "name": "default",
                    "object": "voice",
                    "description": "Configure profiles/voices.json or pass ref_audio_path in the request.",
                }
            ]
        return {"object": "list", "data": voices}

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
            "default_voice": strip_text(default_voice_id) or "default",
            "voices": [profile.to_openai_voice() for profile in registry.list_profiles()],
            "routes": {
                "native_tts": "/tts",
                "openai_speech": "/v1/audio/speech",
                "models": "/v1/models",
                "voices": "/v1/audio/voices",
                "load": "/gpt-sovits/load",
                "unload": "/gpt-sovits/unload",
                "set_gpt_weights": "/set_gpt_weights",
                "set_sovits_weights": "/set_sovits_weights",
            },
        }

    @app.post("/v1/audio/speech")
    def openai_audio_speech(payload: OpenAISpeechRequest):
        if payload.model and payload.model not in {OPENAI_MODEL_ALIAS, "tts-1", "tts-1-hd"}:
            return openai_error(
                f"This launcher serves '{OPENAI_MODEL_ALIAS}'. Received model='{payload.model}'.",
                status_code=400,
            )
        if payload.stream_format not in {"", "audio"}:
            return openai_error("Only stream_format='audio' is supported by this local launcher.")

        try:
            request, profile = request_with_profile(
                payload,
                runtime=runtime,
                registry=registry,
                default_voice_id=default_voice_id,
            )
            runtime.apply_profile_weights(profile)
            sample_rate, audio_data = runtime.synthesize_once(request)
            speaker = profile.id if profile else extract_voice_id(payload.voice) or default_voice_id or "default"
            return audio_response(sample_rate, audio_data, request["media_type"], speaker=speaker)
        except FileNotFoundError as exc:
            return openai_error(str(exc), status_code=404)
        except (ValueError, RuntimeError) as exc:
            return openai_error(str(exc), status_code=400)
        except Exception as exc:
            LOGGER.exception("OpenAI-compatible synthesis failed.")
            return openai_error(str(exc), status_code=500, error_type="server_error")

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
            sample_rate, audio_data = runtime.synthesize_once(request)
            return audio_response(sample_rate, audio_data, request["media_type"], speaker="native")
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
            sample_rate, audio_data = runtime.synthesize_once(request)
            return audio_response(sample_rate, audio_data, response_format, speaker="upload")
        finally:
            cleanup_temp_files([temp_ref])

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
        return command

    def start(self, *, default_voice_id: str = "", preload_model: bool = False) -> str:
        if self.is_running():
            return f"Managed API is already running with PID {self.process.pid}."
        if self.external_health_ok():
            return f"FastAPI is already reachable on port {self.api_port}; using the external process."
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


def build_gradio_admin_blocks(
    api_base: str,
    registry: VoiceRegistry,
    *,
    process_manager: Optional[ManagedApiProcess] = None,
):
    import gradio as gr
    import requests

    base = api_base.rstrip("/")

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
        if ref_audio_file:
            with open(ref_audio_file, "rb") as audio_file:
                files = {"ref_audio": (Path(ref_audio_file).name, audio_file, "audio/wav")}
                data = {
                    "text": text,
                    "text_lang": text_lang or "zh",
                    "prompt_lang": prompt_lang or text_lang or "zh",
                    "prompt_text": prompt_text or "",
                    "response_format": "wav",
                    "speed": str(speed),
                }
                response = requests.post(api_url("/gpt-sovits/speech/upload"), data=data, files=files, timeout=300)
        else:
            payload = {
                "model": OPENAI_MODEL_ALIAS,
                "voice": voice or "default",
                "input": text,
                "response_format": "wav",
                "speed": speed,
                "text_lang": text_lang or None,
                "prompt_lang": prompt_lang or None,
                "prompt_text": prompt_text or "",
                "ref_audio_path": ref_audio_path or None,
            }
            response = requests.post(api_url("/v1/audio/speech"), json=payload, timeout=300)

        response.raise_for_status()
        output_path = write_runtime_output(response.content, voice or "preview", "wav")
        return str(output_path)

    with gr.Blocks(title="Neiroha GPT-SoVITS Admin") as blocks:
        gr.Markdown("# Neiroha GPT-SoVITS Admin")
        gr.Markdown(f"FastAPI endpoint: `{base}`")
        with gr.Tab("API Status"):
            status_box = gr.Code(value=status_text, language="json", label="FastAPI runtime status")
            with gr.Row():
                refresh_btn = gr.Button("Refresh")
                unload_btn = gr.Button("Unload model")
            refresh_btn.click(status_text, outputs=status_box)
            unload_btn.click(unload_model, outputs=status_box)
        with gr.Tab("API Process"):
            process_box = gr.Textbox(value=managed_status_text, label="Managed FastAPI process")
            process_voice = gr.Dropdown(choices=voice_choices(), value=voice_choices()[0], label="Default voice")
            preload_checkbox = gr.Checkbox(value=False, label="Preload selected voice on API start")
            with gr.Row():
                start_api_btn = gr.Button("Start FastAPI")
                stop_api_btn = gr.Button("Stop FastAPI")
                process_refresh_btn = gr.Button("Refresh")
            start_api_btn.click(
                start_api,
                inputs=[process_voice, preload_checkbox],
                outputs=[process_box, status_box],
            )
            stop_api_btn.click(stop_api, outputs=[process_box, status_box])
            process_refresh_btn.click(managed_status_text, outputs=process_box)
        with gr.Tab("Load / Weights"):
            config_input = gr.Textbox(
                value=str(DEFAULT_CONFIG_PATH),
                label="TTS config path",
            )
            gpt_input = gr.Textbox(
                value=str(PRETRAINED_MODELS_DIR / "s1v3.ckpt"),
                label="GPT weights path",
            )
            sovits_input = gr.Textbox(
                value=str(PRETRAINED_MODELS_DIR / "v2Pro" / "s2Gv2ProPlus.pth"),
                label="SoVITS weights path",
            )
            with gr.Row():
                load_btn = gr.Button("Load / Apply")
                gpt_btn = gr.Button("Set GPT only")
                sovits_btn = gr.Button("Set SoVITS only")
            load_btn.click(load_model, inputs=[config_input, gpt_input, sovits_input], outputs=status_box)
            gpt_btn.click(set_gpt_weights, inputs=gpt_input, outputs=status_box)
            sovits_btn.click(set_sovits_weights, inputs=sovits_input, outputs=status_box)
        with gr.Tab("TTS Test"):
            voice_dropdown = gr.Dropdown(choices=voice_choices(), value=voice_choices()[0], label="Voice profile")
            text_input = gr.Textbox(label="Text", lines=4)
            with gr.Row():
                text_lang = gr.Textbox(value="zh", label="Text language")
                prompt_lang = gr.Textbox(value="zh", label="Prompt language")
                speed = gr.Slider(0.25, 4.0, value=1.0, step=0.05, label="Speed")
            ref_audio_file = gr.Audio(type="filepath", label="Reference audio upload")
            ref_audio_path = gr.Textbox(label="Reference audio path")
            prompt_text = gr.Textbox(label="Prompt text", lines=2)
            synth_btn = gr.Button("Synthesize via FastAPI")
            audio_output = gr.Audio(label="Output", type="filepath")
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
        with gr.Tab("Voices"):
            profiles_box = gr.Code(value=profiles_text, language="json", label="/v1/audio/voices")
            gr.Button("Refresh voices").click(profiles_text, outputs=profiles_box)
    return blocks.queue(max_size=8, default_concurrency_limit=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch GPT-SoVITS for Neiroha with native and OpenAI-compatible APIs.",
    )
    parser.add_argument("--mode", choices=["api", "admin", "webui", "combined"], default="api")
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--device", default="config", help="config, auto, cpu, cuda, or cuda:N")
    parser.add_argument("--half", action="store_true", help="Force half precision.")
    parser.add_argument("--no-half", action="store_true", help="Force full precision.")
    parser.add_argument("--preload-model", action="store_true")
    parser.add_argument("--default-voice", default=os.environ.get("NEIROHA_GPT_SOVITS_DEFAULT_VOICE", ""))
    parser.add_argument("--api-base", default="http://127.0.0.1:9880")
    parser.add_argument("--api-host", default="0.0.0.0")
    parser.add_argument("--api-port", type=int, default=None)
    parser.add_argument("--auto-start-api", action="store_true")
    parser.add_argument("--gradio-path", default="/admin")
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def resolve_port(mode: str, port: Optional[int]) -> int:
    if port is not None:
        return port
    return 7860 if mode in {"admin", "webui"} else 9880


def resolve_api_port(api_base: str, explicit_port: Optional[int]) -> int:
    if explicit_port is not None:
        return explicit_port
    parsed = urllib.parse.urlparse(api_base)
    if parsed.port is not None:
        return parsed.port
    return 9880


def main() -> None:
    configure_logging()
    args = parse_args()
    port = resolve_port(args.mode, args.port)
    is_half = True if args.half else False if args.no_half else None

    runtime = GPTSoVITSRuntime(
        repo_dir=args.repo_dir,
        config_path=args.config,
        device=args.device,
        is_half=is_half,
    )
    registry = VoiceRegistry(args.profiles, repo_dir=args.repo_dir.resolve())

    if args.preload_model and args.mode not in {"admin", "webui"}:
        runtime.load()
        default_profile = registry.get(args.default_voice)
        if default_profile is not None:
            runtime.apply_profile_weights(default_profile)

    LOGGER.info(
        "Starting Neiroha GPT-SoVITS mode=%s host=%s port=%s repo=%s config=%s",
        args.mode,
        args.host,
        port,
        args.repo_dir,
        args.config,
    )

    if args.mode in {"admin", "webui"}:
        managed_api_port = resolve_api_port(args.api_base, args.api_port)
        process_manager = ManagedApiProcess(
            api_host=args.api_host,
            api_port=managed_api_port,
            repo_dir=args.repo_dir,
            config_path=args.config,
            profiles_path=args.profiles,
            device=args.device,
            is_half=is_half,
            default_voice_id=args.default_voice,
            log_level=args.log_level,
        )
        if args.auto_start_api or args.mode == "webui":
            LOGGER.info("Auto-starting managed FastAPI on %s:%s", args.api_host, managed_api_port)
            LOGGER.info(process_manager.start(default_voice_id=args.default_voice, preload_model=args.preload_model))
        blocks = build_gradio_admin_blocks(args.api_base, registry, process_manager=process_manager)
        blocks.launch(server_name=args.host, server_port=port, show_error=True)
        return

    app = create_api_app(runtime, registry, default_voice_id=args.default_voice)
    if args.mode == "combined":
        import gradio as gr

        mount_path = args.gradio_path if args.gradio_path.startswith("/") else f"/{args.gradio_path}"
        blocks = build_gradio_blocks(runtime, registry)
        app = gr.mount_gradio_app(app, blocks, path=mount_path, show_error=True)

    uvicorn.run(app, host=args.host, port=port, log_level=args.log_level)


if __name__ == "__main__":
    main()
