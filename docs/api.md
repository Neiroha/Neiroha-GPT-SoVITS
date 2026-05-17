# API Reference

FastAPI is the real runtime API and defaults to `http://127.0.0.1:9880`.
The Gradio admin panel is a separate management UI on `http://127.0.0.1:7860`.
The normal `admin` task starts FastAPI as the primary process and launches the
admin UI as a child process.

## Health

```http
GET /health
```

Returns runtime status, loaded flag, active weights and language/cut-method metadata when available.

## OpenAI-compatible TTS

```http
GET  /v1/models
GET  /v1/audio/voices
POST /v1/audio/speech
```

`/v1/audio/speech` stays as the OpenAI-compatible speech surface. It uses saved
trained voice profiles only; clone mode is exposed through native
`/gpt-sovits/*` routes so Neiroha can model it as a separate capability.

`GET /v1/audio/voices` returns OpenAI-style voice objects and includes local
extension fields (`model_id`, `model_name`, `model_type`, `text_lang`,
`prompt_lang`) so Neiroha can distinguish normal trained voices from shared
trained weights.

`POST /v1/audio/speech` accepts standard fields:

- `model`: `gpt-sovits`, `tts-1`, or `tts-1-hd`
- `input`: text to synthesize
- `voice`: profile id from `profiles/voices.json`
- `response_format`: `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`, `ogg`, or `raw`
- `speed`: mapped to GPT-SoVITS `speed_factor`

Local GPT-SoVITS extensions:

- `ref_audio_path` / `reference_audio` / `reference_audio_path`
- `prompt_text`
- `text_lang`
- `prompt_lang`
- `aux_ref_audio_paths`
- `gpt_weights_path`
- `sovits_weights_path`
- `text_split_method`
- `top_k`, `top_p`, `temperature`, `seed`

For the bundled Hugging Face Genshin demo downloader, the default voice ids are
`genshin-paimon`, `genshin-keqing`, and `genshin-klee`. The generated
`profiles/voices.json` points each voice at its own GPT checkpoint, SoVITS
checkpoint, reference wav, and prompt text.

Every non-streaming synthesis writes a copy under `runtime/outputs/` using
`speaker_YYYYMMDDHHMMSS.ext` naming and returns that path in the
`X-Neiroha-Output-Path` response header.

Non-streaming synthesis also returns performance headers:

- `X-Neiroha-Audio-Seconds`
- `X-Neiroha-Elapsed-Seconds`
- `X-Neiroha-RTF`

## Native GPT-SoVITS

```http
GET  /tts
POST /tts
GET  /set_refer_audio
GET  /set_gpt_weights
GET  /set_sovits_weights
GET  /control
GET  /gpt-sovits/models
GET  /gpt-sovits/voices
GET  /gpt-sovits/capabilities
GET  /gpt-sovits/events
POST /gpt-sovits/clone
POST /gpt-sovits/clone/upload
```

`POST /tts` follows the official `api_v2.py` shape and returns audio bytes.

`/gpt-sovits/models` separates trained voice models from clone models. Trained
models list their real configured voices from `profiles/voices.json`; clone
models do not invent voices and require reference audio plus matching
`prompt_text`.

Shared multi-speaker checkpoints, such as `AI-Hobbyist/GPT-SoVits-V2-models`,
are treated as trained weights plus per-speaker reference audio. GPT-SoVITS
upstream still conditions on `ref_audio_path` and `prompt_text`; this launcher
does not synthesize fake speaker names from the checkpoint alone. The shared
reference downloader can generate profiles such as `shared-genshin-en-furina`
and `shared-genshin-ja-keqing`.

The Gradio admin page also includes an `OpenAI Voice 配置` profile builder. It
can register any combination of:

- GPT weights path
- SoVITS weights path
- reference audio upload or path
- prompt text matching the reference audio
- prompt/text language

The builder writes `profiles/voices.json`; because the FastAPI registry reads
that file on each request, newly saved voices become visible through
`GET /v1/audio/voices` without restarting the API.

v2ProPlus clone mode uses these default local paths:

```text
GPT    models/pretrained/GPT-SoVITS/GPT_SoVITS/pretrained_models/s1v3.ckpt
SoVITS models/pretrained/GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth
SV     models/pretrained/GPT-SoVITS/GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt
```

GPT-SoVITS v2ProPlus upstream requires clone reference audio to be in the
3-10 second range. This launcher does not patch the upstream submodule. Instead,
for `POST /gpt-sovits/clone` and `POST /gpt-sovits/clone/upload` only, it
normalizes the reference audio into a temporary file before inference:

- shorter than `3.05s`: pad trailing silence
- longer than `9.95s`: trim from the start
- already in range: use the original path/upload

The original upload or source file is not modified, and temporary files are
removed after synthesis. For long reference audio, write `prompt_text` for the
first `9.95s`, because that is the effective prompt audio after trimming.

`GET /gpt-sovits/capabilities` exposes this behavior through
`clone_reference_audio.auto_normalize_without_upstream_patch`.

RTF metrics are always returned as response headers and written to the light
runtime log:

```text
runtime/logs/backend.log
```

Read recent summaries, newest first, with:

```http
GET /gpt-sovits/logs?limit=80
```

The Gradio admin page has a `日志` tab that refreshes the same log automatically.
FastAPI terminal RTF lines are off by default; enable them with `--rtf-log` or
`NEIROHA_GPT_SOVITS_RTF_LOG=1`. Raw GPT-SoVITS stdout/stderr is suppressed
during model load and inference. Enable `--debug-runtime-output` to capture that
raw output into `runtime/logs/api-debug.log`.

## Management

```http
GET  /gpt-sovits/meta
POST /gpt-sovits/load
POST /gpt-sovits/unload
POST /gpt-sovits/reload
POST /gpt-sovits/set_gpt_weights
POST /gpt-sovits/set_sovits_weights
POST /gpt-sovits/speech/upload
```

`/gpt-sovits/load` body:

```json
{
  "config_path": "GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml",
  "gpt_weights_path": "",
  "sovits_weights_path": ""
}
```

## Admin Model Downloads

The Gradio admin page includes a `模型下载` tab. It starts downloads in a child
process and writes logs to:

- `runtime/logs/admin-download.out.log`
- `runtime/logs/admin-download.err.log`

Available admin actions:

- download common pretrained base assets
- download v2ProPlus clone base weights
- download one default sample reference voice
- register local trained GPT/SoVITS weights as model presets or per-voice
  overrides

Multi-role trained packages are no longer bundled as first-class download
actions. Download those files yourself, then register the `.ckpt` and `.pth`
paths in the Admin.

The CLI fallback is:

```powershell
pixi run install-assets
pixi run python scripts/download_gpt_sovits_assets.py --source hf --skip-base-assets --v2pro-plus
pixi run python scripts/download_gpt_sovits_assets.py --source hf --skip-base-assets --sample-reference --activate-voices
```
