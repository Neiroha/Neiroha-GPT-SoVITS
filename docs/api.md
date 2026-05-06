# API Reference

FastAPI is the real runtime API and defaults to `http://127.0.0.1:9880`.
The Gradio admin panel is a separate management UI on `http://127.0.0.1:7860`.
It can connect to an existing FastAPI process or start/stop a managed one from
the browser.

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
POST /gpt-sovits/clone
POST /gpt-sovits/clone/upload
```

`POST /tts` follows the official `api_v2.py` shape and returns audio bytes.

`/gpt-sovits/models` separates trained voice models from clone models. Trained
models list their real configured voices from `profiles/voices.json`; clone
models do not invent voices and require reference audio plus matching
`prompt_text`.

RTF logging is enabled by default for non-streaming synthesis:

```text
TTS performance mode=trained speaker=genshin-paimon audio=2.660s elapsed=22.844s rtf=8.588
```

Disable it with `--no-rtf-log` or `NEIROHA_GPT_SOVITS_RTF_LOG=0`.

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
