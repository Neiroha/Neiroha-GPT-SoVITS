# API Reference

FastAPI is the real runtime API and defaults to `http://127.0.0.1:9880`.
The Gradio admin panel is a separate client on `http://127.0.0.1:7860`.

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

## Native GPT-SoVITS

```http
GET  /tts
POST /tts
GET  /set_refer_audio
GET  /set_gpt_weights
GET  /set_sovits_weights
GET  /control
```

`POST /tts` follows the official `api_v2.py` shape and returns audio bytes.

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
