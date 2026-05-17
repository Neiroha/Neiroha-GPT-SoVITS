# Neiroha GPT-SoVITS Local Launcher

Standalone GPT-SoVITS backend for Neiroha. This repository owns its own Pixi
environment, FastAPI server, Gradio Admin, launcher scripts, and configuration
semantics. It does not share a Python environment with other TTS engines.

Chinese documentation: [README_zh.md](README_zh.md)

The current shape is intentionally small:

- OpenAI-style TTS API: `/v1/models`, `/v1/audio/voices`, `/v1/audio/speech`
- `model` means a voice set, not a low-level weight model
- `voice` means a concrete voice profile inside the selected voice set
- low-level GPT-SoVITS runtime weights live in model presets
- only one default voice is exposed: `genshin-keqing`
- multi-role packs and shared community weights are not bundled as bulk
  download actions; download them yourself and register the weights in Admin

## Layout

```text
configs/
  server.toml
  model-presets/
    default.toml
  voice-sets/
    default.toml
runtime/
  voices/
    genshin-keqing/
      voice.toml
      reference.wav
  logs/
  outputs/
models/
  pretrained/
```

Neiroha-owned configuration is TOML. The only YAML file is the generated
GPT-SoVITS runtime cache at `runtime/cache/tts_infer.yaml`, because upstream
GPT-SoVITS expects that shape.

`configs/voice-sets/default.toml` maps to `model=default` in the Neiroha /
OpenAI-compatible API. `runtime/voices/genshin-keqing/voice.toml` maps to
`voice=genshin-keqing`. `configs/model-presets/default.toml` is the low-level
GPT-SoVITS runtime preset.

## Install

```powershell
pixi install
pixi run submodule-init
pixi run install-deps
pixi run install-assets
pixi run install-sample-voice
```

`install-assets` downloads the official pretrained assets. `install-sample-voice`
downloads only one default sample reference voice; it does not download trained
multi-character packs.

## Run

```powershell
start_api_admin.bat
```

Or use Pixi tasks:

```powershell
pixi run api
pixi run api-preload
pixi run admin
pixi run api-admin
pixi run api-admin-preload
```

Default ports come from `configs/server.toml`:

```text
FastAPI  http://127.0.0.1:19880
Admin    http://127.0.0.1:17860
```

Pixi tasks no longer hard-code these ports by default. If a configured port is
busy or Windows refuses the bind, the launcher picks an available random port
and prints the actual FastAPI and Gradio Admin URLs in the terminal and
`runtime/logs/backend.log`. `api-12080` remains an explicit debug override.

`admin` starts only the Gradio Admin and connects to an existing FastAPI server.
`api-admin` starts FastAPI and then launches Gradio Admin as a separate child
process. Gradio is not mounted into FastAPI.

## Admin Language

Gradio does not automatically translate custom labels, but this Admin supports
Chinese and English labels at launch time. Set:

```toml
# configs/server.toml
[ui]
default_language = "en" # zh | en
```

Or set an environment variable:

```powershell
$env:NEIROHA_GPT_SOVITS_UI_LANG="en"
```

Then restart Admin.

## Logs

The Admin Logs tab shows `runtime/logs/backend.log`, newest entries first, with
automatic refresh. Each top-level launcher start rotates the previous file to
`runtime/logs/backend.previous.log` and starts a fresh `backend.log`, so normal
use does not accumulate one endless log file. Download tasks still write fresh
per-run files:

```text
runtime/logs/admin-download.out.log
runtime/logs/admin-download.err.log
```

## API

List voice sets:

```powershell
curl.exe http://127.0.0.1:19880/v1/models
```

List voices:

```powershell
curl.exe http://127.0.0.1:19880/v1/audio/voices
```

Synthesize speech:

```powershell
curl.exe http://127.0.0.1:19880/v1/audio/speech `
  -H "Content-Type: application/json" `
  -d '{ "model":"default", "voice":"genshin-keqing", "input":"Hello, this is a voice cloning test.", "response_format":"wav" }' `
  --output speech.wav
```

Responses include `X-Neiroha-Output-Path`, `X-Neiroha-Audio-Seconds`,
`X-Neiroha-Elapsed-Seconds`, and `X-Neiroha-RTF` headers.

## Add Voices

Use the Admin Voice Config tab to upload reference audio, enter the matching
prompt text, set a voice id/name, and save it.

By default the new voice uses the selected model preset. For community-trained
GPT-SoVITS v2Pro/v2ProPlus weights, either add a reusable preset in the
Model Presets tab, or expand the per-voice override section and fill the `.ckpt`
and `.pth` paths directly.

Manual voice profile path:

```text
runtime/voices/my_voice/voice.toml
```

Then add `my_voice` to:

```text
configs/voice-sets/default.toml
```

Example:

```toml
schema_version = 1
id = "my_voice"
name = "My Voice"
mode = "prompt_clone"
model_preset = "v2proplus-clone"
reference_audio = "runtime/voices/my_voice/reference.wav"
prompt_audio = ""
prompt_text = "Transcript matching the reference audio."
text_lang = "zh"
prompt_lang = "zh"
instruction = ""
speed = 1.0
gpt_weights_path = "models/voices/my-trained/GPT_xxx.ckpt"
sovits_weights_path = "models/voices/my-trained/SV_xxx.pth"

[engine_options]
```

`gpt_weights_path` and `sovits_weights_path` are optional per-voice overrides.
If omitted, the voice uses the weights from `model_preset`.
