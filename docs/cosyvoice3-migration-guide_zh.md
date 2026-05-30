# Neiroha-CosyVoice3 改造范式指导

这份文档给后续 AI / 维护者迁移 CosyVoice3 项目时使用。目标是把 GPT-SoVITS 这轮改造里已经踩过的坑直接迁移过去，不重复犯错。

## 总目标

CosyVoice3 项目保持独立：

- 独立 `.pixi`
- 独立模型目录
- 独立 FastAPI
- 独立 Gradio Admin
- 独立启动脚本

不要和 GPT-SoVITS / VoxCPM 合并 Python 环境。只复用配置语义、OpenAI API 语义、Admin 页面形状和验收标准。

## 核心语义

Flutter / OpenAI API 的 `model` 仍然表示 voice set，不表示底层 CosyVoice3 权重。

映射必须是：

```text
POST /v1/audio/speech
  model=default
    -> configs/voice-sets/default.toml
  voice=local-voice
    -> runtime/voices/local-voice/voice.toml
  model_preset=cosyvoice3-default
    -> configs/model-presets/default.toml
  mode=prompt_clone / zero_shot / sft / instruct
    -> 转换成 CosyVoice3 引擎真实参数
```

API 形状：

- `GET /v1/models` 返回 voice set。
- `GET /v1/audio/voices` 返回 voice profile。
- `POST /v1/audio/speech` 用 `model` 限定 voice set，用 `voice` 选择具体声音。
- 底层 CosyVoice3 模型只放在 model preset，不直接暴露成 OpenAI model。

## 目录与配置

只让用户面对 TOML。JSON 只用于 HTTP 请求 / 响应示例，不再作为默认本地配置。YAML 如果 CosyVoice3 上游必须使用，只能放在 `runtime/cache/` 作为生成缓存，不放在 `configs/` 里混用。

推荐目录：

```text
Neiroha-CosyVoice3/
  start_portable.bat
  pixi.toml
  scripts/
    launch_cosyvoice3.py
    download_cosyvoice3_assets.py
  configs/
    server.toml
    model-presets/
      default.toml
    voice-sets/
      default.toml
  runtime/
    cache/
    voices/
      local-voice/
        voice.toml
        reference.wav
    logs/
      backend.log
      backend.previous.log
      admin-ui.out.log
      admin-ui.err.log
      download.out.log
      download.err.log
    outputs/
```

`configs/server.toml` 统一收口 API、Admin、UI、runtime 默认状态：

```toml
[api]
host = "127.0.0.1"
port = 19890
preload_model = false

[admin]
enabled = true
host = "127.0.0.1"
port = 17870

[ui]
title = "Neiroha CosyVoice3 Admin"
default_language = "zh" # zh | en

[runtime]
active_model_preset = "cosyvoice3-default"
active_voice_set = "default"
default_voice = "local-voice"
```

不要再额外拆 `ui.toml` / `active.json`。如果为了兼容旧配置可以读取旧文件，但默认写入必须回到 `server.toml` 和 TOML 分层文件。

`configs/model-presets/default.toml`：

```toml
schema_version = 1
id = "cosyvoice3-default"
name = "CosyVoice3 Default"
engine = "cosyvoice3"

[cosyvoice3]
model_dir = "models/CosyVoice3"
fp16 = false
load_jit = false
load_trt = false
load_vllm = false
```

`configs/voice-sets/default.toml`：

```toml
schema_version = 1
id = "default"
name = "Default"
description = "Default voices exposed as OpenAI TTS models."
voices = ["local-voice"]
```

`runtime/voices/local-voice/voice.toml`：

```toml
schema_version = 1
id = "local-voice"
name = "Local Voice"
mode = "prompt_clone"
model_preset = "cosyvoice3-default"
reference_audio = "runtime/voices/local-voice/reference.wav"
prompt_audio = ""
prompt_text = "参考音频对应文本"
text_lang = "zh"
prompt_lang = "zh"
instruction = ""
speed = 1.0

[engine_options]
speaker_id = ""
speaker_embedding_path = ""
adapter_path = ""
```

## 启动与端口

`pixi.toml` 的默认 task 不要硬编码端口：

```toml
[tasks]
api = { cmd = "python scripts/launch_cosyvoice3.py --mode api" }
api-preload = { cmd = "python scripts/launch_cosyvoice3.py --mode api-preload" }
admin = { cmd = "python scripts/launch_cosyvoice3.py --mode admin" }
api-admin = { cmd = "python scripts/launch_cosyvoice3.py --mode api-admin" }
api-admin-preload = { cmd = "python scripts/launch_cosyvoice3.py --mode api-admin-preload" }
```

端口读取 `configs/server.toml`。如果配置端口被占用或 Windows 拒绝绑定，launcher 自动挑一个可绑定随机端口，并在终端、`backend.log`、`/health` 里播报真实地址。

必须支持：

```text
api                 只启动 FastAPI
admin               只启动 Gradio Admin，连接已有 FastAPI
api-admin           启动 FastAPI，再拉起独立 Gradio Admin
api-preload         启动 FastAPI 并预加载模型
api-admin-preload   启动 FastAPI + Gradio Admin + 预加载模型
```

不要把 Gradio mount 到 FastAPI。默认是两个独立进程：

```text
launcher
  -> uvicorn FastAPI :实际端口
  -> gradio Admin :实际端口
```

FastAPI 的 `/`、`/health`、`/cosyvoice3/capabilities`、`/cosyvoice3/meta` 应返回：

```json
{
  "api_url": "http://127.0.0.1:19890",
  "admin_url": "http://127.0.0.1:17870"
}
```

## Gradio Admin 页面

Gradio 不会自动翻译自定义 label。CosyVoice3 Admin 要自己维护 `zh` / `en` 文本表：

- 默认读 `configs/server.toml` 的 `[ui] default_language`。
- 环境变量可覆盖，例如 `NEIROHA_COSYVOICE3_UI_LANG=en`。
- 切语言后重启 Admin 生效即可，不需要做运行时热切换。

页面结构：

- 首页
- 试音
- 克隆配置 / Voice Config
- Voice Sets
- Model Presets
- 下载
- 日志

首页不要显示大段 JSON。首页只显示 Admin 真正需要的状态，并自动刷新：

- API online / offline
- API URL
- Admin URL
- 当前实际端口
- 是否发生端口回退
- 当前 model preset
- 当前 voice set
- 默认 voice
- 模型是否 loaded
- device / fp16
- 可用 voice 数
- 进程 PID / 状态

从 `start_portable.bat serve` 或 `pixi run serve` 启动后，首页必须自动刷新 API 状态，不要求用户手动点刷新。

## Gradio 日志页要求

这部分是 GPT-SoVITS 改造里最容易做抽象、做难用的地方，CosyVoice3 必须明确避免。

不要：

- 不要在日志页展示 JSON 大墙。
- 不要展示 `events.jsonl` 作为主日志。
- 不要让用户为了看最新日志滚到最底部。
- 不要把 API、Admin、下载、模型输出全部混成一个不可读面板。
- 不要只提供“刷新”按钮而没有自动刷新。

要做：

- 日志页默认显示普通 `.log` 文本。
- 最新日志在最上方。
- 每 1-2 秒自动刷新。
- 提供手动刷新按钮。
- 日志面板初始高度足够，不要只显示几行。
- 标清楚当前展示的是哪个文件。
- API / Admin / Download 分开看，至少用 tabs 或下拉选择文件。

建议日志文件：

```text
runtime/logs/backend.log              # 本轮 FastAPI/launcher 事件，最新一轮
runtime/logs/backend.previous.log     # 上一轮启动的 backend.log
runtime/logs/admin-ui.out.log         # 本轮 Gradio stdout
runtime/logs/admin-ui.err.log         # 本轮 Gradio stderr
runtime/logs/download.out.log         # 本轮下载 stdout
runtime/logs/download.err.log         # 本轮下载 stderr
```

每次顶层 launcher 启动：

```text
if backend.log exists and not empty:
  backend.previous.log = old backend.log
backend.log = empty new file
```

Admin 子进程日志和下载日志也建议每次启动 / 每次下载使用 `"w"` 模式重写。这样用户看到的是“本次启动 / 本次下载”的量，而不是几天堆在一起的量。

日志页推荐布局：

```text
日志
  [日志源 dropdown: backend.log / backend.previous.log / admin-ui.out.log / admin-ui.err.log / download.out.log / download.err.log]
  [自动刷新 checkbox 默认开]
  [刷新按钮]
  [Textbox/Code: 最新在上，160-300 行]
```

日志内容建议用人能读的行格式：

```text
2026-05-17 22:03:08 | launcher_start | mode=api-admin api_url=http://127.0.0.1:19890 admin_url=http://127.0.0.1:17870
2026-05-17 22:03:09 | port_fallback | service=FastAPI requested_port=19890 selected_port=3048 reason="..."
2026-05-17 22:04:12 | synthesis_complete | voice=local-voice audio_seconds=3.120 elapsed_seconds=1.840 rtf=0.590 output=runtime/outputs/speech_xxx.wav
```

结构化事件如果内部确实需要，可以内部保留；但 Gradio 日志页面对用户时只展示可读 `.log`。

## 上传参考音频与手填路径

Voice Config 页必须支持上传参考音频和手填参考音频路径二选一：

- 如果上传了文件：复制到 `runtime/voices/<voice-id>/reference.<ext>`，并写入 `reference_audio`。
- 如果没上传但填写了路径：写入该路径。
- 两者都没有：保存时报错 `reference audio is required`。
- 保存结果必须显示最终使用的 `Reference audio: ...`，让用户一眼确认。

不要出现“界面上上传了音频，但配置里没写路径”的情况。

## 中文 voice id 与 HTTP header

GPT-SoVITS 改造时踩过一个坑：中文 voice id 会进入输出文件名，再进入 `X-Neiroha-Output-Path` 或 `Content-Disposition` 响应头，Starlette/FastAPI 会按 latin-1 编码 header，导致：

```text
'latin-1' codec can't encode characters ...
```

CosyVoice3 必须从一开始规避：

- voice id / name 可以允许中文。
- 运行输出文件名必须转成 ASCII 安全名。
- `Content-Disposition` 的 filename 必须 ASCII。
- `X-Neiroha-Output-Path` 必须 header-safe。
- 如果需要给 Admin 显示真实本地路径，优先返回 ASCII 输出路径；必要时 Admin 侧把相对路径拼成绝对路径。

推荐输出名策略：

```text
safe_ascii_id = ascii_slug(voice_id)
if empty:
  safe_ascii_id = "speech_" + sha1(voice_id utf8)[0:10]
runtime/outputs/{safe_ascii_id}_YYYYMMDDHHMMSS.wav
```

## 下载边界

默认下载只做：

- CosyVoice3 基座模型。
- 一个示例参考音频 / 示例 voice。

不要默认下载大量社区 voice bank、角色包、别人训练好的模型。用户要更多模型时，让用户自己下载到本地，并在 Admin 新增 model preset / voice profile。

如果 CosyVoice3 支持 adapter、speaker embedding、微调权重：

- 独立模型目录写进 model preset。
- 单 voice 覆盖项写进 `voice.toml` 的 `engine_options` 或明确字段。
- 不要把这些文件混到默认下载里。

## OpenAI API 返回示例

`GET /v1/models`：

```json
{
  "object": "list",
  "data": [
    {
      "id": "default",
      "object": "model",
      "owned_by": "neiroha",
      "name": "Default",
      "voice_count": 1
    }
  ]
}
```

`GET /v1/audio/voices`：

```json
{
  "object": "list",
  "data": [
    {
      "id": "local-voice",
      "voice_id": "local-voice",
      "name": "Local Voice",
      "object": "voice",
      "model": "default",
      "task_mode": "prompt_clone",
      "model_preset": "cosyvoice3-default"
    }
  ],
  "voices": [
    {
      "voice_id": "local-voice",
      "name": "Local Voice",
      "model": "default"
    }
  ]
}
```

`POST /v1/audio/speech`：

```json
{
  "model": "default",
  "voice": "local-voice",
  "input": "你好，这是 CosyVoice3 的语音复刻测试。",
  "response_format": "wav",
  "speed": 1.0
}
```

响应头：

```text
X-Neiroha-Output-Path
X-Neiroha-Audio-Seconds
X-Neiroha-Elapsed-Seconds
X-Neiroha-RTF
```

## 验收清单

改完后至少跑：

1. `python -m py_compile scripts/launch_cosyvoice3.py scripts/download_cosyvoice3_assets.py`
2. 解析所有 TOML：
   - `configs/server.toml`
   - `configs/model-presets/*.toml`
   - `configs/voice-sets/*.toml`
   - `runtime/voices/*/voice.toml`
3. TestClient 检查：
   - `/health`
   - `/v1/models`
   - `/v1/audio/voices`
   - `/v1/audio/speech`
   - `/cosyvoice3/logs`
4. 实际合成一次，确认 `runtime/outputs/*.wav` 生成。
5. 用中文 voice id 合成一次，确认不再触发 latin-1 header 报错。
6. 保存 Voice Config：
   - 上传参考音频路径能写进 `voice.toml`
   - 手填参考音频路径也能写进 `voice.toml`
   - 两者都空会报错
7. 启动时占用配置端口，确认 launcher 自动随机回退端口，并在首页 / 日志 / `/health` 显示实际地址。
8. Gradio Admin Blocks 中文 / 英文各实例化一次。
9. 日志页确认：
   - 显示 `.log`，不是 JSON
   - 最新在上
   - 自动刷新
   - 能切 `backend.previous.log`
   - 新启动不会继续无限追加旧日志

## 非目标

不要做这些事：

- 不要合并三个项目的 Pixi 环境。
- 不要把 Gradio mount 到 FastAPI 作为默认方案。
- 不要让 Pixi 默认 task 硬编码端口。
- 不要让用户配置同时散落在 TOML / JSON / YAML。
- 不要让日志页展示 JSON 大墙。
- 不要默认下载大量第三方训练包。
- 不要因为中文 voice id 让 HTTP header 崩掉。
