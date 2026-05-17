# 将 GPT-SoVITS 改造经验迁移到 Neiroha-CosyVoice 的范式指导

这份文档是给后续 AI / 维护者执行 Neiroha-CosyVoice 改造时使用的工作指令。目标不是复用 GPT-SoVITS 的 Python 环境或代码，而是复用这次整理出的产品语义、配置分层、Admin 交互形状和验收标准。

## 总原则

三个后端项目保持完全独立：

- `Neiroha-GPT-SoVITS`
- `Neiroha-CosyVoice`
- `VoxCPM`

不要把 Python 依赖合进一个环境。每个项目保留自己的 `.pixi`、模型下载目录、FastAPI 服务、Gradio Admin 和启动脚本。可以统一 UI 形状和配置语义，但不要共享运行时依赖。

## 必须保留的语义

Flutter / OpenAI API 侧的 `model` 不是底层权重模型，而是 voice set / 声音组。

- `GET /v1/models` 返回 voice set。
- `GET /v1/audio/voices` 返回某个 voice set 中可用的 voice。
- `POST /v1/audio/speech` 中 `model` 选择 voice set，`voice` 选择具体说话人。
- 底层 CosyVoice 模型放在 `model preset` 中，不直接作为 OpenAI model 暴露。

请求映射必须是：

```text
model=default
  -> configs/voice-sets/default.json
voice=local-voice
  -> runtime/voices/local-voice/voice.json
model_preset=cosyvoice-default
  -> configs/model-presets/cosyvoice-default.toml
mode=prompt_clone / zero_shot / sft
  -> 转换成 CosyVoice 引擎真实参数
```

## 启动模式

CosyVoice 项目内新增或整理为：

```text
Neiroha-CosyVoice/
  start_api_admin.bat
  scripts/launch_cosyvoice.py
```

统一支持这些模式：

```text
api                 只启动 FastAPI
admin               只启动 Gradio Admin，连接已有 FastAPI
api-admin           启动 FastAPI，再拉起独立 Gradio Admin
api-preload         启动 FastAPI 并预加载模型
api-admin-preload   启动 FastAPI + Gradio Admin + 预加载模型
```

不要把 Gradio mount 到 FastAPI。`api-admin` 模式应由 launcher 启动两个独立进程：

```text
launcher
  -> uvicorn FastAPI :9880
  -> gradio Admin :17860
```

不要做成：

```text
FastAPI + gr.mount_gradio_app()
```

`start_api_admin.bat` 可以使用这个形状：

```bat
@echo off
cd /d "%~dp0"
set PY=.pixi\envs\default\python.exe

if not exist "%PY%" (
  pixi install
)

"%PY%" -B scripts\launch_cosyvoice.py --mode api-admin-preload %*
```

## 配置目录

CosyVoice 项目内使用同样的配置语义，但内容只服务 CosyVoice：

```text
configs/
  server.toml
  ui.toml
  model-presets/
    default.toml
  voice-sets/
    default.json
runtime/
  state/
    active.json
  voices/
    local-voice/
      voice.json
      reference.wav
  logs/
    backend.log
    download.log
  outputs/
```

`configs/server.toml`：

```toml
[api]
host = "127.0.0.1"
port = 9880
preload_model = false

[admin]
enabled = true
host = "127.0.0.1"
port = 17860

[runtime]
active_model_preset = "cosyvoice-default"
active_voice_set = "default"
default_voice = "local-voice"
```

`configs/ui.toml`：

```toml
schema_version = 1
title = "Neiroha CosyVoice Admin"
default_language = "zh" # zh | en
```

`configs/model-presets/default.toml`：

```toml
schema_version = 1
id = "cosyvoice-default"
name = "CosyVoice Default"
engine = "cosyvoice"

[cosyvoice]
model_dir = "models/Fun-CosyVoice3-0.5B"
fp16 = false
load_vllm = false
```

如果 CosyVoice 项目使用的是其他官方基座模型，`model_dir` 改成实际默认路径，但语义保持不变。

`configs/voice-sets/default.json`：

```json
{
  "schema_version": 1,
  "id": "default",
  "name": "Default",
  "description": "Default voices exposed as OpenAI TTS models.",
  "voices": ["local-voice"]
}
```

`runtime/voices/local-voice/voice.json`：

```json
{
  "schema_version": 1,
  "id": "local-voice",
  "name": "Local Voice",
  "mode": "prompt_clone",
  "model_preset": "cosyvoice-default",
  "reference_audio": "runtime/voices/local-voice/reference.wav",
  "prompt_audio": "",
  "prompt_text": "参考音频对应文本",
  "text_lang": "zh",
  "prompt_lang": "zh",
  "instruction": "",
  "speed": 1.0,
  "engine_options": {}
}
```

## 下载边界

默认下载只做两件事：

- 下载一个 CosyVoice 基座模型。
- 下载或保留一个示例参考音频 / 单个示例 voice。

不要批量下载社区角色包、别人训练好的大量 voice bank 或无关 demo 文件。用户想要更多声音时，让用户自己下载到本地，然后在 Admin 中新增 voice / preset。

如果 CosyVoice 支持外部微调权重或 speaker embedding，Admin 要允许用户填本地路径，保存到 `voice.json` 或单独的 model preset 里；不要把这些文件内置到项目仓库。

## 训练模型和 voice 的处理

GPT-SoVITS 这次的关键经验是：不能只有“参考音频克隆”，还要允许使用别人训练过的模型。迁移到 CosyVoice 时按 CosyVoice 的能力映射：

- 如果训练产物是一个可独立加载的模型目录，把它保存成新的 model preset。
- 如果训练产物只是 speaker / embedding / adapter，把它保存在 voice profile 的 `engine_options` 或明确字段里。
- voice profile 必须能覆盖默认 preset 的部分引擎参数。

建议 voice profile 扩展方式：

```json
{
  "mode": "prompt_clone",
  "model_preset": "cosyvoice-default",
  "reference_audio": "runtime/voices/local-voice/reference.wav",
  "prompt_text": "参考音频对应文本",
  "engine_options": {
    "speaker_id": "",
    "speaker_embedding_path": "",
    "adapter_path": ""
  }
}
```

字段名要和 CosyVoice 实际加载逻辑对齐，但 Admin 的产品语义保持一致：用户先选 voice set，再选 voice，必要时 voice 内部覆盖底层模型或训练资产。

## Admin 形状

Gradio 不会自动把自定义中文标签翻译成英文；需要项目自己做标签表。CosyVoice Admin 应支持：

- `configs/ui.toml` 中 `default_language = "zh"` 或 `"en"`。
- 环境变量覆盖：`NEIROHA_COSYVOICE_UI_LANG=en`。

首页不要显示大段 JSON。首页只放 Admin 需要看的状态，且自动刷新：

- API online / offline
- API base
- 当前 model preset
- 当前 voice set
- 默认 voice
- 模型是否已加载
- 设备 / fp16
- 可用 model / voice 数量或列表
- API 进程状态

从 `start_api_admin.bat` 启动后，Admin 首页必须自动刷新 API 状态，不要求用户手动点刷新。

页面建议：

- 首页：状态卡 / Markdown 状态，2 秒自动刷新。
- 试音：voice set、voice、文本、格式、速度、输出音频、RTF / elapsed / output path。
- 克隆配置：上传参考音频、prompt text、voice id/name、voice set、model preset、CosyVoice 特有参数。
- Voice Sets：查看、新建、删除、切换、添加 / 移除 voice、设置默认 voice。
- Model Presets：加载、卸载、重载、保存自定义 preset；显示 CosyVoice 的 `model_dir`、`fp16`、`load_vllm` 等。
- 下载：下载默认基座、下载单个示例参考音频、显示下载日志。
- 日志：显示 `runtime/logs/backend.log`，最新在上，自动刷新。

## 日志规则

Admin 上不要把日志做成 JSON 大墙。使用普通 `.log` 文件：

```text
runtime/logs/backend.log
runtime/logs/download.log
```

日志页默认展示最新内容在上方，并自动刷新。结构化事件如果内部确实需要，可以另存内部文件，但 Admin 直接面向用户时只显示可读日志。

## API 返回形状

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
      "model_preset": "cosyvoice-default"
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
  "input": "你好，这是 CosyVoice 的语音复刻测试。",
  "response_format": "wav",
  "speed": 1.0
}
```

返回音频时保留这些响应头，便于 Admin 试音页展示：

```text
X-Neiroha-Output-Path
X-Neiroha-Audio-Seconds
X-Neiroha-Elapsed-Seconds
X-Neiroha-RTF
```

## 验收步骤

改造完成后必须做这些检查：

1. `python -m py_compile scripts/launch_cosyvoice.py scripts/download_cosyvoice_assets.py`
2. 用 FastAPI TestClient 检查：
   - `/health`
   - `/v1/models`
   - `/v1/audio/voices`
   - `/v1/audio/speech`
   - `/cosyvoice/logs`
3. 实际跑一次语音复刻或零样本合成，确认生成 `runtime/outputs/*.wav`。
4. 构建 Admin Blocks 一次中文，一次英文，确认没有未定义变量。
5. 启动 `start_api_admin.bat` 后打开 Admin 首页，确认 API 状态自动刷新。
6. 打开日志页，确认显示 `.log`、最新在上、自动刷新。

## 非目标

不要做这些事：

- 不要合并 GPT-SoVITS、CosyVoice、VoxCPM 的 Python 环境。
- 不要把 Gradio mount 到 FastAPI 当成默认启动方式。
- 不要默认下载一堆社区训练 voice。
- 不要把底层模型 preset 暴露成 OpenAI `model`。
- 不要让用户为了看最新日志滚到页面最底部。
- 不要只支持参考音频克隆而完全堵死训练模型 / adapter / embedding 的本地路径入口。
