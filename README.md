# Neiroha GPT-SoVITS Local Launcher

面向 Neiroha 的 GPT-SoVITS 外层封装仓库。目标是和
[`Neiroha-VoxCPM`](https://github.com/Neiroha/Neiroha-VoxCPM) 一样：

- 外层仓库维护 Pixi 环境、启动器、API 兼容层和本地运行说明
- 官方 [`RVC-Boss/GPT-SoVITS`](https://github.com/RVC-Boss/GPT-SoVITS) 作为 Git submodule 放在 `GPT-SoVITS/`
- FastAPI 主服务默认监听 `9880`，匹配 Neiroha 里 `GPT-SoVITS` provider 的默认配置
- Gradio 管理页默认监听 `7860`，只负责模型/API 管理和简单 TTS 测试
- 提供 GPT-SoVITS 原生 API：`/tts`、`/set_gpt_weights`、`/set_sovits_weights`
- 提供 Neiroha 可直接使用的 OpenAI TTS 风格 API：`/v1/audio/speech`
- 提供 Gradio 管理页，用来查看 API 状态、加载/卸载模型、切换 GPT/SoVITS 权重和试音

OpenAI 兼容路由按官方
[Audio speech API](https://platform.openai.com/docs/api-reference/audio/createSpeech)
的核心字段组织：
`model`、`input`、`voice`、`response_format`、`speed`。GPT-SoVITS 额外需要参考音频，
所以本仓库额外支持 voice profile 和 `ref_audio_path` 扩展字段。

## Quick Start

```powershell
git clone --recurse-submodules https://github.com/neiroha/neiroha-gpt-sovits.git
cd neiroha-gpt-sovits
pixi install
```

如果已经克隆了外层仓库但还没有子模块：

```powershell
pixi run submodule-init
```

安装官方 GPT-SoVITS Python 依赖：

```powershell
pixi run install-deps
```

下载官方预训练资产，默认使用 ModelScope 源：

```powershell
pixi run install-assets
```

下载 Hugging Face 上的官方 v2ProPlus 测试权重：

```powershell
pixi run install-v2pro-hf
```

下载 Hugging Face 上别人训练好的原神多角色 demo，并自动写入
`profiles/voices.json` 方便直接用 `voice` 切换说话人：

```powershell
pixi run install-genshin-demo-hf
```

也可以使用 Hugging Face 或 hf-mirror：

```powershell
pixi run install-assets-hf
pixi run install-assets-hf-mirror
```

## Voice Profiles

GPT-SoVITS 的推理必须有参考音频。OpenAI 风格的 `voice` 字段本身不足以表达
`ref_audio_path`、`prompt_text`、语言和可选权重，因此这里用本地 profile 映射。

复制示例文件：

```powershell
Copy-Item profiles/voices.example.json profiles/voices.json
```

然后把 `ref_audio_path` 和 `prompt_text` 改成真实数据。

如果使用原神 demo 下载任务，它会额外生成
`profiles/voices.genshin.example.json`，并把同样内容写到被 git 忽略的
`profiles/voices.json`。默认包含 `genshin-paimon`、`genshin-keqing`、`genshin-klee`
三个 voice id。

## Run

FastAPI 主服务，默认端口 `9880`：

```powershell
pixi run api
```

Gradio 管理页，默认端口 `7860`，通过 HTTP 管理已经运行在 `9880` 上的 FastAPI：

```powershell
pixi run admin
```

一键启动完整本地栈：FastAPI API 在 `9880`，Gradio 管理页在 `7860`：

```powershell
pixi run webui
```

如果 Windows 把 `9880` 放进了 TCP excluded port range，`9880` 不会显示为被进程占用，
但依然无法绑定。这种情况下可以先用备用端口启动：

```powershell
pixi run webui-12080
```

然后把 Neiroha 里的 Base URL 临时改成 `http://127.0.0.1:12080`。也可以用环境变量覆盖：

```powershell
$env:NEIROHA_GPT_SOVITS_API_PORT = "12080"
pixi run python scripts/serve_neiroha_gpt_sovits.py
```

启动时预加载模型：

```powershell
pixi run api-preload
pixi run webui-preload
```

## Neiroha 配置

Neiroha 里已经有 `gptSovits` adapter：

- Adapter Type: `GPT-SoVITS`
- Base URL: `http://127.0.0.1:9880`
- Default Model: `gpt-sovits`

如果想把它当 OpenAI-compatible provider：

- Adapter Type: `OpenAI TTS API Compatible`
- Base URL: `http://127.0.0.1:9880/v1`
- Model: `gpt-sovits` 或保留 `tts-1`，本地服务会接受 `tts-1` 作为兼容别名
- Voices: 从 `GET /v1/audio/voices` 读取 `profiles/voices.json`

## API Examples

OpenAI-compatible:

```powershell
curl.exe http://127.0.0.1:9880/v1/audio/speech `
  -H "Content-Type: application/json" `
  -d '{ "model":"gpt-sovits", "voice":"default", "input":"你好，欢迎使用 Neiroha。", "response_format":"wav" }' `
  --output speech.wav
```

不使用 profile，直接传参考音频：

```powershell
curl.exe http://127.0.0.1:9880/v1/audio/speech `
  -H "Content-Type: application/json" `
  -d '{ "model":"gpt-sovits", "voice":"default", "input":"你好。", "ref_audio_path":"D:/voices/ref.wav", "prompt_text":"参考音频对应文本", "text_lang":"zh", "prompt_lang":"zh", "response_format":"wav" }' `
  --output speech.wav
```

GPT-SoVITS native:

```powershell
curl.exe http://127.0.0.1:9880/tts `
  -H "Content-Type: application/json" `
  -d '{ "text":"你好。", "text_lang":"zh", "ref_audio_path":"D:/voices/ref.wav", "prompt_text":"参考音频对应文本", "prompt_lang":"zh", "media_type":"wav" }' `
  --output speech.wav
```

## Management

- `GET /health`
- `GET /gpt-sovits/meta`
- `POST /gpt-sovits/load`
- `POST /gpt-sovits/unload`
- `POST /gpt-sovits/reload`
- `GET|POST /set_gpt_weights`
- `GET|POST /set_sovits_weights`
- `GET|POST /control` with `load`、`unload`、`reload`、`restart`、`exit`

Gradio 管理页默认在独立端口：

```text
http://127.0.0.1:7860
```

## CUDA / Pixi Notes

本仓库参考 `Neiroha-VoxCPM` 的 Windows + CUDA + Pixi 工作流：

- `cuda = "12.8"`
- PyTorch / Torchaudio / Torchvision 使用 Aliyun `cu128` wheel URL
- Python 使用 `3.11.*`，符合 GPT-SoVITS README 里的 CUDA 12.8 测试组合

模型权重、运行时缓存、下载包和本地 voice profile 不进入版本控制。
