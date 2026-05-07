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

预训练基座、官方权重和下载好的角色模型都会放在顶层 `models/` 下。
`runtime/` 只保留运行时缓存、上传临时文件和生成语音输出。

更多下载入口放在 Gradio 管理页的 `模型下载` tab：预训练基座、v2ProPlus 克隆基座、
已训练多角色 demo 都可以从网页启动。下载日志写到 `runtime/logs/admin-download.*.log`。
需要纯 CLI 时可以直接调用 `scripts/download_gpt_sovits_assets.py`。公开训练包候选见
`docs/model-sources.md`。

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

FastAPI 主服务，默认端口 `9880`。OpenAI 风格说话人列表始终可从
`/v1/audio/voices` 查看：

```powershell
pixi run api
pixi run api-12080
```

Gradio 管理控制台，默认端口 `7860`。`admin` task 会先启动 FastAPI 主进程，
然后把 Gradio 管理页作为子进程拉起。FastAPI API 在 `9880`，Gradio 管理页在
`7860`：

```powershell
pixi run admin
```

如果 Windows 把 `9880` 放进了 TCP excluded port range，`9880` 不会显示为被进程占用，
但依然无法绑定。这种情况下可以先用备用端口启动：

```powershell
pixi run admin-12080
```

然后把 Neiroha 里的 Base URL 临时改成 `http://127.0.0.1:12080`。

需要预加载或改默认 voice 时直接调用 launcher：

```powershell
pixi run python scripts/launch_gpt_sovits.py --mode api --port 12080 --default-voice genshin-paimon --preload-model
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

## Model Modes

本仓库现在把 GPT-SoVITS 的两种常见用法分开：

- 已训练音色：例如原神角色模型。`GET /gpt-sovits/models` 会按真实配置列出模型和里面的 voices，`/v1/audio/speech` 只走这些已配置 voice。
- 声音克隆：使用 v2ProPlus 基座权重，加参考音频和对应文本。原生接口是 `POST /gpt-sovits/clone` 或上传版 `POST /gpt-sovits/clone/upload`。

Gradio 管理页里也拆成了两个测试页：

- `已训练音色测试`
- `声音克隆测试`

GPT-SoVITS upstream 对 v2ProPlus 克隆参考音频有 3-10 秒限制。本 launcher 不改
`GPT-SoVITS` 子仓库，而是在 clone API 入口临时规整参考音频：短于 `3.05s` 会补尾部静音，
长于 `9.95s` 会从开头裁剪，原始上传或本地文件不会被修改。长音频的 `prompt_text` 应该对应
前 `9.95s`。

非流式推理默认会在终端打印 RTF 性能日志：

```text
TTS performance mode=trained speaker=genshin-paimon audio=2.660s elapsed=22.844s rtf=8.588
```

关闭方式：

```powershell
pixi run python scripts/launch_gpt_sovits.py --mode api --port 12080 --no-rtf-log
```

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
