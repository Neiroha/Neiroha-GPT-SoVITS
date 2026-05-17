# Neiroha GPT-SoVITS Local Launcher

面向 Neiroha 的 GPT-SoVITS 独立后端。这个仓库只维护当前项目自己的 Pixi
环境、FastAPI、Gradio Admin 和配置语义，不和其他 TTS 项目合并 Python 依赖。

当前默认形态已经收敛为：

- OpenAI TTS 风格 API：`/v1/models`、`/v1/audio/voices`、`/v1/audio/speech`
- `model` 表示 voice set，不再表示底层权重
- `voice` 表示 voice set 里的具体声音配置
- 底层 GPT-SoVITS 权重放在 model preset
- 默认只暴露一个 voice：`genshin-keqing`
- 额外角色、共享权重、多角色包不再内置批量下载；需要的人自行下载权重后在 Admin 登记

## 目录语义

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
    genshin-keqing/
      voice.json
      reference.wav
  logs/
  outputs/
models/
  pretrained/
```

`configs/voice-sets/default.json` 对应 Neiroha / OpenAI API 里的 `model=default`。
`runtime/voices/genshin-keqing/voice.json` 对应 `voice=genshin-keqing`。
`configs/model-presets/default.toml` 才是 GPT-SoVITS 的底层权重 preset。

## 安装

```powershell
pixi install
pixi run submodule-init
pixi run install-deps
pixi run install-assets
pixi run install-sample-voice
```

`install-assets` 下载官方预训练资产。`install-sample-voice` 只下载一个默认示例参考音频，
不会下载多角色训练权重。

## 启动

```powershell
start_api_admin.bat
```

或使用 Pixi task：

```powershell
pixi run api
pixi run api-preload
pixi run admin
pixi run api-admin
pixi run api-admin-preload
```

默认端口来自 `configs/server.toml`：

```text
FastAPI  http://127.0.0.1:19880
Admin    http://127.0.0.1:17860
```

这些 Pixi task 默认不再写死端口；如果配置端口被占用或被 Windows 拒绝绑定，
launcher 会自动挑一个可用随机端口，并在终端和 `runtime/logs/backend.log` 里写出实际地址。
`api-12080` 仍然是显式覆盖端口的调试 task。

`admin` 只启动 Gradio Admin，并连接已有 FastAPI。`api-admin` 会启动 FastAPI，
再以独立子进程拉起 Gradio Admin；不再把 Gradio mount 到 FastAPI。

## Admin 语言

Gradio 不会自动翻译自定义 label，但本项目的 Admin 已经支持启动时选择中文或英文。
修改：

```toml
# configs/ui.toml
default_language = "zh" # zh | en
```

或设置环境变量：

```powershell
$env:NEIROHA_GPT_SOVITS_UI_LANG="en"
```

然后重启 Admin。

## 日志

Admin 的“日志”页显示最新的 `runtime/logs/backend.log`，默认最新在上并自动刷新。
下载任务仍写到 `runtime/logs/admin-download.out.log` 和
`runtime/logs/admin-download.err.log`。

## API

列出 voice set：

```powershell
curl.exe http://127.0.0.1:9880/v1/models
```

列出 voice：

```powershell
curl.exe http://127.0.0.1:9880/v1/audio/voices
```

语音复刻：

```powershell
curl.exe http://127.0.0.1:9880/v1/audio/speech `
  -H "Content-Type: application/json" `
  -d '{ "model":"default", "voice":"genshin-keqing", "input":"你好，这是一次语音复刻测试。", "response_format":"wav" }' `
  --output speech.wav
```

响应头会包含 `X-Neiroha-Output-Path`、`X-Neiroha-Audio-Seconds`、
`X-Neiroha-Elapsed-Seconds` 和 `X-Neiroha-RTF`。

## 添加自己的 voice

在 Admin 的“克隆配置”页上传参考音频、填写对应文本、设置 voice id/name 并保存。
默认使用当前 model preset。若要使用别人训练过的 GPT-SoVITS v2Pro/v2ProPlus
权重，可以在“Model Presets”页新增一个 preset，或直接在“克隆配置”页展开
“覆盖为别人训练过的 GPT/SoVITS 权重”并填写 `.ckpt` / `.pth` 路径。

也可以手动创建：

```text
runtime/voices/my_voice/voice.json
```

然后把 `my_voice` 加到：

```text
configs/voice-sets/default.json
```

voice 配置示例：

```json
{
  "schema_version": 1,
  "id": "my_voice",
  "name": "My Voice",
  "mode": "prompt_clone",
  "model_preset": "v2proplus-clone",
  "reference_audio": "runtime/voices/my_voice/reference.wav",
  "prompt_audio": "",
  "prompt_text": "参考音频对应文本",
  "text_lang": "zh",
  "prompt_lang": "zh",
  "instruction": "",
  "speed": 1.0,
  "engine_options": {},
  "gpt_weights_path": "models/voices/my-trained/GPT_xxx.ckpt",
  "sovits_weights_path": "models/voices/my-trained/SV_xxx.pth"
}
```

`gpt_weights_path` 和 `sovits_weights_path` 是可选覆盖项；不填时使用
`model_preset` 指向的底层 preset。
