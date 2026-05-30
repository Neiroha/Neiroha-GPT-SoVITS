# Neiroha GPT-SoVITS API

默认 FastAPI 地址来自 `configs/server.toml` 的 `[api]` 配置，当前为
`http://127.0.0.1:9880`。Gradio Admin 默认地址来自 `[admin]` 配置，当前为
`http://127.0.0.1:7860`。

## 稳定路由

```text
GET  /health
GET  /v1/models
GET  /v1/audio/voices
POST /v1/audio/speech
```

`/v1/audio/speech` 里的 `model` 是 Neiroha voice set id，例如
`default`。底层 GPT-SoVITS 权重由 `configs/model-presets/*.toml` 和 voice profile
间接选择。

## Native 路由

标准 native 前缀是 `/api/gpt-sovits`。旧的 `/gpt-sovits/*` 和 `/tts`
路由保留兼容。

```text
GET  /api/gpt-sovits/models
GET  /api/gpt-sovits/voices
GET  /api/gpt-sovits/capabilities
GET  /api/gpt-sovits/logs
POST /api/gpt-sovits/tts
POST /api/gpt-sovits/clone
POST /api/gpt-sovits/clone/upload
POST /api/gpt-sovits/load
POST /api/gpt-sovits/unload
POST /api/gpt-sovits/reload
```

音频响应会尽量返回 `X-Neiroha-*` 运行时头，包括 backend、model preset、
voice、sample rate、inference time、output format 和 output path。
