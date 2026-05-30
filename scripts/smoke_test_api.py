from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        try:
            from scripts import toml_compat as tomllib
        except ModuleNotFoundError:
            import toml_compat as tomllib

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SERVER_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "server.toml"


def load_default_base_url() -> str:
    with SERVER_CONFIG_PATH.open("rb") as file:
        config = tomllib.load(file)
    api = config.get("api", {}) if isinstance(config.get("api"), dict) else {}
    host = str(api.get("host") or "127.0.0.1")
    port = int(api.get("port") or 9880)
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{port}"


def request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=20) as response:
        text = response.read().decode("utf-8")
    return json.loads(text)


def request_bytes(method: str, url: str, payload: dict[str, Any] | None = None) -> tuple[bytes, str]:
    data = None
    headers = {"Accept": "audio/*"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=300) as response:
        content = response.read()
        content_type = response.headers.get("Content-Type", "")
    return content, content_type


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a running Neiroha GPT-SoVITS API.")
    parser.add_argument("--base-url", default=load_default_base_url())
    parser.add_argument("--synthesize", action="store_true", help="Also POST /v1/audio/speech.")
    parser.add_argument("--text", default="Neiroha GPT-SoVITS smoke test.")
    parser.add_argument("--model", default="default")
    parser.add_argument("--voice", default="genshin-keqing")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    checks = [
        ("GET", "/health", None),
        ("GET", "/v1/models", None),
        ("GET", "/v1/audio/voices", None),
    ]
    if args.synthesize:
        checks.append(
            (
                "POST",
                "/v1/audio/speech",
                {
                    "model": args.model,
                    "voice": args.voice,
                    "input": args.text,
                    "response_format": "wav",
                },
            )
        )

    for method, path, payload in checks:
        url = f"{base}{path}"
        try:
            if path == "/v1/audio/speech":
                content, content_type = request_bytes(method, url, payload)
                if not content:
                    raise RuntimeError("Expected non-empty audio response")
                if not content_type.startswith("audio/"):
                    raise RuntimeError(f"Expected audio response from {path}, got {content_type!r}")
            else:
                result = request_json(method, url)
                if not isinstance(result, dict):
                    raise RuntimeError(f"Expected JSON object from {path}")
        except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"[fail] {method} {path}: {exc}", file=sys.stderr)
            return 1
        print(f"[ok] {method} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
