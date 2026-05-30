from __future__ import annotations

import json
from typing import Any, BinaryIO, TextIO


def _parse_value(raw_value: str) -> Any:
    raw_value = raw_value.strip()
    if raw_value in {"true", "false"}:
        return raw_value == "true"
    if raw_value.startswith('"') or raw_value.startswith("["):
        return json.loads(raw_value)
    try:
        if "." in raw_value:
            return float(raw_value)
        return int(raw_value)
    except ValueError:
        return raw_value.strip('"')


def loads(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current = result
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = result.setdefault(line.strip("[]"), {})
            continue
        if "=" not in line:
            continue
        key, raw_value = [part.strip() for part in line.split("=", 1)]
        current[key] = _parse_value(raw_value)
    return result


def load(file: BinaryIO | TextIO) -> dict[str, Any]:
    data = file.read()
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return loads(data)

