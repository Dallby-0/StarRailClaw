from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_main_tree(path: Path) -> None:
    if path.exists():
        return
    write_json(
        path,
        {
            "type": "Repeat",
            "times": -1,
            "child": {
                "type": "Selector",
                "children": [
                    {"type": "Action", "name": "LLM决策", "do": "llm_decide", "params": {}},
                    {
                        "type": "Action",
                        "name": "空闲",
                        "comment": "无匹配场景时等待3秒",
                        "do": "wait",
                        "params": {"milliseconds": 3000},
                    },
                ],
            },
        },
    )


def node_signature(node: dict[str, Any]) -> str:
    return json.dumps(node, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_json_object(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
