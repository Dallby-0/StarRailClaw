from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import FSM_DEBUG_DIR


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


class FsmRunLogger:
    """Append-only JSONL logger for one FSM run."""

    def __init__(self, run_id: str, debug_dir: Path = FSM_DEBUG_DIR) -> None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = debug_dir / f"fsm_run_{run_id}_{ts}.jsonl"

    def event(self, event: str, **fields: Any) -> None:
        row = {
            "ts": _now_iso(),
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False, separators=(",", ":")) + "\n")

    def text(self, message: str, event: str = "console", **fields: Any) -> None:
        print(message)
        self.event(event, message=message, **fields)


def summarize_match(match: Any) -> dict[str, Any]:
    return {
        "state_id": getattr(match, "state_id", ""),
        "state_dir": str(getattr(match, "state_dir", "")),
        "passed_enabled": getattr(match, "passed_enabled", 0),
        "total_enabled": getattr(match, "total_enabled", 0),
        "passed_all": getattr(match, "passed_all", 0),
        "total_all": getattr(match, "total_all", 0),
        "success": getattr(match, "success", False),
        "conditions": getattr(match, "condition_results", []),
    }
