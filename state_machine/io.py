from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from .constants import (
    EXPERIENCE_PATH,
    FSM_DEBUG_DIR,
    FSM_GRAPH_PATH,
    FSM_RUNTIME_PATH,
    FSM_SCHEMA_PATH,
    FSM_TEMPLATES_DIR,
    LLM_FSM_PROMPT_BASE,
    SCHEMA_VERSION,
)


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _slugify(raw: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(raw))
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("_")
    return safe or "state"


def _normalize_page_type(raw: Any) -> str | None:
    text = str(raw or "").strip()
    if not text or text.lower() in {"none", "null", "unknown", "n/a"}:
        return None
    return _slugify(text)


def _state_page_type(meta: dict[str, Any]) -> str:
    return _normalize_page_type(meta.get("page_type")) or _slugify(str(meta.get("slug", "state")))


def _load_experience_text() -> str:
    if not EXPERIENCE_PATH.exists():
        return ""
    txt = EXPERIENCE_PATH.read_text(encoding="utf-8").strip()
    return txt[-5000:]


def _build_system_prompt_with_experience() -> str:
    exp = _load_experience_text()
    if not exp:
        return LLM_FSM_PROMPT_BASE
    return LLM_FSM_PROMPT_BASE + "\n\n历史经验（仅参考，不要逐字复述）：\n" + exp


def _append_experience(line: str) -> None:
    EXPERIENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not EXPERIENCE_PATH.exists():
        EXPERIENCE_PATH.write_text("# experience\n\n", encoding="utf-8")
    with EXPERIENCE_PATH.open("a", encoding="utf-8") as f:
        f.write(f"- {_now_iso()} {line}\n")


def _ensure_fsm_resources() -> None:
    FSM_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    FSM_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    if not FSM_GRAPH_PATH.exists():
        _save_json(
            FSM_GRAPH_PATH,
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "nodes": [],
                "edges": [],
            },
        )
    if not FSM_RUNTIME_PATH.exists():
        _save_json(
            FSM_RUNTIME_PATH,
            {
                "schema_version": SCHEMA_VERSION,
                "last_state_id": None,
                "run_id": None,
                "llm_session_id": None,
                "llm_turn_count": 0,
                "session_tier": "soft",
                "repair_fail_count": 0,
                "last_transition_ok": False,
                "pending_refresh": False,
                "pending_from_state_id": None,
                "pending_action_id": None,
            },
        )
    _save_json(
        FSM_SCHEMA_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "required_fields": ["page_summary", "slug", "possible_page_type", "elements", "actions"],
            "element_types": ["text_line", "pattern"],
            "element_levels": ["high", "mid", "low"],
            "action_types": ["click", "run_preset"],
        },
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _backup_json(path: Path) -> None:
    if not path.exists():
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    shutil.copy2(path, path.with_name(f"{path.name}.bak_{ts}"))


def _load_runtime() -> dict[str, Any]:
    return _load_json(FSM_RUNTIME_PATH)


def _save_runtime(runtime: dict[str, Any]) -> None:
    _save_json(FSM_RUNTIME_PATH, runtime)


def _save_frame(path: Path, frame_rgb) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))


def _load_frame(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
