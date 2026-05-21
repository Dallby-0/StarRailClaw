from __future__ import annotations

import argparse
import json
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from agent.llm_client import DoubaoClient
from agent.presets import run_preset
from sr_tools.adb import resolve_target_serial
from sr_tools.emulator import EmulatorClient

FSM_DIR = Path("StateMachineResources")
FSM_TEMPLATES_DIR = FSM_DIR / "templates"
FSM_GRAPH_PATH = FSM_DIR / "state_graph.json"
FSM_SCHEMA_PATH = FSM_DIR / "llm_protocol_schema.json"
FSM_RUNTIME_PATH = FSM_DIR / "runtime_state.json"
FSM_DEBUG_DIR = FSM_DIR / "debug"
EXPERIENCE_PATH = FSM_DIR / "experience.md"

SCHEMA_VERSION = "1.1.0"
ACTION_CLICK_WAIT_S = 5.0
LLM_PARSE_RETRY = 2
MAX_RETRY_PER_ACTION = 2
PRESET_WAIT_TIMEOUT_S = 600.0

SOFT_LIMIT = 5
MID_LIMIT = 10
HARD_LIMIT = 15
FORCE_RESET_AT = 16

REPAIR_EFFORTS = ["low", "mid", "high"]
DISCRIMINATION_SCORE = {"high": 4, "mid": 2, "low": 1}
STABILITY_SCORE = {"high": 3, "mid": 2, "low": 1}
MATCH_DISCRIMINATION_TARGET = 4

LLM_FSM_PROMPT_BASE = """你是视觉驱动游戏自动化的状态标注器和动作规划器,当前任务是通关崩坏星穹铁道差分宇宙。

你将收到一张游戏截图。你的任务：
1) 识别并输出用于区分该页面的关键信息，按重要性排序。
2) 输出推进流程的动作序列（1000x1000逻辑坐标）。

强约束：
- 必须输出严格 JSON 对象，字段必须符合约定。
- elements: 每项必须有 type。
  - type=text_line: 必须给 text, bbox[x1,y1,x2,y2], brief。
  - type=pattern: 必须给 bbox[x1,y1,x2,y2], brief。
  - 每项必须额外给 stability 和 discrimination。
    - stability 表示该元素在同类型页面中不变化的程度，只能是 high/mid/low。
    - discrimination 表示该元素能区分当前页面的能力，只能是 high/mid/low。
    - 具体事件名、奖励名、祝福名、长正文通常 stability=low 或 mid，不要高估。
    - 页面标题、固定图标、固定按钮、固定交互控件通常更稳定。
- 注意将标志性的、固定出现且不易变化的元素排在前面，这些元素更可靠，将被优先用于状态匹配。
- slug: 英文小写+下划线，简短可读。
- possible_page_type: 如果当前页面可能属于已知页面类型，输出该类型英文名；否则输出 "none"。不要把具体实例名称当作页面类型。
- actions: 数组，每项仅允许两类：
  - 点击：{"type":"click","x":整数,"y":整数,"brief":"..."}
  - 预置动作：{"type":"run_preset","name":"wait_till_combat_end","brief":"..."}
    - 预置动作 wait_till_combat_end 若当前是战斗状态，则调用该动作，会挂起至战斗结束，此时自动战斗 
    - 预制动作 find_and_interact_with_next_object 会在当前场景寻找并移动至下一个可交互对象并与其交互,只要是在场景中需要与物体交互，都调用这个，包括与前方怪物战斗、与NPC、机关、门互动等
- 注意在3D场景中不要尝试点击物体触发交互，这没有任何效果，若发现需要在3D场景中需要与物体交互，请使用预置动作 find_and_interact_with_next_object。

输出字段：
{
  "page_summary": "...",
  "slug": "...",
  "possible_page_type": "none 或 已知/候选页面类型英文名",
  "elements": [...],
  "actions": [...]
}
"""

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


def _normalize_assistant_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in {"text", "output_text"}]
        return "".join(parts).strip()
    return ""


def _build_user_message_from_frame(image_rgb, text: str = "请按约定输出JSON") -> dict[str, Any]:
    data_url = DoubaoClient.encode_image_to_data_url(image_rgb)
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }


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


def _iter_state_meta() -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    if not FSM_TEMPLATES_DIR.exists():
        return out
    for d in sorted(FSM_TEMPLATES_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "state.json"
        if not meta_path.exists():
            continue
        try:
            out.append((d, _load_json(meta_path)))
        except Exception:
            continue
    return out


def _page_type_summaries(metas: list[tuple[Path, dict[str, Any]]], limit: int = 40) -> list[dict[str, Any]]:
    by_type: dict[str, dict[str, Any]] = {}
    for state_dir, meta in metas:
        ptype = _state_page_type(meta)
        entry = by_type.setdefault(
            ptype,
            {
                "page_type": ptype,
                "example_slug": str(meta.get("slug", "")),
                "description": str(meta.get("description", ""))[:160],
                "sample_count": 0,
                "latest_dir": str(state_dir),
            },
        )
        entry["sample_count"] = int(entry.get("sample_count", 0)) + 1
        try:
            if state_dir.stat().st_mtime >= Path(str(entry["latest_dir"])).stat().st_mtime:
                entry["example_slug"] = str(meta.get("slug", ""))
                entry["description"] = str(meta.get("description", ""))[:160]
                entry["latest_dir"] = str(state_dir)
        except Exception:
            pass
    return sorted(by_type.values(), key=lambda x: str(x.get("page_type", "")))[:limit]


def _states_for_page_type(metas: list[tuple[Path, dict[str, Any]]], page_type: str) -> list[tuple[Path, dict[str, Any]]]:
    norm = _normalize_page_type(page_type)
    if not norm:
        return []
    return [(d, m) for d, m in metas if _state_page_type(m) == norm]


def _latest_state_for_page_type(metas: list[tuple[Path, dict[str, Any]]], page_type: str) -> tuple[Path, dict[str, Any]] | None:
    states = _states_for_page_type(metas, page_type)
    if not states:
        return None
    return max(states, key=lambda item: item[0].stat().st_mtime)


def _sample_screenshot_paths(state_dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for d in state_dirs:
        shots = sorted(d.glob("screenshot_*.png"))
        if shots:
            out.extend(shots)
    return out


def _latest_screenshot_path(state_dir: Path) -> Path | None:
    shots = sorted(state_dir.glob("screenshot_*.png"))
    return shots[-1] if shots else None


@dataclass
class MatchResult:
    state_id: str
    state_dir: Path
    passed_enabled: int
    total_enabled: int
    success: bool
    passed_all: int
    total_all: int


def _ocr_lines(vision: VisionEngine, frame_rgb, rect: list[int] | None) -> list[str]:
    entries = vision.ocr(frame_rgb, rect, white_text=False)
    return [str(e.get("text", "")).strip() for e in entries if str(e.get("text", "")).strip()]


def _condition_passed(cond: dict[str, Any], vision: VisionEngine, frame_rgb) -> bool:
    kind = str(cond.get("kind", ""))
    params = cond.get("params", {})
    if not isinstance(params, dict):
        params = {}
    rect = params.get("rect")
    if not (isinstance(rect, list) and len(rect) == 4):
        bbox = cond.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            rect = bbox
    if kind == "text_line_contains":
        needle = str(params.get("text", "")).strip()
        if not needle:
            return False
        lines = _ocr_lines(vision, frame_rgb, rect)
        return any(needle in line for line in lines)
    if kind == "region_template":
        tpl_path = str(params.get("template_path", "")).strip()
        if not tpl_path:
            return False
        threshold = float(params.get("threshold", 0.8))
        ok, _, _ = vision.match_template(frame_rgb, Path(tpl_path), rect, threshold=threshold)
        return ok
    return False


def _level(raw: Any, default: str = "mid") -> str:
    value = str(raw or default).strip().lower()
    return value if value in {"high", "mid", "low"} else default


def _discrimination_score(cond: dict[str, Any]) -> int:
    return DISCRIMINATION_SCORE[_level(cond.get("discrimination"), "mid")]


def _stability_score(cond: dict[str, Any]) -> int:
    return STABILITY_SCORE[_level(cond.get("stability"), "mid")]


def _condition_sort_key(cond: dict[str, Any]) -> tuple[int, int]:
    return (_stability_score(cond), _discrimination_score(cond))


def _select_enabled_conditions(
    conditions: list[dict[str, Any]],
    vision: VisionEngine,
    frame_rgb,
) -> tuple[list[dict[str, Any]], bool]:
    valid = [c for c in conditions if c.get("kind") in {"text_line_contains", "region_template"}]
    passed = [c for c in valid if _condition_passed(c, vision, frame_rgb)]
    for c in conditions:
        c["enabled"] = False
    if not passed:
        return conditions, False

    ordered = sorted(passed, key=_condition_sort_key, reverse=True)
    selected: list[dict[str, Any]] = []
    score_sum = 0
    for c in ordered:
        selected.append(c)
        score_sum += _discrimination_score(c)
        if score_sum >= MATCH_DISCRIMINATION_TARGET:
            break
    if score_sum < MATCH_DISCRIMINATION_TARGET:
        selected = ordered

    weak = len(selected) == 1 and score_sum >= MATCH_DISCRIMINATION_TARGET
    selected_ids = {id(c) for c in selected}
    for c in conditions:
        c["enabled"] = id(c) in selected_ids
    return conditions, weak


def _eval_state_match(meta: dict[str, Any], state_dir: Path, vision: VisionEngine, frame_rgb) -> MatchResult:
    conds = [c for c in meta.get("match_conditions", []) if isinstance(c, dict)]
    enabled_conds = [c for c in conds if c.get("enabled", False)]
    passed_enabled = sum(1 for c in enabled_conds if _condition_passed(c, vision, frame_rgb))
    passed_all = sum(1 for c in conds if _condition_passed(c, vision, frame_rgb))
    total_enabled = len(enabled_conds)
    total_all = len(conds)
    success = (passed_enabled == total_enabled) if total_enabled > 0 else False
    return MatchResult(
        state_id=str(meta.get("state_id", "")),
        state_dir=state_dir,
        passed_enabled=passed_enabled,
        total_enabled=total_enabled,
        success=success,
        passed_all=passed_all,
        total_all=total_all,
    )


def _select_best_for_unknown(matches: list[MatchResult]) -> MatchResult | None:
    cands = [m for m in matches if m.success]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    # tie-break with all conditions pass count
    cands_sorted = sorted(cands, key=lambda m: (-m.passed_all, -m.passed_enabled))
    return cands_sorted[0]


def _find_match_by_state(matches: list[MatchResult], state_id: str) -> MatchResult | None:
    for m in matches:
        if m.state_id == state_id:
            return m
    return None


def _save_frame(path: Path, frame_rgb) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))


def _load_frame(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _parse_llm_payload(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    required = {"page_summary", "slug", "elements", "actions"}
    if not required.issubset(payload.keys()):
        return None
    payload.setdefault("possible_page_type", "none")
    return payload


def _parse_llm_repair(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    required = {"judgement", "mode", "actions"}
    if not required.issubset(payload.keys()):
        return None
    if payload.get("mode") not in {"override", "append"}:
        return None
    if payload.get("judgement") not in {"invalid", "partial"}:
        return None
    if not isinstance(payload.get("actions"), list):
        return None
    return payload


def _conditions_from_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, e in enumerate(elements):
        if not isinstance(e, dict):
            continue
        etype = e.get("type")
        bbox = e.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        if etype == "text_line":
            text = str(e.get("text", "")).strip()
            if not text:
                continue
            out.append(
                {
                    "id": f"elem_text_{i+1}",
                    "enabled": True,
                    "kind": "text_line_contains",
                    "params": {"text": text, "rect": bbox},
                    "weight": 1.0,
                    "brief": str(e.get("brief", "")),
                    "role": str(e.get("role", "")),
                    "stability": _level(e.get("stability"), "mid"),
                    "discrimination": _level(e.get("discrimination"), "mid"),
                    "condition_status": "active",
                    "bbox": bbox,
                }
            )
        elif etype == "pattern":
            out.append(
                {
                    "id": f"elem_pattern_{i+1}",
                    "enabled": True,
                    "kind": "region_template",
                    "params": {"rect": bbox, "threshold": 0.8},
                    "weight": 1.0,
                    "brief": str(e.get("brief", "")),
                    "role": str(e.get("role", "")),
                    "stability": _level(e.get("stability"), "mid"),
                    "discrimination": _level(e.get("discrimination"), "mid"),
                    "condition_status": "active",
                    "bbox": bbox,
                }
            )
    return out


def _extract_region_templates(frame_rgb, mapper: CoordinateMapper, state_dir: Path, conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    t_idx = 1
    for c in conditions:
        cc = dict(c)
        if cc.get("kind") == "region_template":
            rect = cc.get("params", {}).get("rect")
            if isinstance(rect, list) and len(rect) == 4:
                x, y, w, h = mapper.rect_to_real(rect)
                crop = frame_rgb[y : y + h, x : x + w]
                tpath = state_dir / f"template_{t_idx}.png"
                t_idx += 1
                cv2.imwrite(str(tpath), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                cc["params"]["template_path"] = str(tpath)
                cc["source"] = {
                    "screenshot_path": str(state_dir / "screenshot_1.png"),
                    "bbox_logical": rect,
                    "template_path": str(tpath),
                }
        out.append(cc)
    return out


def _limit_enable_conditions(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enabled_count = 0
    for c in conditions:
        valid = c.get("kind") in {"text_line_contains", "region_template"}
        if valid and enabled_count < 2:
            c["enabled"] = True
            enabled_count += 1
        else:
            c["enabled"] = False
    return conditions


def _ensure_unique_state_dir(slug: str) -> Path:
    base = _slugify(slug)
    d = FSM_TEMPLATES_DIR / base
    n = 1
    while d.exists():
        n += 1
        d = FSM_TEMPLATES_DIR / f"{base}_{n}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def _normalize_actions(raw_actions: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for a in raw_actions:
        if not isinstance(a, dict):
            continue
        atype = str(a.get("type", "click"))
        brief = str(a.get("brief", ""))
        if atype == "run_preset":
            out.append({"type": "run_preset", "name": str(a.get("name", "")), "brief": brief})
            continue
        x = int(a.get("x", 500))
        y = int(a.get("y", 500))
        out.append({"type": "click", "x": x, "y": y, "brief": brief})
    return out


def _create_state_from_llm(
    llm_payload: dict[str, Any],
    frame_rgb,
    mapper: CoordinateMapper,
    vision: VisionEngine,
) -> tuple[str, Path]:
    state_id = uuid.uuid4().hex
    state_dir = _ensure_unique_state_dir(str(llm_payload.get("slug", "state")))
    _save_frame(state_dir / "screenshot_1.png", frame_rgb)
    conds = _conditions_from_elements(llm_payload.get("elements", []))
    conds = _extract_region_templates(frame_rgb, mapper, state_dir, conds)
    conds, weak_match = _select_enabled_conditions(conds, vision, frame_rgb)

    state_meta = {
        "schema_version": SCHEMA_VERSION,
        "state_id": state_id,
        "slug": _slugify(str(llm_payload.get("slug", "state"))),
        "page_type": _normalize_page_type(llm_payload.get("possible_page_type") or llm_payload.get("page_type")),
        "display_name": str(llm_payload.get("slug", "state")),
        "description": str(llm_payload.get("page_summary", "")),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "model_info": {"source": "llm", "weak_match": weak_match},
        "elements": llm_payload.get("elements", []),
        "match_conditions": conds,
        "actions": [
            {
                "action_id": "action_main",
                "version": 1,
                "enabled": True,
                "cooldown_ms": 0,
                "steps": _normalize_actions(llm_payload.get("actions", [])),
            }
        ],
    }
    _save_json(state_dir / "state.json", state_meta)
    return state_id, state_dir


def _append_graph_node(state_id: str, slug: str) -> None:
    graph = _load_json(FSM_GRAPH_PATH)
    graph["nodes"].append({"state_id": state_id, "slug": slug, "enabled": True})
    graph["updated_at"] = _now_iso()
    _save_json(FSM_GRAPH_PATH, graph)


def _append_graph_edge(from_state_id: str | None, action_id: str, to_state_id: str) -> None:
    if not from_state_id:
        return
    graph = _load_json(FSM_GRAPH_PATH)
    for e in graph.get("edges", []):
        if e.get("from_state_id") == from_state_id and e.get("action_id") == action_id and e.get("to_state_id") == to_state_id:
            return
    graph["edges"].append(
        {
            "from_state_id": from_state_id,
            "action_id": action_id,
            "to_state_id": to_state_id,
            "enabled": True,
            "weight": 1.0,
            "created_at": _now_iso(),
        }
    )
    graph["updated_at"] = _now_iso()
    _save_json(FSM_GRAPH_PATH, graph)


def _get_reachable_targets(graph: dict[str, Any], from_state_id: str | None) -> set[str]:
    if not from_state_id:
        return set()
    return {
        str(e.get("to_state_id"))
        for e in graph.get("edges", [])
        if e.get("enabled", True) and str(e.get("from_state_id")) == from_state_id
    }


def _save_llm_raw_debug(session_id: str, attempt: int, text: str, kind: str = "normal") -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    p = FSM_DEBUG_DIR / f"llm_raw_{kind}_{session_id}_attempt{attempt}_{ts}.txt"
    p.write_text(text, encoding="utf-8")


def _apply_reasoning_effort(llm: DoubaoClient, effort: str | None) -> str | None:
    old = llm.reasoning_effort
    llm.reasoning_effort = effort
    return old


def _request_llm_payload(
    llm: DoubaoClient,
    session_id: str,
    frame_rgb,
    system_prompt: str,
    page_summaries: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    context = {
        "mode": "NORMAL",
        "known_page_types": page_summaries or [],
        "instruction": (
            "请按 system 约定输出 JSON。先照常输出用于建立新状态的页面元素信息；"
            "possible_page_type 必须从 known_page_types.page_type 中选择，若都不像则输出 none。"
        ),
    }
    msg = _build_user_message_from_frame(frame_rgb, json.dumps(context, ensure_ascii=False))
    prompt_fix = "上次JSON无效或不完整。只输出一个完整JSON对象，不要省略字段，不要解释文字。"
    for attempt in range(1, LLM_PARSE_RETRY + 2):
        payload_msg = msg if attempt == 1 else {"role": "user", "content": [{"type": "text", "text": prompt_fix}]}
        resp = llm.chat_with_session(
            session_id=session_id,
            system_prompt=system_prompt,
            user_message=payload_msg,
            tools=[],
            tool_choice="none",
        )
        text = _normalize_assistant_text(resp["choices"][0]["message"].get("content"))
        _save_llm_raw_debug(session_id, attempt, text, "normal")
        print(f"[fsm][llm] raw(attempt={attempt})={text[:600]}")
        parsed = _parse_llm_payload(text)
        if parsed is not None:
            return parsed
    return None


def _build_user_message_from_two_frames(image_a_rgb, image_b_rgb, text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": DoubaoClient.encode_image_to_data_url(image_a_rgb)}},
            {"type": "image_url", "image_url": {"url": DoubaoClient.encode_image_to_data_url(image_b_rgb)}},
        ],
    }


def _parse_llm_condition_revision(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("same_page_type") not in {True, False}:
        return None
    if payload.get("same_page_type") is True and not isinstance(payload.get("conditions"), list):
        return None
    return payload


def _request_llm_condition_revision(
    llm: DoubaoClient,
    session_id: str,
    system_prompt: str,
    *,
    latest_sample_rgb,
    current_rgb,
    page_type: str,
    latest_meta: dict[str, Any],
    failed_conditions: list[dict[str, Any]],
    new_payload: dict[str, Any],
) -> dict[str, Any] | None:
    text = json.dumps(
        {
            "mode": "MERGE_CONDITION_REVISION",
            "page_type": page_type,
            "instruction": (
                "第一张图是该 page_type 最新样板截图，第二张图是当前新截图。"
                "请判断第二张是否仍属于同一 page_type。若不是，same_page_type=false。"
                "若是，请逐个修订当前正式条件，使其更像同类页面共有条件。"
                "文本条件只支持 line_contains_text；请提取稳定共有子串，避免具体实例名、奖励名、长正文。"
                "模板条件可修订 rect/threshold；若该元素不是共有元素，请 deprecate。"
                "输出严格 JSON。"
            ),
            "latest_state": {
                "slug": latest_meta.get("slug"),
                "description": latest_meta.get("description"),
                "conditions": latest_meta.get("match_conditions", []),
            },
            "failed_conditions_on_current": failed_conditions,
            "new_page_payload_summary": {
                "slug": new_payload.get("slug"),
                "page_summary": new_payload.get("page_summary"),
                "elements": new_payload.get("elements", []),
            },
            "output_schema": {
                "same_page_type": "boolean",
                "conditions": [
                    {
                        "condition_id": "existing id",
                        "decision": "keep|revise|deprecate",
                        "kind": "text_line_contains|region_template",
                        "params": {},
                        "brief": "string",
                        "stability": "high|mid|low",
                        "discrimination": "high|mid|low",
                    }
                ],
                "note": "string",
            },
        },
        ensure_ascii=False,
    )
    for attempt in range(1, LLM_PARSE_RETRY + 2):
        msg = _build_user_message_from_two_frames(latest_sample_rgb, current_rgb, text)
        resp = llm.chat_with_session(
            session_id=session_id,
            system_prompt=system_prompt,
            user_message=msg,
            tools=[],
            tool_choice="none",
        )
        raw = _normalize_assistant_text(resp["choices"][0]["message"].get("content"))
        _save_llm_raw_debug(session_id, attempt, raw, "merge_condition")
        print(f"[fsm][merge][llm] raw(attempt={attempt})={raw[:600]}")
        parsed = _parse_llm_condition_revision(raw)
        if parsed is not None:
            return parsed
    return None


def _condition_from_revision(raw: dict[str, Any], base: dict[str, Any], state_dir: Path, frame_rgb, mapper: CoordinateMapper) -> dict[str, Any] | None:
    decision = str(raw.get("decision", "keep")).strip().lower()
    if decision == "deprecate":
        cc = dict(base)
        cc["condition_status"] = "deprecated"
        cc["enabled"] = False
        return cc
    if decision not in {"keep", "revise"}:
        return None
    cc = dict(base)
    cc["condition_status"] = "active"
    cc["enabled"] = False
    cc["stability"] = _level(raw.get("stability", cc.get("stability")), "mid")
    cc["discrimination"] = _level(raw.get("discrimination", cc.get("discrimination")), "mid")
    cc["brief"] = str(raw.get("brief", cc.get("brief", "")))
    if decision == "revise":
        kind = str(raw.get("kind", cc.get("kind", ""))).strip()
        if kind == "line_contains_text":
            kind = "text_line_contains"
        params = raw.get("params", {})
        if not isinstance(params, dict):
            params = {}
        if kind == "text_line_contains":
            text = str(params.get("text") or params.get("contains") or raw.get("text") or "").strip()
            rect = params.get("rect") or raw.get("bbox") or cc.get("params", {}).get("rect") or cc.get("bbox")
            if not text or not (isinstance(rect, list) and len(rect) == 4):
                return None
            cc["kind"] = kind
            cc["params"] = {"text": text, "rect": rect}
            cc["bbox"] = rect
        elif kind == "region_template":
            rect = params.get("rect") or raw.get("bbox") or cc.get("params", {}).get("rect") or cc.get("bbox")
            if not (isinstance(rect, list) and len(rect) == 4):
                return None
            x, y, w, h = mapper.rect_to_real(rect)
            crop = frame_rgb[y : y + h, x : x + w]
            tpath = state_dir / f"template_revised_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{cc.get('id', 'cond')}.png"
            cv2.imwrite(str(tpath), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            cc["kind"] = kind
            cc["params"] = {
                "rect": rect,
                "threshold": float(params.get("threshold", cc.get("params", {}).get("threshold", 0.8))),
                "template_path": str(tpath),
            }
            cc["bbox"] = rect
            cc["source"] = {
                "screenshot_path": str(state_dir / "screenshot_latest_merge.png"),
                "bbox_logical": rect,
                "template_path": str(tpath),
            }
        else:
            return None
    return cc


def _conditions_pass_all(conditions: list[dict[str, Any]], vision: VisionEngine, frames: list[Any]) -> bool:
    active = [c for c in conditions if c.get("condition_status", "active") == "active"]
    if not active:
        return False
    for frame in frames:
        for cond in active:
            if not _condition_passed(cond, vision, frame):
                return False
    return True


def _selected_conditions_match_other_page(
    conditions: list[dict[str, Any]],
    vision: VisionEngine,
    metas: list[tuple[Path, dict[str, Any]]],
    page_type: str,
) -> str | None:
    enabled = [c for c in conditions if c.get("enabled", False)]
    if not enabled:
        return None
    for state_dir, meta in metas:
        if _state_page_type(meta) == page_type:
            continue
        shot = _latest_screenshot_path(state_dir)
        if shot is None:
            continue
        frame = _load_frame(shot)
        if frame is None:
            continue
        if all(_condition_passed(c, vision, frame) for c in enabled):
            return f"{_state_page_type(meta)}:{meta.get('slug', '')}"
    return None


def _try_merge_page_type(
    *,
    llm: DoubaoClient,
    session_id: str,
    system_prompt: str,
    frame_rgb,
    llm_payload: dict[str, Any],
    mapper: CoordinateMapper,
    vision: VisionEngine,
    metas: list[tuple[Path, dict[str, Any]]],
) -> tuple[str, Path] | None:
    page_type = _normalize_page_type(llm_payload.get("possible_page_type"))
    if not page_type:
        return None
    latest = _latest_state_for_page_type(metas, page_type)
    if latest is None:
        return None
    latest_dir, latest_meta = latest
    sample_path = _latest_screenshot_path(latest_dir)
    if sample_path is None:
        return None
    latest_rgb = _load_frame(sample_path)
    if latest_rgb is None:
        return None

    existing_conds = [c for c in latest_meta.get("match_conditions", []) if isinstance(c, dict)]
    failed = [c for c in existing_conds if c.get("condition_status", "active") != "deprecated" and not _condition_passed(c, vision, frame_rgb)]
    rev = _request_llm_condition_revision(
        llm,
        session_id,
        system_prompt,
        latest_sample_rgb=latest_rgb,
        current_rgb=frame_rgb,
        page_type=page_type,
        latest_meta=latest_meta,
        failed_conditions=failed,
        new_payload=llm_payload,
    )
    if rev is None or not rev.get("same_page_type", False):
        print(f"[fsm][merge] reject page_type={page_type} reason=llm_not_same_or_invalid")
        return None

    by_id = {str(c.get("id", "")): c for c in existing_conds}
    revised: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in rev.get("conditions", []):
        if not isinstance(item, dict):
            continue
        cid = str(item.get("condition_id") or item.get("id") or "").strip()
        base = by_id.get(cid)
        if base is None:
            continue
        new_cond = _condition_from_revision(item, base, latest_dir, frame_rgb, mapper)
        if new_cond is not None:
            revised.append(new_cond)
            seen_ids.add(cid)
    for cond in existing_conds:
        if str(cond.get("id", "")) not in seen_ids:
            cc = dict(cond)
            if not cc.get("enabled", False):
                cc["condition_status"] = "deprecated"
            revised.append(cc)

    sample_frames: list[Any] = []
    for shot in _sample_screenshot_paths([d for d, _ in _states_for_page_type(metas, page_type)]):
        img = _load_frame(shot)
        if img is not None:
            sample_frames.append(img)
    sample_frames.append(frame_rgb)
    if not _conditions_pass_all(revised, vision, sample_frames):
        print(f"[fsm][merge] reject page_type={page_type} reason=revised_conditions_do_not_cover_positive_samples")
        return None

    revised, weak_match = _select_enabled_conditions(revised, vision, frame_rgb)
    false_positive = _selected_conditions_match_other_page(revised, vision, metas, page_type)
    if false_positive:
        print(f"[fsm][merge] reject page_type={page_type} reason=false_positive other={false_positive}")
        return None

    _backup_json(latest_dir / "state.json")
    next_idx = len(list(latest_dir.glob("screenshot_*.png"))) + 1
    _save_frame(latest_dir / f"screenshot_{next_idx}.png", frame_rgb)
    latest_meta["match_conditions"] = revised
    latest_meta["updated_at"] = _now_iso()
    latest_meta["page_type"] = page_type
    latest_meta.setdefault("model_info", {})
    if isinstance(latest_meta["model_info"], dict):
        latest_meta["model_info"]["weak_match"] = weak_match
        latest_meta["model_info"]["last_merge_note"] = str(rev.get("note", ""))
    _save_json(latest_dir / "state.json", latest_meta)
    print(f"[fsm][merge] accepted page_type={page_type} state_id={latest_meta.get('state_id')} weak={weak_match}")
    return str(latest_meta.get("state_id", "")), latest_dir


def _request_llm_repair(
    llm: DoubaoClient,
    session_id: str,
    frame_rgb,
    system_prompt: str,
    effort: str,
    state_slug: str,
    last_actions: list[dict[str, Any]],
    fail_index: int,
) -> dict[str, Any] | None:
    old_effort = _apply_reasoning_effort(llm, effort)
    text = (
        "MODE=REPAIR\n"
        "输出严格 JSON：{\"judgement\":\"invalid|partial\",\"mode\":\"override|append\",\"actions\":[...],\"note\":\"...\"}\n"
        f"当前状态: {state_slug}\n"
        f"已执行动作: {json.dumps(last_actions, ensure_ascii=False)}\n"
        f"结果: 执行后仍停留原状态（第{fail_index}次纠错）\n"
        "请判断上一步动作是无效还是部分有效，并给出纠正动作。"
    )
    try:
        for attempt in range(1, LLM_PARSE_RETRY + 2):
            msg = _build_user_message_from_frame(frame_rgb, text)
            resp = llm.chat_with_session(
                session_id=session_id,
                system_prompt=system_prompt,
                user_message=msg,
                tools=[],
                tool_choice="none",
            )
            raw = _normalize_assistant_text(resp["choices"][0]["message"].get("content"))
            _save_llm_raw_debug(session_id, attempt, raw, f"repair_{effort}")
            print(f"[fsm][repair][llm] effort={effort} attempt={attempt} raw={raw[:600]}")
            parsed = _parse_llm_repair(raw)
            if parsed is not None:
                parsed["actions"] = _normalize_actions(parsed.get("actions", []))
                return parsed
    finally:
        llm.reasoning_effort = old_effort
    return None


def _execute_action_steps(
    emulator: EmulatorClient,
    mapper: CoordinateMapper,
    vision: VisionEngine,
    state_dir: Path,
    steps: list[dict[str, Any]],
    state_id: str,
    matches_provider,
) -> bool:
    for idx, step in enumerate(steps, start=1):
        stype = str(step.get("type", "click"))
        if stype == "run_preset":
            name = str(step.get("name", ""))
            print(f"[fsm][action][preset] step={idx} name={name}")
            ok = run_preset(
                name,
                emulator=emulator,
                mapper=mapper,
                vision=vision,
                state_id=state_id,
                matches_provider=matches_provider,
                find_match_by_state=_find_match_by_state,
            )
            if not ok:
                return False
            continue

        x = int(step.get("x", 500))
        y = int(step.get("y", 500))
        brief = str(step.get("brief", "")).strip()
        rx, ry = mapper.point_to_real(x, y)
        print(f"[fsm][action][click] step={idx} logical=({x},{y}) real=({rx},{ry}) brief={brief}")
        emulator.tap(rx, ry)
        print(f"[fsm][action][wait] sleep={ACTION_CLICK_WAIT_S}s after step={idx}")
        time.sleep(ACTION_CLICK_WAIT_S)
    return True


def _apply_repair_result(action_steps: list[dict[str, Any]], repair: dict[str, Any]) -> list[dict[str, Any]]:
    new_steps = repair.get("actions", [])
    if repair.get("mode") == "override":
        return list(new_steps)
    return list(action_steps) + list(new_steps)


def _maybe_rotate_session(runtime: dict[str, Any]) -> bool:
    turns = int(runtime.get("llm_turn_count", 0))
    tier = str(runtime.get("session_tier", "soft"))
    if turns >= FORCE_RESET_AT:
        runtime["pending_refresh"] = True
        return True
    if tier == "soft" and turns >= SOFT_LIMIT and runtime.get("last_transition_ok", False):
        runtime["session_tier"] = "mid"
        runtime["pending_refresh"] = True
        return True
    if tier == "mid" and turns >= MID_LIMIT and runtime.get("last_transition_ok", False):
        runtime["session_tier"] = "hard"
        runtime["pending_refresh"] = True
        return True
    if tier == "hard" and turns >= HARD_LIMIT:
        runtime["pending_refresh"] = True
        return True
    return False


def _refresh_session_if_needed(runtime: dict[str, Any], base_session_id: str) -> str:
    if not runtime.get("llm_session_id") or runtime.get("pending_refresh", False):
        runtime["llm_session_id"] = f"{base_session_id}-fsm-{uuid.uuid4().hex[:8]}"
        runtime["llm_turn_count"] = 0
        runtime["pending_refresh"] = False
        runtime["last_transition_ok"] = False
    return str(runtime["llm_session_id"])


def _execute_state_action(
    *,
    emulator: EmulatorClient,
    mapper: CoordinateMapper,
    vision: VisionEngine,
    state_id: str,
    state_dir: Path,
    frame_before,
    matches_provider,
    runtime: dict[str, Any],
    llm: DoubaoClient,
    llm_session_id: str,
    system_prompt: str,
    graph: dict[str, Any],
    prefer_reachable_first: bool,
) -> tuple[bool, Any]:
    state_meta = _load_json(state_dir / "state.json")
    actions = [a for a in state_meta.get("actions", []) if isinstance(a, dict) and a.get("enabled", True)]
    if not actions:
        print(f"[fsm][reason] action not triggered: state={state_id} has no enabled actions")
        return False, frame_before
    action = actions[0]
    action_id = str(action.get("action_id", "action_main"))
    steps = action.get("steps", [])
    if not isinstance(steps, list) or not steps:
        print(f"[fsm][reason] action not triggered: state={state_id} action={action_id} has empty steps")
        return False, frame_before

    # normal attempts
    for attempt in range(1, MAX_RETRY_PER_ACTION + 1):
        print(f"[fsm][action] state={state_id} action={action_id} attempt={attempt}")
        if not _execute_action_steps(emulator, mapper, vision, state_dir, steps, state_id, matches_provider):
            return False, frame_before
        post = emulator.screenshot(prefer_png=True)
        matches = matches_provider(post)
        reachable = _get_reachable_targets(graph, state_id)
        nxt: MatchResult | None = None
        if prefer_reachable_first and reachable:
            reachable_hits = [m for m in matches if m.success and m.state_id in reachable]
            if reachable_hits:
                nxt = sorted(reachable_hits, key=lambda m: (-m.passed_all, -m.passed_enabled))[0]
                print(f"[fsm][transition][reachable] from={state_id} action={action_id} to={nxt.state_id}")
            else:
                print(f"[fsm][transition][reachable-miss] from={state_id} action={action_id} reachable={sorted(reachable)}")
        if nxt is None:
            unknown_pick = _select_best_for_unknown(matches)
            if unknown_pick is not None and unknown_pick.state_id != state_id:
                if unknown_pick.total_enabled > 0 and unknown_pick.passed_enabled == unknown_pick.total_enabled:
                    nxt = unknown_pick
                    print(f"[fsm][transition][unknown-fallback] from={state_id} action={action_id} to={nxt.state_id}")
                    if nxt.state_id not in reachable:
                        _append_graph_edge(state_id, action_id, nxt.state_id)
                        print(f"[fsm][edge][added] from={state_id} action={action_id} to={nxt.state_id} reason=unknown-fallback")
                else:
                    print(
                        f"[fsm][transition][unknown-fallback-rejected] from={state_id} action={action_id} "
                        f"candidate={unknown_pick.state_id} enabled_pass={unknown_pick.passed_enabled}/{unknown_pick.total_enabled}"
                    )
        if nxt is not None and nxt.state_id != state_id:
            runtime["last_state_id"] = nxt.state_id
            runtime["last_transition_ok"] = True
            runtime["pending_from_state_id"] = None
            runtime["pending_action_id"] = None
            _save_runtime(runtime)
            return True, post
        curr_match = _find_match_by_state(matches, state_id)
        if curr_match is None or not curr_match.success:
            # Leave original state but no known target: treat as unknown transition, skip repair.
            runtime["last_state_id"] = None
            runtime["last_transition_ok"] = True
            runtime["pending_from_state_id"] = state_id
            runtime["pending_action_id"] = action_id
            _save_runtime(runtime)
            print(f"[fsm][transition][to-unknown] from={state_id} action={action_id} reason=original-state-no-longer-matched")
            return True, post

    # repair loop
    for idx, effort in enumerate(REPAIR_EFFORTS, start=1):
        post = emulator.screenshot(prefer_png=True)
        repair = _request_llm_repair(
            llm=llm,
            session_id=llm_session_id,
            frame_rgb=post,
            system_prompt=system_prompt,
            effort=effort,
            state_slug=str(state_meta.get("slug", state_id)),
            last_actions=steps,
            fail_index=idx,
        )
        runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0)) + 1
        _save_runtime(runtime)
        if repair is None:
            continue
        if not repair.get("actions"):
            print(f"[fsm][repair] skip empty actions effort={effort}")
            continue
        steps = _apply_repair_result(steps, repair)
        action["version"] = int(action.get("version", 1)) + 1
        action["steps"] = steps
        state_meta["updated_at"] = _now_iso()
        _save_json(state_dir / "state.json", state_meta)

        print(f"[fsm][repair] effort={effort} mode={repair.get('mode')} judgement={repair.get('judgement')}")
        if not _execute_action_steps(emulator, mapper, vision, state_dir, steps, state_id, matches_provider):
            continue
        post2 = emulator.screenshot(prefer_png=True)
        matches2 = matches_provider(post2)
        reachable2 = _get_reachable_targets(graph, state_id)
        nxt2: MatchResult | None = None
        if prefer_reachable_first and reachable2:
            reachable_hits2 = [m for m in matches2 if m.success and m.state_id in reachable2]
            if reachable_hits2:
                nxt2 = sorted(reachable_hits2, key=lambda m: (-m.passed_all, -m.passed_enabled))[0]
                print(f"[fsm][transition][reachable] from={state_id} action={action_id} to={nxt2.state_id} after_repair={effort}")
            else:
                print(f"[fsm][transition][reachable-miss] from={state_id} action={action_id} after_repair={effort}")
        if nxt2 is None:
            unknown_pick2 = _select_best_for_unknown(matches2)
            if unknown_pick2 is not None and unknown_pick2.state_id != state_id:
                if unknown_pick2.total_enabled > 0 and unknown_pick2.passed_enabled == unknown_pick2.total_enabled:
                    nxt2 = unknown_pick2
                    print(f"[fsm][transition][unknown-fallback] from={state_id} action={action_id} to={nxt2.state_id} after_repair={effort}")
                    if nxt2.state_id not in reachable2:
                        _append_graph_edge(state_id, action_id, nxt2.state_id)
                        print(f"[fsm][edge][added] from={state_id} action={action_id} to={nxt2.state_id} reason=unknown-fallback-after-repair")
                else:
                    print(
                        f"[fsm][transition][unknown-fallback-rejected] from={state_id} action={action_id} after_repair={effort} "
                        f"candidate={unknown_pick2.state_id} enabled_pass={unknown_pick2.passed_enabled}/{unknown_pick2.total_enabled}"
                    )
        if nxt2 is not None and nxt2.state_id != state_id:
            runtime["last_state_id"] = nxt2.state_id
            runtime["repair_fail_count"] = 0
            runtime["last_transition_ok"] = True
            runtime["pending_from_state_id"] = None
            runtime["pending_action_id"] = None
            _save_runtime(runtime)
            _append_experience(f"repair success: {state_meta.get('slug', state_id)} -> {nxt2.state_id}; mode={repair.get('mode')} effort={effort}")
            return True, post2
        curr_match2 = _find_match_by_state(matches2, state_id)
        if curr_match2 is None or not curr_match2.success:
            runtime["last_state_id"] = None
            runtime["repair_fail_count"] = 0
            runtime["last_transition_ok"] = True
            runtime["pending_from_state_id"] = state_id
            runtime["pending_action_id"] = action_id
            _save_runtime(runtime)
            _append_experience(f"repair success(to-unknown): {state_meta.get('slug', state_id)}; mode={repair.get('mode')} effort={effort}")
            print(f"[fsm][transition][to-unknown] from={state_id} action={action_id} after_repair={effort}")
            return True, post2

    runtime["repair_fail_count"] = int(runtime.get("repair_fail_count", 0)) + 1
    runtime["last_transition_ok"] = False
    _save_runtime(runtime)
    print("[fsm][repair] failed for low/mid/high, exiting")
    raise SystemExit(1)


def run_agent_loop_fsm(*, session_id: str, serial: str | None = None, adb_path: str | None = None, interval_s: float = 5.0) -> None:
    _ensure_fsm_resources()
    target_serial = resolve_target_serial(serial=serial, adb_path=adb_path, auto_connect=True)
    emulator = EmulatorClient(serial=target_serial, adb_path=adb_path)
    llm = DoubaoClient()
    mapper = CoordinateMapper(logical_w=1000, logical_h=1000, real_w=1280, real_h=720)
    vision = VisionEngine(mapper=mapper)

    runtime = _load_runtime()
    runtime["run_id"] = uuid.uuid4().hex[:8]
    runtime["last_state_id"] = None
    runtime["pending_refresh"] = True
    runtime.setdefault("pending_from_state_id", None)
    runtime.setdefault("pending_action_id", None)
    _save_runtime(runtime)

    prev_frame = None

    while True:
        llm_session_id = _refresh_session_if_needed(runtime, session_id)
        system_prompt = _build_system_prompt_with_experience()

        frame = emulator.screenshot(prefer_png=True)
        if frame.shape[1] != 1280 or frame.shape[0] != 720:
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

        metas = _iter_state_meta()

        def matches_provider(img):
            return [_eval_state_match(meta, sdir, vision, img) for sdir, meta in metas]

        matches = matches_provider(frame)
        dbg = [f"{m.state_id}:{m.passed_enabled}/{m.total_enabled}|all={m.passed_all}/{m.total_all}|ok={m.success}" for m in matches]
        print(f"[fsm][match] candidates={dbg}")

        best = _select_best_for_unknown(matches)
        if best is None:
            print("[fsm] unknown state, requesting llm")
            payload = _request_llm_payload(llm, llm_session_id, frame, system_prompt, _page_type_summaries(metas))
            runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0)) + 1
            _save_runtime(runtime)
            if payload is None:
                print("[fsm][llm] invalid payload")
                prev_frame = frame
                time.sleep(interval_s)
                continue
            merged = _try_merge_page_type(
                llm=llm,
                session_id=llm_session_id,
                system_prompt=system_prompt,
                frame_rgb=frame,
                llm_payload=payload,
                mapper=mapper,
                vision=vision,
                metas=metas,
            )
            if merged is None:
                new_state_id, new_state_dir = _create_state_from_llm(payload, frame, mapper, vision)
                _append_graph_node(new_state_id, str(payload.get("slug", "state")))
                print(f"[fsm] new_state state_id={new_state_id} dir={new_state_dir}")
            else:
                new_state_id, new_state_dir = merged
                print(f"[fsm] merged_state state_id={new_state_id} dir={new_state_dir}")
            pending_from = runtime.get("pending_from_state_id")
            pending_action = runtime.get("pending_action_id")
            if isinstance(pending_from, str) and pending_from and isinstance(pending_action, str) and pending_action:
                _append_graph_edge(pending_from, pending_action, new_state_id)
                print(
                    f"[fsm][edge][added] from={pending_from} action={pending_action} "
                    f"to={new_state_id} reason=pending-unknown-resolution"
                )
                runtime["pending_from_state_id"] = None
                runtime["pending_action_id"] = None
            runtime["last_state_id"] = new_state_id
            runtime["last_transition_ok"] = False
            _save_runtime(runtime)
            ok, post_frame = _execute_state_action(
                emulator=emulator,
                mapper=mapper,
                vision=vision,
                state_id=new_state_id,
                state_dir=new_state_dir,
                frame_before=frame,
                matches_provider=matches_provider,
                runtime=runtime,
                llm=llm,
                llm_session_id=llm_session_id,
                system_prompt=system_prompt,
                graph=_load_json(FSM_GRAPH_PATH),
                prefer_reachable_first=False,
            )
            prev_frame = post_frame if ok else frame
            _maybe_rotate_session(runtime)
            _save_runtime(runtime)
            time.sleep(interval_s)
            continue

        runtime["last_state_id"] = best.state_id
        runtime["last_transition_ok"] = False
        pending_from2 = runtime.get("pending_from_state_id")
        pending_action2 = runtime.get("pending_action_id")
        if isinstance(pending_from2, str) and pending_from2 and isinstance(pending_action2, str) and pending_action2:
            _append_graph_edge(pending_from2, pending_action2, best.state_id)
            print(
                f"[fsm][edge][added] from={pending_from2} action={pending_action2} "
                f"to={best.state_id} reason=pending-unknown-resolved-to-existing"
            )
            runtime["pending_from_state_id"] = None
            runtime["pending_action_id"] = None
        _save_runtime(runtime)

        ok, post_frame = _execute_state_action(
            emulator=emulator,
            mapper=mapper,
            vision=vision,
            state_id=best.state_id,
            state_dir=best.state_dir,
            frame_before=frame,
            matches_provider=matches_provider,
            runtime=runtime,
            llm=llm,
            llm_session_id=llm_session_id,
            system_prompt=system_prompt,
            graph=_load_json(FSM_GRAPH_PATH),
            prefer_reachable_first=True,
        )
        prev_frame = post_frame if ok else frame
        _maybe_rotate_session(runtime)
        _save_runtime(runtime)
        time.sleep(interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="FSM driven emulator agent loop")
    parser.add_argument("--session-id", default=f"session-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--adb-path", default=None)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    run_agent_loop_fsm(session_id=args.session_id, serial=args.serial, adb_path=args.adb_path, interval_s=args.interval)


if __name__ == "__main__":
    main()
