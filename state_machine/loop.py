from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from state_machine.constants import (
    ACTION_CLICK_WAIT_S,
    FORCE_RESET_AT,
    FSM_GRAPH_PATH,
    FSM_TEMPLATES_DIR,
    HARD_LIMIT,
    LOCAL_FLOW_MAX_STEPS,
    LOCAL_FLOW_NO_PROGRESS_LIMIT,
    LLM_PARSE_RETRY,
    MAX_RETRY_PER_ACTION,
    MID_LIMIT,
    REPAIR_EFFORTS,
    SCHEMA_VERSION,
    SCREEN_CHANGE_DIFF_THRESHOLD,
    SOFT_LIMIT,
)
from state_machine.io import (
    _append_experience,
    _backup_json,
    _build_system_prompt_with_experience,
    _ensure_fsm_resources,
    _load_frame,
    _load_json,
    _load_runtime,
    _normalize_page_type,
    _now_iso,
    _save_frame,
    _save_json,
    _save_runtime,
    _slugify,
    _state_page_type,
)
from state_machine.llm_tasks import (
    _apply_reasoning_effort,
    _build_user_message_from_frame,
    _normalize_assistant_text,
    _parse_llm_repair,
    _request_llm_condition_revision,
    _request_llm_payload,
    _save_llm_raw_debug,
)
from state_machine.logger import FsmRunLogger, summarize_match
from state_machine.matching import (
    MatchResult,
    _condition_passed,
    _find_match_by_state,
    _level,
    _select_best_for_unknown,
    _select_enabled_conditions,
    _eval_state_match,
)
from agent.llm_client import DoubaoClient
from state_machine.presets import run_preset
from sr_tools.adb import resolve_target_serial
from sr_tools.emulator import EmulatorClient


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)

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
    logger: FsmRunLogger | None = None,
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
    if logger is not None:
        logger.event(
            "state_created",
            state_id=state_id,
            state_dir=state_dir,
            slug=state_meta["slug"],
            page_type=state_meta["page_type"],
            weak_match=weak_match,
            elements_count=len(llm_payload.get("elements", [])),
            actions_count=len(llm_payload.get("actions", [])),
            conditions=conds,
            screenshot_path=state_dir / "screenshot_1.png",
        )
    return state_id, state_dir


def _append_graph_node(state_id: str, slug: str, logger: FsmRunLogger | None = None) -> None:
    graph = _load_json(FSM_GRAPH_PATH)
    graph["nodes"].append({"state_id": state_id, "slug": slug, "enabled": True})
    graph["updated_at"] = _now_iso()
    _save_json(FSM_GRAPH_PATH, graph)
    if logger is not None:
        logger.event("graph_node_added", state_id=state_id, slug=slug)


def _append_graph_edge(
    from_state_id: str | None,
    action_id: str,
    to_state_id: str,
    logger: FsmRunLogger | None = None,
    reason: str = "",
) -> None:
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
    if logger is not None:
        logger.event(
            "graph_edge_added",
            from_state_id=from_state_id,
            action_id=action_id,
            to_state_id=to_state_id,
            reason=reason,
        )


def _get_reachable_targets(graph: dict[str, Any], from_state_id: str | None) -> set[str]:
    if not from_state_id:
        return set()
    return {
        str(e.get("to_state_id"))
        for e in graph.get("edges", [])
        if e.get("enabled", True) and str(e.get("from_state_id")) == from_state_id
    }


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
    logger: FsmRunLogger | None = None,
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
        _log(logger, f"[fsm][merge] reject page_type={page_type} reason=llm_not_same_or_invalid", "merge_rejected", page_type=page_type, reason="llm_not_same_or_invalid")
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
        _log(logger, f"[fsm][merge] reject page_type={page_type} reason=revised_conditions_do_not_cover_positive_samples", "merge_rejected", page_type=page_type, reason="revised_conditions_do_not_cover_positive_samples")
        return None

    revised, weak_match = _select_enabled_conditions(revised, vision, frame_rgb)
    false_positive = _selected_conditions_match_other_page(revised, vision, metas, page_type)
    if false_positive:
        _log(logger, f"[fsm][merge] reject page_type={page_type} reason=false_positive other={false_positive}", "merge_rejected", page_type=page_type, reason="false_positive", other=false_positive)
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
    _log(
        logger,
        f"[fsm][merge] accepted page_type={page_type} state_id={latest_meta.get('state_id')} weak={weak_match}",
        "merge_accepted",
        page_type=page_type,
        state_id=latest_meta.get("state_id"),
        weak_match=weak_match,
        conditions=revised,
        screenshot_path=latest_dir / f"screenshot_{next_idx}.png",
    )
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
    logger: FsmRunLogger | None = None,
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
            _log(logger, f"[fsm][repair][llm] effort={effort} attempt={attempt} raw={raw[:600]}", "llm_repair_raw", effort=effort, attempt=attempt, raw_preview=raw[:600])
            parsed = _parse_llm_repair(raw)
            if parsed is not None:
                parsed["actions"] = _normalize_actions(parsed.get("actions", []))
                if logger is not None:
                    logger.event(
                        "llm_repair_parsed",
                        effort=effort,
                        attempt=attempt,
                        judgement=parsed.get("judgement"),
                        mode=parsed.get("mode"),
                        actions=parsed.get("actions", []),
                    )
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
    logger: FsmRunLogger | None = None,
    action_id: str = "",
    attempt: int | str = "",
) -> bool:
    for idx, step in enumerate(steps, start=1):
        stype = str(step.get("type", "click"))
        if stype == "run_preset":
            name = str(step.get("name", ""))
            _log(logger, f"[fsm][action][preset] step={idx} name={name}", "action_preset", state_id=state_id, action_id=action_id, attempt=attempt, step=idx, name=name)
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
        _log(
            logger,
            f"[fsm][action][click] step={idx} logical=({x},{y}) real=({rx},{ry}) brief={brief}",
            "action_click",
            state_id=state_id,
            action_id=action_id,
            attempt=attempt,
            step=idx,
            logical=[x, y],
            real=[rx, ry],
            brief=brief,
        )
        emulator.tap(rx, ry)
        _log(logger, f"[fsm][action][wait] sleep={ACTION_CLICK_WAIT_S}s after step={idx}", "action_wait", state_id=state_id, action_id=action_id, attempt=attempt, step=idx, sleep_s=ACTION_CLICK_WAIT_S)
        time.sleep(ACTION_CLICK_WAIT_S)
    return True


def _apply_repair_result(action_steps: list[dict[str, Any]], repair: dict[str, Any]) -> list[dict[str, Any]]:
    new_steps = repair.get("actions", [])
    if repair.get("mode") == "override":
        return list(new_steps)
    return list(action_steps) + list(new_steps)


def _screen_changed(before_rgb, after_rgb, threshold: float = SCREEN_CHANGE_DIFF_THRESHOLD) -> tuple[bool, float]:
    if before_rgb is None or after_rgb is None:
        return False, 0.0
    if before_rgb.shape != after_rgb.shape:
        return True, 1.0
    diff = cv2.absdiff(before_rgb, after_rgb)
    score = float(diff.mean()) / 255.0
    return score >= threshold, score


def _parse_llm_page_local_step(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    payload["done"] = bool(payload.get("done", False))
    action = payload.get("action")
    if action is None:
        payload["action"] = None
        return payload if payload["done"] else None
    if not isinstance(action, dict):
        return None
    normalized = _normalize_actions([action])
    if not normalized:
        return None
    payload["action"] = normalized[0]
    return payload


def _request_llm_page_local_step(
    llm: DoubaoClient,
    session_id: str,
    frame_rgb,
    system_prompt: str,
    *,
    state_slug: str,
    previous_steps: list[dict[str, Any]],
    no_progress_count: int,
    logger: FsmRunLogger | None = None,
) -> dict[str, Any] | None:
    text = json.dumps(
        {
            "mode": "PAGE_LOCAL_STEP",
            "instruction": (
                "当前正在处理同一页面内的多步流程。停留在同一状态不一定是失败；"
                "如果画面/文字/可操作对象仍在变化，说明流程可能正在推进。"
                "请基于当前截图给出下一步最小动作，或在应交还 FSM 重新识别时输出 done=true。"
                "不要重复 previous_steps 中已经证明无进展的动作。"
                "只输出严格 JSON。"
            ),
            "current_state": state_slug,
            "previous_steps": previous_steps[-8:],
            "no_progress_count": no_progress_count,
            "allowed_actions": [
                {"type": "click", "x": "integer", "y": "integer", "brief": "string"},
                {"type": "run_preset", "name": "wait_till_combat_end|find_and_interact_with_next_object", "brief": "string"},
            ],
            "output_schema": {
                "done": "boolean",
                "reason": "string",
                "action": {"type": "click|run_preset", "x": "integer optional", "y": "integer optional", "name": "string optional", "brief": "string"},
                "expected_effect": "screen_changes|text_changes|state_changes|overlay_appears|unknown",
            },
        },
        ensure_ascii=False,
    )
    prompt_fix = "上次JSON无效。只输出一个完整JSON对象，字段为 done, reason, action, expected_effect。"
    for attempt in range(1, LLM_PARSE_RETRY + 2):
        msg = _build_user_message_from_frame(frame_rgb, text if attempt == 1 else prompt_fix)
        resp = llm.chat_with_session(
            session_id=session_id,
            system_prompt=system_prompt,
            user_message=msg,
            tools=[],
            tool_choice="none",
        )
        raw = _normalize_assistant_text(resp["choices"][0]["message"].get("content"))
        _save_llm_raw_debug(session_id, attempt, raw, "page_local")
        _log(logger, f"[fsm][local][llm] attempt={attempt} raw={raw[:600]}", "llm_page_local_raw", attempt=attempt, raw_preview=raw[:600])
        parsed = _parse_llm_page_local_step(raw)
        if parsed is not None:
            if logger is not None:
                logger.event("llm_page_local_parsed", attempt=attempt, parsed=parsed)
            return parsed
    return None


def _resolve_transition_after_progress(
    *,
    state_id: str,
    action_id: str,
    matches: list[MatchResult],
    graph: dict[str, Any],
    runtime: dict[str, Any],
    prefer_reachable_first: bool,
    logger: FsmRunLogger | None,
    reason_suffix: str,
) -> MatchResult | None:
    reachable = _get_reachable_targets(graph, state_id)
    nxt: MatchResult | None = None
    if prefer_reachable_first and reachable:
        reachable_hits = [m for m in matches if m.success and m.state_id in reachable]
        if reachable_hits:
            nxt = sorted(reachable_hits, key=lambda m: (-m.passed_all, -m.passed_enabled))[0]
            _log(logger, f"[fsm][transition][reachable] from={state_id} action={action_id} to={nxt.state_id} {reason_suffix}", "transition", from_state=state_id, action_id=action_id, to_state=nxt.state_id, reason=f"reachable{reason_suffix}")
    if nxt is None:
        unknown_pick = _select_best_for_unknown(matches)
        if unknown_pick is not None and unknown_pick.state_id != state_id:
            if unknown_pick.total_enabled > 0 and unknown_pick.passed_enabled == unknown_pick.total_enabled:
                nxt = unknown_pick
                _log(logger, f"[fsm][transition][unknown-fallback] from={state_id} action={action_id} to={nxt.state_id} {reason_suffix}", "transition", from_state=state_id, action_id=action_id, to_state=nxt.state_id, reason=f"unknown-fallback{reason_suffix}")
                if nxt.state_id not in reachable:
                    _append_graph_edge(state_id, action_id, nxt.state_id, logger=logger, reason=f"unknown-fallback{reason_suffix}")
    if nxt is not None and nxt.state_id != state_id:
        runtime["last_state_id"] = nxt.state_id
        runtime["last_transition_ok"] = True
        runtime["pending_from_state_id"] = None
        runtime["pending_action_id"] = None
        _save_runtime(runtime)
        return nxt
    return None


def _run_page_local_flow(
    *,
    emulator: EmulatorClient,
    mapper: CoordinateMapper,
    vision: VisionEngine,
    state_id: str,
    state_dir: Path,
    action_id: str,
    start_frame,
    matches_provider,
    runtime: dict[str, Any],
    llm: DoubaoClient,
    llm_session_id: str,
    system_prompt: str,
    graph: dict[str, Any],
    prefer_reachable_first: bool,
    state_slug: str,
    logger: FsmRunLogger | None = None,
) -> tuple[bool, Any]:
    current = start_frame
    previous_steps: list[dict[str, Any]] = []
    no_progress_count = 0
    made_progress = False
    for step_idx in range(1, LOCAL_FLOW_MAX_STEPS + 1):
        decision = _request_llm_page_local_step(
            llm,
            llm_session_id,
            current,
            system_prompt,
            state_slug=state_slug,
            previous_steps=previous_steps,
            no_progress_count=no_progress_count,
            logger=logger,
        )
        runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0)) + 1
        _save_runtime(runtime)
        if decision is None:
            _log(logger, f"[fsm][local] no valid llm decision step={step_idx}", "page_local_invalid", state_id=state_id, step=step_idx)
            return made_progress, current
        if decision.get("done", False):
            _log(logger, f"[fsm][local] done step={step_idx} reason={decision.get('reason', '')}", "page_local_done", state_id=state_id, step=step_idx, reason=decision.get("reason", ""))
            return True, current
        action = decision.get("action")
        if not isinstance(action, dict):
            return made_progress, current
        previous_steps.append({"action": action, "reason": decision.get("reason", ""), "expected_effect": decision.get("expected_effect", "")})
        if not _execute_action_steps(
            emulator,
            mapper,
            vision,
            state_dir,
            [action],
            state_id,
            matches_provider,
            logger=logger,
            action_id=action_id,
            attempt=f"local:{step_idx}",
        ):
            return made_progress, current
        post = emulator.screenshot(prefer_png=True)
        changed, diff_score = _screen_changed(current, post)
        matches = matches_provider(post)
        if logger is not None:
            logger.event(
                "page_local_step_result",
                state_id=state_id,
                action_id=action_id,
                step=step_idx,
                changed=changed,
                diff_score=diff_score,
                candidates=[summarize_match(m) for m in matches],
            )
        nxt = _resolve_transition_after_progress(
            state_id=state_id,
            action_id=action_id,
            matches=matches,
            graph=graph,
            runtime=runtime,
            prefer_reachable_first=prefer_reachable_first,
            logger=logger,
            reason_suffix="-page-local",
        )
        if nxt is not None:
            return True, post
        curr_match = _find_match_by_state(matches, state_id)
        if curr_match is None or not curr_match.success:
            runtime["last_state_id"] = None
            runtime["last_transition_ok"] = True
            runtime["pending_from_state_id"] = state_id
            runtime["pending_action_id"] = action_id
            _save_runtime(runtime)
            _log(logger, f"[fsm][local][transition][to-unknown] from={state_id} action={action_id}", "transition", from_state=state_id, action_id=action_id, to_state=None, reason="page-local-original-state-no-longer-matched")
            return True, post
        if changed:
            made_progress = True
            no_progress_count = 0
            current = post
            continue
        no_progress_count += 1
        current = post
        if no_progress_count >= LOCAL_FLOW_NO_PROGRESS_LIMIT:
            _log(logger, f"[fsm][local] no progress limit hit state={state_id}", "page_local_no_progress", state_id=state_id, limit=LOCAL_FLOW_NO_PROGRESS_LIMIT)
            return made_progress, current
    _log(logger, f"[fsm][local] max steps reached state={state_id} progress={made_progress}", "page_local_max_steps", state_id=state_id, max_steps=LOCAL_FLOW_MAX_STEPS, made_progress=made_progress)
    return made_progress, current


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
    logger: FsmRunLogger | None = None,
) -> tuple[bool, Any]:
    state_meta = _load_json(state_dir / "state.json")
    actions = [a for a in state_meta.get("actions", []) if isinstance(a, dict) and a.get("enabled", True)]
    if not actions:
        _log(logger, f"[fsm][reason] action not triggered: state={state_id} has no enabled actions", "action_skipped", state_id=state_id, reason="no_enabled_actions")
        return False, frame_before
    action = actions[0]
    action_id = str(action.get("action_id", "action_main"))
    steps = action.get("steps", [])
    if not isinstance(steps, list) or not steps:
        _log(logger, f"[fsm][reason] action not triggered: state={state_id} action={action_id} has empty steps", "action_skipped", state_id=state_id, action_id=action_id, reason="empty_steps")
        return False, frame_before

    last_attempt_frame = frame_before
    # normal attempts
    for attempt in range(1, MAX_RETRY_PER_ACTION + 1):
        _log(logger, f"[fsm][action] state={state_id} action={action_id} attempt={attempt}", "action_attempt", state_id=state_id, action_id=action_id, attempt=attempt, steps=steps)
        if not _execute_action_steps(emulator, mapper, vision, state_dir, steps, state_id, matches_provider, logger=logger, action_id=action_id, attempt=attempt):
            return False, frame_before
        post = emulator.screenshot(prefer_png=True)
        matches = matches_provider(post)
        reachable = _get_reachable_targets(graph, state_id)
        if logger is not None:
            logger.event(
                "post_action_match",
                state_id=state_id,
                action_id=action_id,
                attempt=attempt,
                reachable=sorted(reachable),
                candidates=[summarize_match(m) for m in matches],
            )
        nxt: MatchResult | None = None
        if prefer_reachable_first and reachable:
            reachable_hits = [m for m in matches if m.success and m.state_id in reachable]
            if reachable_hits:
                nxt = sorted(reachable_hits, key=lambda m: (-m.passed_all, -m.passed_enabled))[0]
                _log(logger, f"[fsm][transition][reachable] from={state_id} action={action_id} to={nxt.state_id}", "transition", from_state=state_id, action_id=action_id, to_state=nxt.state_id, reason="reachable")
            else:
                _log(logger, f"[fsm][transition][reachable-miss] from={state_id} action={action_id} reachable={sorted(reachable)}", "transition_reachable_miss", from_state=state_id, action_id=action_id, reachable=sorted(reachable))
        if nxt is None:
            unknown_pick = _select_best_for_unknown(matches)
            if unknown_pick is not None and unknown_pick.state_id != state_id:
                if unknown_pick.total_enabled > 0 and unknown_pick.passed_enabled == unknown_pick.total_enabled:
                    nxt = unknown_pick
                    _log(logger, f"[fsm][transition][unknown-fallback] from={state_id} action={action_id} to={nxt.state_id}", "transition", from_state=state_id, action_id=action_id, to_state=nxt.state_id, reason="unknown-fallback")
                    if nxt.state_id not in reachable:
                        _append_graph_edge(state_id, action_id, nxt.state_id, logger=logger, reason="unknown-fallback")
                        _log(logger, f"[fsm][edge][added] from={state_id} action={action_id} to={nxt.state_id} reason=unknown-fallback", "edge_added", from_state=state_id, action_id=action_id, to_state=nxt.state_id, reason="unknown-fallback")
                else:
                    _log(
                        logger,
                        f"[fsm][transition][unknown-fallback-rejected] from={state_id} action={action_id} "
                        f"candidate={unknown_pick.state_id} enabled_pass={unknown_pick.passed_enabled}/{unknown_pick.total_enabled}",
                        "transition_rejected",
                        from_state=state_id,
                        action_id=action_id,
                        candidate=unknown_pick.state_id,
                        reason="unknown-fallback-enabled-miss",
                        passed_enabled=unknown_pick.passed_enabled,
                        total_enabled=unknown_pick.total_enabled,
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
            _log(logger, f"[fsm][transition][to-unknown] from={state_id} action={action_id} reason=original-state-no-longer-matched", "transition", from_state=state_id, action_id=action_id, to_state=None, reason="original-state-no-longer-matched")
            return True, post
        changed, diff_score = _screen_changed(last_attempt_frame, post)
        if changed:
            _log(
                logger,
                f"[fsm][local] same state but screen changed diff={diff_score:.4f}; enter page-local flow",
                "page_local_enter",
                state_id=state_id,
                action_id=action_id,
                attempt=attempt,
                diff_score=diff_score,
            )
            local_ok, local_frame = _run_page_local_flow(
                emulator=emulator,
                mapper=mapper,
                vision=vision,
                state_id=state_id,
                state_dir=state_dir,
                action_id=action_id,
                start_frame=post,
                matches_provider=matches_provider,
                runtime=runtime,
                llm=llm,
                llm_session_id=llm_session_id,
                system_prompt=system_prompt,
                graph=graph,
                prefer_reachable_first=prefer_reachable_first,
                state_slug=str(state_meta.get("slug", state_id)),
                logger=logger,
            )
            if local_ok:
                return True, local_frame
            break
        last_attempt_frame = post

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
            logger=logger,
        )
        runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0)) + 1
        _save_runtime(runtime)
        if repair is None:
            continue
        if not repair.get("actions"):
            _log(logger, f"[fsm][repair] skip empty actions effort={effort}", "repair_skipped", effort=effort, reason="empty_actions")
            continue
        steps = _apply_repair_result(steps, repair)
        action["version"] = int(action.get("version", 1)) + 1
        action["steps"] = steps
        state_meta["updated_at"] = _now_iso()
        _save_json(state_dir / "state.json", state_meta)

        _log(logger, f"[fsm][repair] effort={effort} mode={repair.get('mode')} judgement={repair.get('judgement')}", "repair_applied", effort=effort, mode=repair.get("mode"), judgement=repair.get("judgement"), steps=steps)
        if not _execute_action_steps(emulator, mapper, vision, state_dir, steps, state_id, matches_provider, logger=logger, action_id=action_id, attempt=f"repair:{effort}"):
            continue
        post2 = emulator.screenshot(prefer_png=True)
        matches2 = matches_provider(post2)
        reachable2 = _get_reachable_targets(graph, state_id)
        if logger is not None:
            logger.event(
                "post_repair_match",
                state_id=state_id,
                action_id=action_id,
                effort=effort,
                reachable=sorted(reachable2),
                candidates=[summarize_match(m) for m in matches2],
            )
        nxt2: MatchResult | None = None
        if prefer_reachable_first and reachable2:
            reachable_hits2 = [m for m in matches2 if m.success and m.state_id in reachable2]
            if reachable_hits2:
                nxt2 = sorted(reachable_hits2, key=lambda m: (-m.passed_all, -m.passed_enabled))[0]
                _log(logger, f"[fsm][transition][reachable] from={state_id} action={action_id} to={nxt2.state_id} after_repair={effort}", "transition", from_state=state_id, action_id=action_id, to_state=nxt2.state_id, reason="reachable-after-repair", effort=effort)
            else:
                _log(logger, f"[fsm][transition][reachable-miss] from={state_id} action={action_id} after_repair={effort}", "transition_reachable_miss", from_state=state_id, action_id=action_id, reachable=sorted(reachable2), effort=effort)
        if nxt2 is None:
            unknown_pick2 = _select_best_for_unknown(matches2)
            if unknown_pick2 is not None and unknown_pick2.state_id != state_id:
                if unknown_pick2.total_enabled > 0 and unknown_pick2.passed_enabled == unknown_pick2.total_enabled:
                    nxt2 = unknown_pick2
                    _log(logger, f"[fsm][transition][unknown-fallback] from={state_id} action={action_id} to={nxt2.state_id} after_repair={effort}", "transition", from_state=state_id, action_id=action_id, to_state=nxt2.state_id, reason="unknown-fallback-after-repair", effort=effort)
                    if nxt2.state_id not in reachable2:
                        _append_graph_edge(state_id, action_id, nxt2.state_id, logger=logger, reason="unknown-fallback-after-repair")
                        _log(logger, f"[fsm][edge][added] from={state_id} action={action_id} to={nxt2.state_id} reason=unknown-fallback-after-repair", "edge_added", from_state=state_id, action_id=action_id, to_state=nxt2.state_id, reason="unknown-fallback-after-repair")
                else:
                    _log(
                        logger,
                        f"[fsm][transition][unknown-fallback-rejected] from={state_id} action={action_id} after_repair={effort} "
                        f"candidate={unknown_pick2.state_id} enabled_pass={unknown_pick2.passed_enabled}/{unknown_pick2.total_enabled}",
                        "transition_rejected",
                        from_state=state_id,
                        action_id=action_id,
                        candidate=unknown_pick2.state_id,
                        reason="unknown-fallback-enabled-miss-after-repair",
                        effort=effort,
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
            _log(logger, f"[fsm][transition][to-unknown] from={state_id} action={action_id} after_repair={effort}", "transition", from_state=state_id, action_id=action_id, to_state=None, reason="to-unknown-after-repair", effort=effort)
            return True, post2
        changed2, diff_score2 = _screen_changed(post, post2)
        if changed2:
            _log(
                logger,
                f"[fsm][local] repair kept same state but screen changed diff={diff_score2:.4f}; enter page-local flow",
                "page_local_enter_after_repair",
                state_id=state_id,
                action_id=action_id,
                effort=effort,
                diff_score=diff_score2,
            )
            local_ok2, local_frame2 = _run_page_local_flow(
                emulator=emulator,
                mapper=mapper,
                vision=vision,
                state_id=state_id,
                state_dir=state_dir,
                action_id=action_id,
                start_frame=post2,
                matches_provider=matches_provider,
                runtime=runtime,
                llm=llm,
                llm_session_id=llm_session_id,
                system_prompt=system_prompt,
                graph=graph,
                prefer_reachable_first=prefer_reachable_first,
                state_slug=str(state_meta.get("slug", state_id)),
                logger=logger,
            )
            if local_ok2:
                runtime["repair_fail_count"] = 0
                _save_runtime(runtime)
                return True, local_frame2

    runtime["repair_fail_count"] = int(runtime.get("repair_fail_count", 0)) + 1
    runtime["last_transition_ok"] = False
    _save_runtime(runtime)
    _log(logger, "[fsm][repair] failed for low/mid/high, exiting", "repair_failed", state_id=state_id, action_id=action_id, repair_fail_count=runtime["repair_fail_count"])
    raise SystemExit(1)


def run_agent_loop_fsm(*, session_id: str, serial: str | None = None, adb_path: str | None = None, interval_s: float = 5.0) -> None:
    _ensure_fsm_resources()
    target_serial = resolve_target_serial(serial=serial, adb_path=adb_path, auto_connect=True)
    emulator = EmulatorClient(serial=target_serial, adb_path=adb_path)
    llm = DoubaoClient()
    mapper = CoordinateMapper(logical_w=1000, logical_h=1000, real_w=1280, real_h=720)
    vision = VisionEngine(mapper=mapper, log_ocr_calls=False)

    runtime = _load_runtime()
    runtime["run_id"] = uuid.uuid4().hex[:8]
    runtime["last_state_id"] = None
    runtime["pending_refresh"] = True
    runtime.setdefault("pending_from_state_id", None)
    runtime.setdefault("pending_action_id", None)
    _save_runtime(runtime)
    logger = FsmRunLogger(str(runtime["run_id"]))
    logger.event(
        "run_start",
        session_id=session_id,
        serial=target_serial,
        adb_path=adb_path,
        interval_s=interval_s,
        log_path=logger.path,
    )
    logger.text(f"[fsm][log] path={logger.path}", "log_path", log_path=logger.path)

    prev_frame = None

    while True:
        llm_session_id = _refresh_session_if_needed(runtime, session_id)
        system_prompt = _build_system_prompt_with_experience()
        logger.event(
            "loop_start",
            llm_session_id=llm_session_id,
            llm_turn_count=runtime.get("llm_turn_count", 0),
            session_tier=runtime.get("session_tier"),
            last_state_id=runtime.get("last_state_id"),
            pending_from_state_id=runtime.get("pending_from_state_id"),
            pending_action_id=runtime.get("pending_action_id"),
        )

        frame = emulator.screenshot(prefer_png=True)
        if frame.shape[1] != 1280 or frame.shape[0] != 720:
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        logger.event("frame_captured", width=int(frame.shape[1]), height=int(frame.shape[0]))

        metas = _iter_state_meta()

        def matches_provider(img):
            vision.reset_ocr_stats()
            matches = [_eval_state_match(meta, sdir, vision, img) for sdir, meta in metas]
            ocr_stats = vision.consume_ocr_stats()
            calls = int(ocr_stats["calls"])
            if calls > 0:
                elapsed_s = float(ocr_stats["elapsed_s"])
                logger.text(
                    f"[vision][ocr] summary calls={calls} successes={int(ocr_stats['successes'])} errors={int(ocr_stats['errors'])} elapsed={elapsed_s:.2f}s",
                    "ocr_summary",
                    calls=calls,
                    successes=int(ocr_stats["successes"]),
                    errors=int(ocr_stats["errors"]),
                    elapsed_s=elapsed_s,
                )
            return matches

        matches = matches_provider(frame)
        dbg = [f"{m.state_id}:{m.passed_enabled}/{m.total_enabled}|all={m.passed_all}/{m.total_all}|ok={m.success}" for m in matches]
        logger.text(f"[fsm][match] candidates={dbg}", "match_candidates", candidates=[summarize_match(m) for m in matches])

        best = _select_best_for_unknown(matches)
        if best is None:
            logger.text("[fsm] unknown state, requesting llm", "unknown_state")
            payload = _request_llm_payload(llm, llm_session_id, frame, system_prompt, _page_type_summaries(metas))
            runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0)) + 1
            _save_runtime(runtime)
            if payload is None:
                logger.text("[fsm][llm] invalid payload", "llm_payload_invalid", llm_session_id=llm_session_id)
                prev_frame = frame
                time.sleep(interval_s)
                continue
            logger.event(
                "llm_payload_parsed",
                slug=payload.get("slug"),
                possible_page_type=payload.get("possible_page_type"),
                elements_count=len(payload.get("elements", [])),
                actions_count=len(payload.get("actions", [])),
            )
            merged = _try_merge_page_type(
                llm=llm,
                session_id=llm_session_id,
                system_prompt=system_prompt,
                frame_rgb=frame,
                llm_payload=payload,
                mapper=mapper,
                vision=vision,
                metas=metas,
                logger=logger,
            )
            if merged is None:
                new_state_id, new_state_dir = _create_state_from_llm(payload, frame, mapper, vision, logger=logger)
                _append_graph_node(new_state_id, str(payload.get("slug", "state")), logger=logger)
                logger.text(f"[fsm] new_state state_id={new_state_id} dir={new_state_dir}", "state_created_console", state_id=new_state_id, state_dir=new_state_dir)
            else:
                new_state_id, new_state_dir = merged
                logger.text(f"[fsm] merged_state state_id={new_state_id} dir={new_state_dir}", "state_merged_console", state_id=new_state_id, state_dir=new_state_dir)
            pending_from = runtime.get("pending_from_state_id")
            pending_action = runtime.get("pending_action_id")
            if isinstance(pending_from, str) and pending_from and isinstance(pending_action, str) and pending_action:
                _append_graph_edge(pending_from, pending_action, new_state_id, logger=logger, reason="pending-unknown-resolution")
                logger.text(
                    f"[fsm][edge][added] from={pending_from} action={pending_action} "
                    f"to={new_state_id} reason=pending-unknown-resolution",
                    "edge_added",
                    from_state=pending_from,
                    action_id=pending_action,
                    to_state=new_state_id,
                    reason="pending-unknown-resolution",
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
                logger=logger,
            )
            prev_frame = post_frame if ok else frame
            _maybe_rotate_session(runtime)
            _save_runtime(runtime)
            time.sleep(interval_s)
            continue

        runtime["last_state_id"] = best.state_id
        runtime["last_transition_ok"] = False
        logger.event("state_selected", state_id=best.state_id, state_dir=best.state_dir, match=summarize_match(best))
        pending_from2 = runtime.get("pending_from_state_id")
        pending_action2 = runtime.get("pending_action_id")
        if isinstance(pending_from2, str) and pending_from2 and isinstance(pending_action2, str) and pending_action2:
            _append_graph_edge(pending_from2, pending_action2, best.state_id, logger=logger, reason="pending-unknown-resolved-to-existing")
            logger.text(
                f"[fsm][edge][added] from={pending_from2} action={pending_action2} "
                f"to={best.state_id} reason=pending-unknown-resolved-to-existing",
                "edge_added",
                from_state=pending_from2,
                action_id=pending_action2,
                to_state=best.state_id,
                reason="pending-unknown-resolved-to-existing",
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
            logger=logger,
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
