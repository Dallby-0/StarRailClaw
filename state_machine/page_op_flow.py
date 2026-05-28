from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from agent.llm_client import DoubaoClient
from sr_tools.emulator import EmulatorClient
from state_machine.constants import LOCAL_FLOW_MAX_STEPS, LOCAL_FLOW_NO_PROGRESS_LIMIT, LLM_PARSE_RETRY
from state_machine.io import _load_json, _now_iso, _save_json, _save_runtime
from state_machine.llm_tasks import (
    _apply_reasoning_effort,
    _build_user_message_from_frame,
    _build_user_message_from_two_frames,
    _normalize_assistant_text,
    _save_llm_raw_debug,
)
from state_machine.logger import FsmRunLogger, summarize_match
from state_machine.matching import _condition_eval, _find_match_by_state
from state_machine.screen import _screen_changed
from state_machine.transition_policy import _defer_unknown_transition, _resolve_transition_after_progress


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)


def _slug(raw: Any, fallback: str) -> str:
    text = str(raw or "").strip().lower()
    safe = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("_")
    return safe or fallback


def ensure_page_op_flow(meta: dict[str, Any]) -> dict[str, Any]:
    flow = meta.get("page_op_flow")
    if not isinstance(flow, dict):
        flow = {}
        meta["page_op_flow"] = flow
    catalog = flow.get("op_catalog")
    if not isinstance(catalog, dict):
        catalog = {}
        flow["op_catalog"] = catalog
    if not isinstance(catalog.get("ops"), list):
        catalog["ops"] = []
    tree = flow.get("prefix_tree")
    if not isinstance(tree, dict):
        tree = {}
        flow["prefix_tree"] = tree
    nodes = tree.get("nodes")
    if not isinstance(nodes, dict):
        nodes = {}
        tree["nodes"] = nodes
    nodes.setdefault(
        "root",
        {
            "node_id": "root",
            "visits": 0,
            "first_seen_at": _now_iso(),
            "updated_at": _now_iso(),
            "tried_ops": {},
            "children": {},
        },
    )
    return flow


def _op_summary(flow: dict[str, Any], *, limit: int = 24) -> list[dict[str, Any]]:
    ops = [op for op in flow.get("op_catalog", {}).get("ops", []) if isinstance(op, dict)]
    out: list[dict[str, Any]] = []
    for op in ops[:limit]:
        stats = op.get("stats") if isinstance(op.get("stats"), dict) else {}
        out.append(
            {
                "op_id": op.get("op_id"),
                "concrete_name": op.get("concrete_name"),
                "abstract_name": op.get("abstract_name"),
                "status": op.get("status", "active"),
                "effectless": bool(op.get("effectless", False)),
                "effectless_reason": str(op.get("effectless_reason", ""))[:180],
                "has_readiness_conditions": bool(op.get("readiness_conditions")),
                "action": op.get("action"),
                "expected_after_action": op.get("expected_after_action", {}),
                "stats": {
                    "try_count": int(stats.get("try_count", 0) or 0),
                    "success_count": int(stats.get("success_count", 0) or 0),
                    "effectless_count": int(stats.get("effectless_count", 0) or 0),
                },
            }
        )
    return out


def _node(flow: dict[str, Any], node_id: str) -> dict[str, Any]:
    nodes = flow["prefix_tree"]["nodes"]
    if node_id not in nodes or not isinstance(nodes[node_id], dict):
        nodes[node_id] = {
            "node_id": node_id,
            "visits": 0,
            "first_seen_at": _now_iso(),
            "updated_at": _now_iso(),
            "tried_ops": {},
            "children": {},
        }
    return nodes[node_id]


def _active_ops(flow: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        op
        for op in flow.get("op_catalog", {}).get("ops", [])
        if isinstance(op, dict) and op.get("status", "active") == "active" and not bool(op.get("effectless", False))
    ]


def _find_op(flow: dict[str, Any], op_id: str) -> dict[str, Any] | None:
    for op in flow.get("op_catalog", {}).get("ops", []):
        if isinstance(op, dict) and str(op.get("op_id")) == op_id:
            return op
    return None


def _normalize_action(action: Any) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    if str(action.get("type", "click")) != "click":
        return None
    try:
        x = int(action.get("x"))
        y = int(action.get("y"))
    except Exception:
        return None
    return {"type": "click", "x": x, "y": y, "brief": str(action.get("brief", ""))}


def _normalize_condition(cond: Any) -> dict[str, Any] | None:
    if not isinstance(cond, dict):
        return None
    kind = str(cond.get("kind", "")).strip()
    params = cond.get("params") if isinstance(cond.get("params"), dict) else {}
    rect = params.get("rect") or cond.get("bbox")
    if not (isinstance(rect, list) and len(rect) == 4):
        return None
    if kind == "line_contains_text":
        kind = "text_line_contains"
    if kind == "text_line_contains":
        text = str(params.get("text") or params.get("contains") or cond.get("text") or "").strip()
        if not text:
            return None
        return {
            "id": "",
            "enabled": True,
            "kind": "text_line_contains",
            "params": {"text": text, "rect": rect},
            "brief": str(cond.get("brief", "")),
            "stability": str(cond.get("stability", "mid")),
            "discrimination": str(cond.get("discrimination", "mid")),
            "condition_status": "active",
            "bbox": rect,
        }
    if kind == "region_template":
        return {
            "id": "",
            "enabled": True,
            "kind": "region_template",
            "params": {"rect": rect, "threshold": float(params.get("threshold", 0.8))},
            "brief": str(cond.get("brief", "")),
            "stability": str(cond.get("stability", "mid")),
            "discrimination": str(cond.get("discrimination", "mid")),
            "condition_status": "active",
            "bbox": rect,
        }
    return None


def _materialize_condition(cond: dict[str, Any], *, state_dir: Path, frame_rgb, mapper: CoordinateMapper, suffix: str) -> dict[str, Any] | None:
    out = dict(cond)
    params = dict(out.get("params", {}))
    out["params"] = params
    out["id"] = out.get("id") or suffix
    if out.get("kind") != "region_template":
        return out
    rect = params.get("rect")
    if not (isinstance(rect, list) and len(rect) == 4):
        return None
    x, y, w, h = mapper.rect_to_real(rect)
    crop = frame_rgb[y : y + h, x : x + w]
    if crop.size <= 0:
        return None
    tpath = state_dir / f"page_op_template_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{suffix}.png"
    cv2.imwrite(str(tpath), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    params["template_path"] = str(tpath)
    params["threshold"] = float(params.get("threshold", 0.8))
    out["source"] = {"bbox_logical": rect, "template_path": str(tpath)}
    return out


def _dry_run_conditions(
    raw_conditions: Any,
    *,
    vision: VisionEngine,
    frame_rgb,
    mapper: CoordinateMapper,
    state_dir: Path,
    op_id: str,
    bucket: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    if not isinstance(raw_conditions, list):
        return active, rejected
    for idx, raw in enumerate(raw_conditions, start=1):
        cond = _normalize_condition(raw)
        if cond is None:
            continue
        cond = _materialize_condition(cond, state_dir=state_dir, frame_rgb=frame_rgb, mapper=mapper, suffix=f"{op_id}_{bucket}_{idx}")
        if cond is None:
            continue
        ok, detail = _condition_eval(cond, vision, frame_rgb)
        if ok:
            active.append(cond)
            continue
        bad = dict(cond)
        bad["reject_reason"] = "dry_run_failed_on_source_frame"
        bad["dry_run_detail"] = detail
        rejected.append(bad)
    return active, rejected


def _normalize_op(raw: Any, *, vision: VisionEngine, frame_rgb, mapper: CoordinateMapper, state_dir: Path, existing_id: str | None = None) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    action = _normalize_action(raw.get("action"))
    if action is None:
        return None
    op_id = _slug(raw.get("op_id") or raw.get("concrete_name"), f"op_{uuid.uuid4().hex[:8]}")
    if existing_id:
        op_id = existing_id
    visibility, rejected_visibility = _dry_run_conditions(
        raw.get("visibility_conditions") or raw.get("conditions"),
        vision=vision,
        frame_rgb=frame_rgb,
        mapper=mapper,
        state_dir=state_dir,
        op_id=op_id,
        bucket="visibility",
    )
    readiness, rejected_readiness = _dry_run_conditions(
        raw.get("readiness_conditions"),
        vision=vision,
        frame_rgb=frame_rgb,
        mapper=mapper,
        state_dir=state_dir,
        op_id=op_id,
        bucket="readiness",
    )
    expected = raw.get("expected_after_action")
    if not isinstance(expected, dict):
        expected = {}
    return {
        "op_id": op_id,
        "concrete_name": str(raw.get("concrete_name") or raw.get("name") or op_id),
        "abstract_name": _slug(raw.get("abstract_name"), "generic_click"),
        "visibility_conditions": visibility,
        "readiness_conditions": readiness,
        "rejected_conditions": list(raw.get("rejected_conditions", []) if isinstance(raw.get("rejected_conditions"), list) else [])
        + rejected_visibility
        + rejected_readiness,
        "action": action,
        "expected_after_action": {
            "exit_page": bool(expected.get("exit_page", expected.get("exit_likely", False))),
            "same_page_progress": bool(expected.get("same_page_progress", expected.get("same_page_likely", True))),
            "observable_changes": expected.get("observable_changes", []) if isinstance(expected.get("observable_changes"), list) else [],
            "reason": str(expected.get("reason", "")),
        },
        "status": str(raw.get("status", "active")) if str(raw.get("status", "active")) in {"active", "disabled"} else "active",
        "effectless": bool(raw.get("effectless", False)),
        "effectless_reason": str(raw.get("effectless_reason", "")),
        "repair_note": str(raw.get("repair_note", "")),
        "stats": {"try_count": 0, "success_count": 0, "effectless_count": 0},
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


def _merge_catalog_patch(flow: dict[str, Any], patch: dict[str, Any], *, vision: VisionEngine, frame_rgb, mapper: CoordinateMapper, state_dir: Path) -> None:
    catalog = flow["op_catalog"]
    ops = catalog["ops"]
    by_id = {str(op.get("op_id")): op for op in ops if isinstance(op, dict)}
    for raw in patch.get("new_ops", []) if isinstance(patch.get("new_ops"), list) else []:
        op = _normalize_op(raw, vision=vision, frame_rgb=frame_rgb, mapper=mapper, state_dir=state_dir)
        if op is None:
            continue
        if op["op_id"] in by_id:
            continue
        ops.append(op)
        by_id[op["op_id"]] = op
    for raw in patch.get("repaired_ops", []) if isinstance(patch.get("repaired_ops"), list) else []:
        target = _slug(raw.get("op_id"), "")
        if not target or target not in by_id:
            continue
        op = _normalize_op(raw, vision=vision, frame_rgb=frame_rgb, mapper=mapper, state_dir=state_dir, existing_id=target)
        if op is None:
            continue
        old = by_id[target]
        op["created_at"] = old.get("created_at", op["created_at"])
        op["stats"] = old.get("stats", op["stats"]) if isinstance(old.get("stats"), dict) else op["stats"]
        old.update(op)
        old["effectless"] = False
        old["updated_at"] = _now_iso()
    disabled = patch.get("disabled_ops") if isinstance(patch.get("disabled_ops"), list) else []
    for raw_id in disabled:
        op = by_id.get(_slug(raw_id, str(raw_id)))
        if op is not None:
            op["status"] = "disabled"
            op["updated_at"] = _now_iso()


def _parse_page_op_step(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    decision = payload.get("decision")
    if decision is not None and not isinstance(decision, dict):
        return None
    patch = payload.get("catalog_patch")
    if patch is not None and not isinstance(patch, dict):
        return None
    effect = payload.get("effect_judgement")
    if effect is not None and not isinstance(effect, dict):
        return None
    payload.setdefault("catalog_patch", {"should_patch": False, "new_ops": [], "repaired_ops": [], "disabled_ops": []})
    payload.setdefault("effect_judgement", {})
    payload.setdefault("decision", {})
    payload["needs_retry"] = bool(payload.get("needs_retry", False))
    return payload


def _request_page_op_step(
    *,
    llm: DoubaoClient,
    session_id: str,
    frame_rgb,
    previous_frame_rgb,
    system_prompt: str,
    state_slug: str,
    flow: dict[str, Any],
    current_node: str,
    request: dict[str, Any],
    logger: FsmRunLogger | None,
    effort: str | None = None,
) -> dict[str, Any] | None:
    old_effort = _apply_reasoning_effort(llm, effort)
    try:
        context = {
            "mode": "PAGE_OP_STEP",
            "instruction": (
                "你正在同一外部页面状态内推进流程。只输出严格 JSON。"
                "你不能输出 from_node/to_node/edge，也不能维护图结构；局部前缀树由 runtime 维护。"
                "如需补充操作，只输出当前截图下可执行或可探索的 op。"
                "condition 只能是 text_line_contains 或 region_template；如果你给出的 condition 在当前截图 dry-run 不通过，runtime 会拒绝它，"
                "但 op 仍可作为低置信探索操作保留。区分 visibility_conditions 和 readiness_conditions：可见不等于可点击。"
                "如果上一步有进度且是第一次进入当前局部节点，必须尽量补充当前截图下的新候选操作，并给出下一步 decision。"
                "当 request.need_catalog_patch=if_progress_enters_new_prefix_node 时，若你判断 progress=true，就等价于必须补充新节点候选操作；"
                "若 progress=false，catalog_patch 可保持 should_patch=false。"
                "如果上一步无进度，优先判断是否需要 retry；retry 后仍无进度时，给出替代 op 或修复已有 op。"
                "action 当前只支持单次 click，坐标为 1000x1000 逻辑坐标。"
            ),
            "current_state": state_slug,
            "current_prefix_node": current_node,
            "request": request,
            "op_catalog_summary": _op_summary(flow),
            "prefix_node": _node(flow, current_node),
            "output_schema": {
                "effect_judgement": {
                    "progress": "boolean optional",
                    "exit_page": "boolean optional",
                    "effectless": "boolean optional",
                    "confidence": "high|mid|low",
                    "evidence": "short string",
                },
                "catalog_patch": {
                    "should_patch": "boolean",
                    "new_ops": [
                        {
                            "op_id": "stable id",
                            "concrete_name": "具体操作",
                            "abstract_name": "泛化操作名",
                            "visibility_conditions": [],
                            "readiness_conditions": [],
                            "action": {"type": "click", "x": 0, "y": 0, "brief": ""},
                            "expected_after_action": {"exit_page": False, "same_page_progress": True, "observable_changes": [], "reason": ""},
                        }
                    ],
                    "repaired_ops": [],
                    "disabled_ops": [],
                },
                "decision": {"op_id": "existing_or_new_op_id", "reason": "short string"},
                "needs_retry": "boolean",
            },
        }
        text = json.dumps(context, ensure_ascii=False)
        prompt_fix = "上次 JSON 无效。只输出一个完整 JSON 对象，字段遵循 output_schema，不要解释。"
        for attempt in range(1, LLM_PARSE_RETRY + 2):
            if attempt == 1:
                msg = (
                    _build_user_message_from_two_frames(previous_frame_rgb, frame_rgb, text)
                    if previous_frame_rgb is not None
                    else _build_user_message_from_frame(frame_rgb, text)
                )
            else:
                msg = {"role": "user", "content": [{"type": "text", "text": prompt_fix}]}
            resp = llm.chat_with_session(session_id=session_id, system_prompt=system_prompt, user_message=msg, tools=[], tool_choice="none")
            raw = _normalize_assistant_text(resp["choices"][0]["message"].get("content"))
            _save_llm_raw_debug(session_id, attempt, raw, "page_op_step", logger.llm_raw_dir if logger is not None else None)
            _log(logger, f"[fsm][op][llm] attempt={attempt} raw={raw[:600]}", "llm_page_op_raw", attempt=attempt, raw_preview=raw[:600], current_node=current_node)
            parsed = _parse_page_op_step(raw)
            if parsed is not None:
                return parsed
    finally:
        llm.reasoning_effort = old_effort
    return None


def _condition_list_pass(conditions: list[dict[str, Any]], vision: VisionEngine, frame_rgb) -> bool:
    if not conditions:
        return True
    return all(_condition_eval(c, vision, frame_rgb)[0] for c in conditions if isinstance(c, dict))


def _select_decision_op(flow: dict[str, Any], decision: dict[str, Any], vision: VisionEngine, frame_rgb) -> dict[str, Any] | None:
    op_id = str(decision.get("op_id", "")).strip()
    if op_id:
        op = _find_op(flow, _slug(op_id, op_id))
        if op is not None and op.get("status", "active") == "active" and not bool(op.get("effectless", False)):
            readiness = op.get("readiness_conditions")
            if isinstance(readiness, list) and readiness and not _condition_list_pass(readiness, vision, frame_rgb):
                return None
            return op
    active = _active_ops(flow)
    with_readiness = [op for op in active if isinstance(op.get("readiness_conditions"), list) and op.get("readiness_conditions") and _condition_list_pass(op["readiness_conditions"], vision, frame_rgb)]
    if with_readiness:
        return with_readiness[0]
    return active[0] if active else None


def _mark_op_result(op: dict[str, Any], result: str, reason: str = "") -> None:
    stats = op.get("stats")
    if not isinstance(stats, dict):
        stats = {}
        op["stats"] = stats
    stats["try_count"] = int(stats.get("try_count", 0) or 0) + 1
    if result == "progress":
        stats["success_count"] = int(stats.get("success_count", 0) or 0) + 1
    if result == "effectless":
        stats["effectless_count"] = int(stats.get("effectless_count", 0) or 0) + 1
        op["effectless"] = True
        op["effectless_reason"] = reason
    op["updated_at"] = _now_iso()


def _record_tree_result(flow: dict[str, Any], node_id: str, op_id: str, result: str) -> str:
    node = _node(flow, node_id)
    node["visits"] = int(node.get("visits", 0) or 0) + 1
    node["updated_at"] = _now_iso()
    tried = node.setdefault("tried_ops", {})
    tried[op_id] = {"last_result": result, "updated_at": _now_iso()}
    if result != "progress":
        return node_id
    child_id = f"{node_id}/{op_id}:progress" if node_id != "root" else f"{op_id}:progress"
    children = node.setdefault("children", {})
    children[f"{op_id}:progress"] = child_id
    _node(flow, child_id)
    return child_id


def _seed_op_id(seed_step: dict[str, Any]) -> str:
    action = seed_step.get("action") if isinstance(seed_step.get("action"), dict) else {}
    brief = str(action.get("brief") or seed_step.get("brief") or "entry_action")
    return _slug(seed_step.get("op_id") or brief, "entry_action")


def _execute_click(emulator: EmulatorClient, mapper: CoordinateMapper, op: dict[str, Any], logger: FsmRunLogger | None, *, state_id: str, action_id: str, attempt: str) -> bool:
    action = op.get("action") if isinstance(op.get("action"), dict) else {}
    x = int(action.get("x", 500))
    y = int(action.get("y", 500))
    rx, ry = mapper.point_to_real(x, y)
    brief = str(action.get("brief") or op.get("concrete_name") or op.get("op_id"))
    _log(
        logger,
        f"[fsm][op][click] op={op.get('op_id')} logical=({x},{y}) real=({rx},{ry}) brief={brief}",
        "page_op_click",
        state_id=state_id,
        action_id=action_id,
        op_id=op.get("op_id"),
        attempt=attempt,
        logical=[x, y],
        real=[rx, ry],
        brief=brief,
    )
    emulator.tap(rx, ry)
    return True


def _run_page_op_flow(
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
    seed_step: dict[str, Any] | None = None,
    seed_before_frame=None,
    logger: FsmRunLogger | None = None,
) -> tuple[bool, Any]:
    current = start_frame
    state_path = state_dir / "state.json"
    state_meta = _load_json(state_path)
    flow = ensure_page_op_flow(state_meta)
    current_node = "root"
    pending_decision: dict[str, Any] | None = None
    no_progress_count = 0
    made_progress = False

    if isinstance(seed_step, dict):
        response = _request_page_op_step(
            llm=llm,
            session_id=llm_session_id,
            frame_rgb=current,
            previous_frame_rgb=seed_before_frame,
            system_prompt=system_prompt,
            state_slug=state_slug,
            flow=flow,
            current_node=current_node,
            request={
                "has_previous_action": True,
                "previous_action": seed_step.get("action"),
                "need_effect_judgement": True,
                "need_catalog_patch": "if_progress_enters_new_prefix_node",
                "need_decision": True,
                "runtime_signals": seed_step.get("runtime_observation", {}),
                "reason": "external_action_self_loop_enter_page_flow",
                "effect_classes": [
                    "semantic_exit_same_type",
                    "page_internal_progress",
                    "effectless",
                ],
                "instruction": (
                    "判断 previous_action 导致的自环属于哪类："
                    "semantic_exit_same_type 表示操作有效，已进入同类型新实例/阶段；"
                    "page_internal_progress 表示操作有效但仍需同页后续操作；"
                    "effectless 表示操作无效。有效时 progress=true；无效时 effectless=true。"
                ),
            },
            logger=logger,
        )
        runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0)) + 1
        _save_runtime(runtime)
        if response is None:
            return False, current
        _merge_catalog_patch(flow, response.get("catalog_patch", {}), vision=vision, frame_rgb=current, mapper=mapper, state_dir=state_dir)
        effect = response.get("effect_judgement", {}) if isinstance(response.get("effect_judgement"), dict) else {}
        seed_id = _seed_op_id(seed_step)
        if bool(effect.get("progress", False)):
            made_progress = True
            current_node = _record_tree_result(flow, current_node, seed_id, "progress")
        else:
            _record_tree_result(flow, current_node, seed_id, "effectless")
            no_progress_count = 1 if bool(effect.get("effectless", False)) else 0
        decision = response.get("decision", {}) if isinstance(response.get("decision"), dict) else {}
        pending_decision = decision if decision.get("op_id") else None
        state_meta["updated_at"] = _now_iso()
        _save_json(state_path, state_meta)

    request = {
        "has_previous_action": False,
        "previous_action": None,
        "need_effect_judgement": False,
        "need_catalog_patch": True,
        "need_decision": True,
        "reason": "enter_page_op_flow",
    }

    for step_idx in range(1, LOCAL_FLOW_MAX_STEPS + 1):
        node = _node(flow, current_node)
        first_visit = int(node.get("visits", 0) or 0) == 0
        if pending_decision is None:
            request["need_catalog_patch"] = first_visit or not _active_ops(flow) or no_progress_count > 0
            response = _request_page_op_step(
                llm=llm,
                session_id=llm_session_id,
                frame_rgb=current,
                previous_frame_rgb=None,
                system_prompt=system_prompt,
                state_slug=state_slug,
                flow=flow,
                current_node=current_node,
                request=request,
                logger=logger,
            )
            runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0)) + 1
            _save_runtime(runtime)
            if response is None:
                return made_progress, current
            _merge_catalog_patch(flow, response.get("catalog_patch", {}), vision=vision, frame_rgb=current, mapper=mapper, state_dir=state_dir)
            pending_decision = response.get("decision", {})
            state_meta["updated_at"] = _now_iso()
            _save_json(state_path, state_meta)

        op = _select_decision_op(flow, pending_decision or {}, vision, current)
        pending_decision = None
        if op is None:
            _log(logger, f"[fsm][op] no executable op node={current_node}", "page_op_no_executable", state_id=state_id, node=current_node)
            return made_progress, current

        if not _execute_click(emulator, mapper, op, logger, state_id=state_id, action_id=action_id, attempt=f"op:{step_idx}"):
            return made_progress, current
        import time

        from state_machine.constants import ACTION_CLICK_WAIT_S

        _log(logger, f"[fsm][op][wait] sleep={ACTION_CLICK_WAIT_S}s after op={op.get('op_id')}", "page_op_wait", state_id=state_id, action_id=action_id, op_id=op.get("op_id"), sleep_s=ACTION_CLICK_WAIT_S)
        time.sleep(ACTION_CLICK_WAIT_S)
        post = emulator.screenshot(prefer_png=True)
        matches = matches_provider(post)
        changed, diff_score = _screen_changed(current, post)
        if logger is not None:
            logger.event(
                "page_local_step_result",
                state_id=state_id,
                action_id=action_id,
                step=step_idx,
                changed=changed,
                diff_score=diff_score,
                op_id=op.get("op_id"),
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
            reason_suffix="-page-op",
        )
        if nxt is not None:
            _mark_op_result(op, "progress", "")
            _record_tree_result(flow, current_node, str(op.get("op_id")), "exit")
            state_meta["updated_at"] = _now_iso()
            _save_json(state_path, state_meta)
            return True, post
        curr_match = _find_match_by_state(matches, state_id)
        if curr_match is None or not curr_match.success:
            _mark_op_result(op, "progress", "")
            _record_tree_result(flow, current_node, str(op.get("op_id")), "unknown")
            state_meta["updated_at"] = _now_iso()
            _save_json(state_path, state_meta)
            return _defer_unknown_transition(
                runtime=runtime,
                state_id=state_id,
                action_id=action_id,
                logger=logger,
                frame=post,
                reason="page-op-original-state-no-longer-matched",
            )

        response = _request_page_op_step(
            llm=llm,
            session_id=llm_session_id,
            frame_rgb=post,
            previous_frame_rgb=current,
            system_prompt=system_prompt,
            state_slug=state_slug,
            flow=flow,
            current_node=current_node,
            request={
                "has_previous_action": True,
                "previous_op": op.get("op_id"),
                "need_effect_judgement": True,
                "need_catalog_patch": "if_progress_enters_new_prefix_node",
                "need_decision": True,
                "runtime_signals": {"pixel_diff": diff_score, "same_state_still_matches": True},
                "reason": "same_external_state_after_op",
            },
            logger=logger,
        )
        runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0)) + 1
        _save_runtime(runtime)
        if response is None:
            return made_progress, post
        effect = response.get("effect_judgement", {}) if isinstance(response.get("effect_judgement"), dict) else {}
        progress = bool(effect.get("progress", False))
        effectless = bool(effect.get("effectless", False))
        needs_retry = bool(response.get("needs_retry", False))
        if not progress and needs_retry:
            _log(logger, f"[fsm][op][retry] op={op.get('op_id')}", "page_op_retry", state_id=state_id, op_id=op.get("op_id"))
            _execute_click(emulator, mapper, op, logger, state_id=state_id, action_id=action_id, attempt=f"op:{step_idx}:retry")
            from state_machine.constants import ACTION_CLICK_WAIT_S
            import time

            time.sleep(ACTION_CLICK_WAIT_S)
            retry_post = emulator.screenshot(prefer_png=True)
            retry_matches = matches_provider(retry_post)
            nxt_retry = _resolve_transition_after_progress(
                state_id=state_id,
                action_id=action_id,
                matches=retry_matches,
                graph=graph,
                runtime=runtime,
                prefer_reachable_first=prefer_reachable_first,
                logger=logger,
                reason_suffix="-page-op-retry",
            )
            if nxt_retry is not None:
                _mark_op_result(op, "progress", "")
                _record_tree_result(flow, current_node, str(op.get("op_id")), "exit")
                state_meta["updated_at"] = _now_iso()
                _save_json(state_path, state_meta)
                return True, retry_post
            curr_retry = _find_match_by_state(retry_matches, state_id)
            if curr_retry is None or not curr_retry.success:
                _mark_op_result(op, "progress", "")
                _record_tree_result(flow, current_node, str(op.get("op_id")), "unknown")
                state_meta["updated_at"] = _now_iso()
                _save_json(state_path, state_meta)
                return _defer_unknown_transition(
                    runtime=runtime,
                    state_id=state_id,
                    action_id=action_id,
                    logger=logger,
                    frame=retry_post,
                    reason="page-op-retry-original-state-no-longer-matched",
                )
            response = _request_page_op_step(
                llm=llm,
                session_id=llm_session_id,
                frame_rgb=retry_post,
                previous_frame_rgb=post,
                system_prompt=system_prompt,
                state_slug=state_slug,
                flow=flow,
                current_node=current_node,
                request={
                    "has_previous_action": True,
                    "previous_op": op.get("op_id"),
                    "need_effect_judgement": True,
                    "need_catalog_patch": True,
                    "need_decision": True,
                    "retry_after_no_progress": True,
                    "runtime_signals": {"same_state_still_matches": True},
                    "reason": "retry_result",
                },
                logger=logger,
                effort="high",
            )
            runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0)) + 1
            _save_runtime(runtime)
            post = retry_post
            if response is None:
                return made_progress, post
            effect = response.get("effect_judgement", {}) if isinstance(response.get("effect_judgement"), dict) else {}
            progress = bool(effect.get("progress", False))
            effectless = bool(effect.get("effectless", not progress))

        if progress:
            made_progress = True
            no_progress_count = 0
            _mark_op_result(op, "progress", "")
            current_node = _record_tree_result(flow, current_node, str(op.get("op_id")), "progress")
            _merge_catalog_patch(flow, response.get("catalog_patch", {}), vision=vision, frame_rgb=post, mapper=mapper, state_dir=state_dir)
            next_decision = response.get("decision", {}) if isinstance(response.get("decision"), dict) else {}
            pending_decision = next_decision if next_decision.get("op_id") else None
            current = post
        else:
            reason = str(effect.get("evidence") or "llm judged no effective progress")
            if effectless:
                _mark_op_result(op, "effectless", reason)
            _record_tree_result(flow, current_node, str(op.get("op_id")), "effectless")
            _merge_catalog_patch(flow, response.get("catalog_patch", {}), vision=vision, frame_rgb=post, mapper=mapper, state_dir=state_dir)
            next_decision = response.get("decision", {}) if isinstance(response.get("decision"), dict) else {}
            pending_decision = next_decision if next_decision.get("op_id") else None
            no_progress_count += 1
            current = post
            if no_progress_count >= LOCAL_FLOW_NO_PROGRESS_LIMIT:
                _log(logger, f"[fsm][op] no progress limit hit state={state_id}", "page_local_no_progress", state_id=state_id, limit=LOCAL_FLOW_NO_PROGRESS_LIMIT)
                state_meta["updated_at"] = _now_iso()
                _save_json(state_path, state_meta)
                return made_progress, current
        state_meta["updated_at"] = _now_iso()
        _save_json(state_path, state_meta)

    _log(logger, f"[fsm][op] max steps reached state={state_id} progress={made_progress}", "page_local_max_steps", state_id=state_id, max_steps=LOCAL_FLOW_MAX_STEPS, made_progress=made_progress)
    state_meta["updated_at"] = _now_iso()
    _save_json(state_path, state_meta)
    return made_progress, current
