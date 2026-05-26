from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from agent.llm_client import DoubaoClient

from .constants import LLM_PARSE_RETRY
from .io import _now_iso, _save_json
from .llm_tasks import _build_user_message_from_frame, _normalize_assistant_text, _save_llm_raw_debug
from .matching import _condition_eval


def ensure_page_handler(meta: dict[str, Any]) -> dict[str, Any]:
    handler = meta.get("page_handler")
    if not isinstance(handler, dict):
        handler = {}
        meta["page_handler"] = handler
    handler.setdefault("enabled", True)
    handler.setdefault("current_node_default", "root")
    nodes = handler.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        handler["nodes"] = [{"id": "root", "brief": "默认局部阶段"}]
    edges = handler.get("edges")
    if not isinstance(edges, list):
        handler["edges"] = []
    return handler


def get_default_node(handler: dict[str, Any]) -> str:
    return str(handler.get("current_node_default") or "root")


def summarize_page_handler(handler: dict[str, Any], current_node: str, limit: int = 12) -> dict[str, Any]:
    edges = [e for e in handler.get("edges", []) if isinstance(e, dict)]
    nodes = [n for n in handler.get("nodes", []) if isinstance(n, dict)]
    node_summary = [
        {"id": n.get("id"), "action": n.get("action"), "brief": n.get("brief", "")}
        for n in nodes[:limit]
    ]
    nearby = [e for e in edges if str(e.get("from_node", "root")) == current_node]
    if len(nearby) < limit:
        nearby.extend(e for e in edges if e not in nearby)
    compact_edges: list[dict[str, Any]] = []
    for e in nearby[:limit]:
        compact_edges.append(
            {
                "id": e.get("id"),
                "from_node": e.get("from_node", "root"),
                "to_node": e.get("to_node", "root"),
                "conditions": _compact_conditions(e.get("conditions", [])),
                "action": e.get("action"),
                "expected_after_action": e.get("expected_after_action", {}),
                "brief": e.get("brief", ""),
                "success_count": int(e.get("success_count", 0)),
                "fail_count": int(e.get("fail_count", 0)),
            }
        )
    return {"current_node": current_node, "nodes": node_summary, "edges": compact_edges}


def _compact_conditions(conditions: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(conditions, list):
        return out
    for c in conditions:
        if not isinstance(c, dict):
            continue
        params = c.get("params", {})
        if not isinstance(params, dict):
            params = {}
        item = {
            "id": c.get("id"),
            "kind": c.get("kind"),
            "rect": params.get("rect") or c.get("bbox"),
            "brief": c.get("brief", ""),
        }
        if c.get("kind") == "text_line_contains":
            item["text"] = params.get("text", "")
        elif c.get("kind") == "region_template":
            item["threshold"] = params.get("threshold", 0.8)
        out.append(item)
    return out


def match_page_handler_edge(
    handler: dict[str, Any],
    current_node: str,
    vision: VisionEngine,
    frame_rgb,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    candidates: list[tuple[tuple[int, int, int], dict[str, Any], list[dict[str, Any]]]] = []
    default_candidates: list[tuple[tuple[int, int, int], dict[str, Any], list[dict[str, Any]]]] = []
    diagnostics: list[dict[str, Any]] = []
    for edge in handler.get("edges", []):
        if not isinstance(edge, dict) or not edge.get("enabled", True):
            continue
        if str(edge.get("from_node", "root")) != current_node:
            continue
        conditions = [c for c in edge.get("conditions", []) if isinstance(c, dict)]
        if not conditions:
            if edge.get("condition_policy") == "default_if_no_edge_matches":
                priority = int(edge.get("priority", 0))
                success_count = int(edge.get("success_count", 0))
                fail_count = int(edge.get("fail_count", 0))
                default_candidates.append(((priority, success_count - fail_count, 0), edge, []))
            continue
        details: list[dict[str, Any]] = []
        passed = 0
        for cond in conditions:
            ok, detail = _condition_eval(cond, vision, frame_rgb)
            details.append(detail)
            if ok:
                passed += 1
        diag = {"edge_id": edge.get("id"), "passed": passed, "total": len(conditions), "details": details}
        diagnostics.append(diag)
        if passed == len(conditions):
            priority = int(edge.get("priority", 50))
            success_count = int(edge.get("success_count", 0))
            fail_count = int(edge.get("fail_count", 0))
            candidates.append(((priority, success_count - fail_count, len(conditions)), edge, details))
    if not candidates:
        if not default_candidates:
            return None, diagnostics
        default_candidates.sort(key=lambda item: item[0], reverse=True)
        return default_candidates[0][1], diagnostics
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], diagnostics


def update_edge_stats(meta: dict[str, Any], edge_id: str, *, success: bool, state_path: Path) -> None:
    handler = ensure_page_handler(meta)
    for edge in handler.get("edges", []):
        if not isinstance(edge, dict) or str(edge.get("id")) != edge_id:
            continue
        key = "success_count" if success else "fail_count"
        edge[key] = int(edge.get(key, 0)) + 1
        edge["updated_at"] = _now_iso()
        meta["updated_at"] = _now_iso()
        _save_json(state_path, meta)
        return


def add_page_handler_edge(meta: dict[str, Any], edge: dict[str, Any], *, state_path: Path) -> dict[str, Any]:
    handler = ensure_page_handler(meta)
    for existing in handler.get("edges", []):
        if (
            isinstance(existing, dict)
            and str(existing.get("from_node", "root")) == str(edge.get("from_node", "root"))
            and existing.get("condition_policy") == "default_if_no_edge_matches"
            and edge.get("condition_policy") == "default_if_no_edge_matches"
        ):
            existing.update(edge)
            existing["updated_at"] = _now_iso()
            meta["updated_at"] = _now_iso()
            _save_json(state_path, meta)
            return existing
    existing_nodes = {str(n.get("id")): n for n in handler.get("nodes", []) if isinstance(n, dict)}
    for node_id in {str(edge.get("from_node", "root")), str(edge.get("to_node", "root"))}:
        if node_id and node_id not in existing_nodes:
            node = {"id": node_id, "brief": ""}
            handler["nodes"].append(node)
            existing_nodes[node_id] = node
    to_node = existing_nodes.get(str(edge.get("to_node", "root")))
    if isinstance(to_node, dict) and edge.get("action") and not to_node.get("action"):
        to_node["action"] = edge.get("action")
        to_node["brief"] = edge.get("brief", to_node.get("brief", ""))
    handler["edges"].append(edge)
    meta["updated_at"] = _now_iso()
    _save_json(state_path, meta)
    return edge


def action_for_edge(handler: dict[str, Any], edge: dict[str, Any]) -> dict[str, Any] | None:
    to_node = str(edge.get("to_node", "root"))
    for node in handler.get("nodes", []):
        if isinstance(node, dict) and str(node.get("id")) == to_node and isinstance(node.get("action"), dict):
            return node["action"]
    action = edge.get("action")
    return action if isinstance(action, dict) else None


def request_page_handler_edge(
    llm: DoubaoClient,
    session_id: str,
    frame_rgb,
    system_prompt: str,
    *,
    state_slug: str,
    handler_summary: dict[str, Any],
    previous_steps: list[dict[str, Any]],
    logger=None,
    allow_default_edge: bool = True,
) -> dict[str, Any] | None:
    text = json.dumps(
        {
            "mode": "PAGE_HANDLER_EDGE_CREATE",
            "instruction": (
                "当前处于同一页面内的局部操作模式。请只在当前截图上为当前局部节点生成一条可复用的条件边。"
                "边的含义是：当 conditions 全部满足时，runtime 从 from_node 转移到代表操作的 to_node，并执行 action。"
                "runtime 会把 action 持久化到 to_node 上；edge 上的 action 用于创建/兼容。"
                "如果当前动作是该局部深度在没有任何可匹配条件时也应执行的固定默认动作，可以输出空 conditions，"
                "并设置 condition_policy=default_if_no_edge_matches；这类无条件边只会在同节点没有任何有条件边命中时低优先级执行。"
                "conditions 只能使用与主路径完全一致的两种条件："
                "1) text_line_contains：单行 OCR contains 子串。params 必须包含 text 和 rect。rect 必须只覆盖一行文字；"
                "text 是 contains 子串，不是精确匹配，不是正则。优先选择同类页面共有的稳定短词，删除名称、编号、数量、进度等可变部分。"
                "2) region_template：在 rect 内进行模板匹配。params 必须包含 rect，可选 threshold。runtime 会从当前截图裁剪模板。"
                "不要使用其它条件类型，不要写自然语言条件。"
                "action 只能是 click 或 run_preset，坐标使用 1000x1000 逻辑坐标。"
                "如果动作通常留在同一页面推进内容，expected_after_action.same_page_likely=true；如果通常会离开该页面，exit_likely=true。"
                "只输出严格 JSON。"
            ),
            "current_state": state_slug,
            "handler_summary": handler_summary,
            "previous_steps": previous_steps[-8:],
            "output_schema": {
                "from_node": "string",
                "to_node": "string",
                "condition_policy": "all_match|default_if_no_edge_matches",
                "conditions": [
                    {
                        "kind": "text_line_contains|region_template",
                        "params": {"text": "string for text_line_contains", "rect": [0, 0, 0, 0], "threshold": "optional number"},
                        "brief": "string",
                        "stability": "high|mid|low",
                        "discrimination": "high|mid|low",
                    }
                ],
                "action": {"type": "click|run_preset", "x": "integer optional", "y": "integer optional", "name": "string optional", "brief": "string"},
                "expected_after_action": {
                    "same_page_likely": "boolean",
                    "exit_likely": "boolean",
                    "screen_should_change": "boolean",
                    "reason": "string",
                },
                "priority": "integer 0..100",
                "brief": "string",
            },
        },
        ensure_ascii=False,
    )
    prompt_fix = "上次JSON无效。只输出一个完整JSON对象，且 conditions 只能使用 text_line_contains 或 region_template。"
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
        _save_llm_raw_debug(session_id, attempt, raw, "page_handler", logger.llm_raw_dir if logger is not None else None)
        if logger is not None:
            logger.text(f"[fsm][handler][llm] attempt={attempt} raw={raw[:600]}", "llm_page_handler_raw", attempt=attempt, raw_preview=raw[:600])
        parsed = parse_page_handler_edge(raw, allow_default_edge=allow_default_edge)
        if parsed is not None:
            return parsed
    return None


def parse_page_handler_edge(text: str, *, allow_default_edge: bool = True) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    conditions = payload.get("conditions")
    action = payload.get("action")
    condition_policy = str(payload.get("condition_policy") or "all_match")
    if not isinstance(conditions, list) or not isinstance(action, dict):
        return None
    if not conditions and (not allow_default_edge or condition_policy != "default_if_no_edge_matches"):
        return None
    normalized_conditions = [_normalize_condition(c) for c in conditions if isinstance(c, dict)]
    normalized_conditions = [c for c in normalized_conditions if c is not None]
    if conditions and not normalized_conditions:
        return None
    normalized_action = _normalize_action(action)
    if normalized_action is None:
        return None
    expected = payload.get("expected_after_action", {})
    if not isinstance(expected, dict):
        expected = {}
    return {
        "id": f"edge_{uuid4().hex[:12]}",
        "enabled": True,
        "from_node": str(payload.get("from_node") or "root"),
        "to_node": str(payload.get("to_node") or payload.get("from_node") or "root"),
        "condition_policy": condition_policy if condition_policy == "default_if_no_edge_matches" else "all_match",
        "conditions": normalized_conditions,
        "action": normalized_action,
        "expected_after_action": {
            "same_page_likely": bool(expected.get("same_page_likely", True)),
            "exit_likely": bool(expected.get("exit_likely", False)),
            "screen_should_change": bool(expected.get("screen_should_change", True)),
            "reason": str(expected.get("reason", "")),
        },
        "priority": max(0, min(100, int(payload.get("priority", 5 if condition_policy == "default_if_no_edge_matches" else 50)))),
        "brief": str(payload.get("brief", "")),
        "success_count": 0,
        "fail_count": 0,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


def materialize_page_handler_edge(
    edge: dict[str, Any],
    *,
    state_dir: Path,
    frame_rgb,
    mapper: CoordinateMapper,
) -> dict[str, Any] | None:
    out = dict(edge)
    conditions: list[dict[str, Any]] = []
    for idx, cond in enumerate(edge.get("conditions", []), start=1):
        cc = dict(cond)
        params = dict(cc.get("params", {}))
        if cc.get("kind") == "region_template":
            rect = params.get("rect")
            if not (isinstance(rect, list) and len(rect) == 4):
                return None
            x, y, w, h = mapper.rect_to_real(rect)
            crop = frame_rgb[y : y + h, x : x + w]
            if crop.size <= 0:
                return None
            tpath = state_dir / f"page_handler_template_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{idx}.png"
            cv2.imwrite(str(tpath), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            params["template_path"] = str(tpath)
            params["threshold"] = float(params.get("threshold", 0.8))
            cc["source"] = {"bbox_logical": rect, "template_path": str(tpath)}
        cc["params"] = params
        cc["id"] = cc.get("id") or f"{out['id']}_cond_{idx}"
        cc["enabled"] = True
        cc["condition_status"] = "active"
        conditions.append(cc)
    out["conditions"] = conditions
    return out


def edge_conditions_pass(edge: dict[str, Any], vision: VisionEngine, frame_rgb) -> bool:
    conditions = [c for c in edge.get("conditions", []) if isinstance(c, dict)]
    if not conditions and edge.get("condition_policy") == "default_if_no_edge_matches":
        return True
    return bool(conditions) and all(_condition_eval(c, vision, frame_rgb)[0] for c in conditions)


def _normalize_condition(cond: dict[str, Any]) -> dict[str, Any] | None:
    kind = str(cond.get("kind", "")).strip()
    if kind == "line_contains_text":
        kind = "text_line_contains"
    params = cond.get("params", {})
    if not isinstance(params, dict):
        params = {}
    rect = params.get("rect") or cond.get("bbox")
    if not (isinstance(rect, list) and len(rect) == 4):
        return None
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
            "stability": _level(cond.get("stability"), "mid"),
            "discrimination": _level(cond.get("discrimination"), "mid"),
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
            "stability": _level(cond.get("stability"), "mid"),
            "discrimination": _level(cond.get("discrimination"), "mid"),
            "condition_status": "active",
            "bbox": rect,
        }
    return None


def _normalize_action(action: dict[str, Any]) -> dict[str, Any] | None:
    atype = str(action.get("type", "click")).strip()
    brief = str(action.get("brief", ""))
    if atype == "run_preset":
        name = str(action.get("name", "")).strip()
        if not name:
            return None
        return {"type": "run_preset", "name": name, "brief": brief}
    if atype == "click":
        return {"type": "click", "x": int(action.get("x", 500)), "y": int(action.get("y", 500)), "brief": brief}
    return None


def _level(raw: Any, default: str = "mid") -> str:
    value = str(raw or default).strip().lower()
    return value if value in {"high", "mid", "low"} else default
