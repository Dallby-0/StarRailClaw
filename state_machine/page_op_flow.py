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
from state_machine.screen import _screen_changed, _wait_for_screen_stable
from state_machine.transition_policy import _defer_unknown_transition, _resolve_transition_after_progress
from state_machine.weak_guards import WEAK_GUARDS_ENABLED, WEAK_ROI_KIND, evaluate_condition as _eval_weak_guard, materialize_condition as _materialize_weak_guard, normalize_condition as _normalize_weak_guard


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
            "label": "",
            "notes": "",
            "visits": 0,
            "first_seen_at": _now_iso(),
            "updated_at": _now_iso(),
            "candidate_ops": [],
            "blocked_ops": {},
            "last_observed_actions": [],
            "tried_ops": {},
            "children": {},
        },
    )
    _node(flow, "root")
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
                "guard_quality": _op_guard_quality(op),
                "guard_roles": _op_guard_roles(op),
                "guard_generalized_found": op.get("guard_generalized_found"),
                "guard_generalization_note": str(op.get("guard_generalization_note", ""))[:180],
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


def _compact_op_summary(op: dict[str, Any], *, guard_passed: bool | None = None) -> dict[str, Any]:
    stats = op.get("stats") if isinstance(op.get("stats"), dict) else {}
    action = op.get("action") if isinstance(op.get("action"), dict) else {}
    return {
        "op_id": op.get("op_id"),
        "name": op.get("concrete_name") or op.get("abstract_name"),
        "guard_quality": _op_guard_quality(op),
        "guard_roles": _op_guard_roles(op),
        "guard_generalized_found": op.get("guard_generalized_found"),
        "has_readiness_conditions": bool(op.get("readiness_conditions")),
        "guard_passed": guard_passed,
        "action": {"type": action.get("type", "click"), "x": action.get("x"), "y": action.get("y"), "brief": action.get("brief", "")},
        "stats": {
            "try_count": int(stats.get("try_count", 0) or 0),
            "success_count": int(stats.get("success_count", 0) or 0),
            "effectless_count": int(stats.get("effectless_count", 0) or 0),
        },
    }


def _node(flow: dict[str, Any], node_id: str) -> dict[str, Any]:
    nodes = flow["prefix_tree"]["nodes"]
    if node_id not in nodes or not isinstance(nodes[node_id], dict):
        nodes[node_id] = {
            "node_id": node_id,
            "label": "",
            "notes": "",
            "visits": 0,
            "first_seen_at": _now_iso(),
            "updated_at": _now_iso(),
            "candidate_ops": [],
            "blocked_ops": {},
            "last_observed_actions": [],
            "tried_ops": {},
            "children": {},
        }
    node = nodes[node_id]
    node.setdefault("node_id", node_id)
    node.setdefault("label", "")
    node.setdefault("notes", "")
    node.setdefault("visits", 0)
    node.setdefault("first_seen_at", _now_iso())
    node.setdefault("updated_at", _now_iso())
    if not isinstance(node.get("candidate_ops"), list):
        node["candidate_ops"] = []
    if not isinstance(node.get("blocked_ops"), dict):
        node["blocked_ops"] = {}
    if not isinstance(node.get("last_observed_actions"), list):
        node["last_observed_actions"] = []
    if not isinstance(node.get("tried_ops"), dict):
        node["tried_ops"] = {}
    if not isinstance(node.get("children"), dict):
        node["children"] = {}
    return node


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


def _normalize_guard_role(raw: Any) -> str:
    role = str(raw or "").strip()
    if role in {"target_class", "instance_filter", "readiness_hint"}:
        return role
    return ""


def _normalize_condition(cond: Any) -> dict[str, Any] | None:
    if not isinstance(cond, dict):
        return None
    kind = str(cond.get("kind") or cond.get("type") or "").strip()
    params = cond.get("params") if isinstance(cond.get("params"), dict) else {}
    guard_role = _normalize_guard_role(cond.get("guard_role") or params.get("guard_role"))
    if kind in {"any_of", "or"}:
        raw_items = cond.get("conditions") or cond.get("items") or params.get("conditions")
        if not isinstance(raw_items, list):
            return None
        items = [item for item in (_normalize_condition(raw) for raw in raw_items) if item is not None]
        if not items:
            return None
        bboxes = [item.get("bbox") for item in items if isinstance(item.get("bbox"), list) and len(item.get("bbox")) == 4]
        bbox = bboxes[0] if bboxes else []
        return {
            "id": "",
            "enabled": True,
            "kind": "any_of",
            "params": {"conditions": items},
            "guard_role": guard_role or "instance_filter",
            "brief": str(cond.get("brief", "")),
            "stability": str(cond.get("stability", "mid")),
            "discrimination": str(cond.get("discrimination", "mid")),
            "condition_status": "active",
            "bbox": bbox,
        }
    rect = params.get("rect") or cond.get("bbox")
    if not (isinstance(rect, list) and len(rect) == 4):
        return None
    if kind == "line_contains_text":
        kind = "text_line_contains"
    weak = _normalize_weak_guard({**cond, "kind": kind}) if WEAK_GUARDS_ENABLED else None
    if weak is not None:
        weak["guard_role"] = guard_role
        return weak
    if kind == "text_line_contains":
        text = str(params.get("text") or params.get("contains") or cond.get("text") or "").strip()
        if not text:
            return None
        return {
            "id": "",
            "enabled": True,
            "kind": "text_line_contains",
            "params": {"text": text, "rect": rect},
            "guard_role": guard_role,
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
            "guard_role": guard_role,
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
    if out.get("kind") == "any_of":
        raw_items = params.get("conditions") if isinstance(params.get("conditions"), list) else []
        items: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_items, start=1):
            materialized = _materialize_condition(item, state_dir=state_dir, frame_rgb=frame_rgb, mapper=mapper, suffix=f"{suffix}_or_{idx}")
            if materialized is not None:
                items.append(materialized)
        if not items:
            return None
        params["conditions"] = items
        return out
    if out.get("kind") == WEAK_ROI_KIND and WEAK_GUARDS_ENABLED:
        return _materialize_weak_guard(out, frame_rgb=frame_rgb, mapper=mapper)
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
        ok, detail = _page_op_condition_eval(cond, vision, frame_rgb, mapper)
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
    try:
        resume_ttl = int(expected.get("resume_ttl", 8) or 8)
    except Exception:
        resume_ttl = 8
    guard_quality = "target_class" if any(
        "target_class" in _condition_roles(cond)
        for bucket in (visibility, readiness)
        for cond in bucket
        if isinstance(cond, dict)
    ) else "instance_or_unguarded"
    guard_generalized_found = raw.get("guard_generalized_found")
    if not isinstance(guard_generalized_found, bool):
        guard_generalized_found = guard_quality == "target_class"
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
            "temporary_interrupt": bool(expected.get("temporary_interrupt", False)),
            "resume_policy": str(expected.get("resume_policy", "")),
            "resume_node": str(expected.get("resume_node", "")),
            "intent": str(expected.get("intent", "")),
            "resume_ttl": resume_ttl,
            "observable_changes": expected.get("observable_changes", []) if isinstance(expected.get("observable_changes"), list) else [],
            "reason": str(expected.get("reason", "")),
        },
        "guard_generalized_found": guard_generalized_found,
        "guard_generalization_note": str(raw.get("guard_generalization_note", "")),
        "status": str(raw.get("status", "active")) if str(raw.get("status", "active")) in {"active", "disabled"} else "active",
        "effectless": bool(raw.get("effectless", False)),
        "effectless_reason": str(raw.get("effectless_reason", "")),
        "repair_note": str(raw.get("repair_note", "")),
        "stats": {"try_count": 0, "success_count": 0, "effectless_count": 0},
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


def _merge_catalog_patch(flow: dict[str, Any], patch: dict[str, Any], *, vision: VisionEngine, frame_rgb, mapper: CoordinateMapper, state_dir: Path) -> set[str]:
    catalog = flow["op_catalog"]
    ops = catalog["ops"]
    by_id = {str(op.get("op_id")): op for op in ops if isinstance(op, dict)}
    touched: set[str] = set()
    for raw in patch.get("new_ops", []) if isinstance(patch.get("new_ops"), list) else []:
        op = _normalize_op(raw, vision=vision, frame_rgb=frame_rgb, mapper=mapper, state_dir=state_dir)
        if op is None:
            continue
        if op["op_id"] in by_id:
            continue
        ops.append(op)
        by_id[op["op_id"]] = op
        touched.add(op["op_id"])
    for raw in patch.get("repaired_ops", []) if isinstance(patch.get("repaired_ops"), list) else []:
        if not isinstance(raw, dict) or str(raw.get("repair_scope", "")).strip() != "global":
            continue
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
        touched.add(target)
    disabled = patch.get("disabled_ops") if isinstance(patch.get("disabled_ops"), list) else []
    for raw_disabled in disabled:
        if isinstance(raw_disabled, dict):
            if str(raw_disabled.get("disable_scope", "")).strip() != "global":
                continue
            raw_id = raw_disabled.get("op_id")
        else:
            continue
        op = by_id.get(_slug(raw_id, str(raw_id)))
        if op is not None:
            op["status"] = "disabled"
            op["updated_at"] = _now_iso()
            touched.add(str(op.get("op_id")))
    return touched


def _append_unique(target: list[Any], values: list[Any], *, limit: int | None = None) -> None:
    seen = {str(v) for v in target}
    for raw in values:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        target.append(value)
        seen.add(value)
    if limit is not None and len(target) > limit:
        del target[:-limit]


def _apply_node_patch(flow: dict[str, Any], node_id: str, patch: dict[str, Any]) -> None:
    if not isinstance(patch, dict):
        return
    node = _node(flow, node_id)
    is_root = node_id == "root"
    root_scope = str(patch.get("root_scope", "")).strip() == "global_entry"
    if "label" in patch and (not is_root or root_scope):
        node["label"] = str(patch.get("label") or "")[:120]
    if "notes" in patch and (not is_root or root_scope):
        node["notes"] = str(patch.get("notes") or "")[:240]
    add = patch.get("candidate_ops_add") if isinstance(patch.get("candidate_ops_add"), list) else []
    if is_root:
        add = [op_id for op_id in add if (_find_op(flow, _slug(op_id, str(op_id))) or {}).get("guard_generalized_found") is True]
    remove = {_slug(v, str(v)) for v in patch.get("candidate_ops_remove", [])} if isinstance(patch.get("candidate_ops_remove"), list) else set()
    candidates = node.get("candidate_ops") if isinstance(node.get("candidate_ops"), list) else []
    if remove:
        candidates = [op_id for op_id in candidates if _slug(op_id, str(op_id)) not in remove]
        node["candidate_ops"] = candidates
    _append_unique(candidates, [_slug(v, str(v)) for v in add], limit=64)
    blocked = patch.get("blocked_ops") if isinstance(patch.get("blocked_ops"), dict) else {}
    if is_root and str(patch.get("blocked_scope", "")).strip() != "global_entry":
        blocked = {}
    node_blocked = node.get("blocked_ops") if isinstance(node.get("blocked_ops"), dict) else {}
    node["blocked_ops"] = node_blocked
    for raw_id, raw_reason in blocked.items():
        op_id = _slug(raw_id, str(raw_id))
        if op_id:
            node_blocked[op_id] = {"reason": str(raw_reason)[:300], "updated_at": _now_iso()}
    observed = patch.get("last_observed_actions") if isinstance(patch.get("last_observed_actions"), list) else []
    if observed:
        node["last_observed_actions"] = [v for v in observed if isinstance(v, dict) or isinstance(v, str)][-6:]
    node["updated_at"] = _now_iso()


def _attach_fresh_ops_to_node(flow: dict[str, Any], node_id: str, op_ids: set[str]) -> None:
    if not op_ids:
        return
    if node_id == "root":
        op_ids = {op_id for op_id in op_ids if (_find_op(flow, op_id) or {}).get("guard_generalized_found") is True}
        if not op_ids:
            return
    node = _node(flow, node_id)
    candidates = node.get("candidate_ops") if isinstance(node.get("candidate_ops"), list) else []
    node["candidate_ops"] = candidates
    _append_unique(candidates, sorted(op_ids), limit=64)
    node["updated_at"] = _now_iso()


def _block_node_op(flow: dict[str, Any], node_id: str, op_id: str, reason: str) -> None:
    node = _node(flow, node_id)
    blocked = node.get("blocked_ops") if isinstance(node.get("blocked_ops"), dict) else {}
    node["blocked_ops"] = blocked
    blocked[_slug(op_id, op_id)] = {"reason": reason[:300], "updated_at": _now_iso()}
    node["updated_at"] = _now_iso()


def _is_node_blocked(flow: dict[str, Any], node_id: str, op_id: str) -> bool:
    blocked = _node(flow, node_id).get("blocked_ops")
    return isinstance(blocked, dict) and _slug(op_id, op_id) in blocked


def _parse_page_op_step(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if "effect" in payload or "decision_op_id" in payload or "patch_needed" in payload:
        effect = str(payload.get("effect", "")).strip()
        effect_map = {
            "progress": {"progress": True, "effectless": False},
            "same_progress": {"progress": True, "effectless": False},
            "exit": {"progress": True, "exit_page": True, "effectless": False},
            "effectless": {"progress": False, "effectless": True},
            "no_progress": {"progress": False, "effectless": False},
            "unknown": {"progress": False},
        }
        effect_judgement = dict(effect_map.get(effect, {}))
        if "confidence" in payload:
            effect_judgement["confidence"] = str(payload.get("confidence"))
        if "evidence" in payload:
            effect_judgement["evidence"] = str(payload.get("evidence"))
        decision_op_id = str(payload.get("decision_op_id") or "").strip()
        parsed = {
            "effect_judgement": effect_judgement,
            "catalog_patch": {"should_patch": bool(payload.get("patch_needed", False)), "new_ops": [], "repaired_ops": [], "disabled_ops": []},
            "node_patch": {},
            "decision": {"op_id": decision_op_id} if decision_op_id else {},
            "needs_retry": bool(payload.get("needs_retry", False)),
            "after_progress_try": str(payload.get("after_progress_try") or "").strip(),
            "patch_needed": bool(payload.get("patch_needed", False)),
        }
        return parsed
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
    node_patch = payload.get("node_patch")
    if node_patch is not None and not isinstance(node_patch, dict):
        return None
    payload.setdefault("node_patch", {})
    payload.setdefault("effect_judgement", {})
    payload.setdefault("decision", {})
    payload["needs_retry"] = bool(payload.get("needs_retry", False))
    payload["after_progress_try"] = str(payload.get("after_progress_try") or "").strip()
    payload["patch_needed"] = bool(payload.get("patch_needed", False))
    return payload


def _catalog_patch_has_changes(response: dict[str, Any]) -> bool:
    patch = response.get("catalog_patch") if isinstance(response.get("catalog_patch"), dict) else {}
    return any(isinstance(patch.get(key), list) and len(patch.get(key)) > 0 for key in ("new_ops", "repaired_ops", "disabled_ops"))


def _summarize_prefix_node(flow: dict[str, Any], node_id: str, *, compact: bool) -> dict[str, Any]:
    node = _node(flow, node_id)
    if not compact:
        return node
    tried = node.get("tried_ops") if isinstance(node.get("tried_ops"), dict) else {}
    recent_tried = list(tried.items())[-6:]
    return {
        "node_id": node.get("node_id"),
        "label": node.get("label", ""),
        "candidate_ops": node.get("candidate_ops", []) if isinstance(node.get("candidate_ops"), list) else [],
        "blocked_op_ids": sorted((node.get("blocked_ops") or {}).keys()) if isinstance(node.get("blocked_ops"), dict) else [],
        "recent_tried_ops": [{"op_id": op_id, **data} for op_id, data in recent_tried if isinstance(data, dict)],
    }


def _candidate_op_summaries(flow: dict[str, Any], node_id: str, vision: VisionEngine, frame_rgb, mapper: CoordinateMapper, *, limit: int = 10) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for op_id in _node_candidate_ids(flow, node_id):
        op = _find_op(flow, op_id)
        if op is None or str(op.get("op_id")) in seen:
            continue
        seen.add(str(op.get("op_id")))
        passed = _op_applicable(op, vision, frame_rgb, mapper, allow_unguarded=False)
        out.append(_compact_op_summary(op, guard_passed=passed))
    for op in sorted(_active_ops(flow), key=_guard_rank, reverse=True):
        op_id = str(op.get("op_id"))
        if op_id in seen or _is_node_blocked(flow, node_id, op_id):
            continue
        passed = _op_applicable(op, vision, frame_rgb, mapper, allow_unguarded=False)
        if not passed:
            continue
        seen.add(op_id)
        out.append(_compact_op_summary(op, guard_passed=True))
        if len(out) >= limit:
            break
    return out[:limit]


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
        response_mode = str(request.get("response_mode") or "").strip()
        repair_mode = response_mode == "repair" or (
            response_mode != "normal" and (bool(request.get("need_repair", False)) or bool(request.get("need_catalog_patch", False)))
        )
        if not repair_mode:
            context = {
                "mode": "PAGE_OP_NORMAL_DECIDE",
                "instruction": (
                    "你正在同一外部页面状态内推进流程。只输出严格短 JSON。"
                    "本轮只允许判断上一步效果并从 candidate_ops 中选择下一步；不要创建、修复、禁用 op，也不要输出 node_patch/catalog_patch。"
                    "若候选不足或必须新增/修复，输出 patch_needed=true。"
                    "after_progress_try 只能引用已有 op，表示当前 op 进展后可尝试的下一步。"
                ),
                "current_state": state_slug,
                "current_prefix_node": current_node,
                "request": request,
                "candidate_ops": _candidate_op_summaries(flow, current_node, vision, frame_rgb, mapper),
                "prefix_node": _summarize_prefix_node(flow, current_node, compact=True),
                "output_schema": {
                    "effect": "progress|effectless|no_progress|exit|unknown",
                    "confidence": "high|mid|low",
                    "decision_op_id": "candidate op id or empty",
                    "after_progress_try": "optional existing op id",
                    "patch_needed": "boolean",
                    "needs_retry": "boolean",
                },
            }
        else:
            context = {
            "mode": "PAGE_OP_STEP",
            "instruction": (
                "你正在同一外部页面状态内推进流程。只输出严格 JSON。"
                "你不能输出 from_node/to_node/edge，也不能维护图结构；局部前缀树由 runtime 维护。"
                "如需补充操作，只输出当前截图下可执行或可探索的 op。"
                "条件优先级：同类目标共有的结构性 OCR/固定标签(text_line_contains) > 固定控件/控件状态模板(region_template) > 多候选 OR(any_of) > 弱 ROI 存在性(weak_roi_presence)。"
                "请尽量为当前 op 标注潜在可验证的 OCR 区域和模板区域；即使稳定性或区分度不高，也可以给出，"
                "但必须如实把 stability/discrimination 标为 mid 或 low，不要虚高评分。"
                "为一类操作生成 guard 时，优先寻找同类目标共有的结构性特征，例如固定属性标签、固定控件文字、角标、边框、选中态、固定位置关系；"
                "不要把当前实例名称作为唯一 guard。具体名称、对象名、实例描述、可变属性、具体选项文本只能作为辅助条件或 any_of 的一个分支。"
                "每个 condition 必须尽量填写 guard_role：target_class 表示证明目标属于同一类可交互目标；instance_filter 表示当前实例/偏好筛选；readiness_hint 表示控件已可点击或状态已激活。"
                "select/choose 类 op 的 visibility_conditions 必须至少包含一个 guard_role=target_class；具体名称或具体标题只能是 instance_filter，不能作为唯一可见性条件。"
                "confirm/continue/close 类 op 应把按钮文字或固定控件外观标为 target_class，把激活态/选中态模板放入 readiness_conditions 并标为 readiness_hint。"
                "每个 new_ops/repaired_ops 都必须填写 guard_generalized_found。若找不到 target_class 泛化 guard，必须设为 false，并在 guard_generalization_note 说明缺口；不要用实例条件假装泛化。"
                "当 guard_generalized_found=false 时，仍可输出当前可执行 op，但应在 node_patch 中保持局部分支使用，避免污染 root 或全局策略。"
                "需要 new_ops/repaired_ops 时，先在 catalog_patch.interactive_elements 中列出当前可交互元素；new_ops 的 target_class guard 应来自对应元素的 class_features。"
                "repaired_ops 只有在 repair_scope=global 时才会覆盖旧 op；disabled_ops 只有在 disable_scope=global 时才会全局禁用。当前分支不适用请使用 node_patch.blocked_ops。"
                "root 节点表示通用入口；除非 node_patch.root_scope=global_entry，否则不要修改 root 的 label/notes/candidate_ops/blocked_ops。guard_generalized_found=false 的 op 不应加入 root candidate_ops。"
                "如果同一位置可能出现多个稳定候选词，可用 any_of 表达 OR 条件；any_of.conditions 内仍必须是 text_line_contains、region_template 或 weak_roi_presence。"
                "guard 负责证明目标属于可点击类别；选择理由可以说明为什么当前实例优先，但不要把偏好理由混成唯一 guard。"
                "region_template 只能用于看起来稳定的 2D UI 元素、按钮、图标、控件边缘或控件状态；"
                "不要把 3D 场景中的物体、角色、怪物、地面、墙面、背景、光效、可移动目标或摄像机视角相关区域作为 template。"
                "如果当前截图主要是 3D 场景，优先使用稳定 UI 文字/图标作为条件；没有稳定 UI 时宁可使用低置信探索 op 或 preset，不要裁剪场景物体当模板。"
                "weak_roi_presence 只能在 OCR/template 都不合适时作为兜底；不要把具体对象名称、具体收益名称、数值、进度或实例内容作为可复用条件。"
                "描述操作时使用通用术语：选项、条目、按钮、控件、候选目标、可交互区域；不要使用任务领域对象类别。"
                "condition 只能是 text_line_contains、region_template、any_of 或 weak_roi_presence；请使用 kind 字段，兼容 type 但不推荐。"
                "如果你给出的 condition 在当前截图 dry-run 不通过，runtime 会拒绝它，"
                "但 op 仍可作为低置信探索操作保留。区分 visibility_conditions 和 readiness_conditions：可见不等于可点击。"
                "如果上一步有进度且是第一次进入当前局部节点，必须尽量补充当前截图下的新候选操作，并给出下一步 decision。"
                "当 request.need_catalog_patch=if_progress_enters_new_prefix_node 时，若你判断 progress=true，就等价于必须补充新节点候选操作；"
                "若 progress=false，catalog_patch 可保持 should_patch=false。"
                "如果上一步无进度，优先判断是否需要 retry；retry 后仍无进度时，给出替代 op 或修复已有 op。"
                "action 当前只支持单次 click，坐标为 1000x1000 逻辑坐标。"
                "复用已有 op 前必须确认它在当前截图仍可见且可点击；如果相似按钮/目标在不同页面或分支中位置变化，优先 new_ops 新建当前分支 op，"
                "并用 node_patch.candidate_ops_add 绑定到当前节点；只有确认同一个 op 定义本身错了才用 repaired_ops。"
                "如果旧 op 的意图仍正确但坐标或条件明显错误，请用 repaired_ops 修复旧 op；"
                "如果旧 op 只是当前节点/分支不适用，请用 node_patch.blocked_ops 局部屏蔽，并新建适合当前分支的 op。"
                "op_catalog 是全局动作库，不等于当前节点可执行列表；当前节点的 candidate_ops 才表示本分支优先尝试的操作。"
                "操作无效时优先用 node_patch.blocked_ops 隔离到当前节点，只有确认该 op 全局都不应再用时才写 disabled_ops。"
                "last_observed_actions 只作为诊断记录，不作为可直接执行的动作池。"
            ),
            "current_state": state_slug,
            "current_prefix_node": current_node,
            "request": request,
            "op_catalog_summary": _op_summary(flow),
            "prefix_node": _summarize_prefix_node(flow, current_node, compact=False),
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
                    "interactive_elements": [
                        {
                            "element_id": "short id",
                            "role": "selectable_option|confirm_control|close_control|continue_control|other",
                            "class_features": ["common structural features; max 3"],
                            "instance_features": ["current instance features; max 2"],
                            "guard_generalized_found": "boolean",
                        }
                    ],
                    "new_ops": [
                        {
                            "op_id": "stable id",
                            "repair_scope": "global required only for repaired_ops",
                            "concrete_name": "具体操作",
                            "abstract_name": "泛化操作名",
                            "guard_generalized_found": "boolean; true only when a target_class guard exists",
                            "guard_generalization_note": "short note; required when guard_generalized_found=false",
                            "visibility_conditions": [
                                {
                                    "kind": "text_line_contains|region_template|any_of|weak_roi_presence",
                                    "guard_role": "target_class|instance_filter|readiness_hint",
                                    "params": {"text": "optional", "rect": [0, 0, 0, 0]},
                                    "bbox": [0, 0, 0, 0],
                                    "brief": "prefer target_class structural guard; instance text only as auxiliary",
                                }
                            ],
                            "readiness_conditions": ["same condition kinds; use guard_role=readiness_hint for activated/clickable state"],
                            "action": {"type": "click", "x": 0, "y": 0, "brief": ""},
                            "expected_after_action": {
                                "exit_page": False,
                                "same_page_progress": True,
                                "temporary_interrupt": False,
                                "resume_policy": "",
                                "resume_node": "",
                                "intent": "",
                                "resume_ttl": 8,
                                "observable_changes": [],
                                "reason": "",
                            },
                        }
                    ],
                    "repaired_ops": [],
                    "disabled_ops": [{"op_id": "id", "disable_scope": "global", "reason": "required for global disable"}],
                },
                "node_patch": {
                    "root_scope": "global_entry only when intentionally changing root",
                    "blocked_scope": "global_entry only when blocking at root",
                    "label": "optional current branch label",
                    "notes": "optional current branch notes",
                    "candidate_ops_add": ["op ids preferred in this node"],
                    "candidate_ops_remove": ["op ids no longer suitable for this node"],
                    "blocked_ops": {"op_id": "reason blocked only in this node"},
                    "last_observed_actions": ["diagnostic summaries only"],
                },
                "decision": {"op_id": "existing_or_new_op_id", "reason": "short string"},
                "after_progress_try": "optional existing op id",
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


def _page_op_condition_eval(cond: dict[str, Any], vision: VisionEngine, frame_rgb, mapper: CoordinateMapper) -> tuple[bool, dict[str, Any]]:
    if cond.get("kind") == "any_of":
        items = cond.get("params", {}).get("conditions") if isinstance(cond.get("params"), dict) else []
        details: list[dict[str, Any]] = []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            ok, detail = _page_op_condition_eval(item, vision, frame_rgb, mapper)
            details.append(detail)
            if ok:
                return True, {"kind": "any_of", "passed": True, "matched": detail, "details": details}
        return False, {"kind": "any_of", "passed": False, "details": details}
    if cond.get("kind") == WEAK_ROI_KIND:
        if not WEAK_GUARDS_ENABLED:
            return False, {"kind": WEAK_ROI_KIND, "passed": False, "reason": "weak_guards_disabled"}
        return _eval_weak_guard(cond, frame_rgb=frame_rgb, mapper=mapper)
    return _condition_eval(cond, vision, frame_rgb)


def _condition_list_pass(conditions: list[dict[str, Any]], vision: VisionEngine, frame_rgb, mapper: CoordinateMapper) -> bool:
    if not conditions:
        return True
    return all(_page_op_condition_eval(c, vision, frame_rgb, mapper)[0] for c in conditions if isinstance(c, dict))


def _has_conditions(op: dict[str, Any]) -> bool:
    for key in ("visibility_conditions", "readiness_conditions"):
        conditions = op.get(key)
        if isinstance(conditions, list) and any(isinstance(c, dict) for c in conditions):
            return True
    return False


def _condition_roles(cond: dict[str, Any]) -> set[str]:
    roles: set[str] = set()
    role = _normalize_guard_role(cond.get("guard_role"))
    if role:
        roles.add(role)
    if cond.get("kind") == "any_of":
        params = cond.get("params") if isinstance(cond.get("params"), dict) else {}
        items = params.get("conditions") if isinstance(params.get("conditions"), list) else []
        for item in items:
            if isinstance(item, dict):
                roles |= _condition_roles(item)
    return roles


def _op_guard_roles(op: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key in ("visibility_conditions", "readiness_conditions"):
        roles: set[str] = set()
        conditions = op.get(key)
        if isinstance(conditions, list):
            for cond in conditions:
                if isinstance(cond, dict):
                    roles |= _condition_roles(cond)
        out[key] = sorted(roles)
    return out


def _op_guard_quality(op: dict[str, Any]) -> str:
    visibility = op.get("visibility_conditions")
    if not isinstance(visibility, list) or not any(isinstance(c, dict) for c in visibility):
        return "unguarded"
    roles: set[str] = set()
    for cond in visibility:
        if isinstance(cond, dict):
            roles |= _condition_roles(cond)
    if "target_class" in roles:
        return "target_class"
    if "instance_filter" in roles:
        return "instance_only"
    return "unannotated"


def _condition_rank(cond: dict[str, Any]) -> int:
    ranks = {"text_line_contains": 40, "region_template": 35, WEAK_ROI_KIND: 10}
    if cond.get("kind") == "any_of":
        params = cond.get("params") if isinstance(cond.get("params"), dict) else {}
        items = params.get("conditions") if isinstance(params.get("conditions"), list) else []
        child_best = max((_condition_rank(item) for item in items if isinstance(item, dict)), default=25)
        return child_best - 5
    roles = _condition_roles(cond)
    role_bonus = 20 if "target_class" in roles else -15 if "instance_filter" in roles else 0
    return ranks.get(str(cond.get("kind", "")), 0) + role_bonus


def _guard_rank(op: dict[str, Any]) -> int:
    best = 0
    for key in ("visibility_conditions", "readiness_conditions"):
        conditions = op.get(key)
        if not isinstance(conditions, list):
            continue
        for cond in conditions:
            if isinstance(cond, dict):
                best = max(best, _condition_rank(cond))
    return best


def _op_applicable(op: dict[str, Any], vision: VisionEngine, frame_rgb, mapper: CoordinateMapper, *, allow_unguarded: bool = False) -> bool:
    if op.get("status", "active") != "active" or bool(op.get("effectless", False)):
        return False
    visibility = op.get("visibility_conditions")
    readiness = op.get("readiness_conditions")
    has_guards = _has_conditions(op)
    if not has_guards and not allow_unguarded:
        return False
    if isinstance(visibility, list) and visibility and not _condition_list_pass(visibility, vision, frame_rgb, mapper):
        return False
    if isinstance(readiness, list) and readiness and not _condition_list_pass(readiness, vision, frame_rgb, mapper):
        return False
    return True


def _node_candidate_ids(flow: dict[str, Any], node_id: str, resume_hint: dict[str, Any] | None = None) -> list[str]:
    ids: list[str] = []
    if isinstance(resume_hint, dict):
        expected_node = str(resume_hint.get("expected_node") or resume_hint.get("resume_node") or "")
        previous_node = str(resume_hint.get("previous_node") or "")
        for hint_node in (expected_node, previous_node):
            if hint_node and hint_node in flow.get("prefix_tree", {}).get("nodes", {}):
                _append_unique(ids, [str(v) for v in _node(flow, hint_node).get("candidate_ops", [])])
    _append_unique(ids, [str(v) for v in _node(flow, node_id).get("candidate_ops", [])])
    return ids


def _select_decision_op(
    flow: dict[str, Any],
    node_id: str,
    decision: dict[str, Any],
    vision: VisionEngine,
    frame_rgb,
    mapper: CoordinateMapper,
    *,
    fresh_op_ids: set[str] | None = None,
    resume_hint: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    fresh_op_ids = fresh_op_ids or set()
    op_id = str(decision.get("op_id", "")).strip()
    if op_id:
        normalized_id = _slug(op_id, op_id)
        op = _find_op(flow, normalized_id)
        if op is not None and not _is_node_blocked(flow, node_id, normalized_id) and _op_applicable(op, vision, frame_rgb, mapper, allow_unguarded=normalized_id in fresh_op_ids):
            return op
    for candidate_id in _node_candidate_ids(flow, node_id, resume_hint):
        normalized_id = _slug(candidate_id, candidate_id)
        if _is_node_blocked(flow, node_id, normalized_id):
            continue
        op = _find_op(flow, normalized_id)
        if op is not None and _op_applicable(op, vision, frame_rgb, mapper, allow_unguarded=normalized_id in fresh_op_ids):
            return op
    active = _active_ops(flow)
    guarded = [op for op in active if not _is_node_blocked(flow, node_id, str(op.get("op_id"))) and _op_applicable(op, vision, frame_rgb, mapper)]
    if guarded:
        return sorted(guarded, key=_guard_rank, reverse=True)[0]
    fresh = [op for op in active if str(op.get("op_id")) in fresh_op_ids and not _is_node_blocked(flow, node_id, str(op.get("op_id"))) and _op_applicable(op, vision, frame_rgb, mapper, allow_unguarded=True)]
    return fresh[0] if fresh else None


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
        op["last_effectless_reason"] = reason
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


def _push_resume_hint(runtime: dict[str, Any], *, state_id: str, current_node: str, op: dict[str, Any], logger: FsmRunLogger | None) -> None:
    expected = op.get("expected_after_action") if isinstance(op.get("expected_after_action"), dict) else {}
    if not bool(expected.get("temporary_interrupt", False)):
        return
    try:
        ttl = int(expected.get("resume_ttl", 8) or 8)
    except Exception:
        ttl = 8
    stack = runtime.get("resume_stack")
    if not isinstance(stack, list):
        stack = []
        runtime["resume_stack"] = stack
    hint = {
        "return_state_id": state_id,
        "previous_node": current_node,
        "expected_node": str(expected.get("resume_node") or ""),
        "last_op_id": str(op.get("op_id") or ""),
        "intent": str(expected.get("intent") or expected.get("reason") or "resume_after_interrupt"),
        "resume_policy": str(expected.get("resume_policy") or "prefer_expected_node"),
        "ttl": ttl,
        "created_at": _now_iso(),
    }
    stack.append(hint)
    if len(stack) > 8:
        del stack[:-8]
    _save_runtime(runtime)
    _log(logger, f"[fsm][op][resume] pushed state={state_id} node={current_node} op={hint['last_op_id']}", "page_op_resume_pushed", state_id=state_id, node=current_node, hint=hint)


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
    entry_context: dict[str, Any] | None = None,
    logger: FsmRunLogger | None = None,
) -> tuple[bool, Any]:
    current = start_frame
    state_path = state_dir / "state.json"
    state_meta = _load_json(state_path)
    flow = ensure_page_op_flow(state_meta)
    current_node = "root"
    pending_decision: dict[str, Any] | None = None
    pending_after_progress_try: str = ""
    fresh_op_ids: set[str] = set()
    no_progress_count = 0
    made_progress = False
    emergency_repair_used = False
    entry_context = entry_context if isinstance(entry_context, dict) else {"mode": "normal"}
    resume_hint = entry_context.get("resume_hint") if isinstance(entry_context.get("resume_hint"), dict) else None

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
                "entry_context": entry_context,
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
        fresh = _merge_catalog_patch(flow, response.get("catalog_patch", {}), vision=vision, frame_rgb=current, mapper=mapper, state_dir=state_dir)
        fresh_op_ids |= fresh
        effect = response.get("effect_judgement", {}) if isinstance(response.get("effect_judgement"), dict) else {}
        seed_id = _seed_op_id(seed_step)
        if bool(effect.get("progress", False)):
            made_progress = True
            current_node = _record_tree_result(flow, current_node, seed_id, "progress")
        else:
            _record_tree_result(flow, current_node, seed_id, "effectless")
            no_progress_count = 1 if bool(effect.get("effectless", False)) else 0
        _apply_node_patch(flow, current_node, response.get("node_patch", {}))
        _attach_fresh_ops_to_node(flow, current_node, fresh)
        decision = response.get("decision", {}) if isinstance(response.get("decision"), dict) else {}
        pending_decision = decision if decision.get("op_id") else None
        pending_after_progress_try = str(response.get("after_progress_try") or "")
        state_meta["updated_at"] = _now_iso()
        _save_json(state_path, state_meta)

    request = {
        "has_previous_action": False,
        "previous_action": None,
        "need_effect_judgement": False,
        "need_catalog_patch": True,
        "need_decision": True,
        "reason": "enter_page_op_flow",
        "entry_context": entry_context,
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
            if response.get("patch_needed") and not _catalog_patch_has_changes(response):
                repair_request = dict(request)
                repair_request["response_mode"] = "repair"
                repair_request["need_repair"] = True
                repair_request["instruction"] = "短决策认为需要 patch。请只修复当前节点需要的 op，优先 node_patch 局部隔离，避免污染 root。"
                response = _request_page_op_step(
                    llm=llm,
                    session_id=llm_session_id,
                    frame_rgb=current,
                    previous_frame_rgb=None,
                    system_prompt=system_prompt,
                    state_slug=state_slug,
                    flow=flow,
                    current_node=current_node,
                    request=repair_request,
                    logger=logger,
                    effort="high",
                )
                runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0)) + 1
                _save_runtime(runtime)
                if response is None:
                    return made_progress, current
            fresh = _merge_catalog_patch(flow, response.get("catalog_patch", {}), vision=vision, frame_rgb=current, mapper=mapper, state_dir=state_dir)
            fresh_op_ids |= fresh
            _apply_node_patch(flow, current_node, response.get("node_patch", {}))
            _attach_fresh_ops_to_node(flow, current_node, fresh)
            pending_decision = response.get("decision", {})
            pending_after_progress_try = str(response.get("after_progress_try") or "")
            state_meta["updated_at"] = _now_iso()
            _save_json(state_path, state_meta)

        op = _select_decision_op(flow, current_node, pending_decision or {}, vision, current, mapper, fresh_op_ids=fresh_op_ids, resume_hint=resume_hint)
        pending_decision = None
        if op is None:
            if not emergency_repair_used:
                emergency_repair_used = True
                response = _request_page_op_step(
                    llm=llm,
                    session_id=llm_session_id,
                    frame_rgb=current,
                    previous_frame_rgb=None,
                    system_prompt=system_prompt,
                    state_slug=state_slug,
                    flow=flow,
                    current_node=current_node,
                    request={
                        "has_previous_action": False,
                        "need_effect_judgement": False,
                        "need_catalog_patch": True,
                        "need_decision": True,
                        "need_repair": True,
                        "reason": "no_executable_guarded_op",
                        "instruction": "当前没有任何旧 op 在截图上通过可见/可点击条件。请修复 catalog：优先给出当前截图可验证的新 op 或 repaired_ops，不要复用无条件旧坐标。",
                    },
                    logger=logger,
                    effort="high",
                )
                runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0)) + 1
                _save_runtime(runtime)
                if response is not None:
                    fresh = _merge_catalog_patch(flow, response.get("catalog_patch", {}), vision=vision, frame_rgb=current, mapper=mapper, state_dir=state_dir)
                    fresh_op_ids |= fresh
                    _apply_node_patch(flow, current_node, response.get("node_patch", {}))
                    _attach_fresh_ops_to_node(flow, current_node, fresh)
                    pending_decision = response.get("decision", {}) if isinstance(response.get("decision"), dict) else {}
                    pending_after_progress_try = str(response.get("after_progress_try") or "")
                    state_meta["updated_at"] = _now_iso()
                    _save_json(state_path, state_meta)
                    op = _select_decision_op(flow, current_node, pending_decision or {}, vision, current, mapper, fresh_op_ids=fresh_op_ids, resume_hint=resume_hint)
                    pending_decision = None
            if op is None:
                _log(logger, f"[fsm][op] no executable op node={current_node}", "page_op_no_executable", state_id=state_id, node=current_node)
                return made_progress, current
            fresh_op_ids.discard(str(op.get("op_id")))
        else:
            fresh_op_ids.discard(str(op.get("op_id")))
        resume_hint = None

        if not _execute_click(emulator, mapper, op, logger, state_id=state_id, action_id=action_id, attempt=f"op:{step_idx}"):
            return made_progress, current
        import time

        from state_machine.constants import ACTION_CLICK_WAIT_S

        _log(logger, f"[fsm][op][wait] sleep={ACTION_CLICK_WAIT_S}s after op={op.get('op_id')}", "page_op_wait", state_id=state_id, action_id=action_id, op_id=op.get("op_id"), sleep_s=ACTION_CLICK_WAIT_S)
        time.sleep(ACTION_CLICK_WAIT_S)
        post = _wait_for_screen_stable(emulator, logger=logger, label="page-op", event="page_op_stability_check", max_checks=3)
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
            _push_resume_hint(runtime, state_id=state_id, current_node=current_node, op=op, logger=logger)
            state_meta["updated_at"] = _now_iso()
            _save_json(state_path, state_meta)
            return True, post
        curr_match = _find_match_by_state(matches, state_id)
        if curr_match is None or not curr_match.success:
            _mark_op_result(op, "progress", "")
            _record_tree_result(flow, current_node, str(op.get("op_id")), "unknown")
            _push_resume_hint(runtime, state_id=state_id, current_node=current_node, op=op, logger=logger)
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
                "response_mode": "normal",
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
        if response.get("patch_needed") and not _catalog_patch_has_changes(response):
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
                    "response_mode": "repair",
                    "has_previous_action": True,
                    "previous_op": op.get("op_id"),
                    "need_effect_judgement": True,
                    "need_catalog_patch": True,
                    "need_decision": True,
                    "need_repair": True,
                    "runtime_signals": {"pixel_diff": diff_score, "same_state_still_matches": True},
                    "reason": "normal_mode_requested_patch",
                    "instruction": "短决策认为需要 patch。请给出当前截图下可验证的局部 op 修复，优先 node_patch 而非全局 disabled/repaired。",
                },
                logger=logger,
                effort="high",
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
            retry_post = _wait_for_screen_stable(emulator, logger=logger, label="page-op-retry", event="page_op_retry_stability_check", max_checks=3)
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
                _push_resume_hint(runtime, state_id=state_id, current_node=current_node, op=op, logger=logger)
                state_meta["updated_at"] = _now_iso()
                _save_json(state_path, state_meta)
                return True, retry_post
            curr_retry = _find_match_by_state(retry_matches, state_id)
            if curr_retry is None or not curr_retry.success:
                _mark_op_result(op, "progress", "")
                _record_tree_result(flow, current_node, str(op.get("op_id")), "unknown")
                _push_resume_hint(runtime, state_id=state_id, current_node=current_node, op=op, logger=logger)
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
                    "response_mode": "repair",
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
            fresh = _merge_catalog_patch(flow, response.get("catalog_patch", {}), vision=vision, frame_rgb=post, mapper=mapper, state_dir=state_dir)
            fresh_op_ids |= fresh
            _apply_node_patch(flow, current_node, response.get("node_patch", {}))
            _attach_fresh_ops_to_node(flow, current_node, fresh)
            next_decision = response.get("decision", {}) if isinstance(response.get("decision"), dict) else {}
            pending_decision = next_decision if next_decision.get("op_id") else None
            if pending_decision is None and (response.get("after_progress_try") or pending_after_progress_try):
                pending_decision = {"op_id": str(response.get("after_progress_try") or pending_after_progress_try), "reason": "after_progress_try"}
            pending_after_progress_try = ""
            current = post
        else:
            reason = str(effect.get("evidence") or "llm judged no effective progress")
            if effectless:
                _block_node_op(flow, current_node, str(op.get("op_id")), reason)
                _mark_op_result(op, "effectless", reason)
            _record_tree_result(flow, current_node, str(op.get("op_id")), "effectless")
            fresh = _merge_catalog_patch(flow, response.get("catalog_patch", {}), vision=vision, frame_rgb=post, mapper=mapper, state_dir=state_dir)
            fresh_op_ids |= fresh
            _apply_node_patch(flow, current_node, response.get("node_patch", {}))
            _attach_fresh_ops_to_node(flow, current_node, fresh)
            next_decision = response.get("decision", {}) if isinstance(response.get("decision"), dict) else {}
            pending_decision = next_decision if next_decision.get("op_id") else None
            pending_after_progress_try = str(response.get("after_progress_try") or "")
            no_progress_count += 1
            current = post
            if no_progress_count >= LOCAL_FLOW_NO_PROGRESS_LIMIT:
                if not emergency_repair_used:
                    emergency_repair_used = True
                    response = _request_page_op_step(
                        llm=llm,
                        session_id=llm_session_id,
                        frame_rgb=current,
                        previous_frame_rgb=None,
                        system_prompt=system_prompt,
                        state_slug=state_slug,
                        flow=flow,
                        current_node=current_node,
                        request={
                            "has_previous_action": True,
                            "previous_op": op.get("op_id"),
                            "need_effect_judgement": False,
                            "need_catalog_patch": True,
                            "need_decision": True,
                            "need_repair": True,
                            "runtime_signals": {"no_progress_count": no_progress_count},
                            "reason": "local_no_progress_limit",
                            "instruction": "连续操作无进展。请禁用或修复无效 op，并给出当前截图下可验证的新 op 或 repaired_ops。",
                        },
                        logger=logger,
                        effort="high",
                    )
                    runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0)) + 1
                    _save_runtime(runtime)
                    if response is not None:
                        fresh = _merge_catalog_patch(flow, response.get("catalog_patch", {}), vision=vision, frame_rgb=current, mapper=mapper, state_dir=state_dir)
                        fresh_op_ids |= fresh
                        _apply_node_patch(flow, current_node, response.get("node_patch", {}))
                        _attach_fresh_ops_to_node(flow, current_node, fresh)
                        next_decision = response.get("decision", {}) if isinstance(response.get("decision"), dict) else {}
                        pending_decision = next_decision if next_decision.get("op_id") else None
                        pending_after_progress_try = str(response.get("after_progress_try") or "")
                        no_progress_count = 0
                        state_meta["updated_at"] = _now_iso()
                        _save_json(state_path, state_meta)
                        continue
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
