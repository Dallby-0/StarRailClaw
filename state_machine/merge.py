from __future__ import annotations

from datetime import datetime
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from agent.llm_client import DoubaoClient
from state_machine import constants
from state_machine.execution import execution_from_payload, execution_key, reactive_bootstrap_operations
from state_machine.io import _backup_json, _load_frame, _load_json, _normalize_page_type, _now_iso, _save_frame, _save_json, _state_page_type
from state_machine.llm_tasks import _request_llm_condition_revision
from state_machine.logger import FsmRunLogger
from state_machine.matching import _condition_passed, _level, _select_enabled_conditions
from state_machine.page_identity import IDENTITY_ROLES, build_match_clauses
from state_machine.page_handler.store import ensure_page_handler, handler_from_bootstrap, materialize_provider_templates, merge_bootstrap_operations, merge_operation
from state_machine.state_store import (
    _latest_screenshot_path,
    _latest_state_for_page_type,
    _sample_screenshot_paths,
    _states_for_page_type,
)


def _handler_operation_names(meta: dict[str, Any]) -> set[str]:
    handler = meta.get("page_handler") if isinstance(meta.get("page_handler"), dict) else {}
    policies = handler.get("operation_policies") if isinstance(handler.get("operation_policies"), dict) else {}
    return {str(name) for name in policies}


def _rewrite_graph_after_state_merge(winner_id: str, loser_ids: set[str]) -> None:
    graph = _load_json(constants.FSM_GRAPH_PATH)
    nodes = [
        node for node in graph.get("nodes", [])
        if isinstance(node, dict) and str(node.get("state_id")) not in loser_ids
    ]
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in graph.get("edges", []):
        if not isinstance(raw, dict):
            continue
        edge = dict(raw)
        source = str(edge.get("from_state_id") or "")
        target = str(edge.get("to_state_id") or "")
        if source in loser_ids:
            source = winner_id
        if target in loser_ids:
            target = winner_id
        edge["from_state_id"] = source
        edge["to_state_id"] = target
        key = (source, str(edge.get("action_id") or ""), target)
        if key in seen:
            continue
        seen.add(key)
        edges.append(edge)
    graph["nodes"] = nodes
    graph["edges"] = edges
    graph["updated_at"] = _now_iso()
    _save_json(constants.FSM_GRAPH_PATH, graph)


def _try_merge_ambiguous_states(
    *,
    winner,
    losers: list,
    metas: list[tuple[Path, dict[str, Any]]],
    vision: VisionEngine,
    frame_rgb,
    logger: FsmRunLogger | None = None,
) -> set[str]:
    """Conservatively merge indistinguishable duplicate states.

    This is the terminal fallback of disambiguation.  It only merges states
    with the same explicit page type and the same operation surface, and only
    when one common condition set covers every stored positive sample.  A
    failed attempt is logged and leaves both states untouched.
    """
    winner_item = next(((d, m) for d, m in metas if str(m.get("state_id")) == winner.state_id), None)
    if winner_item is None:
        return set()
    winner_dir, winner_meta = winner_item
    merged: set[str] = set()
    for loser in losers:
        loser_item = next(((d, m) for d, m in metas if str(m.get("state_id")) == loser.state_id), None)
        if loser_item is None:
            continue
        loser_dir, loser_meta = loser_item
        winner_type = _normalize_page_type(winner_meta.get("page_family") or _state_page_type(winner_meta))
        loser_type = _normalize_page_type(loser_meta.get("page_family") or _state_page_type(loser_meta))
        reason = ""
        same_reactive_surface = (
            execution_key(winner_meta)[0] == "reactive_2d"
            and execution_key(loser_meta)[0] == "reactive_2d"
            and str(winner_meta.get("scene_mode") or "") == str(loser_meta.get("scene_mode") or "")
        )
        if not winner_type or (winner_type != loser_type and not same_reactive_surface):
            reason = "different_or_missing_page_family"
        if not reason and execution_key(winner_meta) != execution_key(loser_meta):
            reason = "different_execution_route"

        frames: list[Any] = [frame_rgb]
        if not reason:
            for state_dir, meta in ((winner_dir, winner_meta), (loser_dir, loser_meta)):
                for path in _sample_screenshot_paths([state_dir]):
                    image = _load_frame(path)
                    if image is not None:
                        frames.append(image)
        combined = [
            deepcopy(cond)
            for meta in (winner_meta, loser_meta)
            for cond in meta.get("match_conditions", [])
            if isinstance(cond, dict) and cond.get("condition_status", "active") == "active" and str(cond.get("role") or "") in IDENTITY_ROLES
        ]
        common = [cond for cond in combined if not reason and all(_condition_passed(cond, vision, image) for image in frames)]
        if not reason and not common:
            reason = "no_common_condition_set"
        if reason:
            _log(logger, f"[fsm][disambiguation][merge] reject winner={winner.state_id} loser={loser.state_id} reason={reason}", "disambiguation_merge_rejected", winner_state_id=winner.state_id, loser_state_id=loser.state_id, reason=reason)
            continue

        common, weak_match = _select_enabled_conditions(common, vision, frame_rgb)
        if not _conditions_pass_all(common, vision, frames):
            _log(logger, f"[fsm][disambiguation][merge] reject winner={winner.state_id} loser={loser.state_id} reason=common_conditions_failed_validation", "disambiguation_merge_rejected", winner_state_id=winner.state_id, loser_state_id=loser.state_id, reason="common_conditions_failed_validation")
            continue
        false_positive = _selected_conditions_match_other_page(common, vision, metas, winner_type)
        if false_positive:
            _log(logger, f"[fsm][disambiguation][merge] reject winner={winner.state_id} loser={loser.state_id} reason=false_positive other={false_positive}", "disambiguation_merge_rejected", winner_state_id=winner.state_id, loser_state_id=loser.state_id, reason="false_positive", other=false_positive)
            continue

        _backup_json(winner_dir / "state.json")
        _backup_json(loser_dir / "state.json")
        winner_meta["match_conditions"] = common
        winner_meta["match_clauses"] = build_match_clauses(common)
        if execution_key(winner_meta)[0] == "reactive_2d":
            winner_handler = ensure_page_handler(winner_meta)
            loser_handler = ensure_page_handler(loser_meta)
            winner_policies = winner_handler.get("operation_policies") if isinstance(winner_handler.get("operation_policies"), dict) else {}
            loser_policies = loser_handler.get("operation_policies") if isinstance(loser_handler.get("operation_policies"), dict) else {}
            winner_default = winner_handler.get("default_operation") if isinstance(winner_handler.get("default_operation"), dict) else {}
            target_name = str(winner_default.get("operation") or "")
            target_policy = winner_policies.get(target_name) if target_name else None
            if isinstance(target_policy, dict):
                for loser_policy in loser_policies.values():
                    if isinstance(loser_policy, dict):
                        target_policy, _ = merge_operation(target_policy, loser_policy)
                        winner_policies[target_name] = target_policy
        winner_samples = winner_meta.setdefault("samples", [])
        known_paths = {str(item.get("path")) for item in winner_samples if isinstance(item, dict)} if isinstance(winner_samples, list) else set()
        if isinstance(winner_samples, list):
            for sample in loser_meta.get("samples", []):
                if isinstance(sample, dict) and str(sample.get("path")) not in known_paths:
                    winner_samples.append(dict(sample))
        winner_meta.setdefault("model_info", {})
        if isinstance(winner_meta["model_info"], dict):
            winner_meta["model_info"]["weak_match"] = weak_match
        merged_ids: list[str] = []
        merged_names: list[str] = []
        for source_meta, source_id in ((winner_meta, winner.state_id), (loser_meta, loser.state_id)):
            source_history = source_meta.get("merge_history") if isinstance(source_meta.get("merge_history"), dict) else {}
            for value in source_history.get("state_ids", []):
                value = str(value or "")
                if value and value not in merged_ids:
                    merged_ids.append(value)
            for value in source_history.get("state_names", []):
                value = str(value or "")
                if value and value not in merged_names:
                    merged_names.append(value)
            if source_id and source_id not in merged_ids:
                merged_ids.append(source_id)
            source_name = str(source_meta.get("slug") or source_meta.get("display_name") or source_id)
            if source_name and source_name not in merged_names:
                merged_names.append(source_name)
        winner_meta["merge_history"] = {
            "state_ids": merged_ids,
            "state_names": merged_names,
            "reason": "indistinguishable_reactive_surface",
            "updated_at": _now_iso(),
        }
        winner_meta["updated_at"] = _now_iso()
        _save_json(winner_dir / "state.json", winner_meta)

        for condition in loser_meta.get("match_conditions", []):
            if isinstance(condition, dict):
                condition["enabled"] = False
        loser_meta.setdefault("model_info", {})
        if isinstance(loser_meta["model_info"], dict):
            loser_meta["model_info"]["merged_into_state_id"] = winner.state_id
        loser_meta["updated_at"] = _now_iso()
        _save_json(loser_dir / "state.json", loser_meta)
        merged.add(loser.state_id)
        _log(logger, f"[fsm][disambiguation][merge] accepted winner={winner.state_id} loser={loser.state_id}", "disambiguation_merge_accepted", winner_state_id=winner.state_id, loser_state_id=loser.state_id, page_type=winner_type)

    if merged:
        _rewrite_graph_after_state_merge(winner.state_id, merged)
    return merged


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)


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
        raw_debug_dir=logger.llm_raw_dir if logger is not None else None,
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
    samples = latest_meta.setdefault("samples", [])
    if isinstance(samples, list):
        samples.append(
            {
                "path": str(latest_dir / f"screenshot_{next_idx}.png"),
                "role": "positive",
                "source": "merge",
                "confidence": 0.8,
                "created_at": _now_iso(),
            }
        )
    latest_meta["match_conditions"] = revised
    incoming_execution = execution_from_payload(llm_payload)
    bootstrap_raw = reactive_bootstrap_operations(llm_payload)
    if incoming_execution["kind"] == "reactive_2d":
        if execution_key(latest_meta)[0] == "reactive_2d" and isinstance(latest_meta.get("page_handler"), dict):
            handler = ensure_page_handler(latest_meta)
            touched = merge_bootstrap_operations(handler, bootstrap_raw)
            bootstrap = handler_from_bootstrap(bootstrap_raw)
            if handler.get("default_operation") is None and bootstrap.get("default_operation") is not None:
                handler["default_operation"] = bootstrap["default_operation"]
            known_routes = {str(item) for item in handler.get("intent_routes", []) if isinstance(item, dict)}
            for route in bootstrap.get("intent_routes", []):
                if isinstance(route, dict) and str(route) not in known_routes:
                    handler["intent_routes"].append(route)
        else:
            handler = handler_from_bootstrap(bootstrap_raw)
            latest_meta["page_handler"] = handler
            touched = {
                str(provider.get("provider_id"))
                for policy in handler.get("operation_policies", {}).values()
                if isinstance(policy, dict)
                for provider in policy.get("providers", [])
                if isinstance(provider, dict)
            }
        materialize_provider_templates(handler, latest_dir, frame_rgb, vision, touched)
    else:
        latest_meta.pop("page_handler", None)
    latest_meta["updated_at"] = _now_iso()
    latest_meta["page_type"] = page_type
    latest_meta["scene_mode"] = str(llm_payload.get("scene_mode") or latest_meta.get("scene_mode") or "unknown")
    latest_meta["execution"] = incoming_execution
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
