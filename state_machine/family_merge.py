from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from state_machine.io import _backup_json, _load_frame, _now_iso, _save_json, _slugify
from state_machine.execution import execution_from_payload, execution_key, reactive_bootstrap_operations
from state_machine.logger import FsmRunLogger
from state_machine.page_handler.store import ensure_page_handler, handler_from_bootstrap, materialize_provider_templates, merge_operation
from state_machine.page_identity import build_match_clauses, normalize_elements, record_condition_observations
from state_machine.matching import _condition_passed
from state_machine.state_builder import _conditions_from_elements, _extract_region_templates
from state_machine.state_store import _add_state_sample, _latest_screenshot_path, _meta_for_state


def _log(logger: FsmRunLogger | None, message: str, event: str, **fields: Any) -> None:
    if logger is None:
        print(message)
    else:
        logger.text(message, event, **fields)


def _merge_bootstrap_into_default(handler: dict[str, Any], operations: Any) -> set[str]:
    incoming = handler_from_bootstrap(operations)
    incoming_policies = incoming.get("operation_policies") if isinstance(incoming.get("operation_policies"), dict) else {}
    policies = handler.get("operation_policies") if isinstance(handler.get("operation_policies"), dict) else {}
    handler["operation_policies"] = policies
    default = handler.get("default_operation") if isinstance(handler.get("default_operation"), dict) else None
    default_name = str(default.get("operation") or "") if default else ""
    touched: set[str] = set()
    for name, policy in incoming_policies.items():
        if not isinstance(policy, dict):
            continue
        target_name = default_name or str(name)
        target = policies.get(target_name)
        if not isinstance(target, dict):
            policies[target_name] = policy
            target = policy
            if not default_name:
                handler["default_operation"] = {
                    "operation": target_name,
                    "intent_scope": policy.get("intent_scope", "intent_specific"),
                    "intent_effect": policy.get("intent_effect", "none"),
                }
                default_name = target_name
        else:
            existing_ids = {str(p.get("provider_id")) for p in target.get("providers", []) if isinstance(p, dict)}
            for provider in policy.get("providers", []):
                if not isinstance(provider, dict):
                    continue
                base = str(provider.get("provider_id") or "provider")
                candidate = base
                suffix = 2
                while candidate in existing_ids:
                    candidate = f"{base}_variant_{suffix}"
                    suffix += 1
                provider["provider_id"] = candidate
                existing_ids.add(candidate)
            target, _ = merge_operation(target, policy)
            policies[target_name] = target
        touched.update(str(p.get("provider_id")) for p in target.get("providers", []) if isinstance(p, dict))
    handler["updated_at"] = _now_iso()
    return touched


def try_merge_same_surface(
    *,
    llm_payload: dict[str, Any],
    frame_rgb,
    mapper: CoordinateMapper,
    vision: VisionEngine,
    metas: list[tuple[Path, dict[str, Any]]],
    predecessor_state_id: str | None,
    logger: FsmRunLogger | None = None,
) -> tuple[str, Path] | None:
    """Merge a page-local step into its temporal predecessor without another LLM call."""
    if str(llm_payload.get("surface_relation") or "") != "same_surface_step" or not predecessor_state_id:
        return None
    item = _meta_for_state(metas, predecessor_state_id)
    if item is None:
        return None
    state_dir, meta = item
    previous_family = _slugify(str(meta.get("page_family") or meta.get("page_type") or meta.get("slug") or ""))
    current_family = _slugify(str(llm_payload.get("page_family") or ""))
    if not previous_family or previous_family != current_family:
        _log(logger, f"[fsm][surface] reject predecessor={predecessor_state_id} family={previous_family}->{current_family}", "surface_merge_rejected", predecessor_state_id=predecessor_state_id, reason="family_mismatch")
        return None
    incoming_execution = execution_from_payload(llm_payload)
    if execution_key(meta) != (incoming_execution["kind"], str(incoming_execution.get("tool_name") or "")):
        _log(logger, f"[fsm][surface] reject predecessor={predecessor_state_id} reason=execution_mismatch", "surface_merge_rejected", predecessor_state_id=predecessor_state_id, reason="execution_mismatch")
        return None
    common = {str(value).strip() for value in llm_payload.get("common_identity", []) if str(value).strip()}
    if not common:
        _log(logger, f"[fsm][surface] reject predecessor={predecessor_state_id} reason=no_common_identity", "surface_merge_rejected", predecessor_state_id=predecessor_state_id, reason="no_common_identity")
        return None

    _backup_json(state_dir / "state.json")
    _add_state_sample(state_dir, meta, frame_rgb, role="positive", source="same_surface_step", confidence=0.9, logger=logger)
    normalized = normalize_elements(llm_payload.get("elements", []), vision, frame_rgb)
    known = {(str(e.get("type")), str(e.get("text", "")), tuple(e.get("bbox", []))) for e in meta.get("elements", []) if isinstance(e, dict)}
    for element in normalized:
        key = (str(element.get("type")), str(element.get("text", "")), tuple(element.get("bbox", [])))
        if key not in known:
            meta.setdefault("elements", []).append(element)
            known.add(key)
    previous_path = _latest_screenshot_path(state_dir)
    previous_frame = _load_frame(previous_path) if previous_path is not None else None
    existing_condition_ids = {str(c.get("id") or "") for c in meta.get("match_conditions", []) if isinstance(c, dict)}
    candidates = _conditions_from_elements(normalized)
    for index, candidate in enumerate(candidates, start=1):
        base = str(candidate.get("id") or f"surface_identity_{index}")
        candidate["id"] = f"{base}_surface_{len(existing_condition_ids) + index}"
    candidates = _extract_region_templates(frame_rgb, mapper, state_dir, candidates)
    for candidate in candidates:
        descriptor = " ".join([
            str(candidate.get("brief") or ""),
            str((candidate.get("params") or {}).get("text") or ""),
        ])
        declared_common = any(token in descriptor or descriptor in token for token in common if descriptor)
        if not declared_common or not _condition_passed(candidate, vision, frame_rgb):
            continue
        if previous_frame is not None and not _condition_passed(candidate, vision, previous_frame):
            continue
        meta.setdefault("match_conditions", []).append(candidate)
        existing_condition_ids.add(str(candidate["id"]))
    meta["match_clauses"] = build_match_clauses([
        c for c in meta.get("match_conditions", []) if isinstance(c, dict)
    ])
    if incoming_execution["kind"] == "reactive_2d":
        handler = ensure_page_handler(meta)
        touched = _merge_bootstrap_into_default(handler, reactive_bootstrap_operations(llm_payload))
        materialize_provider_templates(handler, state_dir, frame_rgb, vision, touched)
    meta["page_family"] = previous_family
    meta.setdefault("model_info", {})
    if isinstance(meta["model_info"], dict):
        meta["model_info"]["last_surface_merge"] = {"relation": "same_surface_step", "common_identity": sorted(common), "created_at": _now_iso()}
    meta["updated_at"] = _now_iso()
    record_condition_observations(meta, vision, frame_rgb, cohort="after_action")
    record_condition_observations(meta, vision, frame_rgb, cohort="family_positive")
    _save_json(state_dir / "state.json", meta)
    _log(logger, f"[fsm][surface] merged predecessor={predecessor_state_id} family={previous_family}", "surface_merge_accepted", predecessor_state_id=predecessor_state_id, page_family=previous_family, common_identity=sorted(common))
    return predecessor_state_id, state_dir
