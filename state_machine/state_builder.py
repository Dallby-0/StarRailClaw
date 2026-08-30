from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from state_machine.constants import SCHEMA_VERSION
from state_machine.io import _normalize_page_type, _now_iso, _save_frame, _save_json, _slugify
from state_machine.logger import FsmRunLogger
from state_machine.matching import _level, _select_enabled_conditions
from state_machine.page_handler.store import handler_from_bootstrap, materialize_strategy_templates
from state_machine.state_store import _allocate_state_id, _ensure_unique_state_dir


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


def _create_state_from_llm(
    llm_payload: dict[str, Any],
    frame_rgb,
    mapper: CoordinateMapper,
    vision: VisionEngine,
    logger: FsmRunLogger | None = None,
) -> tuple[str, Path]:
    state_id = _allocate_state_id()
    state_dir = _ensure_unique_state_dir(str(llm_payload.get("slug", "state")))
    _save_frame(state_dir / "screenshot_1.png", frame_rgb)
    conds = _conditions_from_elements(llm_payload.get("elements", []))
    conds = _extract_region_templates(frame_rgb, mapper, state_dir, conds)
    conds, weak_match = _select_enabled_conditions(conds, vision, frame_rgb)
    handler = handler_from_bootstrap(llm_payload.get("bootstrap_operations", []))
    strategy_ids = {
        str(strategy.get("strategy_id"))
        for policy in handler.get("operation_policies", {}).values()
        if isinstance(policy, dict)
        for strategy in policy.get("strategies", [])
        if isinstance(strategy, dict)
    }
    strategy_ids.update(
        str(provider.get("provider_id"))
        for policy in handler.get("operation_policies", {}).values()
        if isinstance(policy, dict) and isinstance(policy.get("controller"), dict)
        for provider in policy["controller"].get("providers", [])
        if isinstance(provider, dict)
    )
    materialize_strategy_templates(handler, state_dir, frame_rgb, vision, strategy_ids)

    state_meta = {
        "schema_version": SCHEMA_VERSION,
        "state_id": state_id,
        "slug": _slugify(str(llm_payload.get("slug", "state"))),
        "page_type": _normalize_page_type(llm_payload.get("possible_page_type") or llm_payload.get("page_type")),
        "page_family": _slugify(str(llm_payload.get("page_family") or _normalize_page_type(llm_payload.get("possible_page_type")) or llm_payload.get("slug") or "generic_page")),
        "display_name": str(llm_payload.get("slug", "state")),
        "description": str(llm_payload.get("page_summary", "")),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "model_info": {"source": "llm", "weak_match": weak_match},
        "elements": llm_payload.get("elements", []),
        "samples": [
            {
                "path": str(state_dir / "screenshot_1.png"),
                "role": "prototype",
                "source": "created",
                "confidence": 1.0,
                "created_at": _now_iso(),
            }
        ],
        "match_conditions": conds,
        "page_handler": handler,
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
            bootstrap_operations_count=len(llm_payload.get("bootstrap_operations", [])),
            conditions=conds,
            screenshot_path=state_dir / "screenshot_1.png",
        )
    return state_id, state_dir
