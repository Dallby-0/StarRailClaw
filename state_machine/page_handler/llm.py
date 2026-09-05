from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from agent.llm_client import DoubaoClient
from state_machine.constants import LLM_PARSE_RETRY
from state_machine.io import _save_frame
from state_machine.llm_tasks import (
    _apply_reasoning_effort,
    _build_user_message_from_frame,
    _build_user_message_from_two_frames,
    _normalize_assistant_text,
    _save_llm_raw_debug,
)
from state_machine.logger import FsmRunLogger
from state_machine.page_handler.store import handler_summary


MAX_QUERY_CELLS = 2


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)


def parse_repair_response(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    decision = str(payload.get("decision") or "").strip().lower()
    if decision == "give_up":
        return {"decision": decision, "reason": str(payload.get("reason") or "")[:500]}
    if decision == "query":
        queries = payload.get("image_queries") if isinstance(payload.get("image_queries"), list) else []
        cell_ids = [str(item.get("cell_id") or "") for item in queries if isinstance(item, dict) and str(item.get("cell_id") or "")]
        if not cell_ids:
            return None
        return {"decision": decision, "image_queries": cell_ids[:MAX_QUERY_CELLS], "reason": str(payload.get("reason") or "")[:500]}
    if decision != "repair" or not isinstance(payload.get("local_patch"), dict):
        return None
    local_patch = payload["local_patch"]
    if not isinstance(local_patch.get("providers"), list):
        return None
    candidates = payload.get("generalization_candidates")
    if candidates is not None and not isinstance(candidates, list):
        return None
    return {
        "decision": decision,
        "local_patch": {"providers": local_patch["providers"]},
        "generalization_candidates": candidates or [],
        "reason": str(payload.get("reason") or "")[:500],
    }


def _make_contact_sheet(history: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    available = [item for item in history if isinstance(item, dict) and item.get("frame") is not None]
    family = next((item for item in available if item.get("kind") == "stored_family_sample"), None)
    selected = ([family] if family is not None else []) + [item for item in available[-5:] if item is not family]
    if not selected:
        raise ValueError("repair history has no frames")
    cell_w, cell_h = 320, 180
    cells: list[np.ndarray] = []
    manifest: list[dict[str, Any]] = []
    frame_index: dict[str, Any] = {}
    for index, item in enumerate(selected, 1):
        frame = item["frame"]
        cell_id = str(item.get("cell_id") or f"cell_{index}")
        resized = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(resized, (0, 0), (cell_w, 24), (0, 0, 0), -1)
        cv2.putText(resized, cell_id, (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(resized)
        entry = {key: value for key, value in item.items() if key != "frame"}
        entry["cell_id"] = cell_id
        manifest.append(entry)
        frame_index[cell_id] = frame
    columns = 2
    rows = (len(cells) + columns - 1) // columns
    sheet = np.zeros((rows * cell_h, columns * cell_w, 3), dtype=np.uint8)
    for index, cell in enumerate(cells):
        row, column = divmod(index, columns)
        sheet[row * cell_h : (row + 1) * cell_h, column * cell_w : (column + 1) * cell_w] = cell
    return sheet, manifest, frame_index


def _save_bundle(logger: FsmRunLogger | None, context: dict[str, Any], current, sheet, history: list[dict[str, Any]]) -> str | None:
    if logger is None:
        return None
    try:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        root = Path(logger.llm_raw_dir) / f"bundle_reactive_repair_{stamp}"
        root.mkdir(parents=True, exist_ok=True)
        (root / "request.json").write_text(json.dumps(context, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        _save_frame(root / "current_full.png", current)
        _save_frame(root / "contact_sheet.png", sheet)
        available = [item for item in history if isinstance(item, dict) and item.get("frame") is not None]
        family = next((item for item in available if item.get("kind") == "stored_family_sample"), None)
        selected = ([family] if family is not None else []) + [item for item in available[-5:] if item is not family]
        for item in selected:
            if isinstance(item, dict) and item.get("frame") is not None:
                _save_frame(root / f"{item.get('cell_id', 'frame')}.png", item["frame"])
        return str(root)
    except Exception:
        return None


def _repair_context(
    *,
    state_meta: dict[str, Any],
    handler: dict[str, Any],
    command: dict[str, Any],
    failures: list[dict[str, Any]],
    cursor_summary: dict[str, Any],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "mode": "REACTIVE_HANDLER_V2_REPAIR",
        "instruction": (
            "Repair only the current operation. The first image is the current full-resolution stable frame; "
            "the second is a chronological contact sheet described by frame_manifest. "
            "Return either a bounded image query, give_up, or a repair. A repair should add the smallest local provider set "
            "that can advance the current instance. When two or more variants reveal a real shared visual invariant, also "
            "propose a separate family-scoped generalization candidate. Never replace a working instance provider and never "
            "generalize by averaging coordinates. Core providers must remain domain-neutral. All persisted points and rectangles "
            "use 1000x1000 logical coordinates. A point locator must explicitly say coordinate_space=logical. "
            "Hints are soft ranking evidence. Effect hints are observable hypotheses, not promises. Deferred hints describe "
            "features that can only be materialized after a predecessor provider. Do not output safety classifications."
        ),
        "state": {
            "state_id": state_meta.get("state_id"),
            "page_family": state_meta.get("page_family"),
            "description": str(state_meta.get("description") or "")[:300],
        },
        "effective_command": command,
        "handler": handler_summary(handler),
        "failed_attempts": failures[-12:],
        "runtime": cursor_summary,
        "frame_manifest": manifest,
        "query_budget": {"max_cells": MAX_QUERY_CELLS, "one_query_round": True},
        "output_schema": {
            "decision": "query|repair|give_up",
            "image_queries": [{"cell_id": "id from frame_manifest"}],
            "local_patch": {"providers": ["provider objects"]},
            "generalization_candidates": ["provider objects with a visual invariant, or empty"],
            "reason": "short string",
            "provider_object": {
                "provider_id": "stable domain-neutral id",
                "base_priority": 0,
                "repeat_policy": "once_per_visit|after_confirmed_effect",
                "locators": [
                    {"type": "point", "x": 0, "y": 0, "coordinate_space": "logical", "source": "bootstrap|verified"},
                    {"type": "region_template", "template_bbox": [0, 0, 0, 0], "search_rect": [0, 0, 0, 0], "threshold": 0.82, "coordinate_space": "logical"},
                    {"type": "text_target", "rect": [0, 0, 0, 0], "texts": ["visible text"], "coordinate_space": "logical"},
                ],
                "hints": [
                    {"id": "id", "type": "template", "template_bbox": [0, 0, 0, 0], "rect": [0, 0, 0, 0], "coordinate_space": "logical"},
                    {"id": "id", "type": "text", "rect": [0, 0, 0, 0], "texts": ["text"], "coordinate_space": "logical"},
                    {"id": "id", "type": "line_count", "rect": [0, 0, 0, 0], "min": 1, "max": 2, "coordinate_space": "logical"},
                ],
                "deferred_hints": [{"id": "id", "type": "template|text|line_count", "materialize_after": "provider_id", "rect": [0, 0, 0, 0], "template_bbox": [0, 0, 0, 0], "coordinate_space": "logical"}],
                "effect_hints": [{"id": "id", "probe": {"id": "id", "type": "template|text|line_count"}, "expected": "pass|becomes_pass|becomes_fail"}],
                "successors": ["provider_id"],
                "emits_on_success": {"type": "optional event"},
                "brief": "short description",
            },
        },
    }


def request_handler_repair(
    *,
    llm: DoubaoClient,
    session_id: str,
    current_frame,
    history: list[dict[str, Any]],
    system_prompt: str,
    state_meta: dict[str, Any],
    handler: dict[str, Any],
    command: dict[str, Any],
    failed_attempts: list[dict[str, Any]],
    cursor_summary: dict[str, Any],
    logger: FsmRunLogger | None,
    effort: str | None = "high",
) -> dict[str, Any] | None:
    sheet, manifest, frame_index = _make_contact_sheet(history)
    context = _repair_context(
        state_meta=state_meta,
        handler=handler,
        command=command,
        failures=failed_attempts,
        cursor_summary=cursor_summary,
        manifest=manifest,
    )
    bundle = _save_bundle(logger, context, current_frame, sheet, history)
    old_effort = _apply_reasoning_effort(llm, effort)
    try:
        prompt = json.dumps(context, ensure_ascii=False)
        for attempt in range(1, LLM_PARSE_RETRY + 2):
            message = _build_user_message_from_two_frames(current_frame, sheet, prompt) if attempt == 1 else {
                "role": "user",
                "content": [{"type": "text", "text": "The previous response was invalid. Return one strict JSON object matching output_schema."}],
            }
            response = llm.chat_with_session(session_id=session_id, system_prompt=system_prompt, user_message=message, tools=[], tool_choice="none")
            raw = _normalize_assistant_text(response["choices"][0]["message"].get("content"))
            _save_llm_raw_debug(session_id, attempt, raw, "reactive_repair", logger.llm_raw_dir if logger is not None else None)
            parsed = parse_repair_response(raw)
            if parsed is None:
                continue
            if parsed["decision"] != "query":
                if bundle:
                    parsed["_input_bundle"] = bundle
                return parsed
            requested = [cell_id for cell_id in parsed["image_queries"] if cell_id in frame_index]
            if not requested:
                continue
            query_text = json.dumps({
                "instruction": "These are the requested full-resolution cells. Now return decision=repair or decision=give_up; no more queries.",
                "cells": requested,
            }, ensure_ascii=False)
            query_message = (
                _build_user_message_from_two_frames(frame_index[requested[0]], frame_index[requested[1]], query_text)
                if len(requested) > 1
                else _build_user_message_from_frame(frame_index[requested[0]], query_text)
            )
            response = llm.chat_with_session(session_id=session_id, system_prompt=system_prompt, user_message=query_message, tools=[], tool_choice="none")
            raw = _normalize_assistant_text(response["choices"][0]["message"].get("content"))
            parsed = parse_repair_response(raw)
            if parsed is not None and parsed["decision"] != "query":
                if bundle:
                    parsed["_input_bundle"] = bundle
                return parsed
    finally:
        llm.reasoning_effort = old_effort
    return None
