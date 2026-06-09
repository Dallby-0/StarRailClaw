from __future__ import annotations

import json
from typing import Any

from agent.llm_client import DoubaoClient
from state_machine.constants import LLM_PARSE_RETRY
from state_machine.llm_tasks import (
    _apply_reasoning_effort,
    _build_user_message_from_frame,
    _build_user_message_from_two_frames,
    _normalize_assistant_text,
    _save_llm_raw_debug,
)
from state_machine.logger import FsmRunLogger
from state_machine.page_handler.store import handler_summary


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)


def parse_handler_response(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    decision = payload.get("decision")
    if decision is not None and not isinstance(decision, dict):
        return None
    payload.setdefault("decision", {})
    handler_patch = payload.get("handler_patch")
    if handler_patch is not None and not isinstance(handler_patch, dict):
        return None
    payload.setdefault("handler_patch", {})
    intent_patch = payload.get("intent_patch")
    if intent_patch is not None and not isinstance(intent_patch, dict):
        return None
    if "intent" in payload and not isinstance(payload.get("intent"), dict):
        return None
    outcome = payload.get("outcome")
    if outcome is not None and not isinstance(outcome, dict):
        return None
    payload.setdefault("outcome", {})
    payload["needs_repair"] = bool(payload.get("needs_repair", False))
    return payload


def request_handler_step(
    *,
    llm: DoubaoClient,
    session_id: str,
    frame_rgb,
    previous_frame_rgb,
    system_prompt: str,
    state_meta: dict[str, Any],
    handler: dict[str, Any],
    active_intent: dict[str, Any] | None,
    request: dict[str, Any],
    logger: FsmRunLogger | None,
    effort: str | None = None,
) -> dict[str, Any] | None:
    old_effort = _apply_reasoning_effort(llm, effort)
    try:
        context = {
            "mode": "PAGE_HANDLER_STEP",
            "instruction": (
                "你在维护一个 intent-driven 页面处理器。高层 state 已经确认当前主页面类型；"
                "不要把随机生成的名称、描述、奖励文字当作可复用前条件。"
                "你可以维护轻量 intent，表示当前来到该页面的目的；也可以修复 intent。"
                "优先使用或创建抽象 action template：select_from_slots、click_active_confirm、click_button、"
                "click_close/continue 可用 click_button 表达、click_blank_to_dismiss、run_preset。"
                "初见时允许低条件盲点，但必须把这类模板 confidence 标为 low 或 mid。"
                "可复用模板应依赖结构位置、按钮/槽位、选中态/激活态等稳定特征；随机文字只能作为排序/偏好。"
                "只输出严格 JSON。"
            ),
            "state": {
                "state_id": state_meta.get("state_id"),
                "slug": state_meta.get("slug"),
                "page_type": state_meta.get("page_type"),
                "description": str(state_meta.get("description", ""))[:240],
                "controller": state_meta.get("controller", {}),
            },
            "active_intent": active_intent or {},
            "handler": handler_summary(handler),
            "request": request,
            "output_schema": {
                "intent": {
                    "kind": "short intent kind when creating/replacing intent",
                    "params": {},
                    "status": "active|candidate_satisfied|satisfied|repairing|abandoned",
                    "confidence": "high|mid|low",
                    "expected": {},
                    "reason": "short string",
                },
                "intent_patch": {
                    "kind": "optional",
                    "params": {},
                    "status": "active|candidate_satisfied|satisfied|repairing|abandoned",
                    "reason": "short string",
                },
                "handler_patch": {
                    "supported_intents": ["intent kinds"],
                    "action_templates": [
                        {
                            "template_id": "stable short id",
                            "kind": "click|click_button|click_active_confirm|select_from_slots|click_blank_to_dismiss|run_preset",
                            "label": "short label",
                            "confidence": "high|mid|low",
                            "bbox": [0, 0, 0, 0],
                            "x": 0,
                            "y": 0,
                            "slots": [{"slot_id": "left", "bbox": [0, 0, 0, 0], "label": "optional"}],
                            "ranking": "how to choose slot using intent/current text; random text is preference only",
                            "name": "preset name for run_preset",
                        }
                    ],
                    "disable_templates": ["template ids"],
                    "memory": {},
                },
                "decision": {
                    "template_id": "existing or newly patched template id",
                    "slot_id": "optional slot id for select_from_slots",
                    "action": {"type": "click|run_preset", "x": 0, "y": 0, "name": "", "brief": ""},
                    "reason": "short string",
                },
                "outcome": {
                    "previous_action_effect": "progress|effectless|exit|unknown",
                    "intent_satisfied": "boolean optional",
                    "evidence": "short string",
                },
                "needs_repair": "boolean",
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
            _save_llm_raw_debug(session_id, attempt, raw, "page_handler_step", logger.llm_raw_dir if logger is not None else None)
            _log(logger, f"[fsm][handler][llm] attempt={attempt} raw={raw[:600]}", "llm_page_handler_raw", attempt=attempt, raw_preview=raw[:600])
            parsed = parse_handler_response(raw)
            if parsed is not None:
                return parsed
    finally:
        llm.reasoning_effort = old_effort
    return None
