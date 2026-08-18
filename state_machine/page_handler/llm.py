from __future__ import annotations

import json
from typing import Any

from agent.llm_client import DoubaoClient
from state_machine.constants import LLM_PARSE_RETRY
from state_machine.llm_tasks import _apply_reasoning_effort, _build_user_message_from_frame, _build_user_message_from_two_frames, _normalize_assistant_text, _save_llm_raw_debug
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
    if not isinstance(payload, dict) or not isinstance(payload.get("handler_patch"), dict):
        return None
    event = payload.get("event")
    if event is not None and not isinstance(event, dict):
        return None
    payload.setdefault("event", {})
    return payload


def request_handler_repair(
    *,
    llm: DoubaoClient,
    session_id: str,
    frame_rgb,
    previous_frame_rgb,
    system_prompt: str,
    state_meta: dict[str, Any],
    handler: dict[str, Any],
    command: dict[str, Any],
    failed_attempts: list[dict[str, Any]],
    logger: FsmRunLogger | None,
    effort: str | None = "high",
) -> dict[str, Any] | None:
    old_effort = _apply_reasoning_effort(llm, effort)
    try:
        context = {
            "mode": "PROGRESSIVE_HANDLER_REPAIR",
            "instruction": (
                "当前页面状态已确认。请只修复 effective_command 对应的页面操作策略，不要创建或修改 intent。"
                "优先提出比已失败策略更可靠的 resolver：固定点失败后可给 region_template，使用当前图上的 template_bbox 和受限 search_rect。"
                "最多给 2 个条件步骤；每步执行后系统都会重新截图验证，禁止无条件连点。"
                "低风险提示页可以固定坐标；提交或破坏性动作必须带明确安全等级和视觉定位。只输出严格 JSON。"
            ),
            "state": {"state_id": state_meta.get("state_id"), "slug": state_meta.get("slug"), "page_type": state_meta.get("page_type"), "description": str(state_meta.get("description", ""))[:240]},
            "effective_command": command,
            "handler": handler_summary(handler),
            "failed_attempts": failed_attempts[-4:],
            "output_schema": {
                "handler_patch": {
                    "operations": [{
                        "operation": "must equal effective_command.operation",
                        "intent_scope": "intent_specific|intent_invariant",
                        "intent_effect": "advance|preserve|complete|none",
                        "safety": "low_risk|reversible|commit|destructive",
                        "expected_event": "optional semantic event",
                        "strategies": [{
                            "strategy_id": "stable id",
                            "level": "integer; stronger than failed strategy",
                            "status": "proposed",
                            "steps": [{
                                "step_id": "short id",
                                "resolver": {"type": "fixed_point|region_template|run_preset", "x": 0, "y": 0, "template_bbox": [0, 0, 0, 0], "search_rect": [0, 0, 0, 0], "threshold": 0.82, "name": ""},
                                "expected_after": {"screen_should_change": True, "same_page_likely": False, "exit_likely": True},
                                "emits_on_success": {"type": "semantic event"},
                                "brief": "short string",
                            }],
                        }],
                    }],
                },
                "event": {"type": "optional event already evident on current page", "facts": {}},
                "reason": "short string",
            },
        }
        text = json.dumps(context, ensure_ascii=False)
        for attempt in range(1, LLM_PARSE_RETRY + 2):
            if attempt == 1:
                msg = _build_user_message_from_two_frames(previous_frame_rgb, frame_rgb, text) if previous_frame_rgb is not None else _build_user_message_from_frame(frame_rgb, text)
            else:
                msg = {"role": "user", "content": [{"type": "text", "text": "上次 JSON 无效。只输出符合 output_schema 的完整 JSON，不要解释。"}]}
            resp = llm.chat_with_session(session_id=session_id, system_prompt=system_prompt, user_message=msg, tools=[], tool_choice="none")
            raw = _normalize_assistant_text(resp["choices"][0]["message"].get("content"))
            _save_llm_raw_debug(session_id, attempt, raw, "page_handler_repair", logger.llm_raw_dir if logger is not None else None)
            _log(logger, f"[fsm][handler][repair-llm] attempt={attempt} raw={raw[:600]}", "llm_page_handler_repair_raw", attempt=attempt, raw_preview=raw[:600])
            parsed = parse_handler_response(raw)
            if parsed is not None:
                return parsed
    finally:
        llm.reasoning_effort = old_effort
    return None
