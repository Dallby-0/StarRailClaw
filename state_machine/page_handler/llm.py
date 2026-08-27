from __future__ import annotations

import json
from typing import Any

from agent.llm_client import DoubaoClient
from state_machine.constants import LLM_PARSE_RETRY
from state_machine.llm_tasks import _apply_reasoning_effort, _build_user_message_from_frame, _build_user_message_from_two_frames, _normalize_assistant_text, _save_llm_raw_debug
from state_machine.logger import FsmRunLogger
from state_machine.page_handler.store import handler_summary
from state_machine.page_handler.review_protocol import parse_same_state_review


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


def request_same_state_review(
    *,
    llm: DoubaoClient,
    session_id: str,
    before_frame_rgb,
    after_frame_rgb,
    system_prompt: str,
    state_meta: dict[str, Any],
    handler: dict[str, Any],
    command: dict[str, Any],
    executed_action: dict[str, Any],
    logger: FsmRunLogger | None,
) -> dict[str, Any] | None:
    """Resolve the semantic ambiguity when an action ends in the same state class.

    This is deliberately one model call. The same response both judges the
    previous action and supplies a corrected/continuation strategy when needed.
    """
    context = {
        "mode": "SAME_STATE_ACTION_REVIEW",
        "instruction": (
            "第一张图是执行前，第二张图是执行后。外部 matcher 认为两张图属于同一页面状态。"
            "请判断上一个动作对 effective_command 的完整目标是否有效，而不是仅判断是否出现像素变化。"
            "即使 effective_command 的旧名称只写了 select，也要以完成当前页面的一次推进为目标，而不是拘泥于窄名称。"
            "若动作已经推进页面但仍需重复同一动作，应判 partial_needs_continue，continuation.reuse_strategy=true，"
            "无需生成同义 handler_patch；若只是选中了卡片且下一步需要点击不同的确认按钮，也判 partial_needs_continue，"
            "但 continuation.reuse_strategy=false，并在 handler_patch 中给出后续策略。若点击完全无效则判 ineffective 并给出替代策略。"
            "若旧页面已经完成、第二张图是连续弹出的同类新事件，判 completed_new_visit。"
            "如果第二张图仍有已启用的确认、继续、提交按钮，禁止判 completed_same_visit；必须判 partial_needs_continue。"
            "只有操作的完整页面推进语义已经完成且确实应停留在当前页面实例时才判 completed_same_visit。"
            "不得创建或修改 intent，patch 中 operation 必须等于 effective_command.operation。只输出严格 JSON。"
        ),
        "state": {"state_id": state_meta.get("state_id"), "slug": state_meta.get("slug"), "description": str(state_meta.get("description", ""))[:240]},
        "effective_command": command,
        "executed_action": executed_action,
        "handler": handler_summary(handler),
        "output_schema": {
            "verdict": "completed_same_visit|completed_new_visit|partial_needs_continue|ineffective|wrong_effect|uncertain",
            "reason": "short visual/semantic reason",
            "event": {"type": "optional observed semantic event, including page-local progress"},
            "continuation": {
                "reuse_strategy": "true only when repeating the executed strategy is the correct next action",
                "max_additional_actions": "integer 1..12; omit unless partial_needs_continue"
            },
            "handler_patch": {
                "operations": [{
                    "operation": "must equal effective_command.operation",
                    "intent_scope": "intent_specific|intent_invariant",
                    "intent_effect": "advance|preserve|complete|none",
                    "safety": "low_risk|reversible|commit|destructive",
                    "expected_event": "semantic event",
                    "strategies": [{
                        "strategy_id": "new stable id",
                        "level": "integer greater than failed strategy when correcting",
                        "status": "proposed",
                        "steps": [{
                            "step_id": "short id",
                            "resolver": {"type": "fixed_point|region_template|run_preset", "x": 0, "y": 0, "template_bbox": [0, 0, 0, 0], "search_rect": [0, 0, 0, 0], "threshold": 0.82, "name": ""},
                            "expected_after": {"state_relation": "must_leave|must_remain|may_leave", "reentry_policy": "forbid|new_visit|same_visit"},
                            "emits_on_success": {"type": "semantic event"},
                            "brief": "short string"
                        }]
                    }]
                }]
            }
        },
    }
    msg = _build_user_message_from_two_frames(before_frame_rgb, after_frame_rgb, json.dumps(context, ensure_ascii=False))
    resp = llm.chat_with_session(session_id=session_id, system_prompt=system_prompt, user_message=msg, tools=[], tool_choice="none")
    raw = _normalize_assistant_text(resp["choices"][0]["message"].get("content"))
    _save_llm_raw_debug(session_id, 1, raw, "same_state_review", logger.llm_raw_dir if logger is not None else None)
    _log(logger, f"[fsm][handler][same-state-review] raw={raw[:600]}", "llm_same_state_review_raw", raw_preview=raw[:600])
    return parse_same_state_review(raw)


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
                "expected_after 必须使用 state_relation 和 reentry_policy。页面内步骤用 must_remain/same_visit；"
                "必须退出且同类页面可能连续出现时用 must_leave/new_visit；普通关闭用 must_leave/forbid。"
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
                                "expected_after": {"state_relation": "must_leave|must_remain|may_leave", "reentry_policy": "forbid|new_visit|same_visit"},
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
