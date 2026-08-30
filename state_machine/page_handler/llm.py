from __future__ import annotations

import json
from typing import Any

from agent.llm_client import DoubaoClient
from agent.responses_protocol import response_output_text
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


def parse_progress_audit(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    verdict = str(payload.get("verdict") or "").strip().lower()
    if verdict not in {"progressing", "new_instance", "stalled", "looping", "uncertain"}:
        return None
    confidence = str(payload.get("confidence") or "low").strip().lower()
    if confidence not in {"high", "mid", "low"}:
        confidence = "low"
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else []
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reason": str(payload.get("reason") or "")[:500],
        "evidence": [str(item)[:240] for item in evidence[:8]],
    }


def request_progress_audit(
    *,
    llm: DoubaoClient,
    session_id: str,
    state_meta: dict[str, Any],
    command: dict[str, Any],
    cursor: dict[str, Any],
    observations: list[dict[str, Any]],
    logger: FsmRunLogger | None,
) -> dict[str, Any] | None:
    """Judge a short visual trajectory; runtime alone owns capacity values."""
    usable = [item for item in observations if isinstance(item, dict) and item.get("frame") is not None]
    if len(usable) < 2:
        return None
    selected = usable if len(usable) <= 4 else [usable[0], *usable[-3:]]
    trajectory = []
    for index, item in enumerate(selected):
        trajectory.append({
            "frame_index": index,
            "provider_id": item.get("provider_id"),
            "provider_kind": item.get("provider_kind"),
            "outcome": item.get("outcome"),
            "changed": item.get("changed"),
            "diff_score": item.get("diff_score"),
            "action_count": item.get("action_count", 0),
            "observation_epoch": item.get("observation_epoch"),
            "note": str(item.get("note") or "")[:240],
        })
    context = {
        "mode": "REACTIVE_PROGRESS_AUDIT",
        "instruction": (
            "这些图片按时间顺序展示同一 page family 最近的一段动作轨迹。"
            "请判断流程是否在产生实际业务推进，不要把动画、闪烁、选中高亮或无意义来回切换当成推进。"
            "如果完成了一轮操作后出现新的同类页面实例，输出 new_instance；"
            "如果步骤或内容持续向目标前进，输出 progressing；若没有有效推进输出 stalled；"
            "若画面和动作序列重复形成循环输出 looping；证据不足输出 uncertain。"
            "你只负责分类，不得建议、猜测或输出任何动作次数、上限、扩容量或新操作。只输出严格 JSON。"
        ),
        "state": {
            "state_id": state_meta.get("state_id"),
            "slug": state_meta.get("slug"),
            "page_type": state_meta.get("page_type"),
            "page_family": state_meta.get("page_family"),
            "description": str(state_meta.get("description") or "")[:240],
        },
        "effective_command": command,
        "runtime_summary": {
            "total_actions": int(cursor.get("total_actions", 0) or 0),
            "observation_epoch": int(cursor.get("observation_epoch", 1) or 1),
            "capacity_extensions": int(cursor.get("capacity_extensions", 0) or 0),
        },
        "trajectory": trajectory,
        "output_schema": {
            "verdict": "progressing|new_instance|stalled|looping|uncertain",
            "confidence": "high|mid|low",
            "reason": "short reason based on temporal evidence",
            "evidence": ["short visible or action-sequence evidence"],
        },
    }
    audit_system_prompt = (
        "你是游戏自动化运行轨迹审计器。你只根据按时间排序的截图和动作摘要判断是否实际推进、"
        "进入同类新实例、停滞或循环。不要规划动作，不要输出任何次数或容量。必须只输出约定 JSON。"
    )
    try:
        resp = llm.responses_json_schema(
            system_prompt=audit_system_prompt,
            user_text=json.dumps(context, ensure_ascii=False),
            image_data_urls=[DoubaoClient.encode_image_to_data_url(item["frame"]) for item in selected],
            schema_name="reactive_progress_audit",
            schema={
                "type": "object",
                "properties": {
                    "verdict": {"type": "string", "enum": ["progressing", "new_instance", "stalled", "looping", "uncertain"]},
                    "confidence": {"type": "string", "enum": ["high", "mid", "low"]},
                    "reason": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                },
                "required": ["verdict", "confidence", "reason", "evidence"],
                "additionalProperties": False,
            },
        )
    except Exception as exc:  # noqa: BLE001
        _log(
            logger,
            f"[fsm][handler][progress-audit] request failed type={type(exc).__name__}",
            "llm_progress_audit_error",
            error_type=type(exc).__name__,
            error_message=str(exc)[:300],
        )
        return None
    raw = response_output_text(resp)
    _save_llm_raw_debug(session_id, 1, raw, "progress_audit", logger.llm_raw_dir if logger is not None else None)
    _log(logger, f"[fsm][handler][progress-audit] raw={raw[:600]}", "llm_progress_audit_raw", raw_preview=raw[:600])
    return parse_progress_audit(raw)


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
                "当前 handler 使用 reactive_local 控制器时，优先向同一 operation 的 controller.providers 追加一个最小立即动作，"
                "并尽可能把本次布局差异提炼为 fallback_profiles（受限区域 OCR、水平/垂直探测、template_offset），"
                "使同族页面的其他布局也能本地推进，而不是只增加当前截图专用的完整多步 strategy。"
                "provider 按 cost 从低到高惰性执行；画面变化后 runtime 会重新观察，禁止生成无条件连点宏。"
                "固定点使用 once_per_visit；OCR、模板或探索可以使用 once_per_observation。"
                "精确动作 expected_after 必须使用 state_relation 和 reentry_policy。"
                "页面内推进可用 may_leave/same_visit；普通退出使用 must_leave/forbid；同类页面连续出现用 must_leave/new_visit。"
                "低风险提示页可以固定坐标；提交或破坏性动作必须带明确安全等级和视觉定位。只输出严格 JSON。"
            ),
            "state": {"state_id": state_meta.get("state_id"), "slug": state_meta.get("slug"), "page_type": state_meta.get("page_type"), "page_family": state_meta.get("page_family"), "description": str(state_meta.get("description", ""))[:240]},
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
                        "controller": {
                            "type": "reactive_local",
                            "max_actions": 12,
                            "max_observation_rounds": 6,
                            "providers": [{
                                "provider_id": "stable id",
                                "kind": "action",
                                "cost": "0 for direct point, 1-10 for exact visual action",
                                "repeat_policy": "once_per_visit|once_per_observation|repeatable",
                                "status": "proposed",
                                "resolver": {"type": "fixed_point|region_template|run_preset", "x": 0, "y": 0, "template_bbox": [0, 0, 0, 0], "search_rect": [0, 0, 0, 0], "threshold": 0.82, "name": ""},
                                "expected_after": {"state_relation": "must_leave|must_remain|may_leave", "reentry_policy": "forbid|new_visit|same_visit"},
                                "emits_on_success": {"type": "semantic event"},
                                "brief": "short string"
                            }],
                            "fallback_profiles": [{
                                "id": "stable id",
                                "kind": "horizontal_probe|confirm_search|template_offset",
                                "region": [0, 0, 0, 0],
                                "keywords": ["确定", "确认", "继续", "前往"],
                                "x_start": 0,
                                "x_end": 0,
                                "y": 0,
                                "samples": 5,
                                "template_path": "",
                                "template_bbox": [0, 0, 0, 0],
                                "click_offset": [0, 0]
                            }]
                        },
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
