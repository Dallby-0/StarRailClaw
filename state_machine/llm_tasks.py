from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.llm_client import DoubaoClient

from . import constants
from .constants import LLM_PARSE_RETRY


def _normalize_assistant_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in {"text", "output_text"}]
        return "".join(parts).strip()
    return ""


def _build_user_message_from_frame(image_rgb, text: str = "请按约定输出JSON") -> dict[str, Any]:
    data_url = DoubaoClient.encode_image_to_data_url(image_rgb)
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }


def _build_user_message_from_two_frames(image_a_rgb, image_b_rgb, text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": DoubaoClient.encode_image_to_data_url(image_a_rgb)}},
            {"type": "image_url", "image_url": {"url": DoubaoClient.encode_image_to_data_url(image_b_rgb)}},
        ],
    }


def _parse_llm_payload(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    required = {"page_summary", "slug", "elements", "actions"}
    if not required.issubset(payload.keys()):
        return None
    payload.setdefault("possible_page_type", "none")
    return payload


def _parse_llm_repair(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    required = {"judgement", "mode", "actions"}
    if not required.issubset(payload.keys()):
        return None
    if payload.get("mode") not in {"override", "append"}:
        return None
    if payload.get("judgement") not in {"invalid", "partial"}:
        return None
    if not isinstance(payload.get("actions"), list):
        return None
    return payload


def _parse_llm_failure_diagnosis(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    diagnosis = payload.get("diagnosis")
    if diagnosis not in {"state_misidentified", "action_wrong", "action_partial", "wait_needed", "unknown"}:
        return None
    return payload


def _parse_llm_disambiguation(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    winner = str(payload.get("winner_state_id", "")).strip()
    if not winner:
        return None
    confidence = str(payload.get("confidence", "mid")).strip().lower()
    if confidence not in {"high", "mid", "low"}:
        confidence = "mid"
    payload["confidence"] = confidence
    return payload


def _parse_llm_condition_revision(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("same_page_type") not in {True, False}:
        return None
    if payload.get("same_page_type") is True and not isinstance(payload.get("conditions"), list):
        return None
    return payload


def _save_llm_raw_debug(session_id: str, attempt: int, text: str, kind: str = "normal", debug_dir: Path | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = debug_dir or constants.FSM_DEBUG_DIR
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"llm_raw_{kind}_{session_id}_attempt{attempt}_{ts}.txt"
    p.write_text(text, encoding="utf-8")
    return p


def _apply_reasoning_effort(llm: DoubaoClient, effort: str | None) -> str | None:
    old = llm.reasoning_effort
    llm.reasoning_effort = effort
    return old


def _request_llm_payload(
    llm: DoubaoClient,
    session_id: str,
    frame_rgb,
    system_prompt: str,
    page_summaries: list[dict[str, Any]] | None = None,
    raw_debug_dir: Path | None = None,
) -> dict[str, Any] | None:
    context = {
        "mode": "NORMAL",
        "known_page_types": page_summaries or [],
        "instruction": (
            "请按 system 约定输出 JSON。先照常输出用于建立新状态的页面元素信息；"
            "possible_page_type 必须从 known_page_types.page_type 中选择，若都不像则输出 none。"
        ),
    }
    msg = _build_user_message_from_frame(frame_rgb, json.dumps(context, ensure_ascii=False))
    prompt_fix = "上次JSON无效或不完整。只输出一个完整JSON对象，不要省略字段，不要解释文字。"
    for attempt in range(1, LLM_PARSE_RETRY + 2):
        payload_msg = msg if attempt == 1 else {"role": "user", "content": [{"type": "text", "text": prompt_fix}]}
        resp = llm.chat_with_session(
            session_id=session_id,
            system_prompt=system_prompt,
            user_message=payload_msg,
            tools=[],
            tool_choice="none",
        )
        text = _normalize_assistant_text(resp["choices"][0]["message"].get("content"))
        _save_llm_raw_debug(session_id, attempt, text, "normal", raw_debug_dir)
        print(f"[fsm][llm] raw(attempt={attempt})={text[:600]}")
        parsed = _parse_llm_payload(text)
        if parsed is not None:
            return parsed
    return None


def _request_llm_condition_revision(
    llm: DoubaoClient,
    session_id: str,
    system_prompt: str,
    *,
    latest_sample_rgb,
    current_rgb,
    page_type: str,
    latest_meta: dict[str, Any],
    failed_conditions: list[dict[str, Any]],
    new_payload: dict[str, Any],
    raw_debug_dir: Path | None = None,
) -> dict[str, Any] | None:
    text = json.dumps(
        {
            "mode": "MERGE_CONDITION_REVISION",
            "page_type": page_type,
            "instruction": (
                "第一张图是该 page_type 最新样板截图，第二张图是当前新截图。"
                "请判断第二张是否仍属于同一 page_type。若不是，same_page_type=false。"
                "若是，请逐个修订当前正式条件，使其更像同类页面共有条件。"
                "文本条件只支持 line_contains_text，其含义是：在单行 OCR 结果中 contains 某个子串。"
                "每个文本条件的 rect 必须只覆盖一行文本，不能覆盖多行、段落或相邻文字。"
                "请提取同类页面共有的稳定短子串；当一行文字同时包含可变实例信息和稳定页面类型词时，"
                "应丢弃名称、编号、进度、计数、序号、轮次、等级、数量等可变部分，只保留稳定类别词。"
                "避免使用长正文、叙事描述、具体实例名称、对象名称、奖励名称、数值、编号、进度、计数等容易变化的内容。"
                "如果某个子串过于通用，应降低 discrimination；如果不确定是否稳定，应降低 stability。"
                "模板条件可修订 rect/threshold；若该元素不是共有元素，请 deprecate。"
                "输出严格 JSON。"
            ),
            "latest_state": {
                "slug": latest_meta.get("slug"),
                "description": latest_meta.get("description"),
                "conditions": latest_meta.get("match_conditions", []),
            },
            "failed_conditions_on_current": failed_conditions,
            "new_page_payload_summary": {
                "slug": new_payload.get("slug"),
                "page_summary": new_payload.get("page_summary"),
                "elements": new_payload.get("elements", []),
            },
            "output_schema": {
                "same_page_type": "boolean",
                "conditions": [
                    {
                        "condition_id": "existing id",
                        "decision": "keep|revise|deprecate",
                        "kind": "text_line_contains|region_template",
                        "params": {},
                        "brief": "string",
                        "stability": "high|mid|low",
                        "discrimination": "high|mid|low",
                    }
                ],
                "note": "string",
            },
        },
        ensure_ascii=False,
    )
    for attempt in range(1, LLM_PARSE_RETRY + 2):
        msg = _build_user_message_from_two_frames(latest_sample_rgb, current_rgb, text)
        resp = llm.chat_with_session(
            session_id=session_id,
            system_prompt=system_prompt,
            user_message=msg,
            tools=[],
            tool_choice="none",
        )
        raw = _normalize_assistant_text(resp["choices"][0]["message"].get("content"))
        _save_llm_raw_debug(session_id, attempt, raw, "merge_condition", raw_debug_dir)
        print(f"[fsm][merge][llm] raw(attempt={attempt})={raw[:600]}")
        parsed = _parse_llm_condition_revision(raw)
        if parsed is not None:
            return parsed
    return None


def _request_llm_failure_diagnosis(
    llm: DoubaoClient,
    session_id: str,
    frame_rgb,
    system_prompt: str,
    *,
    believed_state: dict[str, Any],
    action_steps: list[dict[str, Any]],
    runtime_observation: dict[str, Any],
    raw_debug_dir: Path | None = None,
) -> dict[str, Any] | None:
    text = json.dumps(
        {
            "mode": "FAILURE_DIAGNOSIS",
            "instruction": (
                "当前系统执行动作后没有得到明确转移。请先诊断失败原因，而不是直接修动作。"
                "重点判断：系统认为的当前状态是否可能和截图不一致。只输出严格 JSON。"
            ),
            "believed_state": believed_state,
            "action_steps": action_steps,
            "runtime_observation": runtime_observation,
            "output_schema": {
                "diagnosis": "state_misidentified|action_wrong|action_partial|wait_needed|unknown",
                "same_as_believed_state": "boolean",
                "possible_existing_state_id": "string or none",
                "needs_new_state": "boolean",
                "reason": "string",
            },
        },
        ensure_ascii=False,
    )
    for attempt in range(1, LLM_PARSE_RETRY + 2):
        msg = _build_user_message_from_frame(frame_rgb, text)
        resp = llm.chat_with_session(
            session_id=session_id,
            system_prompt=system_prompt,
            user_message=msg,
            tools=[],
            tool_choice="none",
        )
        raw = _normalize_assistant_text(resp["choices"][0]["message"].get("content"))
        _save_llm_raw_debug(session_id, attempt, raw, "failure_diagnosis", raw_debug_dir)
        parsed = _parse_llm_failure_diagnosis(raw)
        if parsed is not None:
            return parsed
    return None


def _request_llm_disambiguation(
    llm: DoubaoClient,
    session_id: str,
    frame_rgb,
    system_prompt: str,
    *,
    candidates: list[dict[str, Any]],
    raw_debug_dir: Path | None = None,
) -> dict[str, Any] | None:
    text = json.dumps(
        {
            "mode": "DISAMBIGUATE_STATE",
            "instruction": (
                "当前截图同时满足多个候选状态。请只根据当前截图和候选摘要选择最符合的一个。"
                "如果都不像，winner_state_id 输出 none。不要生成或修改匹配条件。只输出严格 JSON。"
            ),
            "candidates": candidates,
            "output_schema": {
                "winner_state_id": "state_id|none",
                "confidence": "high|mid|low",
                "reason": "string",
                "visible_evidence": ["string"],
            },
        },
        ensure_ascii=False,
    )
    for attempt in range(1, LLM_PARSE_RETRY + 2):
        msg = _build_user_message_from_frame(frame_rgb, text)
        resp = llm.chat_with_session(
            session_id=session_id,
            system_prompt=system_prompt,
            user_message=msg,
            tools=[],
            tool_choice="none",
        )
        raw = _normalize_assistant_text(resp["choices"][0]["message"].get("content"))
        _save_llm_raw_debug(session_id, attempt, raw, "disambiguation", raw_debug_dir)
        parsed = _parse_llm_disambiguation(raw)
        if parsed is not None:
            return parsed
    return None
