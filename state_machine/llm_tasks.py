from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.llm_client import DoubaoClient
from agent.responses_protocol import response_output_text

from . import constants
from .constants import LLM_PARSE_RETRY
from .protocol import parse_state_payload, state_bootstrap_json_schema


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
    return parse_state_payload(text)


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
    active_intent: dict[str, Any] | None = None,
    previous_surface: dict[str, Any] | None = None,
    previous_frame_rgb=None,
    raw_debug_dir: Path | None = None,
) -> dict[str, Any] | None:
    context = {
        "mode": "NORMAL",
        "known_page_types": page_summaries or [],
        "active_intent": active_intent or {},
        "previous_surface": previous_surface or {},
        "instruction": (
            "请按 system 约定输出 JSON。先照常输出用于建立新状态的页面元素信息；"
            "必须先判断 scene_mode：ui_2d、scene_3d 或 unknown。scene_3d 时仅规划 find_and_interact_with_next_object 预置；"
            "possible_page_type 必须从 known_page_types.page_type 中选择，若都不像则输出 none。"
            "同时给出最多两个 bootstrap operation；每个 operation 直接包含 reactive providers。"
            "operation 表达当前页面的一次完整推进；当前画面中明显的短动作链可以在一次调用中给出，"
            "但必须拆成独立 provider，并用 successors 表达后继先验。只有前一步后才出现的特征使用 deferred_hints。"
            "禁止把顺序必做的动作拆成并列 operations；多个 operations 只表示不同 intent 下互斥的操作。"
            "如果 active_intent 非空，判断页面与 intent 的关系，并只为该 intent 或透明阻塞层提出操作。"
            "如果 active_intent 为空，仅当页面存在同识别异操作或操作必须跨多个页面保持语义时，才输出 intent_proposal；"
            "普通唯一推进页面不得创建 intent_proposal。"
            "若 previous_surface 非空，第一张图片是前驱交互表面的样本，第二张是当前图片；"
            "优先判断当前图是否只是同一交互表面的页内下一步，并填写 surface_relation/common_identity。"
            "若是 same_surface_step，bootstrap operation 必须沿用 previous_surface 的完整页面目标语义，"
            "不要按当前临时元素或具体实例重新命名 operation。core provider 必须保持领域无关。"
        ),
    }
    # State recognition and bootstrap action planning share exactly one image
    # request. Responses JSON Schema enforces syntax and shape server-side;
    # this call is deliberately independent from accumulated chat history.
    images = [DoubaoClient.encode_image_to_data_url(frame_rgb)]
    if previous_frame_rgb is not None:
        images.insert(0, DoubaoClient.encode_image_to_data_url(previous_frame_rgb))
    resp = llm.responses_json_schema(
        system_prompt=system_prompt,
        user_text=json.dumps(context, ensure_ascii=False),
        image_data_urls=images,
        schema_name="fsm_state_bootstrap",
        schema=state_bootstrap_json_schema(),
    )
    text = response_output_text(resp)
    _save_llm_raw_debug(session_id, 1, text, "normal", raw_debug_dir)
    print(f"[fsm][llm][responses][json_schema] raw={text[:600]}")
    return _parse_llm_payload(text)


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
