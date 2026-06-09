from __future__ import annotations

from typing import Any

from state_machine.io import _now_iso

HANDLER_SCHEMA_VERSION = "page_handler.v1"
INTENT_RUNTIME_KEY = "page_handler_intents"
TRACE_LIMIT = 40


def _slug(raw: Any, fallback: str) -> str:
    text = str(raw or "").strip().lower()
    safe = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("_")
    return safe or fallback


def ensure_page_handler(meta: dict[str, Any]) -> dict[str, Any]:
    handler = meta.get("page_handler")
    if not isinstance(handler, dict):
        handler = {}
        meta["page_handler"] = handler
    handler.setdefault("schema_version", HANDLER_SCHEMA_VERSION)
    handler.setdefault("handler_kind", "intent_driven")
    handler.setdefault("created_at", _now_iso())
    handler["updated_at"] = _now_iso()
    if not isinstance(handler.get("supported_intents"), list):
        handler["supported_intents"] = []
    if not isinstance(handler.get("action_templates"), dict):
        handler["action_templates"] = {}
    if not isinstance(handler.get("memory"), dict):
        handler["memory"] = {}
    if not isinstance(handler.get("failure_stats"), dict):
        handler["failure_stats"] = {}
    if not isinstance(handler.get("episode_trace"), list):
        handler["episode_trace"] = []
    return handler


def handler_summary(handler: dict[str, Any], *, limit_templates: int = 12, limit_trace: int = 8) -> dict[str, Any]:
    templates = handler.get("action_templates") if isinstance(handler.get("action_templates"), dict) else {}
    out_templates: list[dict[str, Any]] = []
    for template_id, raw in list(templates.items())[:limit_templates]:
        if not isinstance(raw, dict) or raw.get("status", "active") != "active":
            continue
        item = {
            "template_id": template_id,
            "kind": raw.get("kind"),
            "label": raw.get("label", ""),
            "confidence": raw.get("confidence", "mid"),
            "success_count": int(raw.get("success_count", 0) or 0),
            "fail_count": int(raw.get("fail_count", 0) or 0),
        }
        if isinstance(raw.get("slots"), list):
            item["slots"] = [
                {
                    "slot_id": str(slot.get("slot_id") or slot.get("id") or idx),
                    "bbox": slot.get("bbox"),
                    "label": slot.get("label", ""),
                }
                for idx, slot in enumerate(raw.get("slots") or [], start=1)
                if isinstance(slot, dict)
            ][:8]
        if raw.get("bbox") is not None:
            item["bbox"] = raw.get("bbox")
        if raw.get("x") is not None and raw.get("y") is not None:
            item["point"] = [raw.get("x"), raw.get("y")]
        out_templates.append(item)
    trace = handler.get("episode_trace") if isinstance(handler.get("episode_trace"), list) else []
    return {
        "schema_version": handler.get("schema_version"),
        "handler_kind": handler.get("handler_kind"),
        "supported_intents": handler.get("supported_intents", []),
        "action_templates": out_templates,
        "memory": handler.get("memory", {}),
        "failure_stats": handler.get("failure_stats", {}),
        "recent_trace": trace[-limit_trace:],
    }


def active_intent(runtime: dict[str, Any], state_id: str) -> dict[str, Any] | None:
    intents = runtime.get(INTENT_RUNTIME_KEY)
    if not isinstance(intents, dict):
        return None
    intent = intents.get(str(state_id))
    return intent if isinstance(intent, dict) else None


def set_active_intent(runtime: dict[str, Any], state_id: str, intent: dict[str, Any] | None) -> None:
    intents = runtime.get(INTENT_RUNTIME_KEY)
    if not isinstance(intents, dict):
        intents = {}
        runtime[INTENT_RUNTIME_KEY] = intents
    key = str(state_id)
    if intent is None:
        intents.pop(key, None)
        return
    out = dict(intent)
    out.setdefault("status", "active")
    out.setdefault("source", "llm")
    out["updated_at"] = _now_iso()
    intents[key] = out


def merge_intent_patch(current: dict[str, Any] | None, patch: Any) -> dict[str, Any] | None:
    if not isinstance(patch, dict):
        return current
    status = str(patch.get("status", "")).strip()
    if status in {"satisfied", "abandoned"}:
        out = dict(current or {})
        out.update(patch)
        out["updated_at"] = _now_iso()
        return out
    out = dict(current or {})
    for key, value in patch.items():
        if key == "params" and isinstance(value, dict):
            params = out.get("params") if isinstance(out.get("params"), dict) else {}
            params = dict(params)
            params.update(value)
            out["params"] = params
        elif key == "runtime" and isinstance(value, dict):
            rt = out.get("runtime") if isinstance(out.get("runtime"), dict) else {}
            rt = dict(rt)
            rt.update(value)
            out["runtime"] = rt
        else:
            out[key] = value
    out.setdefault("status", "active")
    out.setdefault("source", "llm")
    out["updated_at"] = _now_iso()
    return out


def normalize_template(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind", "")).strip()
    if kind not in {"click", "click_button", "click_active_confirm", "select_from_slots", "click_blank_to_dismiss", "run_preset"}:
        return None
    template_id = _slug(raw.get("template_id") or raw.get("id") or raw.get("label") or kind, kind)
    out = dict(raw)
    out["template_id"] = template_id
    out["kind"] = kind
    out["label"] = str(out.get("label") or out.get("brief") or template_id)
    out["status"] = str(out.get("status", "active")) if str(out.get("status", "active")) in {"active", "disabled"} else "active"
    out["confidence"] = str(out.get("confidence", "mid")) if str(out.get("confidence", "mid")) in {"high", "mid", "low"} else "mid"
    out.setdefault("success_count", 0)
    out.setdefault("fail_count", 0)
    out["updated_at"] = _now_iso()
    if kind == "select_from_slots":
        slots = []
        for idx, slot in enumerate(out.get("slots") if isinstance(out.get("slots"), list) else [], start=1):
            if not isinstance(slot, dict):
                continue
            bbox = slot.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            item = dict(slot)
            item["slot_id"] = _slug(item.get("slot_id") or item.get("id") or idx, f"slot_{idx}")
            item["bbox"] = [int(v) for v in bbox]
            item["label"] = str(item.get("label", item["slot_id"]))
            slots.append(item)
        out["slots"] = slots
    return out


def apply_handler_patch(handler: dict[str, Any], patch: Any) -> set[str]:
    touched: set[str] = set()
    if not isinstance(patch, dict):
        return touched
    templates = handler.get("action_templates") if isinstance(handler.get("action_templates"), dict) else {}
    handler["action_templates"] = templates
    raw_templates = patch.get("action_templates")
    if isinstance(raw_templates, list):
        for raw in raw_templates:
            template = normalize_template(raw)
            if template is None:
                continue
            old = templates.get(template["template_id"])
            if isinstance(old, dict):
                template["success_count"] = int(old.get("success_count", template.get("success_count", 0)) or 0)
                template["fail_count"] = int(old.get("fail_count", template.get("fail_count", 0)) or 0)
                template["created_at"] = old.get("created_at", old.get("updated_at", _now_iso()))
            else:
                template["created_at"] = _now_iso()
            templates[template["template_id"]] = template
            touched.add(template["template_id"])
    disabled = patch.get("disable_templates")
    if isinstance(disabled, list):
        for raw_id in disabled:
            template_id = _slug(raw_id, str(raw_id))
            if template_id in templates and isinstance(templates[template_id], dict):
                templates[template_id]["status"] = "disabled"
                templates[template_id]["updated_at"] = _now_iso()
                touched.add(template_id)
    memory_patch = patch.get("memory")
    if isinstance(memory_patch, dict):
        memory = handler.get("memory") if isinstance(handler.get("memory"), dict) else {}
        memory.update(memory_patch)
        handler["memory"] = memory
    supported = patch.get("supported_intents")
    if isinstance(supported, list):
        seen = {str(v) for v in handler.get("supported_intents", []) if str(v)}
        for raw in supported:
            value = str(raw).strip()
            if value and value not in seen:
                handler["supported_intents"].append(value)
                seen.add(value)
    handler["updated_at"] = _now_iso()
    return touched


def append_episode(handler: dict[str, Any], episode: dict[str, Any]) -> None:
    trace = handler.get("episode_trace") if isinstance(handler.get("episode_trace"), list) else []
    handler["episode_trace"] = trace
    item = dict(episode)
    item.setdefault("created_at", _now_iso())
    trace.append(item)
    if len(trace) > TRACE_LIMIT:
        del trace[:-TRACE_LIMIT]
    handler["updated_at"] = _now_iso()


def mark_template_result(handler: dict[str, Any], template_id: str | None, success: bool) -> None:
    if not template_id:
        return
    templates = handler.get("action_templates") if isinstance(handler.get("action_templates"), dict) else {}
    template = templates.get(_slug(template_id, template_id))
    if not isinstance(template, dict):
        return
    key = "success_count" if success else "fail_count"
    template[key] = int(template.get(key, 0) or 0) + 1
    template["updated_at"] = _now_iso()
