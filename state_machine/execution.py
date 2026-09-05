from __future__ import annotations

from typing import Any


EXECUTION_KINDS = {"reactive_2d", "invoke_tool", "cannot_handle"}


def execution_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("execution") if isinstance(payload.get("execution"), dict) else {}
    kind = str(raw.get("kind") or "")
    if kind not in EXECUTION_KINDS:
        raise ValueError(f"invalid execution kind: {kind!r}")
    if kind == "invoke_tool":
        name = str(raw.get("tool_name") or "").strip()
        if not name:
            raise ValueError("invoke_tool requires tool_name")
        return {"kind": kind, "tool_name": name}
    if kind == "cannot_handle":
        return {"kind": kind, "reason": str(raw.get("reason") or "").strip()}
    return {"kind": kind}


def reactive_bootstrap_operations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("execution") if isinstance(payload.get("execution"), dict) else {}
    if raw.get("kind") != "reactive_2d":
        return []
    operations = raw.get("bootstrap_operations")
    return [item for item in operations if isinstance(item, dict)] if isinstance(operations, list) else []


def execution_key(meta: dict[str, Any]) -> tuple[str, str]:
    raw = meta.get("execution") if isinstance(meta.get("execution"), dict) else {}
    return str(raw.get("kind") or ""), str(raw.get("tool_name") or "")
