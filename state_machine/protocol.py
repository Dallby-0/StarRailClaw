from __future__ import annotations

import json
from typing import Any


def parse_state_payload(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    required = {"page_summary", "slug", "elements", "bootstrap_operations"}
    if not required.issubset(payload.keys()):
        return None
    if not isinstance(payload.get("elements"), list) or not isinstance(payload.get("bootstrap_operations"), list):
        return None
    payload.setdefault("possible_page_type", "none")
    return payload
