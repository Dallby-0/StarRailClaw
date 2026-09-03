from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EffectiveCommand:
    operation: str
    source: str
    intent_id: str | None = None
    intent_kind: str | None = None
    intent_phase: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    persistent: bool = False
    intent_effect: str = "none"
    expected_event: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _matches_route(route: dict[str, Any], intent: dict[str, Any]) -> bool:
    intent_kind = str(intent.get("kind", ""))
    intent_phase = str(intent.get("phase", ""))
    kinds = route.get("intent_kinds")
    if not isinstance(kinds, list):
        kinds = [route.get("intent_kind", "*")]
    phases = route.get("phases")
    if not isinstance(phases, list):
        phases = [route.get("phase", "*")]
    return ("*" in kinds or intent_kind in {str(v) for v in kinds}) and ("*" in phases or intent_phase in {str(v) for v in phases})


def resolve_effective_command(state_meta: dict[str, Any], intent: dict[str, Any] | None) -> EffectiveCommand | None:
    handler = state_meta.get("page_handler") if isinstance(state_meta.get("page_handler"), dict) else {}
    if isinstance(intent, dict):
        routes = handler.get("intent_routes") if isinstance(handler.get("intent_routes"), list) else []
        for route in routes:
            if not isinstance(route, dict) or not _matches_route(route, intent):
                continue
            operation = str(route.get("operation") or "").strip()
            if not operation:
                continue
            params = dict(intent.get("params") if isinstance(intent.get("params"), dict) else {})
            if isinstance(intent.get("facts"), dict):
                params["facts"] = dict(intent["facts"])
            if isinstance(route.get("params"), dict):
                params.update(route["params"])
            return EffectiveCommand(
                operation=operation,
                source="active_intent",
                intent_id=str(intent.get("intent_id") or "") or None,
                intent_kind=str(intent.get("kind") or "") or None,
                intent_phase=str(intent.get("phase") or "") or None,
                params=params,
                persistent=True,
                intent_effect=str(route.get("intent_effect", "advance")),
                expected_event=str(route.get("expected_event") or "") or None,
            )

    default = handler.get("default_operation") if isinstance(handler.get("default_operation"), dict) else None
    if not isinstance(default, dict):
        return None
    operation = str(default.get("operation") or "").strip()
    if not operation:
        return None
    intent_effect = str(default.get("intent_effect", "none"))
    intent_scope = str(default.get("intent_scope", "intent_specific"))
    if isinstance(intent, dict):
        if intent_scope != "intent_invariant":
            return None
        return EffectiveCommand(
            operation=operation,
            source="state_default",
            intent_id=str(intent.get("intent_id") or "") or None,
            intent_kind=str(intent.get("kind") or "") or None,
            intent_phase=str(intent.get("phase") or "") or None,
            params=dict(default.get("params") or {}),
            persistent=False,
            intent_effect=intent_effect,
            expected_event=str(default.get("expected_event") or "") or None,
        )
    return EffectiveCommand(
        operation=operation,
        source="state_default",
        params=dict(default.get("params") or {}),
        persistent=False,
        intent_effect=intent_effect,
        expected_event=str(default.get("expected_event") or "") or None,
    )
