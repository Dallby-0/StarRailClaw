from __future__ import annotations

from copy import deepcopy
from typing import Any

from state_machine.time_utils import now_iso as _now_iso


REACTIVE_CONTROLLER_TYPE = "reactive_local"
PROVIDER_RESULTS = {"resolver_miss", "no_change", "changed_same_state", "state_left", "action_error"}
PROGRESS_ACTION_INCREMENT = 8
PROGRESS_OBSERVATION_INCREMENT = 4
PROGRESS_UNCERTAIN_ACTION_GRANT = 2
PROGRESS_UNCERTAIN_OBSERVATION_GRANT = 1
PROGRESS_MAX_EXTENSIONS = 4
PROGRESS_HARD_MAX_ACTIONS = 64
PROGRESS_HARD_MAX_OBSERVATIONS = 32


def _slug(raw: Any, fallback: str) -> str:
    text = str(raw or "").strip().lower()
    safe = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or fallback


def normalize_provider(raw: Any, *, index: int = 1, default_cost: int = 0) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind") or "action").strip()
    if kind not in {"action", "exploration"}:
        return None
    provider_id = _slug(raw.get("provider_id") or raw.get("action_id") or raw.get("id"), f"provider_{index}")
    repeat_policy = str(raw.get("repeat_policy") or "once_per_visit")
    if repeat_policy not in {"once_per_visit", "once_per_observation", "repeatable"}:
        repeat_policy = "once_per_visit"
    status = str(raw.get("status") or "proposed")
    if status not in {"proposed", "active", "degraded", "quarantined"}:
        status = "proposed"
    provider: dict[str, Any] = {
        "provider_id": provider_id,
        "kind": kind,
        "cost": max(0, int(raw.get("cost", default_cost) or 0)),
        "repeat_policy": repeat_policy,
        "status": status,
        "brief": str(raw.get("brief") or raw.get("label") or ""),
        "expected_after": dict(raw.get("expected_after") or {"state_relation": "may_leave", "reentry_policy": "forbid"}),
        "emits_on_success": dict(raw.get("emits_on_success") or {}),
        "success_count": int(raw.get("success_count", 0) or 0),
        "fail_count": int(raw.get("fail_count", 0) or 0),
        "result_counts": dict(raw.get("result_counts") or {}),
        "created_at": str(raw.get("created_at") or _now_iso()),
        "updated_at": _now_iso(),
    }
    if kind == "action":
        resolver = raw.get("resolver") if isinstance(raw.get("resolver"), dict) else None
        if resolver is None:
            return None
        provider["resolver"] = deepcopy(resolver)
    else:
        profiles = raw.get("profiles") if isinstance(raw.get("profiles"), list) else []
        if not profiles and isinstance(raw.get("profile"), dict):
            profiles = [raw["profile"]]
        provider["profiles"] = [deepcopy(item) for item in profiles if isinstance(item, dict)]
        if not provider["profiles"]:
            return None
    return provider


def controller_from_steps(steps: Any, *, exploration: Any = None) -> dict[str, Any] | None:
    if not isinstance(steps, list) or not steps:
        return None
    providers: list[dict[str, Any]] = []
    for index, step in enumerate(steps[:4], start=1):
        if not isinstance(step, dict) or not isinstance(step.get("resolver"), dict):
            continue
        provider = normalize_provider(
            {
                "provider_id": step.get("step_id") or f"direct_{index}",
                "kind": "action",
                "cost": index - 1,
                "repeat_policy": step.get("repeat_policy") or "once_per_visit",
                "resolver": step["resolver"],
                "expected_after": step.get("expected_after"),
                "emits_on_success": step.get("emits_on_success"),
                "brief": step.get("brief"),
            },
            index=index,
            default_cost=index - 1,
        )
        if provider is not None:
            providers.append(provider)
    profiles = [deepcopy(item) for item in exploration if isinstance(item, dict)] if isinstance(exploration, list) else []
    if profiles:
        fallback = normalize_provider(
            {
                "provider_id": "local_exploration",
                "kind": "exploration",
                "cost": 20,
                "repeat_policy": "once_per_observation",
                "profiles": profiles,
                "brief": "bounded local exploration",
            },
            index=len(providers) + 1,
            default_cost=20,
        )
        if fallback is not None:
            providers.append(fallback)
    if not providers:
        return None
    return {
        "type": REACTIVE_CONTROLLER_TYPE,
        "providers": providers,
        "max_actions": 12,
        "max_observation_rounds": 6,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


def normalize_controller(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or str(raw.get("type") or REACTIVE_CONTROLLER_TYPE) != REACTIVE_CONTROLLER_TYPE:
        return None
    providers: list[dict[str, Any]] = []
    for index, item in enumerate(raw.get("providers") if isinstance(raw.get("providers"), list) else [], start=1):
        provider = normalize_provider(item, index=index, default_cost=index - 1)
        if provider is not None:
            providers.append(provider)
    fallback_profiles = raw.get("fallback_profiles") if isinstance(raw.get("fallback_profiles"), list) else []
    if fallback_profiles:
        fallback = normalize_provider(
            {
                "provider_id": "learned_exploration",
                "kind": "exploration",
                "cost": 20,
                "repeat_policy": "once_per_observation",
                "profiles": fallback_profiles,
                "brief": "learned bounded exploration",
            },
            index=len(providers) + 1,
            default_cost=20,
        )
        if fallback is not None:
            providers.append(fallback)
    if not providers:
        return None
    return {
        "type": REACTIVE_CONTROLLER_TYPE,
        "providers": providers,
        "max_actions": max(1, min(int(raw.get("max_actions", 12) or 12), 24)),
        "max_observation_rounds": max(1, min(int(raw.get("max_observation_rounds", 6) or 6), 12)),
        "created_at": str(raw.get("created_at") or _now_iso()),
        "updated_at": _now_iso(),
    }


def merge_controller(existing: Any, incoming: Any) -> dict[str, Any] | None:
    new = normalize_controller(incoming)
    if new is None:
        return normalize_controller(existing)
    old = normalize_controller(existing)
    if old is None:
        return new
    by_id = {str(item.get("provider_id")): item for item in old["providers"] if isinstance(item, dict)}
    for provider in new["providers"]:
        provider_id = str(provider.get("provider_id"))
        previous = by_id.get(provider_id)
        if isinstance(previous, dict):
            if provider.get("kind") == "exploration" and previous.get("kind") == "exploration":
                merged_profiles = [deepcopy(item) for item in previous.get("profiles", []) if isinstance(item, dict)]
                profile_by_id = {str(item.get("id") or ""): item for item in merged_profiles}
                for profile in provider.get("profiles", []):
                    if not isinstance(profile, dict):
                        continue
                    profile_id = str(profile.get("id") or "")
                    old_profile = profile_by_id.get(profile_id) if profile_id else None
                    if old_profile is not None:
                        merged_profiles.remove(old_profile)
                    merged_profiles.append(deepcopy(profile))
                provider["profiles"] = merged_profiles
            provider["success_count"] = int(previous.get("success_count", 0) or 0)
            provider["fail_count"] = int(previous.get("fail_count", 0) or 0)
            provider["result_counts"] = dict(previous.get("result_counts") or {})
            provider["created_at"] = previous.get("created_at", provider["created_at"])
            old["providers"].remove(previous)
        old["providers"].append(provider)
    old["max_actions"] = new["max_actions"]
    old["max_observation_rounds"] = new["max_observation_rounds"]
    old["updated_at"] = _now_iso()
    return old


def initial_cursor(*, visit_id: str, operation: str) -> dict[str, Any]:
    return {
        "visit_id": visit_id,
        "operation": operation,
        "observation_epoch": 1,
        "attempted_visit": [],
        "attempted_by_epoch": {"1": []},
        "total_actions": 0,
        "action_capacity": 0,
        "observation_capacity": 0,
        "capacity_extensions": 0,
        "uncertain_probe_used": False,
        "last_progress_audit_action": -1,
        "last_progress_audit_observation": -1,
        "progress_audits": [],
        "updated_at": _now_iso(),
    }


def ensure_progress_capacity(cursor: dict[str, Any], controller: dict[str, Any]) -> dict[str, int]:
    base_actions = max(1, min(int(controller.get("max_actions", 12) or 12), 24))
    base_observations = max(1, min(int(controller.get("max_observation_rounds", 6) or 6), 12))
    if int(cursor.get("action_capacity", 0) or 0) <= 0:
        cursor["action_capacity"] = base_actions
    if int(cursor.get("observation_capacity", 0) or 0) <= 0:
        cursor["observation_capacity"] = base_observations
    cursor.setdefault("capacity_extensions", 0)
    cursor.setdefault("uncertain_probe_used", False)
    cursor.setdefault("last_progress_audit_action", -1)
    cursor.setdefault("last_progress_audit_observation", -1)
    if not isinstance(cursor.get("progress_audits"), list):
        cursor["progress_audits"] = []
    return {
        "actions": int(cursor["action_capacity"]),
        "observations": int(cursor["observation_capacity"]),
    }


def progress_capacity_exhausted(cursor: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if int(cursor.get("total_actions", 0) or 0) >= int(cursor.get("action_capacity", 0) or 0):
        reasons.append("actions")
    if int(cursor.get("observation_epoch", 1) or 1) > int(cursor.get("observation_capacity", 0) or 0):
        reasons.append("observations")
    return reasons


def apply_progress_audit(cursor: dict[str, Any], verdict: str, confidence: str) -> dict[str, Any]:
    """Apply a model verdict using fixed runtime-owned capacity rules."""
    verdict = str(verdict or "uncertain").strip().lower()
    confidence = str(confidence or "low").strip().lower()
    before_actions = int(cursor.get("action_capacity", 0) or 0)
    before_observations = int(cursor.get("observation_capacity", 0) or 0)
    grant_kind = "none"

    reliable_progress = verdict in {"progressing", "new_instance"} and confidence in {"high", "mid"}
    if reliable_progress and int(cursor.get("capacity_extensions", 0) or 0) < PROGRESS_MAX_EXTENSIONS:
        cursor["action_capacity"] = min(PROGRESS_HARD_MAX_ACTIONS, before_actions + PROGRESS_ACTION_INCREMENT)
        cursor["observation_capacity"] = min(PROGRESS_HARD_MAX_OBSERVATIONS, before_observations + PROGRESS_OBSERVATION_INCREMENT)
        if int(cursor["action_capacity"]) > before_actions or int(cursor["observation_capacity"]) > before_observations:
            cursor["capacity_extensions"] = int(cursor.get("capacity_extensions", 0) or 0) + 1
            grant_kind = "progress_extension"
    elif verdict == "uncertain" or (verdict in {"progressing", "new_instance"} and confidence == "low"):
        if not bool(cursor.get("uncertain_probe_used", False)):
            cursor["action_capacity"] = min(PROGRESS_HARD_MAX_ACTIONS, before_actions + PROGRESS_UNCERTAIN_ACTION_GRANT)
            cursor["observation_capacity"] = min(PROGRESS_HARD_MAX_OBSERVATIONS, before_observations + PROGRESS_UNCERTAIN_OBSERVATION_GRANT)
            cursor["uncertain_probe_used"] = True
            if int(cursor["action_capacity"]) > before_actions or int(cursor["observation_capacity"]) > before_observations:
                grant_kind = "uncertain_probe"

    cursor["updated_at"] = _now_iso()
    return {
        "granted": grant_kind != "none",
        "grant_kind": grant_kind,
        "before": {"actions": before_actions, "observations": before_observations},
        "after": {
            "actions": int(cursor.get("action_capacity", before_actions) or before_actions),
            "observations": int(cursor.get("observation_capacity", before_observations) or before_observations),
        },
        "extensions": int(cursor.get("capacity_extensions", 0) or 0),
    }


def select_provider(controller: dict[str, Any], cursor: dict[str, Any]) -> dict[str, Any] | None:
    epoch = str(int(cursor.get("observation_epoch", 1) or 1))
    attempted_visit = {str(value) for value in cursor.get("attempted_visit", [])}
    attempted_epoch = {
        str(value)
        for value in (cursor.get("attempted_by_epoch", {}).get(epoch, []) if isinstance(cursor.get("attempted_by_epoch"), dict) else [])
    }
    candidates = []
    for provider in controller.get("providers", []):
        if not isinstance(provider, dict) or str(provider.get("status", "proposed")) not in {"proposed", "active"}:
            continue
        provider_id = str(provider.get("provider_id") or "")
        repeat = str(provider.get("repeat_policy") or "once_per_visit")
        if repeat == "once_per_visit" and provider_id in attempted_visit:
            continue
        if repeat == "once_per_observation" and provider_id in attempted_epoch:
            continue
        candidates.append(provider)
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            int(item.get("cost", 0) or 0),
            0 if item.get("status") == "active" else 1,
            -int(item.get("success_count", 0) or 0),
            str(item.get("provider_id") or ""),
        )
    )
    return candidates[0]


def record_provider_attempt(cursor: dict[str, Any], provider: dict[str, Any], *, action_count: int = 0) -> None:
    provider_id = str(provider.get("provider_id") or "")
    epoch = str(int(cursor.get("observation_epoch", 1) or 1))
    attempted_visit = cursor.get("attempted_visit") if isinstance(cursor.get("attempted_visit"), list) else []
    cursor["attempted_visit"] = attempted_visit
    if provider_id not in attempted_visit:
        attempted_visit.append(provider_id)
    by_epoch = cursor.get("attempted_by_epoch") if isinstance(cursor.get("attempted_by_epoch"), dict) else {}
    cursor["attempted_by_epoch"] = by_epoch
    attempted_epoch = by_epoch.get(epoch) if isinstance(by_epoch.get(epoch), list) else []
    by_epoch[epoch] = attempted_epoch
    if provider_id not in attempted_epoch:
        attempted_epoch.append(provider_id)
    cursor["total_actions"] = int(cursor.get("total_actions", 0) or 0) + max(0, int(action_count or 0))
    cursor["updated_at"] = _now_iso()


def advance_observation(cursor: dict[str, Any]) -> int:
    epoch = int(cursor.get("observation_epoch", 1) or 1) + 1
    cursor["observation_epoch"] = epoch
    by_epoch = cursor.get("attempted_by_epoch") if isinstance(cursor.get("attempted_by_epoch"), dict) else {}
    cursor["attempted_by_epoch"] = by_epoch
    by_epoch.setdefault(str(epoch), [])
    if len(by_epoch) > 12:
        for key in sorted(by_epoch, key=lambda value: int(value))[:-12]:
            by_epoch.pop(key, None)
    cursor["updated_at"] = _now_iso()
    return epoch


def mark_provider_result(controller: dict[str, Any], provider_id: str, result: str) -> None:
    if result not in PROVIDER_RESULTS:
        return
    provider = next(
        (item for item in controller.get("providers", []) if isinstance(item, dict) and str(item.get("provider_id")) == provider_id),
        None,
    )
    if not isinstance(provider, dict):
        return
    counts = provider.get("result_counts") if isinstance(provider.get("result_counts"), dict) else {}
    provider["result_counts"] = counts
    counts[result] = int(counts.get(result, 0) or 0) + 1
    if result == "state_left":
        provider["success_count"] = int(provider.get("success_count", 0) or 0) + 1
        provider["status"] = "active"
    elif result in {"no_change", "action_error"}:
        provider["fail_count"] = int(provider.get("fail_count", 0) or 0) + 1
        # A provider may be layout-specific. A miss on one same-family visit is
        # local evidence, not a reason to disable it globally.
    provider["updated_at"] = _now_iso()
    controller["updated_at"] = _now_iso()
