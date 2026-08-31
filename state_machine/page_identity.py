from __future__ import annotations

import math
from copy import deepcopy
from numbers import Real
from typing import Any

from agent.behavior_tree.vision import VisionEngine


ELEMENT_ROLES = {"identity", "identity_support", "interaction", "instance", "diagnostic"}
IDENTITY_ROLES = {"identity", "identity_support"}


def _role(raw: Any) -> str:
    value = str(raw or "diagnostic").strip().lower()
    return value if value in ELEMENT_ROLES else "diagnostic"


def _logical_bbox(vision: VisionEngine, real_bbox: tuple[int, int, int, int]) -> list[int]:
    x, y, w, h = real_bbox
    x1, y1 = vision.mapper.point_to_logical(x, y)
    x2, y2 = vision.mapper.point_to_logical(x + max(1, w), y + max(1, h))
    return [x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)]


def _valid_bbox(raw: Any) -> list[int] | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    values: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            return None
        integer = int(value)
        if integer != value or not 0 <= integer <= 1000:
            return None
        values.append(integer)
    x1, y1, x2, y2 = values
    return values if x1 < x2 and y1 < y2 else None


def normalize_elements(elements: Any, vision: VisionEngine, frame_rgb) -> list[dict[str, Any]]:
    """Normalize LLM elements into runtime-observable primitives.

    Text boxes are passed through multi-line OCR exactly once.  A box covering
    multiple detected lines is split; a one-line box is tightened.  The LLM's
    semantic role is retained, while detected text becomes the actual matcher
    needle so an impossible cross-line string is never enabled.
    """
    normalized: list[dict[str, Any]] = []
    for raw in elements if isinstance(elements, list) else []:
        if not isinstance(raw, dict):
            continue
        item = deepcopy(raw)
        item["role"] = _role(item.get("role"))
        bbox = _valid_bbox(item.get("bbox"))
        if bbox is None:
            print(f"[fsm][elements][warning] ignored invalid bbox: {item.get('bbox')!r}")
            continue
        item["bbox"] = bbox
        if item.get("type") != "text_line":
            normalized.append(item)
            continue
        blocks = [entry for entry in vision.ocr_blocks(frame_rgb, item["bbox"]) if str(entry.get("text", "")).strip()]
        if not blocks:
            # An unobservable text proposal may still be useful diagnostically,
            # but it must never become state identity.
            item["observable"] = False
            item["role"] = "diagnostic"
            normalized.append(item)
            continue
        for index, block in enumerate(sorted(blocks, key=lambda e: (e["bbox"][1], e["bbox"][0])), start=1):
            split = deepcopy(item)
            split["text"] = str(block["text"]).strip()
            split["bbox"] = _logical_bbox(vision, block["bbox"])
            split["observable"] = True
            split["ocr_confidence"] = float(block.get("conf", 0.0) or 0.0)
            if len(blocks) > 1:
                split["normalized_from_multiline"] = True
                split["brief"] = f"{str(item.get('brief', '')).strip()} line {index}".strip()
            normalized.append(split)
    return normalized


def initial_observations() -> dict[str, Any]:
    return {
        "family_positive_pass": 0,
        "family_positive_fail": 0,
        "other_page_pass": 0,
        "other_page_fail": 0,
        "after_action_pass": 0,
        "after_action_fail": 0,
    }


def record_condition_observations(meta: dict[str, Any], vision: VisionEngine, frame_rgb, *, cohort: str) -> None:
    """Accumulate observable evidence; LLM levels remain priors only."""
    from state_machine.matching import _condition_eval

    pass_key = f"{cohort}_pass"
    fail_key = f"{cohort}_fail"
    for cond in meta.get("match_conditions", []):
        if not isinstance(cond, dict):
            continue
        observations = cond.get("observations") if isinstance(cond.get("observations"), dict) else initial_observations()
        cond["observations"] = observations
        ok, detail = _condition_eval(cond, vision, frame_rgb)
        observations[pass_key if ok else fail_key] = int(observations.get(pass_key if ok else fail_key, 0) or 0) + 1
        positive_total = int(observations.get("family_positive_pass", 0) or 0) + int(observations.get("family_positive_fail", 0) or 0)
        other_total = int(observations.get("other_page_pass", 0) or 0) + int(observations.get("other_page_fail", 0) or 0)
        # Beta(1,1) smoothing prevents 1/1 observations from becoming hard
        # certainty. These values are diagnostic/ranking inputs, not gates.
        cond["empirical"] = {
            "family_coverage": (int(observations.get("family_positive_pass", 0) or 0) + 1) / (positive_total + 2),
            "other_page_false_positive_rate": (int(observations.get("other_page_pass", 0) or 0) + 1) / (other_total + 2),
        }
        if cohort == "family_positive" and cond.get("kind") == "region_template" and detail.get("similarity") is not None:
            similarities = observations.get("family_positive_similarities") if isinstance(observations.get("family_positive_similarities"), list) else []
            observations["family_positive_similarities"] = similarities
            similarities.append(float(detail["similarity"]))
            del similarities[:-20]
            if len(similarities) >= 2:
                ordered = sorted(similarities)
                lower = ordered[max(0, int(len(ordered) * 0.1) - 1)]
                calibrated = max(0.65, min(0.90, lower - 0.03))
                params = cond.get("params") if isinstance(cond.get("params"), dict) else {}
                cond["params"] = params
                params["threshold"] = calibrated
                cond["threshold_policy"] = "positive_sample_lower_quantile_minus_margin"


def build_match_clauses(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compile roles into a small DNF matcher (OR of AND clauses)."""
    identities = [c for c in conditions if c.get("role") == "identity" and c.get("condition_status", "active") == "active"]
    supports = [c for c in conditions if c.get("role") == "identity_support" and c.get("condition_status", "active") == "active"]
    clauses = [{"all": [str(c["id"])]} for c in identities]
    if not clauses and supports:
        # Supporting anchors are deliberately conjunctive.  A generic title
        # such as a game mode name must not identify a state by itself.
        clauses.append({"all": [str(c["id"]) for c in supports]})
    return clauses


def temporal_continuation_evidence(meta: dict[str, Any], vision: VisionEngine, frame_rgb) -> dict[str, Any]:
    """Cheap evidence that an action revealed another step on the same surface."""
    from state_machine.matching import _condition_eval

    passed: list[str] = []
    score = 0
    for cond in meta.get("match_conditions", []):
        if not isinstance(cond, dict) or cond.get("condition_status", "active") != "active":
            continue
        role = str(cond.get("role") or "")
        if role not in IDENTITY_ROLES:
            continue
        ok, _ = _condition_eval(cond, vision, frame_rgb)
        if not ok:
            continue
        passed.append(str(cond.get("id") or ""))
        if role == "identity":
            score += 3
        elif str(cond.get("stability") or "mid") == "high":
            score += 2
        else:
            score += 1
    return {"accepted": score >= 2, "score": score, "passed_condition_ids": passed}
