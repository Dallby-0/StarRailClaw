from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from agent.behavior_tree.vision import VisionEngine


DIFF_THRESHOLD = 18
MATCH_THRESHOLD = 0.88
MIN_CHANGED_FRACTION = 0.002


def _provider(operation: dict[str, Any], provider_id: str) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in operation.get("providers", [])
            if isinstance(item, dict) and str(item.get("provider_id") or "") == provider_id
        ),
        None,
    )


def _learned_guard(provider: dict[str, Any], watch_id: str) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in provider.get("guards", [])
            if isinstance(item, dict) and str(item.get("learned_from_watch") or "") == watch_id
        ),
        None,
    )


def _changed_bbox(before, after, rect: list[int], vision: VisionEngine) -> list[int] | None:
    before_crop = vision.crop_rect(before, rect)
    after_crop = vision.crop_rect(after, rect)
    if before_crop.shape != after_crop.shape or before_crop.size == 0:
        return None
    channel_diff = cv2.absdiff(before_crop, after_crop)
    mask = (channel_diff.max(axis=2) >= DIFF_THRESHOLD).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    points = cv2.findNonZero(mask)
    if points is None or int(cv2.countNonZero(mask)) < max(1, int(mask.size * MIN_CHANGED_FRACTION)):
        return None
    x, y, width, height = cv2.boundingRect(points)
    pad = max(2, min(before_crop.shape[:2]) // 50)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(before_crop.shape[1], x + width + pad)
    y2 = min(before_crop.shape[0], y + height + pad)
    watch_x, watch_y, _, _ = vision.mapper.rect_to_real(rect)
    logical_a = vision.mapper.point_to_logical(watch_x + x1, watch_y + y1)
    logical_b = vision.mapper.point_to_logical(watch_x + x2, watch_y + y2)
    if logical_b[0] <= logical_a[0] or logical_b[1] <= logical_a[1]:
        return None
    return [logical_a[0], logical_a[1], logical_b[0], logical_b[1]]


def _matches(guard: dict[str, Any], frame, vision: VisionEngine) -> bool:
    path = Path(str(guard.get("template_path") or ""))
    if not path.is_file():
        return False
    matcher = getattr(vision, "match_appearance_template", vision.match_template)
    matched, _, _ = matcher(
        frame,
        path,
        guard.get("rect"),
        threshold=float(guard.get("threshold", MATCH_THRESHOLD)),
    )
    return bool(matched)


def _record_evidence(guard: dict[str, Any], positive, negative, visit_id: str, vision: VisionEngine) -> None:
    evidence = guard.setdefault(
        "evidence",
        {"positive_visits": [], "negative_checks": 0, "false_positive_count": 0},
    )
    visits = evidence.setdefault("positive_visits", [])
    if _matches(guard, positive, vision) and visit_id not in visits:
        visits.append(visit_id)
        del visits[:-8]
    evidence["negative_checks"] = int(evidence.get("negative_checks", 0) or 0) + 1
    if _matches(guard, negative, vision):
        evidence["false_positive_count"] = int(evidence.get("false_positive_count", 0) or 0) + 1
    guard["status"] = (
        "active"
        if len(visits) >= 2
        and int(evidence.get("negative_checks", 0) or 0) >= 2
        and int(evidence.get("false_positive_count", 0) or 0) == 0
        else "provisional"
    )


def _observe_guard(
    *,
    provider: dict[str, Any],
    watch_id: str,
    watch_rect: list[int],
    template_bbox: list[int] | None,
    positive,
    negative,
    visit_id: str,
    state_dir: Path,
    vision: VisionEngine,
    phase: str,
) -> dict[str, Any] | None:
    guards = provider.setdefault("guards", [])
    guard = _learned_guard(provider, watch_id)
    if guard is None:
        if template_bbox is None:
            return None
        provider_id = str(provider.get("provider_id") or "provider")
        path = state_dir / f"reactive_guard_{provider_id}_{watch_id}_{phase}.png"
        vision.save_template_from_rect(positive, template_bbox, path)
        guard = {
            "id": f"learned_{watch_id}",
            "type": "template",
            "status": "provisional",
            "rect": watch_rect,
            "template_bbox": template_bbox,
            "template_path": str(path),
            "threshold": MATCH_THRESHOLD,
            "cost": "cheap",
            "coordinate_space": "logical",
            "learned_from_watch": watch_id,
            "group": f"learned_{watch_id}",
            "evidence": {"positive_visits": [], "negative_checks": 0, "false_positive_count": 0},
        }
        guards.append(guard)
    _record_evidence(guard, positive, negative, visit_id, vision)
    return guard


def observe_successful_transition(
    operation: dict[str, Any],
    provider: dict[str, Any],
    before,
    after,
    *,
    visit_id: str,
    state_dir: Path,
    vision: VisionEngine,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    successors = [str(value) for value in provider.get("successors", [])]
    for watch in provider.get("watches", []):
        if not isinstance(watch, dict) or "appearance" not in watch.get("modalities", []):
            continue
        watch_id = str(watch.get("id") or "watch")
        rect = watch.get("rect")
        if not isinstance(rect, list):
            continue
        bbox = _changed_bbox(before, after, rect, vision)
        current = _observe_guard(
            provider=provider,
            watch_id=watch_id,
            watch_rect=rect,
            template_bbox=bbox,
            positive=before,
            negative=after,
            visit_id=visit_id,
            state_dir=state_dir,
            vision=vision,
            phase="before",
        )
        after_provider_id = str(watch.get("after_provider") or "")
        if not after_provider_id and len(successors) == 1:
            after_provider_id = successors[0]
        target = _provider(operation, after_provider_id) if after_provider_id else None
        successor = None
        if target is not None:
            successor = _observe_guard(
                provider=target,
                watch_id=watch_id,
                watch_rect=rect,
                template_bbox=bbox,
                positive=after,
                negative=before,
                visit_id=visit_id,
                state_dir=state_dir,
                vision=vision,
                phase="after",
            )
        observations.append(
            {
                "watch_id": watch_id,
                "template_bbox": bbox,
                "current_guard_status": current.get("status") if current else None,
                "after_provider": after_provider_id or None,
                "successor_guard_status": successor.get("status") if successor else None,
            }
        )
    return observations
