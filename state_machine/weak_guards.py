from __future__ import annotations

from typing import Any

import cv2
import numpy as np


WEAK_ROI_KIND = "weak_roi_presence"
WEAK_GUARDS_ENABLED = True


def _rect(raw: Any) -> list[int] | None:
    if not (isinstance(raw, list) and len(raw) == 4):
        return None
    try:
        x, y, w, h = [int(v) for v in raw]
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def _clip_crop(frame_rgb, rect_real: list[int]):
    h, w = frame_rgb.shape[:2]
    x, y, rw, rh = rect_real
    x0 = max(0, min(w, x))
    y0 = max(0, min(h, y))
    x1 = max(0, min(w, x + rw))
    y1 = max(0, min(h, y + rh))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame_rgb[y0:y1, x0:x1]


def _features(crop) -> dict[str, float]:
    if crop is None or crop.size <= 0:
        return {"edge_density": 0.0, "color_std": 0.0, "brightness_std": 0.0, "foreground_ratio": 0.0}
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    color_std = float(np.mean(np.std(crop.reshape(-1, 3), axis=0)))
    brightness_std = float(np.std(gray))
    foreground = (gray > 24) & (gray < 235)
    foreground_ratio = float(np.count_nonzero(foreground)) / float(gray.size)
    return {
        "edge_density": edge_density,
        "color_std": color_std,
        "brightness_std": brightness_std,
        "foreground_ratio": foreground_ratio,
    }


def normalize_condition(raw: dict[str, Any]) -> dict[str, Any] | None:
    kind = str(raw.get("kind", "")).strip()
    if kind not in {WEAK_ROI_KIND, "roi_presence", "region_presence"}:
        return None
    params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
    rect = _rect(params.get("rect") or raw.get("bbox"))
    if rect is None:
        return None
    return {
        "id": "",
        "enabled": True,
        "kind": WEAK_ROI_KIND,
        "params": {"rect": rect},
        "brief": str(raw.get("brief", "")),
        "stability": str(raw.get("stability", "low")),
        "discrimination": str(raw.get("discrimination", "low")),
        "condition_status": "active",
        "bbox": rect,
        "guard_strength": "weak_roi",
    }


def materialize_condition(cond: dict[str, Any], *, frame_rgb, mapper) -> dict[str, Any] | None:
    params = dict(cond.get("params", {}))
    rect = _rect(params.get("rect") or cond.get("bbox"))
    if rect is None:
        return None
    x, y, w, h = mapper.rect_to_real(rect)
    feats = _features(_clip_crop(frame_rgb, [x, y, w, h]))
    out = dict(cond)
    params["rect"] = rect
    params["baseline"] = feats
    params.setdefault("min_edge_density", max(0.002, feats["edge_density"] * 0.35))
    params.setdefault("min_color_std", max(3.0, feats["color_std"] * 0.35))
    params.setdefault("min_brightness_std", max(3.0, feats["brightness_std"] * 0.35))
    params.setdefault("min_foreground_ratio", max(0.05, feats["foreground_ratio"] * 0.35))
    out["params"] = params
    out["guard_strength"] = "weak_roi"
    return out


def evaluate_condition(cond: dict[str, Any], *, frame_rgb, mapper) -> tuple[bool, dict[str, Any]]:
    params = cond.get("params") if isinstance(cond.get("params"), dict) else {}
    rect = _rect(params.get("rect") or cond.get("bbox"))
    detail: dict[str, Any] = {
        "id": str(cond.get("id", "")),
        "kind": WEAK_ROI_KIND,
        "enabled": bool(cond.get("enabled", False)),
        "status": str(cond.get("condition_status", "active")),
        "rect": rect,
        "brief": str(cond.get("brief", "")),
    }
    if rect is None:
        detail["reason"] = "invalid_rect"
        detail["passed"] = False
        return False, detail
    x, y, w, h = mapper.rect_to_real(rect)
    feats = _features(_clip_crop(frame_rgb, [x, y, w, h]))
    detail["features"] = feats
    checks = {
        "edge_density": feats["edge_density"] >= float(params.get("min_edge_density", 0.002)),
        "color_std": feats["color_std"] >= float(params.get("min_color_std", 3.0)),
        "brightness_std": feats["brightness_std"] >= float(params.get("min_brightness_std", 3.0)),
        "foreground_ratio": feats["foreground_ratio"] >= float(params.get("min_foreground_ratio", 0.05)),
    }
    detail["checks"] = checks
    # Weak ROI is intentionally loose: require any two signals so simple UI controls,
    # irregular targets, and text-heavy choices can all pass without semantic detection.
    passed = sum(1 for ok in checks.values() if ok) >= 2
    detail["passed"] = passed
    return passed, detail
