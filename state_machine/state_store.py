from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from state_machine import constants
from state_machine.io import (
    _load_json,
    _normalize_page_type,
    _now_iso,
    _save_frame,
    _save_json,
    _slugify,
    _state_page_type,
)
from state_machine.logger import FsmRunLogger


def _iter_state_meta() -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    if not constants.FSM_TEMPLATES_DIR.exists():
        return out
    for d in sorted(constants.FSM_TEMPLATES_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "state.json"
        if not meta_path.exists():
            continue
        try:
            out.append((d, _load_json(meta_path)))
        except Exception:
            continue
    return out


def _page_type_summaries(metas: list[tuple[Path, dict[str, Any]]], limit: int = 40) -> list[dict[str, Any]]:
    by_type: dict[str, dict[str, Any]] = {}
    for state_dir, meta in metas:
        ptype = _state_page_type(meta)
        entry = by_type.setdefault(
            ptype,
            {
                "page_type": ptype,
                "page_family": str(meta.get("page_family") or ptype),
                "scene_mode": str(meta.get("scene_mode") or "unknown"),
                "execution": dict(meta.get("execution") or {}),
                "example_slug": str(meta.get("slug", "")),
                "description": str(meta.get("description", ""))[:160],
                "sample_count": 0,
                "latest_dir": str(state_dir),
            },
        )
        entry["sample_count"] = int(entry.get("sample_count", 0)) + 1
        try:
            if state_dir.stat().st_mtime >= Path(str(entry["latest_dir"])).stat().st_mtime:
                entry["example_slug"] = str(meta.get("slug", ""))
                entry["description"] = str(meta.get("description", ""))[:160]
                entry["scene_mode"] = str(meta.get("scene_mode") or "unknown")
                entry["execution"] = dict(meta.get("execution") or {})
                entry["latest_dir"] = str(state_dir)
        except Exception:
            pass
    return sorted(by_type.values(), key=lambda x: str(x.get("page_type", "")))[:limit]


def _states_for_page_type(metas: list[tuple[Path, dict[str, Any]]], page_type: str) -> list[tuple[Path, dict[str, Any]]]:
    norm = _normalize_page_type(page_type)
    if not norm:
        return []
    return [(d, m) for d, m in metas if _state_page_type(m) == norm]


def _latest_state_for_page_type(metas: list[tuple[Path, dict[str, Any]]], page_type: str) -> tuple[Path, dict[str, Any]] | None:
    states = _states_for_page_type(metas, page_type)
    if not states:
        return None
    return max(states, key=lambda item: item[0].stat().st_mtime)


def _sample_screenshot_paths(state_dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for d in state_dirs:
        shots = sorted(d.glob("screenshot_*.png"))
        if shots:
            out.extend(shots)
    return out


def _latest_screenshot_path(state_dir: Path) -> Path | None:
    shots = sorted(state_dir.glob("screenshot_*.png"))
    return shots[-1] if shots else None


def _add_state_sample(
    state_dir: Path,
    meta: dict[str, Any],
    frame_rgb,
    *,
    role: str,
    source: str,
    confidence: float = 0.5,
    logger: FsmRunLogger | None = None,
) -> None:
    samples = meta.setdefault("samples", [])
    if not isinstance(samples, list):
        samples = []
        meta["samples"] = samples
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = state_dir / f"sample_{role}_{source}_{ts}.png"
    _save_frame(path, frame_rgb)
    samples.append(
        {
            "path": str(path),
            "role": role,
            "source": source,
            "confidence": max(0.0, min(1.0, float(confidence))),
            "created_at": _now_iso(),
        }
    )
    meta["updated_at"] = _now_iso()
    _save_json(state_dir / "state.json", meta)
    if logger is not None:
        logger.event("state_sample_added", state_id=meta.get("state_id"), role=role, source=source, confidence=confidence, path=path)


def _sample_paths_from_meta(state_dir: Path, meta: dict[str, Any]) -> list[Path]:
    out: list[Path] = []
    samples = meta.get("samples")
    if isinstance(samples, list):
        for s in samples:
            if not isinstance(s, dict):
                continue
            p = Path(str(s.get("path", "")))
            if p.exists():
                out.append(p)
    latest = _latest_screenshot_path(state_dir)
    if latest is not None:
        out.append(latest)
    return out


def _ensure_unique_state_dir(slug: str) -> Path:
    base = _slugify(slug)
    d = constants.FSM_TEMPLATES_DIR / base
    n = 1
    while d.exists():
        n += 1
        d = constants.FSM_TEMPLATES_DIR / f"{base}_{n}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def _parse_decimal_state_id(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text.isdecimal():
        return None
    return int(text)


def _allocate_state_id(width: int = 3) -> str:
    graph = _load_json(constants.FSM_GRAPH_PATH) if constants.FSM_GRAPH_PATH.exists() else {}
    used: set[str] = set()
    max_seen = -1

    for node in graph.get("nodes", []) if isinstance(graph.get("nodes"), list) else []:
        if not isinstance(node, dict):
            continue
        state_id = str(node.get("state_id", "")).strip()
        if not state_id:
            continue
        used.add(state_id)
        parsed = _parse_decimal_state_id(state_id)
        if parsed is not None:
            max_seen = max(max_seen, parsed)

    for _, meta in _iter_state_meta():
        state_id = str(meta.get("state_id", "")).strip()
        if not state_id:
            continue
        used.add(state_id)
        parsed = _parse_decimal_state_id(state_id)
        if parsed is not None:
            max_seen = max(max_seen, parsed)

    try:
        next_seq = int(graph.get("next_state_seq", 0) or 0)
    except Exception:
        next_seq = 0
    next_seq = max(next_seq, max_seen + 1, 0)

    while True:
        state_id = f"{next_seq:0{width}d}"
        if state_id not in used:
            break
        next_seq += 1

    graph.setdefault("schema_version", constants.SCHEMA_VERSION)
    graph.setdefault("created_at", _now_iso())
    graph.setdefault("nodes", [])
    graph.setdefault("edges", [])
    graph["next_state_seq"] = next_seq + 1
    graph["updated_at"] = _now_iso()
    _save_json(constants.FSM_GRAPH_PATH, graph)
    return state_id


def _meta_for_state(metas: list[tuple[Path, dict[str, Any]]], state_id: str) -> tuple[Path, dict[str, Any]] | None:
    for d, m in metas:
        if str(m.get("state_id", "")) == state_id:
            return d, m
    return None
