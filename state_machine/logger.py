from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import FSM_DEBUG_DIR


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


class FsmRunLogger:
    """Session/run scoped logger with machine and human readable outputs."""

    def __init__(self, session_id: str, run_id: str, debug_dir: Path = FSM_DEBUG_DIR) -> None:
        self.session_id = _safe_name(session_id)
        self.run_id = run_id
        self.started_at = _now_iso()
        self.session_dir = debug_dir / "sessions" / self.session_id
        self.run_dir = self.session_dir / "runs" / f"run_{run_id}"
        self.llm_raw_dir = self.run_dir / "llm_raw"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.llm_raw_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.report_path = self.run_dir / "session_report.txt"
        self.latest_report_path = self.session_dir / "latest_session_report.txt"
        self.path = self.events_path
        self._event_counts: dict[str, int] = {}
        self._summary: dict[str, Any] = {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "last_event_at": self.started_at,
            "total_events": 0,
            "loops": 0,
            "llm_turns": 0,
            "unknown_states": 0,
            "states_created": 0,
            "states_merged": 0,
            "merge_rejected": 0,
            "actions": 0,
            "clicks": 0,
            "presets": 0,
            "transitions": 0,
            "edges_added": 0,
            "repairs": 0,
            "repair_failures": 0,
            "ocr_calls": 0,
            "ocr_errors": 0,
            "last_state_id": None,
            "last_transition": None,
            "event_counts": self._event_counts,
            "paths": {
                "run_dir": str(self.run_dir),
                "events": str(self.events_path),
                "summary": str(self.summary_path),
                "report": str(self.report_path),
                "latest_report": str(self.latest_report_path),
                "llm_raw": str(self.llm_raw_dir),
            },
        }
        self._report_lines: list[str] = []
        self._write_summary_and_report()

    def event(self, event: str, **fields: Any) -> None:
        ts = _now_iso()
        row = {
            "ts": ts,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        safe_row = _json_safe(row)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe_row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._update_summary(event, safe_row)
        self._report_lines.append(_format_report_line(safe_row))
        self._write_summary_and_report()

    def text(self, message: str, event: str = "console", **fields: Any) -> None:
        print(message)
        self.event(event, message=message, **fields)

    def _update_summary(self, event: str, row: dict[str, Any]) -> None:
        _update_counts(self._event_counts, event)
        _touch_total(self._summary, str(row.get("ts", "")))
        _apply_delta(self._summary, _event_delta(event, row))

    def _write_summary_and_report(self) -> None:
        _write_json(self.summary_path, self._summary)
        _write_report(self.report_path, self._summary, self._report_lines)
        _write_report(self.latest_report_path, self._summary, self._report_lines)


def summarize_match(match: Any) -> dict[str, Any]:
    return {
        "state_id": getattr(match, "state_id", ""),
        "state_dir": str(getattr(match, "state_dir", "")),
        "passed_enabled": getattr(match, "passed_enabled", 0),
        "total_enabled": getattr(match, "total_enabled", 0),
        "passed_all": getattr(match, "passed_all", 0),
        "total_all": getattr(match, "total_all", 0),
        "success": getattr(match, "success", False),
        "conditions": getattr(match, "condition_results", []),
    }


def _safe_name(raw: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(raw))
    safe = safe.strip("._")
    return safe or "session"


def _format_report_line(row: dict[str, Any]) -> str:
    event = str(row.get("event", ""))
    ts = str(row.get("ts", ""))
    message = row.get("message")
    payload = {k: v for k, v in row.items() if k not in {"ts", "session_id", "run_id", "event", "message"}}
    payload_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) if payload else ""
    if message:
        line = f"{ts} {event}: {message}"
        return f"{line}\n  data: {payload_text}" if payload_text else line
    parts = []
    for key in ("state_id", "from_state", "to_state", "action_id", "reason", "slug", "possible_page_type"):
        if key in row and row.get(key) is not None:
            parts.append(f"{key}={row.get(key)}")
    line = f"{ts} {event}" + (f": {' '.join(parts)}" if parts else "")
    return f"{line}\n  data: {payload_text}" if payload_text else line


def _summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "FSM Session Report",
        "",
        "Summary",
        f"session_id: {summary.get('session_id')}",
        f"run_id: {summary.get('run_id')}",
        f"started_at: {summary.get('started_at')}",
        f"last_event_at: {summary.get('last_event_at')}",
        f"total_events: {summary.get('total_events')}",
        f"loops: {summary.get('loops')}",
        f"llm_turns: {summary.get('llm_turns')}",
        f"unknown_states: {summary.get('unknown_states')}",
        f"states_created: {summary.get('states_created')}",
        f"states_merged: {summary.get('states_merged')}",
        f"merge_rejected: {summary.get('merge_rejected')}",
        f"actions: {summary.get('actions')}",
        f"clicks: {summary.get('clicks')}",
        f"presets: {summary.get('presets')}",
        f"transitions: {summary.get('transitions')}",
        f"edges_added: {summary.get('edges_added')}",
        f"repairs: {summary.get('repairs')}",
        f"repair_failures: {summary.get('repair_failures')}",
        f"ocr_calls: {summary.get('ocr_calls')}",
        f"ocr_errors: {summary.get('ocr_errors')}",
        f"last_state_id: {summary.get('last_state_id')}",
        f"last_transition: {summary.get('last_transition')}",
        "",
        "Paths",
    ]
    paths = summary.get("paths", {})
    if isinstance(paths, dict):
        for key, value in paths.items():
            lines.append(f"{key}: {value}")
    lines.extend(["", "Events"])
    return "\n".join(lines) + "\n"


def _event_delta(event: str, row: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if event == "loop_start":
        delta["loops"] = 1
        delta["llm_turns_value"] = row.get("llm_turn_count")
    elif event == "unknown_state":
        delta["unknown_states"] = 1
    elif event == "state_created":
        delta["states_created"] = 1
        delta["last_state_id"] = row.get("state_id")
    elif event == "state_merged_console":
        delta["states_merged"] = 1
        delta["last_state_id"] = row.get("state_id")
    elif event == "merge_rejected":
        delta["merge_rejected"] = 1
    elif event == "action_attempt":
        delta["actions"] = 1
    elif event == "action_click":
        delta["clicks"] = 1
    elif event == "action_preset":
        delta["presets"] = 1
    elif event == "transition":
        delta["transitions"] = 1
        delta["last_transition"] = {
            "from_state": row.get("from_state"),
            "to_state": row.get("to_state"),
            "action_id": row.get("action_id"),
            "reason": row.get("reason"),
        }
        if row.get("to_state"):
            delta["last_state_id"] = row.get("to_state")
    elif event == "graph_edge_added":
        delta["edges_added"] = 1
    elif event == "repair_applied":
        delta["repairs"] = 1
    elif event == "repair_failed":
        delta["repair_failures"] = 1
    elif event == "ocr_summary":
        delta["ocr_calls"] = int(row.get("calls", 0) or 0)
        delta["ocr_errors"] = int(row.get("errors", 0) or 0)
    elif event == "state_selected":
        delta["last_state_id"] = row.get("state_id")
    return delta


def _add(summary: dict[str, Any], key: str, value: int) -> None:
    summary[key] = int(summary.get(key, 0) or 0) + value


def _set_if_present(summary: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        summary[key] = value


def _apply_delta(summary: dict[str, Any], delta: dict[str, Any]) -> None:
    for key in (
        "loops",
        "unknown_states",
        "states_created",
        "states_merged",
        "merge_rejected",
        "actions",
        "clicks",
        "presets",
        "transitions",
        "edges_added",
        "repairs",
        "repair_failures",
        "ocr_calls",
        "ocr_errors",
    ):
        if key in delta:
            _add(summary, key, int(delta[key]))
    if "llm_turns_value" in delta:
        try:
            summary["llm_turns"] = max(int(summary.get("llm_turns", 0) or 0), int(delta["llm_turns_value"]))
        except Exception:
            pass
    _set_if_present(summary, "last_state_id", delta.get("last_state_id"))
    _set_if_present(summary, "last_transition", delta.get("last_transition"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_report(path: Path, summary: dict[str, Any], lines: list[str]) -> None:
    path.write_text(_summary_text(summary) + "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _update_counts(counts: dict[str, int], event: str) -> None:
    counts[event] = int(counts.get(event, 0)) + 1


def _touch_total(summary: dict[str, Any], ts: str) -> None:
    summary["total_events"] = int(summary.get("total_events", 0) or 0) + 1
    summary["last_event_at"] = ts
