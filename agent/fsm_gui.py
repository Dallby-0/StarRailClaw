from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import tkinter as tk


FSM_DIR = Path("StateMachineResources")
GRAPH_PATH = FSM_DIR / "state_graph.json"
RUNTIME_PATH = FSM_DIR / "runtime_state.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class FsmGui:
    def __init__(self, graph_path: Path, runtime_path: Path, poll_ms: int = 700) -> None:
        self.graph_path = graph_path
        self.runtime_path = runtime_path
        self.poll_ms = poll_ms

        self.root = tk.Tk()
        self.root.title("FSM Live View")
        self.root.geometry("1200x820")

        self.info_var = tk.StringVar(value="waiting for data...")
        self.info = tk.Label(self.root, textvariable=self.info_var, anchor="w", justify="left")
        self.info.pack(fill="x", padx=8, pady=6)

        self.canvas = tk.Canvas(self.root, bg="#101418", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.node_positions: dict[str, tuple[float, float]] = {}
        self.last_state_id: str | None = None
        self.prev_state_id: str | None = None
        self.last_graph_signature = ""

        self.root.after(50, self._tick)

    def _graph_signature(self, graph: dict[str, Any]) -> str:
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        return f"{len(nodes)}-{len(edges)}-{graph.get('updated_at', '')}"

    def _ensure_layout(self, graph: dict[str, Any], width: int, height: int) -> None:
        sig = self._graph_signature(graph)
        if sig == self.last_graph_signature and self.node_positions:
            return
        self.last_graph_signature = sig

        nodes = [n for n in graph.get("nodes", []) if isinstance(n, dict) and n.get("enabled", True)]
        ids = [str(n.get("state_id", "")) for n in nodes if str(n.get("state_id", ""))]
        self.node_positions = {}
        if not ids:
            return

        cx = width / 2
        cy = height / 2
        radius = max(120, min(width, height) * 0.36)
        for i, sid in enumerate(ids):
            angle = (2 * math.pi * i / len(ids)) - math.pi / 2
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            self.node_positions[sid] = (x, y)

    def _draw(self) -> None:
        graph = _load_json(self.graph_path)
        runtime = _load_json(self.runtime_path)

        current_state = runtime.get("last_state_id")
        if isinstance(current_state, str) and current_state:
            if self.last_state_id != current_state:
                self.prev_state_id = self.last_state_id
                self.last_state_id = current_state

        width = max(self.canvas.winfo_width(), 400)
        height = max(self.canvas.winfo_height(), 300)
        self._ensure_layout(graph, width, height)

        self.canvas.delete("all")

        nodes = [n for n in graph.get("nodes", []) if isinstance(n, dict) and n.get("enabled", True)]
        edges = [e for e in graph.get("edges", []) if isinstance(e, dict) and e.get("enabled", True)]
        node_by_id = {str(n.get("state_id", "")): n for n in nodes}

        active_edge = None
        if self.prev_state_id and self.last_state_id:
            for e in edges:
                fs = str(e.get("from_state_id", ""))
                ts = str(e.get("to_state_id", ""))
                if fs == self.prev_state_id and ts == self.last_state_id:
                    active_edge = (fs, ts, str(e.get("action_id", "")))
                    break

        for e in edges:
            fs = str(e.get("from_state_id", ""))
            ts = str(e.get("to_state_id", ""))
            if fs not in self.node_positions or ts not in self.node_positions:
                continue
            x1, y1 = self.node_positions[fs]
            x2, y2 = self.node_positions[ts]
            color = "#4f5d75"
            width_px = 2
            if active_edge and fs == active_edge[0] and ts == active_edge[1]:
                color = "#ff8c42"
                width_px = 4
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width_px, arrow=tk.LAST, arrowshape=(14, 16, 6))
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            self.canvas.create_text(mx, my - 8, text=str(e.get("action_id", "")), fill="#cfd8e3", font=("Consolas", 10))

        r = 34
        for sid, (x, y) in self.node_positions.items():
            n = node_by_id.get(sid, {})
            slug = str(n.get("slug", sid))[:18]
            is_current = sid == self.last_state_id
            fill = "#1f77b4" if not is_current else "#e63946"
            outline = "#9fb3c8" if not is_current else "#ffd166"
            ow = 2 if not is_current else 4
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=outline, width=ow)
            self.canvas.create_text(x, y - 3, text=slug, fill="#ffffff", font=("Consolas", 10, "bold"))
            self.canvas.create_text(x, y + 15, text=sid[:8], fill="#dbe6f2", font=("Consolas", 9))

        info = (
            f"nodes={len(nodes)} edges={len(edges)} "
            f"current={self.last_state_id or '-'} prev={self.prev_state_id or '-'} "
            f"run_id={runtime.get('run_id', '-')}"
        )
        self.info_var.set(info)

    def _tick(self) -> None:
        self._draw()
        self.root.after(self.poll_ms, self._tick)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live FSM graph viewer")
    parser.add_argument("--graph", default=str(GRAPH_PATH))
    parser.add_argument("--runtime", default=str(RUNTIME_PATH))
    parser.add_argument("--poll-ms", type=int, default=700)
    args = parser.parse_args()

    gui = FsmGui(graph_path=Path(args.graph), runtime_path=Path(args.runtime), poll_ms=max(200, args.poll_ms))
    gui.run()


if __name__ == "__main__":
    main()
