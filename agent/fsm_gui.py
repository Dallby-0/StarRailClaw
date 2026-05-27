from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tkinter as tk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from state_machine.tasks import task_workspace_path


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


@dataclass
class NodeDraw:
    state_id: str
    slug: str
    x: float
    y: float
    w: float
    h: float


@dataclass
class EdgeDraw:
    from_state_id: str
    to_state_id: str
    action_id: str
    points: list[tuple[float, float]]


def _q(s: str) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


class FsmGui:
    def __init__(self, graph_path: Path, runtime_path: Path, poll_ms: int = 700, engine: str = "layered") -> None:
        self.graph_path = graph_path
        self.runtime_path = runtime_path
        self.poll_ms = poll_ms
        self.engine = engine

        self.root = tk.Tk()
        self.root.title("FSM Live View")
        self.root.geometry("1360x900")

        self.task_var = tk.StringVar(value=self.graph_path.parent.name if self.graph_path.parent != FSM_DIR else "")
        self.task_frame = tk.Frame(self.root)
        self.task_frame.pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(self.task_frame, text="Task").pack(side="left")
        self.task_entry = tk.Entry(self.task_frame, textvariable=self.task_var, width=28)
        self.task_entry.pack(side="left", padx=(6, 4))
        tk.Button(self.task_frame, text="Load", command=self._load_task).pack(side="left")
        self.task_status_var = tk.StringVar(value=f"workspace: {self.graph_path.parent}")
        tk.Label(self.task_frame, textvariable=self.task_status_var, anchor="w").pack(side="left", padx=(8, 0), fill="x", expand=True)

        self.info_var = tk.StringVar(value="loading...")
        self.info = tk.Label(self.root, textvariable=self.info_var, anchor="w", justify="left")
        self.info.pack(fill="x", padx=8, pady=6)

        self.canvas = tk.Canvas(self.root, bg="#0b0f14", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.last_state_id: str | None = None
        self.prev_state_id: str | None = None
        self.pulse_phase = 0

        self.last_graph_signature = ""
        self.nodes: dict[str, NodeDraw] = {}
        self.edges: list[EdgeDraw] = []
        self.layout_ok = False
        self.layout_error = ""

        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._drag_last: tuple[int, int] | None = None

        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_drag_end)
        self.canvas.bind("<Double-Button-1>", self._on_reset_view)

        self.root.after(50, self._tick)

    def _load_task(self) -> None:
        name = self.task_var.get().strip()
        workspace = task_workspace_path(name or None)
        self.graph_path = workspace / "state_graph.json"
        self.runtime_path = workspace / "runtime_state.json"
        self.task_status_var.set(f"workspace: {workspace}")
        self.last_graph_signature = ""
        self.nodes = {}
        self.edges = []
        self.layout_ok = False
        self.layout_error = ""
        self.last_state_id = None
        self.prev_state_id = None

    def _on_wheel(self, event) -> None:
        factor = 1.12 if event.delta > 0 else 0.9
        old = self.scale
        self.scale = max(0.05, min(240.0, self.scale * factor))
        if old == self.scale:
            return
        mx = event.x
        my = event.y
        wx = (mx - self.offset_x) / old
        wy = (my - self.offset_y) / old
        self.offset_x = mx - wx * self.scale
        self.offset_y = my - wy * self.scale

    def _on_drag_start(self, event) -> None:
        self._drag_last = (event.x, event.y)

    def _on_drag_move(self, event) -> None:
        if self._drag_last is None:
            return
        lx, ly = self._drag_last
        self.offset_x += event.x - lx
        self.offset_y += event.y - ly
        self._drag_last = (event.x, event.y)

    def _on_drag_end(self, _event) -> None:
        self._drag_last = None

    def _on_reset_view(self, _event) -> None:
        self._fit_view()

    def _graph_signature(self, graph: dict[str, Any]) -> str:
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        return f"{len(nodes)}-{len(edges)}-{graph.get('updated_at', '')}"

    def _build_dot(self, graph: dict[str, Any]) -> str:
        nodes = [n for n in graph.get("nodes", []) if isinstance(n, dict) and n.get("enabled", True)]
        edges = [e for e in graph.get("edges", []) if isinstance(e, dict) and e.get("enabled", True)]

        lines = [
            "digraph FSM {",
            "  graph [overlap=false, splines=true, outputorder=edgesfirst, ranksep=1.0, nodesep=0.5];",
            "  node [shape=box, style=filled, fillcolor=\"#1f2733\", color=\"#637a96\", fontname=\"Consolas\", fontsize=11];",
            "  edge [color=\"#7f95b2\", penwidth=1.8, arrowsize=0.8, fontname=\"Consolas\", fontsize=9];",
        ]

        for n in nodes:
            sid = str(n.get("state_id", ""))
            if not sid:
                continue
            slug = str(n.get("slug", sid))
            label = f"{slug}\\n{sid[:8]}"
            lines.append(f"  {_q(sid)} [label={_q(label)}];")

        for e in edges:
            fs = str(e.get("from_state_id", ""))
            ts = str(e.get("to_state_id", ""))
            if not fs or not ts:
                continue
            action = str(e.get("action_id", ""))
            lines.append(f"  {_q(fs)} -> {_q(ts)} [label={_q(action)}];")

        lines.append("}")
        return "\n".join(lines)

    def _layout_with_graphviz(self, graph: dict[str, Any]) -> tuple[dict[str, NodeDraw], list[EdgeDraw], str]:
        if self.engine == "layered":
            return {}, [], "graphviz disabled"

        dot = self._build_dot(graph)
        cmd = [self.engine, "-Tplain"]
        try:
            proc = subprocess.run(cmd, input=dot, text=True, capture_output=True, check=False)
        except FileNotFoundError:
            return {}, [], f"Graphviz engine not found: {self.engine}"
        if proc.returncode != 0:
            err = proc.stderr.strip() or f"graphviz exit code={proc.returncode}"
            return {}, [], err

        nodes: dict[str, NodeDraw] = {}
        edges: list[EdgeDraw] = []

        for raw in proc.stdout.splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = shlex.split(line)
            if not parts:
                continue
            typ = parts[0]
            if typ == "node" and len(parts) >= 7:
                sid = parts[1]
                try:
                    x = float(parts[2])
                    y = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                except ValueError:
                    continue
                label = parts[6] if len(parts) >= 7 else sid
                slug = label.split("\\n", 1)[0] if "\\n" in label else label.split("\n", 1)[0]
                nodes[sid] = NodeDraw(state_id=sid, slug=slug, x=x, y=y, w=w, h=h)
            elif typ == "edge" and len(parts) >= 6:
                fs = parts[1]
                ts = parts[2]
                try:
                    n = int(parts[3])
                except ValueError:
                    continue
                need = 4 + n * 2
                if len(parts) < need:
                    continue
                pts: list[tuple[float, float]] = []
                ok = True
                for i in range(n):
                    try:
                        px = float(parts[4 + i * 2])
                        py = float(parts[5 + i * 2])
                    except ValueError:
                        ok = False
                        break
                    pts.append((px, py))
                if not ok:
                    continue
                action_id = ""
                if len(parts) > need:
                    action_id = parts[need]
                edges.append(EdgeDraw(from_state_id=fs, to_state_id=ts, action_id=action_id, points=pts))

        return nodes, edges, ""

    def _layout_layered(self, graph: dict[str, Any]) -> tuple[dict[str, NodeDraw], list[EdgeDraw]]:
        nodes_raw = [n for n in graph.get("nodes", []) if isinstance(n, dict) and n.get("enabled", True)]
        edges_raw = [e for e in graph.get("edges", []) if isinstance(e, dict) and e.get("enabled", True)]
        ids = [str(n.get("state_id", "")) for n in nodes_raw if str(n.get("state_id", ""))]
        if not ids:
            return {}, []

        id_set = set(ids)
        graph_index = {sid: i for i, sid in enumerate(ids)}
        slug_by_id = {str(n.get("state_id", "")): str(n.get("slug", n.get("state_id", ""))) for n in nodes_raw}

        out_edges: dict[str, list[str]] = {sid: [] for sid in ids}
        in_edges: dict[str, list[str]] = {sid: [] for sid in ids}
        weak_neighbors: dict[str, set[str]] = {sid: set() for sid in ids}
        for e in edges_raw:
            fs = str(e.get("from_state_id", ""))
            ts = str(e.get("to_state_id", ""))
            if fs not in id_set or ts not in id_set:
                continue
            out_edges[fs].append(ts)
            in_edges[ts].append(fs)
            if fs != ts:
                weak_neighbors[fs].add(ts)
                weak_neighbors[ts].add(fs)

        components: list[list[str]] = []
        seen: set[str] = set()
        for sid in ids:
            if sid in seen:
                continue
            q: deque[str] = deque([sid])
            seen.add(sid)
            comp: list[str] = []
            while q:
                cur = q.popleft()
                comp.append(cur)
                for nxt in sorted(weak_neighbors[cur], key=lambda item: graph_index[item]):
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append(nxt)
            components.append(comp)

        pos: dict[str, tuple[float, float]] = {}
        y_cursor = 0.0
        x_gap = 320.0
        y_gap = 92.0
        component_gap = 150.0

        for comp in components:
            comp_set = set(comp)
            roots = [sid for sid in comp if not [p for p in in_edges[sid] if p in comp_set and p != sid]]
            if not roots:
                roots = [max(comp, key=lambda sid: (len([t for t in out_edges[sid] if t in comp_set]), -graph_index[sid]))]

            layer: dict[str, int] = {}
            q = deque()
            for root in sorted(roots, key=lambda sid: graph_index[sid]):
                layer[root] = 0
                q.append(root)
            while q:
                cur = q.popleft()
                for nxt in sorted(out_edges[cur], key=lambda sid: graph_index[sid]):
                    if nxt not in comp_set or nxt == cur:
                        continue
                    wanted = layer[cur] + 1
                    if nxt not in layer or wanted < layer[nxt]:
                        layer[nxt] = wanted
                        q.append(nxt)

            for sid in comp:
                if sid not in layer:
                    preds = [layer[p] for p in in_edges[sid] if p in layer]
                    layer[sid] = (max(preds) + 1) if preds else 0

            layer_map: dict[int, list[str]] = defaultdict(list)
            for sid in comp:
                layer_map[layer[sid]].append(sid)

            orders: dict[int, list[str]] = {depth: sorted(items, key=lambda sid: graph_index[sid]) for depth, items in layer_map.items()}
            depths = sorted(orders)
            for _ in range(8):
                for depth in depths[1:]:
                    rank = {sid: i for i, sid in enumerate(orders.get(depth - 1, []))}
                    orders[depth].sort(
                        key=lambda sid: (
                            self._neighbor_barycenter([p for p in in_edges[sid] if layer.get(p) == depth - 1], rank),
                            graph_index[sid],
                        )
                    )
                for depth in reversed(depths[:-1]):
                    rank = {sid: i for i, sid in enumerate(orders.get(depth + 1, []))}
                    orders[depth].sort(
                        key=lambda sid: (
                            self._neighbor_barycenter([t for t in out_edges[sid] if layer.get(t) == depth + 1], rank),
                            graph_index[sid],
                        )
                    )

            max_rows = max(len(items) for items in orders.values())
            comp_height = max(1.0, (max_rows - 1) * y_gap)
            for depth in depths:
                row = orders[depth]
                row_height = (len(row) - 1) * y_gap
                start_y = y_cursor + (comp_height - row_height) / 2.0
                for row_index, sid in enumerate(row):
                    pos[sid] = (depth * x_gap, start_y + row_index * y_gap)
            y_cursor += comp_height + component_gap

        nodes: dict[str, NodeDraw] = {}
        for n in nodes_raw:
            sid = str(n.get("state_id", ""))
            if not sid or sid not in pos:
                continue
            px, py = pos[sid]
            slug = slug_by_id.get(sid, sid)
            width = max(170.0, min(260.0, 8.0 * len(slug) + 42.0))
            nodes[sid] = NodeDraw(state_id=sid, slug=slug, x=px, y=py, w=width, h=58.0)

        edges: list[EdgeDraw] = []
        parallel_count: dict[tuple[str, str], int] = defaultdict(int)
        for e in edges_raw:
            fs = str(e.get("from_state_id", ""))
            ts = str(e.get("to_state_id", ""))
            if fs not in nodes or ts not in nodes:
                continue
            pair = (fs, ts)
            parallel_count[pair] += 1
            lane = parallel_count[pair] - 1
            points = self._route_layered_edge(nodes[fs], nodes[ts], lane)
            edges.append(EdgeDraw(from_state_id=fs, to_state_id=ts, action_id=str(e.get("action_id", "")), points=points))
        return nodes, edges

    @staticmethod
    def _neighbor_barycenter(neighbor_ids: list[str], rank: dict[str, int]) -> float:
        vals = [rank[sid] for sid in neighbor_ids if sid in rank]
        if not vals:
            return 1_000_000.0
        return sum(vals) / len(vals)

    @staticmethod
    def _route_layered_edge(source: NodeDraw, target: NodeDraw, lane: int) -> list[tuple[float, float]]:
        sx, sy = source.x, source.y
        tx, ty = target.x, target.y
        lane_offset = (lane % 4) * 0.16
        if source.state_id == target.state_id:
            r = max(source.w, source.h) * 0.75 + 0.25 + lane_offset
            return [(sx + source.w / 2, sy), (sx + source.w / 2 + r, sy - r), (sx, sy - r * 1.3), (sx - source.w / 2, sy)]

        if tx > sx:
            start = (sx + source.w / 2, sy)
            end = (tx - target.w / 2, ty)
        elif tx < sx:
            start = (sx - source.w / 2, sy)
            end = (tx + target.w / 2, ty)
        else:
            side = 1 if ty >= sy else -1
            start = (sx, sy + side * source.h / 2)
            end = (tx, ty - side * target.h / 2)

        if abs(start[1] - end[1]) < 0.18:
            return [start, end]

        mid_x = (start[0] + end[0]) / 2.0
        if tx <= sx:
            mid_x -= 0.45 + lane_offset
        return [start, (mid_x, start[1]), (mid_x, end[1]), end]

    def _fit_view(self) -> None:
        if not self.nodes:
            self.scale = 1.0
            self.offset_x = 40.0
            self.offset_y = 40.0
            return
        min_x = min(n.x - n.w / 2 for n in self.nodes.values())
        max_x = max(n.x + n.w / 2 for n in self.nodes.values())
        min_y = min(n.y - n.h / 2 for n in self.nodes.values())
        max_y = max(n.y + n.h / 2 for n in self.nodes.values())
        w = max(0.1, max_x - min_x)
        h = max(0.1, max_y - min_y)
        cw = max(400, self.canvas.winfo_width())
        ch = max(300, self.canvas.winfo_height())
        pad = 60.0
        sx = (cw - 2 * pad) / w
        sy = (ch - 2 * pad) / h
        self.scale = max(0.72, min(2.0, min(sx, sy)))
        self.offset_x = pad - min_x * self.scale
        self.offset_y = pad - min_y * self.scale

    def _ensure_layout(self, graph: dict[str, Any]) -> None:
        sig = self._graph_signature(graph)
        if sig == self.last_graph_signature and self.nodes:
            return
        self.last_graph_signature = sig
        if self.engine == "layered":
            self.nodes, self.edges = self._layout_layered(graph)
            self.layout_error = ""
            self.layout_ok = bool(self.nodes)
        else:
            self.nodes, self.edges, err = self._layout_with_graphviz(graph)
            if err:
                fb_nodes, fb_edges = self._layout_layered(graph)
                self.nodes = fb_nodes
                self.edges = fb_edges
                self.layout_error = f"{err}; fallback=layered"
                self.layout_ok = bool(self.nodes)
            else:
                self.layout_error = ""
                self.layout_ok = bool(self.nodes)
        self._fit_view()

    def _to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y

    def _draw_arrow_head(self, x1: float, y1: float, x2: float, y2: float, color: str, size: float = 9.0) -> None:
        dx = x2 - x1
        dy = y2 - y1
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return
        ux = dx / d
        uy = dy / d
        px = -uy
        py = ux
        bx = x2 - ux * size
        by = y2 - uy * size
        p1 = (x2, y2)
        p2 = (bx + px * size * 0.6, by + py * size * 0.6)
        p3 = (bx - px * size * 0.6, by - py * size * 0.6)
        self.canvas.create_polygon([p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]], fill=color, outline=color)

    def _draw(self) -> None:
        graph = _load_json(self.graph_path)
        runtime = _load_json(self.runtime_path)
        self._ensure_layout(graph)

        current_state = runtime.get("last_state_id")
        if isinstance(current_state, str) and current_state:
            if self.last_state_id != current_state:
                self.prev_state_id = self.last_state_id
                self.last_state_id = current_state

        self.canvas.delete("all")
        cw = max(400, self.canvas.winfo_width())
        ch = max(300, self.canvas.winfo_height())
        self.canvas.create_rectangle(0, 0, cw, ch, fill="#0b0f14", outline="")

        active_edge = None
        if self.prev_state_id and self.last_state_id:
            active_edge = (self.prev_state_id, self.last_state_id)

        for e in self.edges:
            if len(e.points) < 2:
                continue
            pts = [self._to_canvas(px, py) for px, py in e.points]
            flat = [v for p in pts for v in p]
            color = "#89a0bd"
            width_px = 2.4
            is_active = active_edge is not None and e.from_state_id == active_edge[0] and e.to_state_id == active_edge[1]
            if is_active:
                glow = 8 + (self.pulse_phase % 3)
                self.canvas.create_line(*flat, fill="#ffd166", width=glow, smooth=True)
                color = "#ff7a00"
                width_px = 4.5
            self.canvas.create_line(*flat, fill=color, width=width_px, smooth=True)

            x1, y1 = pts[-2]
            x2, y2 = pts[-1]
            self._draw_arrow_head(x1, y1, x2, y2, color=color, size=8.0 if not is_active else 11.0)

            mid = len(pts) // 2
            mx, my = pts[mid]
            lbl = e.action_id or "action"
            tw = max(32, min(150, len(lbl) * 7 + 12))
            self.canvas.create_rectangle(mx - tw / 2, my - 13, mx + tw / 2, my + 4, fill="#172233", outline="#2e425e")
            self.canvas.create_text(mx, my - 5, text=lbl, fill="#f4f8ff", font=("Consolas", 9, "bold"))

        for sid, n in self.nodes.items():
            x, y = self._to_canvas(n.x, n.y)
            w = n.w * self.scale
            h = n.h * self.scale
            is_current = sid == self.last_state_id
            fill = "#233345" if not is_current else "#a52834"
            outline = "#8aa2be" if not is_current else "#ffd166"
            ow = 2 if not is_current else 4
            self.canvas.create_rectangle(x - w / 2, y - h / 2, x + w / 2, y + h / 2, fill=fill, outline=outline, width=ow)
            self.canvas.create_text(x, y - 8, text=n.slug[:24], fill="#ffffff", font=("Consolas", 10, "bold"))
            self.canvas.create_text(x, y + 9, text=sid[:8], fill="#cdd8e8", font=("Consolas", 9))

        status = "layout=ok"
        if self.layout_error:
            status = f"layout=error({self.layout_error})"
        info = (
            f"nodes={len(self.nodes)} edges={len(self.edges)} "
            f"current={self.last_state_id or '-'} prev={self.prev_state_id or '-'} "
            f"run_id={runtime.get('run_id', '-')} engine={self.engine} scale={self.scale:.2f} {status}"
        )
        self.info_var.set(info)

    def _tick(self) -> None:
        self.pulse_phase = (self.pulse_phase + 1) % 30
        self._draw()
        self.root.after(self.poll_ms, self._tick)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live FSM graph viewer")
    parser.add_argument("--graph", default=str(GRAPH_PATH))
    parser.add_argument("--runtime", default=str(RUNTIME_PATH))
    parser.add_argument("--task", default=None, help="task name under StateMachineTasks")
    parser.add_argument("--poll-ms", type=int, default=700)
    parser.add_argument("--engine", choices=["layered", "sfdp", "dot", "neato"], default="layered")
    args = parser.parse_args()
    graph_path = Path(args.graph)
    runtime_path = Path(args.runtime)
    if args.task:
        workspace = task_workspace_path(args.task)
        graph_path = workspace / "state_graph.json"
        runtime_path = workspace / "runtime_state.json"

    gui = FsmGui(
        graph_path=graph_path,
        runtime_path=runtime_path,
        poll_ms=max(200, args.poll_ms),
        engine=args.engine,
    )
    gui.run()


if __name__ == "__main__":
    main()
