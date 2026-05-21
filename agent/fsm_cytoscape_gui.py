from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


FSM_DIR = Path("StateMachineResources")
GRAPH_PATH = FSM_DIR / "state_graph.json"
RUNTIME_PATH = FSM_DIR / "runtime_state.json"


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FSM Live View</title>
  <script src="https://unpkg.com/cytoscape@3.29.2/dist/cytoscape.min.js"></script>
  <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
  <script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
  <style>
    :root {
      color-scheme: dark;
      --bg: #090d12;
      --panel: #111821;
      --panel2: #172231;
      --text: #e7edf7;
      --muted: #97a6ba;
      --line: #2a3a4e;
      --edge: #7f94ad;
      --accent: #f5b642;
      --current: #d94a57;
    }

    * {
      box-sizing: border-box;
    }

    html,
    body {
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font: 13px/1.4 Consolas, "Segoe UI", sans-serif;
    }

    .app {
      display: grid;
      grid-template-rows: auto 1fr;
      width: 100%;
      height: 100%;
    }

    .toolbar {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      background: #0d131b;
      min-width: 0;
    }

    button {
      height: 30px;
      padding: 0 10px;
      border: 1px solid #33465f;
      border-radius: 6px;
      background: var(--panel2);
      color: var(--text);
      font: inherit;
      cursor: pointer;
    }

    button:hover {
      border-color: #527092;
      background: #1d2b3d;
    }

    button.active {
      border-color: var(--accent);
      color: #fff4da;
    }

    .status {
      flex: 1;
      min-width: 0;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
      color: var(--muted);
    }

    .wrap {
      display: grid;
      grid-template-columns: 1fr 340px;
      min-height: 0;
    }

    #cy {
      min-width: 0;
      min-height: 0;
      background:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px),
        var(--bg);
      background-size: 32px 32px;
    }

    .side {
      min-width: 0;
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 12px;
      overflow: auto;
    }

    .section {
      padding: 10px 0;
      border-bottom: 1px solid var(--line);
    }

    .section:first-child {
      padding-top: 0;
    }

    .label {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .04em;
      margin-bottom: 6px;
    }

    .value {
      overflow-wrap: anywhere;
      color: var(--text);
    }

    .kv {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 4px 8px;
      align-items: start;
    }

    .kv div:nth-child(odd) {
      color: var(--muted);
    }

    .edge-list {
      display: grid;
      gap: 6px;
    }

    .edge-item {
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0d141d;
      overflow-wrap: anywhere;
    }

    .missing {
      color: #ff8b8b;
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="toolbar">
      <button id="fitBtn">Fit</button>
      <button id="layoutBtn">Layout</button>
      <button id="followBtn" class="active">Follow</button>
      <button id="pauseBtn">Pause</button>
      <div id="status" class="status">loading...</div>
    </div>
    <div class="wrap">
      <div id="cy"></div>
      <aside class="side">
        <div class="section">
          <div class="label">Selected</div>
          <div id="selected" class="value">-</div>
        </div>
        <div class="section">
          <div class="label">Runtime</div>
          <div id="runtime" class="kv"></div>
        </div>
        <div class="section">
          <div class="label">Outgoing</div>
          <div id="outgoing" class="edge-list"></div>
        </div>
        <div class="section">
          <div class="label">Incoming</div>
          <div id="incoming" class="edge-list"></div>
        </div>
      </aside>
    </div>
  </div>

  <script>
    cytoscape.use(cytoscapeDagre);

    const cy = cytoscape({
      container: document.getElementById('cy'),
      wheelSensitivity: 0.18,
      minZoom: 0.08,
      maxZoom: 2.5,
      elements: [],
      style: [
        {
          selector: 'node',
          style: {
            'shape': 'round-rectangle',
            'width': 190,
            'height': 58,
            'background-color': '#233345',
            'border-color': '#6f86a0',
            'border-width': 2,
            'label': 'data(label)',
            'font-family': 'Consolas, monospace',
            'font-size': 11,
            'font-weight': 700,
            'color': '#eef4ff',
            'text-wrap': 'wrap',
            'text-max-width': 170,
            'text-valign': 'center',
            'text-halign': 'center',
            'overlay-opacity': 0
          }
        },
        {
          selector: 'node.current',
          style: {
            'background-color': '#b73342',
            'border-color': '#f5b642',
            'border-width': 4,
            'box-shadow-blur': 8
          }
        },
        {
          selector: 'node.pending',
          style: {
            'border-color': '#58c4dd',
            'border-style': 'dashed',
            'border-width': 4
          }
        },
        {
          selector: 'node:selected',
          style: {
            'border-color': '#ffffff',
            'border-width': 4
          }
        },
        {
          selector: 'edge',
          style: {
            'curve-style': 'bezier',
            'control-point-step-size': 42,
            'width': 2,
            'line-color': '#7f94ad',
            'target-arrow-color': '#7f94ad',
            'target-arrow-shape': 'triangle',
            'arrow-scale': 1.15,
            'label': 'data(action_id)',
            'font-family': 'Consolas, monospace',
            'font-size': 9,
            'color': '#d8e2f0',
            'text-background-color': '#111821',
            'text-background-opacity': 0.92,
            'text-background-padding': 3,
            'text-rotation': 'autorotate',
            'overlay-opacity': 0
          }
        },
        {
          selector: 'edge.active',
          style: {
            'width': 5,
            'line-color': '#ff9f1c',
            'target-arrow-color': '#ff9f1c',
            'color': '#fff4da',
            'z-index': 20
          }
        },
        {
          selector: 'edge.fallback',
          style: {
            'line-style': 'dashed',
            'line-color': '#c084fc',
            'target-arrow-color': '#c084fc'
          }
        },
        {
          selector: 'edge.loop',
          style: {
            'curve-style': 'loop',
            'loop-direction': '-45deg',
            'loop-sweep': '80deg'
          }
        }
      ]
    });

    const state = {
      signature: '',
      runtime: {},
      graph: { nodes: [], edges: [] },
      selectedId: null,
      follow: true,
      paused: false,
      didInitialLayout: false,
      prevCurrent: null
    };

    const statusEl = document.getElementById('status');
    const selectedEl = document.getElementById('selected');
    const runtimeEl = document.getElementById('runtime');
    const outgoingEl = document.getElementById('outgoing');
    const incomingEl = document.getElementById('incoming');
    const fitBtn = document.getElementById('fitBtn');
    const layoutBtn = document.getElementById('layoutBtn');
    const followBtn = document.getElementById('followBtn');
    const pauseBtn = document.getElementById('pauseBtn');

    function shortId(id) {
      return id ? String(id).slice(0, 8) : '-';
    }

    function nodeLabel(node) {
      const slug = node.slug || node.state_id || 'state';
      return `${slug}\n${shortId(node.state_id)}`;
    }

    function edgeId(edge, index) {
      return `${edge.from_state_id}->${edge.to_state_id}:${edge.action_id || 'action'}:${index}`;
    }

    function isFallbackAction(action) {
      const s = String(action || '').toLowerCase();
      return s.includes('fallback') || s.includes('unknown') || s.includes('repair');
    }

    function runLayout(animate = true) {
      cy.layout({
        name: 'dagre',
        rankDir: 'LR',
        nodeSep: 70,
        rankSep: 150,
        edgeSep: 28,
        animate,
        animationDuration: 280,
        fit: false
      }).run();
    }

    function placeNewNode(ele, graphEdgeMap) {
      const id = ele.id();
      const incoming = graphEdgeMap.in.get(id) || [];
      const outgoing = graphEdgeMap.out.get(id) || [];
      const parentEdge = incoming.find(e => cy.getElementById(e.from_state_id).nonempty());
      const childEdge = outgoing.find(e => cy.getElementById(e.to_state_id).nonempty());
      if (parentEdge) {
        const p = cy.getElementById(parentEdge.from_state_id).position();
        const siblings = incoming.length + outgoing.length;
        ele.position({ x: p.x + 300, y: p.y + (siblings % 5 - 2) * 85 });
        return;
      }
      if (childEdge) {
        const p = cy.getElementById(childEdge.to_state_id).position();
        ele.position({ x: p.x - 300, y: p.y });
        return;
      }
      const count = cy.nodes().length;
      ele.position({ x: (count % 6) * 260, y: Math.floor(count / 6) * 110 });
    }

    function makeEdgeMaps(edges) {
      const maps = { in: new Map(), out: new Map() };
      for (const edge of edges) {
        if (!maps.in.has(edge.to_state_id)) maps.in.set(edge.to_state_id, []);
        if (!maps.out.has(edge.from_state_id)) maps.out.set(edge.from_state_id, []);
        maps.in.get(edge.to_state_id).push(edge);
        maps.out.get(edge.from_state_id).push(edge);
      }
      return maps;
    }

    function applyGraph(payload) {
      const graph = payload.graph || { nodes: [], edges: [] };
      const runtime = payload.runtime || {};
      state.graph = graph;
      state.runtime = runtime;

      const nodeIds = new Set(graph.nodes.map(n => n.state_id));
      const edgeIds = new Set(graph.edges.map((e, i) => edgeId(e, i)));
      const maps = makeEdgeMaps(graph.edges);

      cy.batch(() => {
        cy.nodes().forEach(ele => {
          if (!nodeIds.has(ele.id())) ele.remove();
        });
        cy.edges().forEach(ele => {
          if (!edgeIds.has(ele.id())) ele.remove();
        });

        const newNodes = [];
        for (const node of graph.nodes) {
          const id = node.state_id;
          if (!id) continue;
          const existing = cy.getElementById(id);
          const data = {
            id,
            label: nodeLabel(node),
            slug: node.slug || id,
            state_id: id
          };
          if (existing.nonempty()) {
            existing.data(data);
          } else {
            newNodes.push(cy.add({ group: 'nodes', data }));
          }
        }

        for (const ele of newNodes) {
          placeNewNode(ele, maps);
        }

        graph.edges.forEach((edge, index) => {
          if (!nodeIds.has(edge.from_state_id) || !nodeIds.has(edge.to_state_id)) return;
          const id = edgeId(edge, index);
          const existing = cy.getElementById(id);
          const classes = [
            isFallbackAction(edge.action_id) ? 'fallback' : '',
            edge.from_state_id === edge.to_state_id ? 'loop' : ''
          ].join(' ');
          const data = {
            id,
            source: edge.from_state_id,
            target: edge.to_state_id,
            action_id: edge.action_id || 'action',
            created_at: edge.created_at || ''
          };
          if (existing.nonempty()) {
            existing.data(data);
            existing.classes(classes);
          } else {
            cy.add({ group: 'edges', data, classes });
          }
        });

        cy.elements().removeClass('current pending active');
        if (runtime.last_state_id) {
          cy.getElementById(runtime.last_state_id).addClass('current');
        }
        if (runtime.pending_from_state_id) {
          cy.getElementById(runtime.pending_from_state_id).addClass('pending');
        }
        if (state.prevCurrent && runtime.last_state_id) {
          cy.edges().filter(edge => (
            edge.source().id() === state.prevCurrent &&
            edge.target().id() === runtime.last_state_id
          )).addClass('active');
        }
      });

      if (!state.didInitialLayout) {
        runLayout(false);
        cy.fit(cy.elements(), 60);
        state.didInitialLayout = true;
      }

      if (state.follow && runtime.last_state_id) {
        const current = cy.getElementById(runtime.last_state_id);
        if (current.nonempty()) {
          cy.animate({ center: { eles: current }, zoom: Math.max(cy.zoom(), 0.75) }, { duration: 260 });
        }
      }

      if (runtime.last_state_id) {
        state.prevCurrent = runtime.last_state_id;
      }
      renderSide();
      statusEl.textContent = [
        `nodes=${graph.nodes.length}`,
        `edges=${graph.edges.length}`,
        `current=${shortId(runtime.last_state_id)}`,
        `run=${runtime.run_id || '-'}`,
        `updated=${payload.graph_updated_at || '-'}`
      ].join('  ');
    }

    async function poll() {
      if (state.paused) return;
      try {
        const res = await fetch(`/api/state?signature=${encodeURIComponent(state.signature)}`, { cache: 'no-store' });
        if (res.status === 304) return;
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = await res.json();
        if (payload.signature !== state.signature) {
          state.signature = payload.signature || '';
          applyGraph(payload);
        } else {
          state.runtime = payload.runtime || {};
          applyGraph(payload);
        }
      } catch (err) {
        statusEl.innerHTML = `<span class="missing">${String(err)}</span>`;
      }
    }

    function renderRuntime(runtime) {
      runtimeEl.innerHTML = '';
      const keys = [
        'run_id',
        'last_state_id',
        'pending_from_state_id',
        'pending_action_id',
        'last_transition_ok',
        'llm_turn_count',
        'session_tier',
        'repair_fail_count'
      ];
      for (const key of keys) {
        const k = document.createElement('div');
        const v = document.createElement('div');
        k.textContent = key;
        v.textContent = runtime[key] === undefined || runtime[key] === null ? '-' : String(runtime[key]);
        runtimeEl.append(k, v);
      }
    }

    function edgeHtml(edge, direction) {
      const el = document.createElement('div');
      el.className = 'edge-item';
      const other = direction === 'out' ? edge.to_state_id : edge.from_state_id;
      const node = state.graph.nodes.find(n => n.state_id === other);
      el.textContent = `${edge.action_id || 'action'} -> ${node ? node.slug : shortId(other)} (${shortId(other)})`;
      return el;
    }

    function renderSide() {
      renderRuntime(state.runtime || {});
      const selected = state.selectedId || state.runtime.last_state_id;
      const node = state.graph.nodes.find(n => n.state_id === selected);
      selectedEl.textContent = node ? `${node.slug}\n${node.state_id}` : '-';

      outgoingEl.innerHTML = '';
      incomingEl.innerHTML = '';
      if (!selected) return;
      const outgoing = state.graph.edges.filter(e => e.from_state_id === selected);
      const incoming = state.graph.edges.filter(e => e.to_state_id === selected);
      for (const edge of outgoing) outgoingEl.append(edgeHtml(edge, 'out'));
      for (const edge of incoming) incomingEl.append(edgeHtml(edge, 'in'));
      if (!outgoing.length) outgoingEl.textContent = '-';
      if (!incoming.length) incomingEl.textContent = '-';
    }

    cy.on('tap', 'node', event => {
      state.selectedId = event.target.id();
      renderSide();
    });

    cy.on('tap', event => {
      if (event.target === cy) {
        state.selectedId = null;
        renderSide();
      }
    });

    fitBtn.addEventListener('click', () => cy.fit(cy.elements(), 60));
    layoutBtn.addEventListener('click', () => runLayout(true));
    followBtn.addEventListener('click', () => {
      state.follow = !state.follow;
      followBtn.classList.toggle('active', state.follow);
    });
    pauseBtn.addEventListener('click', () => {
      state.paused = !state.paused;
      pauseBtn.classList.toggle('active', state.paused);
      pauseBtn.textContent = state.paused ? 'Resume' : 'Pause';
    });

    poll();
    setInterval(poll, 700);
  </script>
</body>
</html>
"""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


def _file_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _enabled_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for node in graph.get("nodes", []):
        if not isinstance(node, dict) or not node.get("enabled", True):
            continue
        state_id = str(node.get("state_id", ""))
        if not state_id:
            continue
        result.append(
            {
                "state_id": state_id,
                "slug": str(node.get("slug", state_id)),
                "enabled": True,
            }
        )
    return result


def _enabled_edges(graph: dict[str, Any], node_ids: set[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict) or not edge.get("enabled", True):
            continue
        from_state_id = str(edge.get("from_state_id", ""))
        to_state_id = str(edge.get("to_state_id", ""))
        if from_state_id not in node_ids or to_state_id not in node_ids:
            continue
        result.append(
            {
                "from_state_id": from_state_id,
                "to_state_id": to_state_id,
                "action_id": str(edge.get("action_id", "")),
                "weight": edge.get("weight", 1.0),
                "created_at": edge.get("created_at", ""),
            }
        )
    return result


class FsmStateHandler(BaseHTTPRequestHandler):
    graph_path = GRAPH_PATH
    runtime_path = RUNTIME_PATH

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[fsm-cyto] {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            self._send_state(parse_qs(parsed.query))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def _send_state(self, query: dict[str, list[str]]) -> None:
        graph_mtime = _file_mtime_ns(self.graph_path)
        runtime_mtime = _file_mtime_ns(self.runtime_path)
        signature = f"{graph_mtime}:{runtime_mtime}"
        client_signature = query.get("signature", [""])[0]
        if client_signature == signature:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return

        graph = _load_json(self.graph_path)
        runtime = _load_json(self.runtime_path)
        nodes = _enabled_nodes(graph)
        node_ids = {node["state_id"] for node in nodes}
        edges = _enabled_edges(graph, node_ids)
        payload = {
            "signature": signature,
            "graph_path": str(self.graph_path),
            "runtime_path": str(self.runtime_path),
            "graph_updated_at": graph.get("updated_at", ""),
            "graph": {
                "schema_version": graph.get("schema_version", ""),
                "nodes": nodes,
                "edges": edges,
            },
            "runtime": runtime,
            "server_time": time.time(),
        }
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8")

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or mimetypes.types_map.get(".txt", "text/plain"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cytoscape.js live FSM graph viewer")
    parser.add_argument("--graph", default=str(GRAPH_PATH))
    parser.add_argument("--runtime", default=str(RUNTIME_PATH))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the viewer in the default browser")
    args = parser.parse_args()

    handler = type(
        "ConfiguredFsmStateHandler",
        (FsmStateHandler,),
        {
            "graph_path": Path(args.graph),
            "runtime_path": Path(args.runtime),
        },
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[fsm-cyto] serving {url}")
    print(f"[fsm-cyto] graph={Path(args.graph)} runtime={Path(args.runtime)}")

    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[fsm-cyto] stopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
