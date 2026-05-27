from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from state_machine.tasks import list_task_workspaces, task_workspace_path


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

    .search-input {
      width: 220px;
      height: 30px;
      min-width: 120px;
      padding: 0 9px;
      border: 1px solid #33465f;
      border-radius: 6px;
      background: #0a1017;
      color: var(--text);
      font: inherit;
      outline: none;
    }

    .search-input:focus {
      border-color: var(--accent);
    }

    .task-select {
      width: 150px;
      height: 30px;
      min-width: 100px;
      padding: 0 6px;
      border: 1px solid #33465f;
      border-radius: 6px;
      background: #0a1017;
      color: var(--text);
      font: inherit;
      outline: none;
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
      grid-template-columns: 1fr 420px;
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

    .annotation-frame {
      position: relative;
      width: 100%;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #070b10;
      cursor: zoom-in;
    }

    .annotation-frame img {
      display: block;
      width: 100%;
      height: auto;
      user-select: none;
    }

    .bbox {
      position: absolute;
      border: 2px solid rgba(160, 170, 185, .82);
      background: rgba(160, 170, 185, .08);
      box-shadow: 0 0 0 1px rgba(0, 0, 0, .45);
      pointer-events: none;
    }

    .bbox.active {
      border-color: #ff4057;
      background: rgba(255, 64, 87, .12);
      box-shadow: 0 0 0 1px rgba(0, 0, 0, .55), 0 0 14px rgba(255, 64, 87, .42);
    }

    .bbox-label {
      position: absolute;
      left: -2px;
      top: -23px;
      max-width: 260px;
      padding: 2px 5px;
      border-radius: 4px 4px 4px 0;
      background: rgba(18, 26, 36, .94);
      color: #eef4ff;
      font-size: 10px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      border: 1px solid rgba(160, 170, 185, .7);
    }

    .bbox.active .bbox-label {
      border-color: #ff4057;
      color: #fff2f4;
    }

    .condition-list {
      display: grid;
      gap: 6px;
      margin-top: 8px;
    }

    .condition-item {
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-left-width: 4px;
      border-radius: 6px;
      background: #0d141d;
      overflow-wrap: anywhere;
    }

    .condition-item.active {
      border-left-color: #ff4057;
    }

    .condition-item.inactive {
      border-left-color: #8b96a5;
      color: #c3ccd8;
    }

    .condition-meta {
      color: var(--muted);
      font-size: 11px;
      margin-top: 2px;
    }

    .lightbox {
      position: fixed;
      inset: 0;
      display: none;
      z-index: 1000;
      background: rgba(3, 6, 10, .92);
    }

    .lightbox.open {
      display: grid;
      grid-template-rows: auto 1fr;
    }

    .lightbox-bar {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 9px 12px;
      border-bottom: 1px solid #27384c;
      background: rgba(13, 19, 27, .96);
    }

    .lightbox-title {
      flex: 1;
      min-width: 0;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
      color: var(--muted);
    }

    .lightbox-stage {
      position: relative;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
      cursor: grab;
    }

    .lightbox-stage.dragging {
      cursor: grabbing;
    }

    .lightbox-content {
      position: absolute;
      left: 50%;
      top: 50%;
      transform-origin: 0 0;
      border: 1px solid #31445c;
      background: #070b10;
      box-shadow: 0 18px 60px rgba(0, 0, 0, .48);
    }

    .lightbox-content img {
      display: block;
      width: 100%;
      height: 100%;
      user-select: none;
      pointer-events: none;
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="toolbar">
      <select id="taskSelect" class="task-select" title="Known task workspaces">
        <option value="">default</option>
      </select>
      <input id="taskInput" class="search-input" type="search" placeholder="task name" />
      <button id="taskLoadBtn">Load Task</button>
      <button id="fitBtn">Fit</button>
      <button id="layoutBtn">Layout</button>
      <button id="thumbnailBtn">Thumbnails</button>
      <button id="labelBtn">Label Below</button>
      <input id="searchInput" class="search-input" type="search" placeholder="slug or node id" />
      <button id="searchBtn">Search</button>
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
          <div class="label">Annotation</div>
          <div id="annotation" class="value">-</div>
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
  <div id="lightbox" class="lightbox">
    <div class="lightbox-bar">
      <button id="lightboxFitBtn">Fit</button>
      <button id="lightboxOneBtn">100%</button>
      <button id="lightboxCloseBtn">Close</button>
      <div id="lightboxTitle" class="lightbox-title">-</div>
    </div>
    <div id="lightboxStage" class="lightbox-stage">
      <div id="lightboxContent" class="lightbox-content"></div>
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
          selector: 'node.thumbnail',
          style: {
            'width': 190,
            'height': 107,
            'background-image': 'data(thumbnail_url)',
            'background-fit': 'cover',
            'background-clip': 'node',
            'background-opacity': .72,
            'background-color': '#0e1722',
            'text-background-color': '#111821',
            'text-background-opacity': .82,
            'text-background-padding': 3
          }
        },
        {
          selector: 'node.label-below',
          style: {
            'text-valign': 'bottom',
            'text-halign': 'center',
            'text-margin-y': 22,
            'text-background-color': '#101720',
            'text-background-opacity': .92,
            'text-background-padding': 4,
            'text-border-color': '#2d4058',
            'text-border-opacity': .9,
            'text-border-width': 1,
            'text-max-width': 210
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
      templateCache: new Map(),
      templateLoadingId: null,
      lightboxTemplate: null,
      lightboxScale: 1,
      lightboxX: 0,
      lightboxY: 0,
      lightboxDrag: null,
      thumbnails: false,
      labelsBelow: false,
      follow: true,
      paused: false,
      didInitialLayout: false,
      prevCurrent: null
    };

    const statusEl = document.getElementById('status');
    const taskSelect = document.getElementById('taskSelect');
    const taskInput = document.getElementById('taskInput');
    const taskLoadBtn = document.getElementById('taskLoadBtn');
    const selectedEl = document.getElementById('selected');
    const annotationEl = document.getElementById('annotation');
    const runtimeEl = document.getElementById('runtime');
    const outgoingEl = document.getElementById('outgoing');
    const incomingEl = document.getElementById('incoming');
    const fitBtn = document.getElementById('fitBtn');
    const layoutBtn = document.getElementById('layoutBtn');
    const thumbnailBtn = document.getElementById('thumbnailBtn');
    const labelBtn = document.getElementById('labelBtn');
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const followBtn = document.getElementById('followBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    const lightboxEl = document.getElementById('lightbox');
    const lightboxStageEl = document.getElementById('lightboxStage');
    const lightboxContentEl = document.getElementById('lightboxContent');
    const lightboxTitleEl = document.getElementById('lightboxTitle');
    const lightboxFitBtn = document.getElementById('lightboxFitBtn');
    const lightboxOneBtn = document.getElementById('lightboxOneBtn');
    const lightboxCloseBtn = document.getElementById('lightboxCloseBtn');

    function shortId(id) {
      return id ? String(id).slice(0, 8) : '-';
    }

    function nodeLabel(node) {
      const slug = node.slug || node.state_id || 'state';
      return `${slug}\n${shortId(node.state_id)}`;
    }

    function templateImageUrl(node) {
      const stateId = node.state_id || '';
      const slug = node.slug || '';
      return `/api/template-image?state_id=${encodeURIComponent(stateId)}&slug=${encodeURIComponent(slug)}`;
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
            thumbnail_url: templateImageUrl(node),
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
      updateNodeViewMode(false);
      renderSide();
      statusEl.textContent = [
        `nodes=${graph.nodes.length}`,
        `edges=${graph.edges.length}`,
        `current=${shortId(runtime.last_state_id)}`,
        `run=${runtime.run_id || '-'}`,
        `task=${payload.task_name || 'default'}`,
        `updated=${payload.graph_updated_at || '-'}`
      ].join('  ');
    }

    async function loadTaskList() {
      try {
        const res = await fetch('/api/tasks', { cache: 'no-store' });
        if (!res.ok) return;
        const payload = await res.json();
        taskSelect.innerHTML = '<option value="">default</option>';
        for (const task of payload.tasks || []) {
          const opt = document.createElement('option');
          opt.value = task.name || '';
          opt.textContent = task.name || task.path || '';
          opt.title = task.summary || task.path || '';
          taskSelect.appendChild(opt);
        }
      } catch (_err) {
      }
    }

    async function loadTask(name) {
      const task = String(name || '').trim();
      const res = await fetch(`/api/task?name=${encodeURIComponent(task)}`, { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const payload = await res.json();
      state.signature = '';
      state.runtime = {};
      state.graph = { nodes: [], edges: [] };
      state.selectedId = null;
      state.templateCache.clear();
      state.didInitialLayout = false;
      state.prevCurrent = null;
      taskInput.value = payload.task_name || '';
      taskSelect.value = payload.task_name || '';
      statusEl.textContent = `workspace=${payload.workspace}`;
      await poll();
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

    function conditionTitle(condition) {
      const type = condition.display_type || condition.type || condition.kind || 'condition';
      const stability = condition.stability || '-';
      const discrimination = condition.discrimination || '-';
      return `${type}  stability:${stability}  discrimination:${discrimination}`;
    }

    function renderAnnotation(template) {
      annotationEl.innerHTML = '';
      if (!template || template.missing) {
        annotationEl.textContent = template && template.message ? template.message : '-';
        return;
      }

      const frame = document.createElement('div');
      frame.className = 'annotation-frame';
      if (!template.screenshot_url) {
        annotationEl.textContent = 'screenshot not found';
        return;
      }
      const img = document.createElement('img');
      img.src = template.screenshot_url;
      img.alt = template.slug || 'state screenshot';
      frame.append(img);

      appendBboxes(frame, template);
      frame.addEventListener('dblclick', () => openLightbox(template));
      annotationEl.append(frame);

      const list = document.createElement('div');
      list.className = 'condition-list';
      for (const condition of template.conditions || []) {
        const item = document.createElement('div');
        item.className = `condition-item ${condition.active ? 'active' : 'inactive'}`;
        const title = document.createElement('div');
        title.textContent = `${condition.active ? 'active' : 'inactive'} · ${condition.id || '-'} · ${condition.kind || '-'}`;
        const meta = document.createElement('div');
        meta.className = 'condition-meta';
        meta.textContent = `${conditionTitle(condition)} · ${condition.brief || ''}`;
        item.append(title, meta);
        list.append(item);
      }
      annotationEl.append(list);
    }

    function appendBboxes(parent, template) {
      const width = template.image_width || 1000;
      const height = template.image_height || 1000;
      for (const condition of template.conditions || []) {
        const bbox = condition.bbox;
        if (!Array.isArray(bbox) || bbox.length !== 4) continue;
        const [x1, y1, x2, y2] = bbox.map(Number);
        const box = document.createElement('div');
        box.className = `bbox ${condition.active ? 'active' : ''}`;
        box.style.left = `${x1 / width * 100}%`;
        box.style.top = `${y1 / height * 100}%`;
        box.style.width = `${Math.max(1, x2 - x1) / width * 100}%`;
        box.style.height = `${Math.max(1, y2 - y1) / height * 100}%`;
        const label = document.createElement('div');
        label.className = 'bbox-label';
        label.textContent = conditionTitle(condition);
        box.title = `${condition.id || ''}\n${condition.brief || ''}`;
        box.append(label);
        parent.append(box);
      }
    }

    function renderLightboxContent(template) {
      lightboxContentEl.innerHTML = '';
      const img = document.createElement('img');
      img.src = template.screenshot_url;
      img.alt = template.slug || 'state screenshot';
      lightboxContentEl.append(img);
      appendBboxes(lightboxContentEl, template);
      lightboxTitleEl.textContent = `${template.slug || 'state'} · double-click image preview · wheel zoom · drag pan · Esc close`;
      const width = template.screenshot_width || template.image_width || 1000;
      const height = template.screenshot_height || template.image_height || 1000;
      lightboxContentEl.style.width = `${width}px`;
      lightboxContentEl.style.height = `${height}px`;
    }

    function updateLightboxTransform() {
      lightboxContentEl.style.transform = `translate(${state.lightboxX}px, ${state.lightboxY}px) scale(${state.lightboxScale}) translate(-50%, -50%)`;
    }

    function fitLightbox() {
      const template = state.lightboxTemplate;
      if (!template) return;
      const width = template.screenshot_width || template.image_width || 1000;
      const height = template.screenshot_height || template.image_height || 1000;
      const rect = lightboxStageEl.getBoundingClientRect();
      state.lightboxScale = Math.min((rect.width - 48) / width, (rect.height - 48) / height, 1.8);
      state.lightboxX = 0;
      state.lightboxY = 0;
      updateLightboxTransform();
    }

    function openLightbox(template) {
      if (!template || !template.screenshot_url) return;
      state.lightboxTemplate = template;
      renderLightboxContent(template);
      lightboxEl.classList.add('open');
      requestAnimationFrame(fitLightbox);
    }

    function closeLightbox() {
      lightboxEl.classList.remove('open');
      state.lightboxTemplate = null;
      state.lightboxDrag = null;
      lightboxStageEl.classList.remove('dragging');
    }

    function updateNodeViewMode(relayout = true) {
      cy.nodes().toggleClass('thumbnail', state.thumbnails);
      cy.nodes().toggleClass('label-below', state.labelsBelow);
      thumbnailBtn.classList.toggle('active', state.thumbnails);
      labelBtn.classList.toggle('active', state.labelsBelow);
      labelBtn.textContent = state.labelsBelow ? 'Label Inside' : 'Label Below';
      if (relayout) {
        runLayout(true);
        window.setTimeout(() => cy.fit(cy.elements(), 60), 320);
      }
    }

    function searchNode() {
      const query = searchInput.value.trim().toLowerCase();
      if (!query) return;
      const matches = state.graph.nodes.filter(node => {
        const id = String(node.state_id || '').toLowerCase();
        const slug = String(node.slug || '').toLowerCase();
        return id.includes(query) || slug.includes(query);
      });
      if (!matches.length) {
        statusEl.textContent = `node not found: ${searchInput.value.trim()}`;
        return;
      }
      const node = matches[0];
      const ele = cy.getElementById(node.state_id);
      if (ele.empty()) {
        statusEl.textContent = `node not rendered: ${node.slug || node.state_id}`;
        return;
      }
      state.selectedId = node.state_id;
      ele.select();
      cy.animate({ center: { eles: ele }, zoom: Math.max(cy.zoom(), 0.9) }, { duration: 280 });
      renderSide();
      statusEl.textContent = `found ${matches.length}: ${node.slug || node.state_id} (${shortId(node.state_id)})`;
    }

    async function loadTemplateForSelection(selected, node) {
      if (!selected) {
        renderAnnotation(null);
        return;
      }
      if (state.templateCache.has(selected)) {
        renderAnnotation(state.templateCache.get(selected));
        return;
      }
      if (state.templateLoadingId === selected) return;
      state.templateLoadingId = selected;
      annotationEl.textContent = 'loading...';
      try {
        const slug = node && node.slug ? node.slug : '';
        const res = await fetch(`/api/template?state_id=${encodeURIComponent(selected)}&slug=${encodeURIComponent(slug)}`, { cache: 'no-store' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = await res.json();
        state.templateCache.set(selected, payload);
        if ((state.selectedId || state.runtime.last_state_id) === selected) {
          renderAnnotation(payload);
        }
      } catch (err) {
        const payload = { missing: true, message: String(err) };
        state.templateCache.set(selected, payload);
        renderAnnotation(payload);
      } finally {
        state.templateLoadingId = null;
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
      loadTemplateForSelection(selected, node);

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
    thumbnailBtn.addEventListener('click', () => {
      state.thumbnails = !state.thumbnails;
      updateNodeViewMode(true);
    });
    labelBtn.addEventListener('click', () => {
      state.labelsBelow = !state.labelsBelow;
      updateNodeViewMode(true);
    });
    searchBtn.addEventListener('click', searchNode);
    searchInput.addEventListener('keydown', event => {
      if (event.key === 'Enter') searchNode();
    });
    taskLoadBtn.addEventListener('click', () => {
      loadTask(taskInput.value).catch(err => {
        statusEl.innerHTML = `<span class="missing">${String(err)}</span>`;
      });
    });
    taskInput.addEventListener('keydown', event => {
      if (event.key === 'Enter') taskLoadBtn.click();
    });
    taskSelect.addEventListener('change', () => {
      taskInput.value = taskSelect.value;
      taskLoadBtn.click();
    });
    followBtn.addEventListener('click', () => {
      state.follow = !state.follow;
      followBtn.classList.toggle('active', state.follow);
    });
    pauseBtn.addEventListener('click', () => {
      state.paused = !state.paused;
      pauseBtn.classList.toggle('active', state.paused);
      pauseBtn.textContent = state.paused ? 'Resume' : 'Pause';
    });
    lightboxCloseBtn.addEventListener('click', closeLightbox);
    lightboxFitBtn.addEventListener('click', fitLightbox);
    lightboxOneBtn.addEventListener('click', () => {
      state.lightboxScale = 1;
      state.lightboxX = 0;
      state.lightboxY = 0;
      updateLightboxTransform();
    });
    lightboxEl.addEventListener('dblclick', event => {
      if (event.target === lightboxStageEl) closeLightbox();
    });
    lightboxStageEl.addEventListener('wheel', event => {
      if (!state.lightboxTemplate) return;
      event.preventDefault();
      const oldScale = state.lightboxScale;
      const factor = event.deltaY < 0 ? 1.12 : 0.9;
      state.lightboxScale = Math.max(0.08, Math.min(8, state.lightboxScale * factor));
      const rect = lightboxStageEl.getBoundingClientRect();
      const mx = event.clientX - rect.left - rect.width / 2;
      const my = event.clientY - rect.top - rect.height / 2;
      state.lightboxX = mx - (mx - state.lightboxX) * (state.lightboxScale / oldScale);
      state.lightboxY = my - (my - state.lightboxY) * (state.lightboxScale / oldScale);
      updateLightboxTransform();
    }, { passive: false });
    lightboxStageEl.addEventListener('mousedown', event => {
      if (!state.lightboxTemplate || event.button !== 0) return;
      state.lightboxDrag = { x: event.clientX, y: event.clientY, ox: state.lightboxX, oy: state.lightboxY };
      lightboxStageEl.classList.add('dragging');
    });
    window.addEventListener('mousemove', event => {
      if (!state.lightboxDrag) return;
      state.lightboxX = state.lightboxDrag.ox + event.clientX - state.lightboxDrag.x;
      state.lightboxY = state.lightboxDrag.oy + event.clientY - state.lightboxDrag.y;
      updateLightboxTransform();
    });
    window.addEventListener('mouseup', () => {
      state.lightboxDrag = null;
      lightboxStageEl.classList.remove('dragging');
    });
    window.addEventListener('keydown', event => {
      if (event.key === 'Escape' && lightboxEl.classList.contains('open')) {
        closeLightbox();
      }
    });
    window.addEventListener('resize', () => {
      if (lightboxEl.classList.contains('open')) fitLightbox();
    });

    loadTaskList();
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


def _templates_dir_for_graph(graph_path: Path) -> Path:
    return graph_path.parent / "templates"


def _png_size(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as f:
            header = f.read(24)
    except OSError:
        return None
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def _resolve_under(path: Path, root: Path) -> Path | None:
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
    except OSError:
        return None
    if resolved == root_resolved or root_resolved in resolved.parents:
        return resolved
    return None


def _find_template_state(templates_dir: Path, state_id: str, slug: str = "") -> tuple[Path, dict[str, Any]] | None:
    if state_id:
        for path in sorted(templates_dir.glob("*/state.json")):
            if not path.exists():
                continue
            data = _load_json(path)
            if isinstance(data, dict) and str(data.get("state_id", "")) == state_id:
                return path, data
        return None

    candidates: list[Path] = []
    if slug:
        candidates.append(templates_dir / slug / "state.json")
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        if slug and str(data.get("slug", "")) == slug:
            return path, data
    return None


def _path_from_state_value(value: Any, template_dir: Path, templates_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    raw = Path(value)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([Path.cwd() / raw, template_dir / raw.name])
    for candidate in candidates:
        safe = _resolve_under(candidate, templates_dir)
        if safe and safe.exists():
            return safe
    return None


def _find_screenshot_path(state: dict[str, Any], state_path: Path, templates_dir: Path) -> Path | None:
    template_dir = state_path.parent
    for condition in state.get("match_conditions", []):
        if not isinstance(condition, dict):
            continue
        source = condition.get("source")
        if not isinstance(source, dict):
            continue
        path = _path_from_state_value(source.get("screenshot_path"), template_dir, templates_dir)
        if path:
            return path
    for path in sorted(template_dir.glob("screenshot*.png")):
        safe = _resolve_under(path, templates_dir)
        if safe and safe.exists():
            return safe
    return None


def _condition_display_type(condition: dict[str, Any]) -> str:
    kind = str(condition.get("kind", "")).lower()
    if "text" in kind:
        return "text"
    if "template" in kind or "pattern" in kind or "region" in kind:
        return "template"
    condition_id = str(condition.get("id", "")).lower()
    if "text" in condition_id:
        return "text"
    if "pattern" in condition_id or "template" in condition_id:
        return "template"
    return kind or "condition"


def _normalized_conditions(state: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, condition in enumerate(state.get("match_conditions", []), start=1):
        if not isinstance(condition, dict):
            continue
        bbox = condition.get("bbox") or (condition.get("params") or {}).get("rect")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            bbox_nums = [float(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        result.append(
            {
                "id": str(condition.get("id", f"condition_{index}")),
                "kind": str(condition.get("kind", "")),
                "display_type": _condition_display_type(condition),
                "active": bool(condition.get("enabled", True)),
                "enabled": bool(condition.get("enabled", True)),
                "condition_status": str(condition.get("condition_status", "")),
                "stability": str(condition.get("stability", "")),
                "discrimination": str(condition.get("discrimination", "")),
                "brief": str(condition.get("brief", "")),
                "role": str(condition.get("role", "")),
                "weight": condition.get("weight", 1.0),
                "bbox": bbox_nums,
            }
        )
    return result


def _condition_coordinate_size(conditions: list[dict[str, Any]], image_size: tuple[int, int] | None) -> tuple[int, int]:
    max_x = 0.0
    max_y = 0.0
    for condition in conditions:
        bbox = condition.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        max_x = max(max_x, float(bbox[0]), float(bbox[2]))
        max_y = max(max_y, float(bbox[1]), float(bbox[3]))
    if max_x <= 1000 and max_y <= 1000:
        return 1000, 1000
    if image_size:
        return image_size
    return int(max(1000, max_x)), int(max(1000, max_y))


class FsmStateHandler(BaseHTTPRequestHandler):
    graph_path = GRAPH_PATH
    runtime_path = RUNTIME_PATH
    task_name = ""

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
        if parsed.path == "/api/tasks":
            self._send_tasks()
            return
        if parsed.path == "/api/task":
            self._set_task(parse_qs(parsed.query))
            return
        if parsed.path == "/api/template":
            self._send_template(parse_qs(parsed.query))
            return
        if parsed.path == "/api/template-image":
            self._send_template_image(parse_qs(parsed.query))
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
            "task_name": self.task_name,
            "workspace": str(self.graph_path.parent),
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

    def _send_tasks(self) -> None:
        payload = {
            "tasks": list_task_workspaces(),
            "current": self.task_name,
            "workspace": str(self.graph_path.parent),
        }
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8")

    def _set_task(self, query: dict[str, list[str]]) -> None:
        name = query.get("name", [""])[0].strip()
        workspace = task_workspace_path(name or None)
        type(self).task_name = name
        type(self).graph_path = workspace / "state_graph.json"
        type(self).runtime_path = workspace / "runtime_state.json"
        payload = {
            "task_name": name,
            "workspace": str(workspace),
            "graph_path": str(type(self).graph_path),
            "runtime_path": str(type(self).runtime_path),
        }
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8")

    def _send_template(self, query: dict[str, list[str]]) -> None:
        state_id = query.get("state_id", [""])[0]
        slug = query.get("slug", [""])[0]
        templates_dir = _templates_dir_for_graph(self.graph_path)
        found = _find_template_state(templates_dir, state_id, slug)
        if not found:
            body = json.dumps(
                {
                    "missing": True,
                    "message": f"template state not found: {slug or state_id or '-'}",
                    "conditions": [],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8")
            return

        state_path, state = found
        screenshot_path = _find_screenshot_path(state, state_path, templates_dir)
        size = _png_size(screenshot_path) if screenshot_path else None
        conditions = _normalized_conditions(state)
        coord_size = _condition_coordinate_size(conditions, size)
        payload = {
            "missing": False,
            "state_id": str(state.get("state_id", "")),
            "slug": str(state.get("slug", state_path.parent.name)),
            "display_name": str(state.get("display_name", "")),
            "description": str(state.get("description", "")),
            "state_path": str(state_path),
            "screenshot_path": str(screenshot_path) if screenshot_path else "",
            "screenshot_url": (
                f"/api/template-image?state_id={quote(state_id)}&slug={quote(slug)}" if screenshot_path else ""
            ),
            "image_width": coord_size[0],
            "image_height": coord_size[1],
            "screenshot_width": size[0] if size else None,
            "screenshot_height": size[1] if size else None,
            "conditions": conditions,
            "actions": state.get("actions", []),
        }
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8")

    def _send_template_image(self, query: dict[str, list[str]]) -> None:
        state_id = query.get("state_id", [""])[0]
        slug = query.get("slug", [""])[0]
        templates_dir = _templates_dir_for_graph(self.graph_path)
        found = _find_template_state(templates_dir, state_id, slug)
        if not found:
            self.send_error(HTTPStatus.NOT_FOUND, "template state not found")
            return
        state_path, state = found
        screenshot_path = _find_screenshot_path(state, state_path, templates_dir)
        if not screenshot_path:
            self.send_error(HTTPStatus.NOT_FOUND, "screenshot not found")
            return
        try:
            body = screenshot_path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "screenshot not readable")
            return
        self._send_bytes(body, "image/png")

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
    parser.add_argument("--task", default=None, help="task name under StateMachineTasks")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the viewer in the default browser")
    args = parser.parse_args()
    graph_path = Path(args.graph)
    runtime_path = Path(args.runtime)
    task_name = args.task or ""
    if args.task:
        workspace = task_workspace_path(args.task)
        graph_path = workspace / "state_graph.json"
        runtime_path = workspace / "runtime_state.json"

    handler = type(
        "ConfiguredFsmStateHandler",
        (FsmStateHandler,),
        {
            "graph_path": graph_path,
            "runtime_path": runtime_path,
            "task_name": task_name,
        },
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[fsm-cyto] serving {url}")
    print(f"[fsm-cyto] graph={graph_path} runtime={runtime_path} task={task_name or 'default'}")

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
