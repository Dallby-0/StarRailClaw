from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_FSM_DIR = Path("StateMachineResources")
TASKS_ROOT = Path("StateMachineTasks")
TASK_SUMMARY_FILENAME = "task_summary.md"


def task_slug(raw: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(raw).strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("_")
    return safe or "task"


def task_workspace_path(task: str | None = None, task_dir: str | Path | None = None) -> Path:
    if task_dir:
        return Path(task_dir)
    if task:
        return TASKS_ROOT / task_slug(task)
    return DEFAULT_FSM_DIR


def list_task_workspaces(tasks_root: Path = TASKS_ROOT) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not tasks_root.exists():
        return out
    for d in sorted(tasks_root.iterdir()):
        if not d.is_dir():
            continue
        summary_path = d / TASK_SUMMARY_FILENAME
        summary = ""
        if summary_path.exists():
            try:
                summary = summary_path.read_text(encoding="utf-8").strip().splitlines()[0][:160]
            except OSError:
                summary = ""
        out.append({"name": d.name, "path": str(d), "summary": summary})
    return out


def create_task_workspace(
    task: str,
    *,
    summary: str = "",
    tasks_root: str | Path = TASKS_ROOT,
) -> Path:
    name = task_slug(task)
    workspace = Path(tasks_root) / name
    workspace.mkdir(parents=True, exist_ok=True)

    summary_path = workspace / TASK_SUMMARY_FILENAME
    if not summary_path.exists():
        text = summary.strip() or f"# {name}\n\nDescribe this FSM task in English.\n"
        summary_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return workspace


def configure_fsm_workspace(workspace: str | Path) -> Path:
    root = Path(workspace)
    import state_machine.constants as constants

    paths = {
        "FSM_DIR": root,
        "FSM_TEMPLATES_DIR": root / "templates",
        "FSM_GRAPH_PATH": root / "state_graph.json",
        "FSM_SCHEMA_PATH": root / "llm_protocol_schema.json",
        "FSM_RUNTIME_PATH": root / "runtime_state.json",
        "FSM_DEBUG_DIR": root / "debug",
        "EXPERIENCE_PATH": root / "experience.md",
        "TASK_SUMMARY_PATH": root / TASK_SUMMARY_FILENAME,
    }
    for name, value in paths.items():
        setattr(constants, name, value)

    for module_name in ("state_machine.io", "state_machine.loop", "state_machine.llm_tasks", "state_machine.logger"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for name, value in paths.items():
            if hasattr(module, name):
                setattr(module, name, value)

    return root
