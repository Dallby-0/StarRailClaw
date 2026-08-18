from __future__ import annotations

from state_machine.page_handler.store import ensure_page_handler


def run_page_handler(*args, **kwargs):
    from state_machine.page_handler.runner import run_page_handler as _run_page_handler

    return _run_page_handler(*args, **kwargs)


__all__ = ["ensure_page_handler", "run_page_handler"]
