from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional


class AdbNotFoundError(FileNotFoundError):
    pass


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def bundled_adb_path() -> Path:
    return _project_root() / "adbutils" / "binaries" / "adb.exe"


def resolve_adb_path(adb_path: Optional[str] = None) -> str:
    if adb_path:
        p = Path(adb_path)
        if p.exists():
            return str(p)
        hit = shutil.which(adb_path)
        if hit:
            return hit
        raise AdbNotFoundError(f"ADB not found from --adb-path: {adb_path}")

    bundled = bundled_adb_path()
    if bundled.exists():
        return str(bundled)

    hit = shutil.which("adb")
    if hit:
        return hit

    raise AdbNotFoundError(
        "ADB not found. Put adb at ./adbutils/binaries/adb.exe or add adb to PATH."
    )

