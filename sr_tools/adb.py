from __future__ import annotations

import re
import socket
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional


class AdbNotFoundError(FileNotFoundError):
    pass


class AdbDeviceNotFoundError(RuntimeError):
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


def _run_adb(adb_path: str, *args: str, timeout_s: float = 5.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [adb_path, *map(str, args)],
        check=False,
        capture_output=True,
        timeout=timeout_s,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )


def _list_device_serials(adb_path: str) -> List[str]:
    proc = _run_adb(adb_path, "devices")
    if proc.returncode != 0:
        return []
    serials: List[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("List of devices attached"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            serials.append(parts[0])
    return serials


def _is_local_port_open(port: int, timeout_s: float = 0.15) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _parse_port_from_serial(serial: str) -> Optional[int]:
    m = re.match(r"^(?:127\.0\.0\.1|localhost):(\d+)$", serial)
    if not m:
        return None
    return int(m.group(1))


def mumu_candidate_ports() -> List[int]:
    # MuMu ADB ports start from 16384, and multi-instance ports usually increase by 32.
    ports = [7555, 16384]
    ports.extend(range(16384, 17409, 32))
    return sorted(set(ports))


def auto_connect_mumu(adb_path: Optional[str] = None, timeout_s: float = 1.5) -> List[str]:
    adb = resolve_adb_path(adb_path)
    connected = set(_list_device_serials(adb))
    for port in mumu_candidate_ports():
        if not _is_local_port_open(port):
            continue
        target = f"127.0.0.1:{port}"
        if target in connected:
            continue
        _run_adb(adb, "connect", target, timeout_s=timeout_s)
    return _list_device_serials(adb)


def resolve_target_serial(
    serial: Optional[str] = None,
    adb_path: Optional[str] = None,
    *,
    auto_connect: bool = True,
) -> str:
    adb = resolve_adb_path(adb_path)
    if serial:
        existing = set(_list_device_serials(adb))
        if serial in existing:
            return serial
        if _parse_port_from_serial(serial) is not None:
            _run_adb(adb, "connect", serial, timeout_s=2.0)
        return serial

    serials = _list_device_serials(adb)
    if auto_connect:
        serials = auto_connect_mumu(adb_path=adb)
    if len(serials) == 1:
        return serials[0]
    if len(serials) > 1:
        loopbacks = [s for s in serials if _parse_port_from_serial(s) is not None]
        if loopbacks:
            return max(loopbacks, key=lambda s: _parse_port_from_serial(s) or -1)
        return serials[0]
    raise AdbDeviceNotFoundError(
        "No ADB devices found. Start emulator and ensure adb is enabled; "
        "or pass --serial explicitly."
    )
