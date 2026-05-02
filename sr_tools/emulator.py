from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Generator, Optional

import cv2
import numpy as np
import requests

from .adb import resolve_adb_path


class EmulatorError(RuntimeError):
    pass


def _decode_screencap_png(data: bytes) -> np.ndarray:
    # Device newline behaviors are inconsistent: \r\n or \r\r\n.
    candidates = (data, data.replace(b"\r\n", b"\n"), data.replace(b"\r\r\n", b"\n"))
    for raw in candidates:
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)
            return image
    raise EmulatorError("Unable to decode PNG screenshot from adb screencap -p")


def _decode_screencap_raw(data: bytes) -> np.ndarray:
    if len(data) < 16:
        raise EmulatorError("Raw screencap payload is too short")
    header = np.frombuffer(data[:12], dtype=np.uint32)
    width, height, _ = header
    rgba_size = int(width * height * 4)
    rgba = np.frombuffer(data[-rgba_size:], dtype=np.uint8)
    try:
        rgba = rgba.reshape(height, width, 4)
    except ValueError as exc:
        raise EmulatorError(f"Raw screencap payload shape mismatch: {exc}") from exc
    return cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGB)


class EmulatorClient:
    """ADB-based emulator interaction with stable screenshot decode fallbacks."""

    def __init__(self, serial: str, adb_path: str | None = None, timeout_s: int = 10) -> None:
        self.serial = serial
        self.adb_path = resolve_adb_path(adb_path)
        self.timeout_s = timeout_s

    def _run_adb(self, *args: str, binary: bool = True) -> bytes:
        cmd = [self.adb_path, "-s", self.serial, *map(str, args)]
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            timeout=self.timeout_s,
        )
        if proc.returncode != 0:
            raise EmulatorError(f"ADB command failed: {' '.join(cmd)}\n{proc.stderr.decode('utf-8', 'ignore')}")
        return proc.stdout if binary else proc.stdout.decode("utf-8", "ignore").encode("utf-8")

    def screenshot(self, prefer_png: bool = True) -> np.ndarray:
        if prefer_png:
            data = self._run_adb("shell", "screencap", "-p")
            return _decode_screencap_png(data)
        data = self._run_adb("shell", "screencap")
        return _decode_screencap_raw(data)

    def tap(self, x: int, y: int) -> None:
        self._run_adb("shell", "input", "tap", str(x), str(y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 120) -> None:
        self._run_adb(
            "shell",
            "input",
            "swipe",
            str(x1),
            str(y1),
            str(x2),
            str(y2),
            str(duration_ms),
        )

    def forward(self, local_port: int, remote_port: int) -> None:
        self._run_adb("forward", f"tcp:{local_port}", f"tcp:{remote_port}")


@dataclass
class DroidCastStream:
    """Simple DroidCast frame stream wrapper.

    Usage:
        stream = DroidCastStream(port=53516)
        for frame in stream.frames():
            ...
    """

    port: int = 53516
    endpoint: str = "/preview"
    timeout_s: int = 3
    _session: Optional[requests.Session] = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}{self.endpoint}"

    def _get_session(self) -> requests.Session:
        if self._session is None:
            session = requests.Session()
            session.trust_env = False
            self._session = session
        return self._session

    def frame(self) -> np.ndarray:
        resp = self._get_session().get(self.url, timeout=self.timeout_s)
        resp.raise_for_status()
        image = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise EmulatorError("DroidCast frame decode failed")
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)
        return image

    def frames(self) -> Generator[np.ndarray, None, None]:
        while True:
            yield self.frame()
