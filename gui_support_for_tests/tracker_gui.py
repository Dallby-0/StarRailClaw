from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sr_tools import EmulatorClient
from sr_tools.adb import AdbDeviceNotFoundError, AdbNotFoundError, resolve_adb_path, resolve_target_serial
from sr_tools.emulator import EmulatorError


BBox = Tuple[int, int, int, int]


@dataclass
class SharedState:
    frame: Optional[np.ndarray] = None
    predicted_bbox: Optional[BBox] = None
    error: Optional[str] = None
    capture_count: int = 0
    error_count: int = 0
    last_capture_ts: float = 0.0
    last_frame_shape: Optional[Tuple[int, int, int]] = None


def map_bbox_between_sizes(bbox: BBox, src_wh: Tuple[int, int], dst_wh: Tuple[int, int]) -> BBox:
    sx = dst_wh[0] / max(1, src_wh[0])
    sy = dst_wh[1] / max(1, src_wh[1])
    x, y, w, h = bbox
    return int(round(x * sx)), int(round(y * sy)), int(round(w * sx)), int(round(h * sy))


def clamp_bbox(bbox: BBox, width: int, height: int) -> Optional[BBox]:
    x, y, w, h = bbox
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    if w <= 1 or h <= 1:
        return None
    return x, y, w, h


def make_nanotracker(backbone_path: str, neckhead_path: str):
    if not hasattr(cv2, "TrackerNano_create") or not hasattr(cv2, "TrackerNano_Params"):
        raise RuntimeError("Current OpenCV build does not provide TrackerNano. Install opencv-contrib-python >= 4.7.")
    params = cv2.TrackerNano_Params()
    params.backbone = backbone_path
    params.neckhead = neckhead_path
    return cv2.TrackerNano_create(params)


def to_tracker_input(frame_rgb: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(frame_rgb, size_wh, interpolation=cv2.INTER_LINEAR)
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(bgr)


class NanoTrackGui:
    def __init__(
        self,
        serial: str,
        backbone: str,
        neckhead: str,
        interval_s: float,
        adb_path: str,
        track_width: int,
        track_height: int,
    ) -> None:
        self.serial = serial
        self.backbone = backbone
        self.neckhead = neckhead
        self.interval_s = interval_s
        self.adb_path = adb_path
        self.track_width = track_width
        self.track_height = track_height

        self.client = EmulatorClient(serial=serial, adb_path=adb_path)
        self.state = SharedState()
        self.lock = threading.Lock()

        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._capture_loop, daemon=True)
        self.tracker_lock = threading.Lock()

        self.tracking_enabled = False
        self.tracker = None
        self.init_bbox: Optional[BBox] = None
        self.last_norm_shape: Optional[Tuple[int, int, int]] = None
        self.capture_fps: float = 0.0
        self._last_fps_t = 0.0
        self._last_fps_n = 0

        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_current: Optional[Tuple[int, int]] = None
        self.tk_image = None

        self.root = tk.Tk()
        self.root.title("NanoTrack Template")
        self.root.geometry("980x680")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        controls = tk.Frame(self.root)
        controls.pack(fill=tk.X, padx=8, pady=8)

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Idle")
        status_label = tk.Label(controls, textvariable=self.status_var, anchor="w")
        status_label.pack(side=tk.LEFT, padx=(0, 12))

        self.start_btn = tk.Button(controls, text="Start", command=self.start_tracking)
        self.start_btn.pack(side=tk.LEFT)

        self.stop_btn = tk.Button(controls, text="Stop", command=self.stop_tracking)
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.debug_var = tk.StringVar(value="debug: waiting for first frame")
        debug_label = tk.Label(controls, textvariable=self.debug_var, anchor="w")
        debug_label.pack(side=tk.LEFT, padx=(16, 0))

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self._log(
            f"GUI init serial={self.serial} interval={self.interval_s}s "
            f"backbone={self.backbone} neckhead={self.neckhead} adb_path={self.adb_path} "
            f"track_size=({self.track_width}x{self.track_height})"
        )

    def _log(self, msg: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{stamp}] {msg}")

    def _capture_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self.client.screenshot(prefer_png=True)
            except FileNotFoundError:
                msg = (
                    f"ADB executable not found: {self.adb_path}. "
                    "Please install adb or pass --adb-path <full_path_to_adb.exe>."
                )
                with self.lock:
                    self.state.error = msg
                    self.state.error_count += 1
                self._log(msg)
                time.sleep(1.0)
                continue
            except EmulatorError as exc:
                with self.lock:
                    self.state.error = str(exc)
                    self.state.error_count += 1
                self._log(f"capture error: {exc}")
                time.sleep(max(0.2, self.interval_s))
                continue
            except Exception as exc:
                with self.lock:
                    self.state.error = f"Unexpected capture error: {exc}"
                    self.state.error_count += 1
                self._log(f"unexpected capture error: {exc}")
                time.sleep(max(0.2, self.interval_s))
                continue

            predicted_bbox: Optional[BBox] = None
            norm_frame = to_tracker_input(frame, (self.track_width, self.track_height))
            self.last_norm_shape = norm_frame.shape
            if self.tracking_enabled and self.tracker is not None:
                try:
                    with self.tracker_lock:
                        ok, box = self.tracker.update(norm_frame)
                except cv2.error as exc:
                    msg = f"Tracker update failed: {exc}"
                    self._log(msg)
                    self.tracking_enabled = False
                    with self.tracker_lock:
                        self.tracker = None
                    self.status_var.set("Tracking stopped (tracker update failed)")
                    with self.lock:
                        self.state.error = msg
                        self.state.error_count += 1
                    ok = False
                    box = None
                if ok and box is not None:
                    x, y, w, h = [int(v) for v in box]
                    raw_bbox = map_bbox_between_sizes(
                        (x, y, w, h),
                        (self.track_width, self.track_height),
                        (frame.shape[1], frame.shape[0]),
                    )
                    predicted_bbox = clamp_bbox(raw_bbox, frame.shape[1], frame.shape[0])

            with self.lock:
                self.state.frame = frame
                self.state.predicted_bbox = predicted_bbox
                self.state.error = None
                self.state.capture_count += 1
                self.state.last_capture_ts = time.time()
                self.state.last_frame_shape = frame.shape

            now = time.time()
            if self._last_fps_t == 0.0:
                self._last_fps_t = now
                self._last_fps_n = 1
            else:
                self._last_fps_n += 1
                dt = now - self._last_fps_t
                if dt >= 1.0:
                    self.capture_fps = self._last_fps_n / dt
                    self._last_fps_t = now
                    self._last_fps_n = 0

            time.sleep(max(0.02, self.interval_s))

    def on_mouse_down(self, event) -> None:
        self.drag_start = (int(event.x), int(event.y))
        self.drag_current = self.drag_start

    def on_mouse_drag(self, event) -> None:
        if self.drag_start is None:
            return
        self.drag_current = (int(event.x), int(event.y))

    def on_mouse_up(self, event) -> None:
        if self.drag_start is None:
            return
        x1, y1 = self.drag_start
        x2, y2 = int(event.x), int(event.y)
        self.drag_start = None
        self.drag_current = None

        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        w, h = right - left, bottom - top
        if w < 2 or h < 2:
            self.status_var.set("Selection too small")
            return

        with self.lock:
            frame = self.state.frame
        if frame is None:
            self.status_var.set("No frame yet")
            return

        bbox = clamp_bbox((left, top, w, h), frame.shape[1], frame.shape[0])
        if bbox is None:
            self.status_var.set("Invalid selection")
            return
        self.init_bbox = bbox
        self.status_var.set(f"Init bbox set: {bbox}")

    def start_tracking(self) -> None:
        with self.lock:
            frame = None if self.state.frame is None else self.state.frame.copy()
        if frame is None:
            messagebox.showwarning("Start failed", "No frame available yet.")
            return
        if self.init_bbox is None:
            messagebox.showwarning("Start failed", "Please drag a rectangle first.")
            return
        try:
            tracker = make_nanotracker(self.backbone, self.neckhead)
            norm_frame = to_tracker_input(frame, (self.track_width, self.track_height))
            init_bbox_norm = map_bbox_between_sizes(
                self.init_bbox,
                (frame.shape[1], frame.shape[0]),
                (self.track_width, self.track_height),
            )
            init_bbox_norm = clamp_bbox(init_bbox_norm, self.track_width, self.track_height)
            if init_bbox_norm is None:
                messagebox.showwarning("Start failed", "Initial bbox becomes invalid after resizing.")
                return
            with self.tracker_lock:
                tracker.init(norm_frame, init_bbox_norm)
                # Quick sanity check: incompatible model pairs often fail on first update.
                tracker.update(norm_frame)
                self.tracker = tracker
        except Exception as exc:
            messagebox.showerror(
                "Tracker init failed",
                f"{exc}\n\nPossible cause: incompatible NanoTrack ONNX model pair for OpenCV TrackerNano.",
            )
            return
        self.tracking_enabled = True
        self.status_var.set("Tracking")

    def stop_tracking(self) -> None:
        self.tracking_enabled = False
        with self.tracker_lock:
            self.tracker = None
        with self.lock:
            self.state.predicted_bbox = None
        self.status_var.set("Stopped")

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        view = frame.copy()

        if self.init_bbox is not None and not self.tracking_enabled:
            x, y, w, h = self.init_bbox
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 220, 255), 2)

        with self.lock:
            predicted = self.state.predicted_bbox
            err = self.state.error
        if predicted is not None:
            x, y, w, h = predicted
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if err:
            cv2.putText(view, err[:90], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 70, 70), 2, cv2.LINE_AA)

        if self.drag_start and self.drag_current:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_current
            cv2.rectangle(view, (x1, y1), (x2, y2), (255, 160, 0), 2)

        return view

    def refresh_ui(self) -> None:
        with self.lock:
            frame = None if self.state.frame is None else self.state.frame.copy()
            err = self.state.error
            capture_count = self.state.capture_count
            error_count = self.state.error_count
            last_capture_ts = self.state.last_capture_ts
            last_frame_shape = self.state.last_frame_shape

        debug_text = (
            f"capture={capture_count} err={error_count} fps={self.capture_fps:.1f} "
            f"shape={last_frame_shape} norm=({self.track_width},{self.track_height}) last_ts={last_capture_ts:.2f}"
        )
        if err:
            debug_text += f" last_error={err[:80]}"
        self.debug_var.set(debug_text)

        if frame is not None:
            frame = self._draw_overlay(frame)
            image = Image.fromarray(frame)
            self.tk_image = ImageTk.PhotoImage(image=image)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        else:
            self.canvas.delete("all")
            self.canvas.create_text(
                16,
                20,
                anchor=tk.NW,
                text="Waiting for screenshot...\nCheck adb serial / emulator state.",
                fill="#dddddd",
                font=("Consolas", 13),
            )
            if err:
                self.canvas.create_text(
                    16,
                    70,
                    anchor=tk.NW,
                    text=f"Last error: {err}",
                    fill="#ff7a7a",
                    font=("Consolas", 11),
                )
        self.root.after(30, self.refresh_ui)

    def _on_close(self) -> None:
        self.stop_event.set()
        self.root.destroy()

    def run(self) -> None:
        self.worker.start()
        self.refresh_ui()
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    models = here / "models"
    default_backbone = models / "nanotrack_backbone_sim.onnx"
    default_head = models / "nanotrack_head_sim.onnx"
    if not default_backbone.exists():
        default_backbone = models / "nanotrack_backbone.onnx"
    if not default_head.exists():
        default_head = models / "nanotrack_head.onnx"
    parser = argparse.ArgumentParser(description="NanoTrack GUI template based on emulator screenshots.")
    parser.add_argument("--serial", default=None, help="ADB serial, e.g. 127.0.0.1:5555 (optional)")
    parser.add_argument("--interval", type=float, default=0.1, help="Screenshot interval seconds.")
    parser.add_argument("--backbone", default=str(default_backbone))
    parser.add_argument("--neckhead", default=str(default_head))
    parser.add_argument("--adb-path", default=None, help="Path to adb executable, e.g. C:\\Android\\platform-tools\\adb.exe")
    parser.add_argument("--track-width", type=int, default=1280, help="Tracker input width.")
    parser.add_argument("--track-height", type=int, default=720, help="Tracker input height.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.backbone).exists() or not Path(args.neckhead).exists():
        raise SystemExit(
            "NanoTrack model files not found.\n"
            f"Expected backbone: {args.backbone}\n"
            f"Expected neckhead: {args.neckhead}"
        )
    try:
        adb_path = resolve_adb_path(args.adb_path)
        serial = resolve_target_serial(serial=args.serial, adb_path=adb_path, auto_connect=True)
    except (AdbNotFoundError, AdbDeviceNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    app = NanoTrackGui(
        serial=serial,
        backbone=args.backbone,
        neckhead=args.neckhead,
        interval_s=args.interval,
        adb_path=adb_path,
        track_width=args.track_width,
        track_height=args.track_height,
    )
    app.run()


if __name__ == "__main__":
    main()
