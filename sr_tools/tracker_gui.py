from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
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
from sr_tools.emulator import EmulatorError


BBox = Tuple[int, int, int, int]


@dataclass
class SharedState:
    frame: Optional[np.ndarray] = None
    predicted_bbox: Optional[BBox] = None
    error: Optional[str] = None


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


class NanoTrackGui:
    def __init__(self, serial: str, backbone: str, neckhead: str, interval_s: float) -> None:
        self.serial = serial
        self.backbone = backbone
        self.neckhead = neckhead
        self.interval_s = interval_s

        self.client = EmulatorClient(serial=serial)
        self.state = SharedState()
        self.lock = threading.Lock()

        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._capture_loop, daemon=True)

        self.tracking_enabled = False
        self.tracker = None
        self.init_bbox: Optional[BBox] = None

        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_current: Optional[Tuple[int, int]] = None
        self.tk_image = None

        self.root = tk.Tk()
        self.root.title("NanoTrack Template")
        self.root.geometry("980x680")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        controls = tk.Frame(self.root)
        controls.pack(fill=tk.X, padx=8, pady=8)

        self.status_var = tk.StringVar(value="Idle")
        status_label = tk.Label(controls, textvariable=self.status_var, anchor="w")
        status_label.pack(side=tk.LEFT, padx=(0, 12))

        self.start_btn = tk.Button(controls, text="Start", command=self.start_tracking)
        self.start_btn.pack(side=tk.LEFT)

        self.stop_btn = tk.Button(controls, text="Stop", command=self.stop_tracking)
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def _capture_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self.client.screenshot(prefer_png=True)
            except EmulatorError as exc:
                with self.lock:
                    self.state.error = str(exc)
                time.sleep(max(0.2, self.interval_s))
                continue
            except Exception as exc:
                with self.lock:
                    self.state.error = f"Unexpected capture error: {exc}"
                time.sleep(max(0.2, self.interval_s))
                continue

            predicted_bbox: Optional[BBox] = None
            if self.tracking_enabled and self.tracker is not None:
                ok, box = self.tracker.update(frame)
                if ok:
                    x, y, w, h = [int(v) for v in box]
                    predicted_bbox = clamp_bbox((x, y, w, h), frame.shape[1], frame.shape[0])

            with self.lock:
                self.state.frame = frame
                self.state.predicted_bbox = predicted_bbox
                self.state.error = None

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
            self.tracker = make_nanotracker(self.backbone, self.neckhead)
            self.tracker.init(frame, self.init_bbox)
        except Exception as exc:
            messagebox.showerror("Tracker init failed", str(exc))
            return
        self.tracking_enabled = True
        self.status_var.set("Tracking")

    def stop_tracking(self) -> None:
        self.tracking_enabled = False
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
        if frame is not None:
            frame = self._draw_overlay(frame)
            image = Image.fromarray(frame)
            self.tk_image = ImageTk.PhotoImage(image=image)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            self.canvas.config(width=frame.shape[1], height=frame.shape[0])
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
    parser = argparse.ArgumentParser(description="NanoTrack GUI template based on emulator screenshots.")
    parser.add_argument("--serial", required=True, help="ADB serial, e.g. 127.0.0.1:5555")
    parser.add_argument("--interval", type=float, default=0.1, help="Screenshot interval seconds.")
    parser.add_argument("--backbone", default=str(models / "nanotrack_backbone_sim.onnx"))
    parser.add_argument("--neckhead", default=str(models / "nanotrack_head_sim.onnx"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.backbone).exists() or not Path(args.neckhead).exists():
        raise SystemExit(
            "NanoTrack model files not found.\n"
            f"Expected backbone: {args.backbone}\n"
            f"Expected neckhead: {args.neckhead}"
        )
    app = NanoTrackGui(
        serial=args.serial,
        backbone=args.backbone,
        neckhead=args.neckhead,
        interval_s=args.interval,
    )
    app.run()


if __name__ == "__main__":
    main()
