from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox

from sr_tools import EmulatorClient
from sr_tools.adb import AdbDeviceNotFoundError, AdbNotFoundError, resolve_adb_path, resolve_target_serial
from sr_tools.emulator import EmulatorError


Point = Tuple[int, int]


class EmulatorTestGui:
    def __init__(self, serial: str, interval_s: float, adb_path: str) -> None:
        self.client = EmulatorClient(serial=serial, adb_path=adb_path)
        self.interval_s = interval_s

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._capture_loop, daemon=True)

        self.latest_frame: Optional[np.ndarray] = None
        self.last_error: Optional[str] = None
        self.capture_count = 0

        self.last_clicked: Optional[Point] = None
        self.tap_point: Optional[Point] = None
        self.swipe_start: Optional[Point] = None
        self.swipe_end: Optional[Point] = None

        self._display_w = 1
        self._display_h = 1
        self._render_offset_x = 0
        self._render_offset_y = 0
        self._render_scale = 1.0
        self.tk_image = None

        self.root = tk.Tk()
        self.root.title("Emulator Test GUI")
        self.root.geometry("1060x760")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        panel = tk.Frame(self.root)
        panel.pack(fill=tk.X, padx=8, pady=(0, 8))

        self.status_var = tk.StringVar(value="waiting frame...")
        tk.Label(panel, textvariable=self.status_var, anchor="w").grid(row=0, column=0, columnspan=8, sticky="w", pady=(0, 4))

        self.point_var = tk.StringVar(value="last click: -")
        tk.Label(panel, textvariable=self.point_var, anchor="w").grid(row=1, column=0, columnspan=8, sticky="w", pady=(0, 6))

        tk.Button(panel, text="填入Tap", command=self._fill_tap).grid(row=2, column=0, padx=4, pady=2, sticky="ew")
        tk.Button(panel, text="填入Swipe起点", command=self._fill_swipe_start).grid(row=2, column=1, padx=4, pady=2, sticky="ew")
        tk.Button(panel, text="填入Swipe终点", command=self._fill_swipe_end).grid(row=2, column=2, padx=4, pady=2, sticky="ew")

        tk.Button(panel, text="执行Tap", command=self._run_tap).grid(row=3, column=0, padx=4, pady=2, sticky="ew")
        tk.Button(panel, text="执行Swipe", command=self._run_swipe).grid(row=3, column=1, padx=4, pady=2, sticky="ew")

        tk.Label(panel, text="Swipe毫秒").grid(row=3, column=2, padx=4, pady=2, sticky="e")
        self.duration_var = tk.StringVar(value="120")
        tk.Entry(panel, textvariable=self.duration_var, width=8).grid(row=3, column=3, padx=4, pady=2, sticky="w")

        self.tap_var = tk.StringVar(value="tap: -")
        self.swipe_var = tk.StringVar(value="swipe: - -> -")
        tk.Label(panel, textvariable=self.tap_var, anchor="w").grid(row=4, column=0, columnspan=8, sticky="w", pady=(6, 0))
        tk.Label(panel, textvariable=self.swipe_var, anchor="w").grid(row=5, column=0, columnspan=8, sticky="w")

        for col in range(8):
            panel.grid_columnconfigure(col, weight=1 if col < 4 else 0)

    def _capture_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self.client.screenshot(prefer_png=True)
                with self.lock:
                    self.latest_frame = frame
                    self.last_error = None
                    self.capture_count += 1
            except (EmulatorError, FileNotFoundError) as exc:
                with self.lock:
                    self.last_error = str(exc)
            except Exception as exc:
                with self.lock:
                    self.last_error = f"unexpected: {exc}"
            time.sleep(max(0.03, self.interval_s))

    def _on_canvas_click(self, event) -> None:
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame
        if frame is None:
            return
        x = int(round((event.x - self._render_offset_x) / max(self._render_scale, 1e-8)))
        y = int(round((event.y - self._render_offset_y) / max(self._render_scale, 1e-8)))
        h, w = frame.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return
        self.last_clicked = (x, y)
        self._refresh_point_labels()
        self.status_var.set(f"captured point ({x}, {y})")

    def _fill_tap(self) -> None:
        if self.last_clicked is None:
            self.status_var.set("no captured point")
            return
        self.tap_point = self.last_clicked
        self._refresh_point_labels()

    def _fill_swipe_start(self) -> None:
        if self.last_clicked is None:
            self.status_var.set("no captured point")
            return
        self.swipe_start = self.last_clicked
        self._refresh_point_labels()

    def _fill_swipe_end(self) -> None:
        if self.last_clicked is None:
            self.status_var.set("no captured point")
            return
        self.swipe_end = self.last_clicked
        self._refresh_point_labels()

    def _run_tap(self) -> None:
        if self.tap_point is None:
            messagebox.showwarning("Tap", "请先填入 tap 坐标")
            return
        try:
            self.client.tap(self.tap_point[0], self.tap_point[1])
            self.status_var.set(f"tap sent: {self.tap_point}")
        except Exception as exc:
            messagebox.showerror("Tap failed", str(exc))

    def _run_swipe(self) -> None:
        if self.swipe_start is None or self.swipe_end is None:
            messagebox.showwarning("Swipe", "请先填入 swipe 起点和终点")
            return
        try:
            duration_ms = int(self.duration_var.get().strip())
            if duration_ms < 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Swipe", "毫秒必须是非负整数")
            return
        try:
            self.client.swipe(
                self.swipe_start[0],
                self.swipe_start[1],
                self.swipe_end[0],
                self.swipe_end[1],
                duration_ms=duration_ms,
            )
            self.status_var.set(f"swipe sent: {self.swipe_start} -> {self.swipe_end}, {duration_ms}ms")
        except Exception as exc:
            messagebox.showerror("Swipe failed", str(exc))

    def _refresh_point_labels(self) -> None:
        self.point_var.set(f"last click: {self.last_clicked if self.last_clicked else '-'}")
        self.tap_var.set(f"tap: {self.tap_point if self.tap_point else '-'}")
        start = self.swipe_start if self.swipe_start else "-"
        end = self.swipe_end if self.swipe_end else "-"
        self.swipe_var.set(f"swipe: {start} -> {end}")

    def _draw_points(self, frame: np.ndarray) -> np.ndarray:
        image = frame.copy()

        def draw_cross(point: Optional[Point], color: Tuple[int, int, int], label: str) -> None:
            if point is None:
                return
            x, y = point
            cv = int(8)
            import cv2

            cv2.line(image, (x - cv, y), (x + cv, y), color, 2)
            cv2.line(image, (x, y - cv), (x, y + cv), color, 2)
            cv2.putText(image, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        draw_cross(self.last_clicked, (255, 255, 0), "P")
        draw_cross(self.tap_point, (0, 255, 0), "T")
        draw_cross(self.swipe_start, (255, 165, 0), "S")
        draw_cross(self.swipe_end, (255, 80, 80), "E")
        return image

    def _fit_size(self, src_w: int, src_h: int, dst_w: int, dst_h: int) -> Tuple[int, int, float]:
        scale = min(dst_w / max(1, src_w), dst_h / max(1, src_h))
        out_w = max(1, int(round(src_w * scale)))
        out_h = max(1, int(round(src_h * scale)))
        return out_w, out_h, scale

    def _refresh_ui(self) -> None:
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            err = self.last_error
            count = self.capture_count

        if frame is None:
            self.canvas.delete("all")
            text = "Waiting screenshot..."
            if err:
                text += f"\nLast error: {err}"
            self.canvas.create_text(16, 16, anchor=tk.NW, text=text, fill="#dddddd", font=("Consolas", 12))
            self.root.after(30, self._refresh_ui)
            return

        frame = self._draw_points(frame)
        src_h, src_w = frame.shape[:2]
        dst_w = max(1, self.canvas.winfo_width())
        dst_h = max(1, self.canvas.winfo_height())
        out_w, out_h, scale = self._fit_size(src_w, src_h, dst_w, dst_h)
        self._render_scale = scale
        self._render_offset_x = (dst_w - out_w) // 2
        self._render_offset_y = (dst_h - out_h) // 2
        self._display_w, self._display_h = out_w, out_h

        image = Image.fromarray(frame).resize((out_w, out_h), Image.Resampling.BILINEAR)
        self.tk_image = ImageTk.PhotoImage(image=image)
        self.canvas.delete("all")
        self.canvas.create_image(self._render_offset_x, self._render_offset_y, anchor=tk.NW, image=self.tk_image)

        if err:
            self.canvas.create_text(10, 10, anchor=tk.NW, text=f"ERR: {err}", fill="#ff8080", font=("Consolas", 11))
        self.status_var.set(f"capture={count} frame={src_w}x{src_h}")
        self.root.after(30, self._refresh_ui)

    def _on_close(self) -> None:
        self.stop_event.set()
        self.root.destroy()

    def run(self) -> None:
        self.worker.start()
        self._refresh_ui()
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emulator tap/swipe interactive test GUI")
    parser.add_argument("--serial", default=None, help="ADB serial, e.g. 127.0.0.1:5555 (optional)")
    parser.add_argument("--interval", type=float, default=0.1, help="Screenshot interval seconds")
    parser.add_argument("--adb-path", default=None, help="Path to adb executable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        adb_path = resolve_adb_path(args.adb_path)
        serial = resolve_target_serial(serial=args.serial, adb_path=adb_path, auto_connect=True)
    except (AdbNotFoundError, AdbDeviceNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    app = EmulatorTestGui(serial=serial, interval_s=args.interval, adb_path=adb_path)
    app.run()


if __name__ == "__main__":
    main()
