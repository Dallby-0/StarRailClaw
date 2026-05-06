from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from sr_tools import EmulatorClient
from sr_tools.adb import AdbDeviceNotFoundError, AdbNotFoundError, resolve_adb_path, resolve_target_serial
from sr_tools.emulator import EmulatorError


class OcrTestGui:
    def __init__(self, serial: str, adb_path: str) -> None:
        self.client = EmulatorClient(serial=serial, adb_path=adb_path)
        self.current_frame: Optional[np.ndarray] = None
        self.mapper = CoordinateMapper(real_w=1280, real_h=720)
        self.vision = VisionEngine(self.mapper, log_fn=self._on_ocr_log)

        self.drag_start: Optional[tuple[int, int]] = None
        self.drag_current: Optional[tuple[int, int]] = None
        self.selection_real: Optional[tuple[int, int, int, int]] = None
        self.selection_logical: Optional[list[int]] = None
        self.ocr_words: list[dict] = []

        self._render_scale = 1.0
        self._render_offset_x = 0
        self._render_offset_y = 0
        self.tk_image = None

        self.root = tk.Tk()
        self.root.title("OCR Test GUI")
        self.root.geometry("1240x780")

        container = tk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(container, bg="black", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 4), pady=8)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        side = tk.Frame(container, width=360)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 8), pady=8)
        side.pack_propagate(False)

        self.status_var = tk.StringVar(value="Ready")
        self.threshold_var = tk.StringVar(value="0.8")
        self.white_text_var = tk.BooleanVar(value=False)
        self.selection_var = tk.StringVar(value="区域: -")

        tk.Label(side, textvariable=self.status_var, anchor="w", justify=tk.LEFT, wraplength=340).pack(fill=tk.X, pady=(0, 6))
        tk.Button(side, text="刷新截图", command=self.refresh_screenshot).pack(fill=tk.X, pady=2)
        tk.Button(side, text="执行OCR", command=self.run_ocr).pack(fill=tk.X, pady=2)
        tk.Checkbutton(side, text="白字模式(text_match_white)", variable=self.white_text_var).pack(anchor="w", pady=(4, 2))

        controls = tk.Frame(side)
        controls.pack(fill=tk.X, pady=(2, 8))
        tk.Label(controls, text="置信度阈值").grid(row=0, column=0, sticky="w")
        tk.Entry(controls, textvariable=self.threshold_var, width=8).grid(row=0, column=1, sticky="w", padx=(8, 0))

        tk.Label(side, textvariable=self.selection_var, anchor="w", justify=tk.LEFT, wraplength=340).pack(fill=tk.X, pady=(0, 8))
        tk.Label(side, text="OCR结果").pack(anchor="w")

        self.result_text = tk.Text(side, wrap="word", height=28)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    def _on_ocr_log(self, msg: str) -> None:
        self.result_text.insert(tk.END, f"{msg}\n")
        self.result_text.see(tk.END)

    def _fit_size(self, src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int, float]:
        scale = min(dst_w / max(1, src_w), dst_h / max(1, src_h))
        out_w = max(1, int(round(src_w * scale)))
        out_h = max(1, int(round(src_h * scale)))
        return out_w, out_h, scale

    def _canvas_to_image(self, canvas_x: int, canvas_y: int) -> Optional[tuple[int, int]]:
        if self.current_frame is None:
            return None
        x = int(round((canvas_x - self._render_offset_x) / max(self._render_scale, 1e-8)))
        y = int(round((canvas_y - self._render_offset_y) / max(self._render_scale, 1e-8)))
        h, w = self.current_frame.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return x, y

    def _on_mouse_down(self, event) -> None:
        p = self._canvas_to_image(int(event.x), int(event.y))
        if p is None:
            return
        self.drag_start = p
        self.drag_current = p

    def _on_mouse_drag(self, event) -> None:
        if self.drag_start is None:
            return
        p = self._canvas_to_image(int(event.x), int(event.y))
        if p is None:
            return
        self.drag_current = p
        self._refresh_canvas()

    def _on_mouse_up(self, event) -> None:
        if self.drag_start is None:
            return
        p = self._canvas_to_image(int(event.x), int(event.y))
        if p is None:
            self.drag_start = None
            self.drag_current = None
            return
        x1, y1 = self.drag_start
        x2, y2 = p
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right - left < 2 or bottom - top < 2:
            self.selection_real = None
            self.selection_logical = None
            self.selection_var.set("区域: 框选太小，请重试")
            self.drag_start = None
            self.drag_current = None
            self._refresh_canvas()
            return
        self.selection_real = (left, top, right, bottom)
        lx1, ly1 = self.mapper.point_to_logical(left, top)
        lx2, ly2 = self.mapper.point_to_logical(right, bottom)
        self.selection_logical = [lx1, ly1, lx2, ly2]
        self.selection_var.set(
            f"区域 logical={self.selection_logical} real={[left, top, right, bottom]}"
        )
        self.drag_start = None
        self.drag_current = None
        self._refresh_canvas()

    def refresh_screenshot(self) -> None:
        try:
            frame = self.client.screenshot(prefer_png=True)
        except (EmulatorError, FileNotFoundError) as exc:
            messagebox.showerror("截图失败", str(exc))
            return
        h, w = frame.shape[:2]
        self.mapper = CoordinateMapper(real_w=w, real_h=h)
        self.vision = VisionEngine(self.mapper, log_fn=self._on_ocr_log)
        self.current_frame = frame
        self.selection_real = None
        self.selection_logical = None
        self.ocr_words.clear()
        self.selection_var.set("区域: -")
        self.result_text.delete("1.0", tk.END)
        self.status_var.set(f"截图成功: {w}x{h}")
        self._refresh_canvas()

    def run_ocr(self) -> None:
        if self.current_frame is None:
            messagebox.showwarning("OCR", "请先刷新截图")
            return
        if self.selection_logical is None:
            messagebox.showwarning("OCR", "请先框选区域")
            return
        try:
            threshold = float(self.threshold_var.get().strip())
        except ValueError:
            messagebox.showwarning("OCR", "阈值必须是数字")
            return
        self.result_text.delete("1.0", tk.END)
        words = self.vision.ocr(self.current_frame, self.selection_logical, white_text=bool(self.white_text_var.get()))
        self.ocr_words = words
        self.result_text.insert(
            tk.END,
            f"threshold={threshold:g}, white_text={self.white_text_var.get()}, candidates={len(words)}\n",
        )
        kept = [w for w in words if float(w.get("conf", 0.0)) >= threshold]
        self.result_text.insert(tk.END, f"above_threshold={len(kept)}\n")
        for i, item in enumerate(words, start=1):
            text = str(item.get("text", "")).replace("\n", " ").strip()
            conf = float(item.get("conf", 0.0))
            bbox = item.get("bbox")
            self.result_text.insert(tk.END, f"{i:02d}. conf={conf:.3f} bbox={bbox} text={text!r}\n")
        self.status_var.set(f"OCR完成: candidates={len(words)} above_threshold={len(kept)}")
        self._refresh_canvas()

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        view = frame.copy()
        if self.selection_real is not None:
            x1, y1, x2, y2 = self.selection_real
            cv2.rectangle(view, (x1, y1), (x2, y2), (255, 180, 0), 2)

        if self.drag_start and self.drag_current:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_current
            cv2.rectangle(view, (x1, y1), (x2, y2), (255, 220, 0), 2)

        for w in self.ocr_words:
            x, y, bw, bh = [int(v) for v in w.get("bbox", (0, 0, 0, 0))]
            cv2.rectangle(view, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            label = f"{float(w.get('conf', 0.0)):.2f} {str(w.get('text', '')).strip()[:16]}"
            cv2.putText(view, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        return view

    def _refresh_canvas(self) -> None:
        self.canvas.delete("all")
        if self.current_frame is None:
            self.canvas.create_text(16, 16, anchor=tk.NW, text="请点击“刷新截图”获取工作区截图", fill="#dddddd", font=("Consolas", 12))
            return

        frame = self._draw_overlay(self.current_frame)
        src_h, src_w = frame.shape[:2]
        dst_w = max(1, self.canvas.winfo_width())
        dst_h = max(1, self.canvas.winfo_height())
        out_w, out_h, scale = self._fit_size(src_w, src_h, dst_w, dst_h)
        self._render_scale = scale
        self._render_offset_x = (dst_w - out_w) // 2
        self._render_offset_y = (dst_h - out_h) // 2

        image = Image.fromarray(frame).resize((out_w, out_h), Image.Resampling.BILINEAR)
        self.tk_image = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(self._render_offset_x, self._render_offset_y, anchor=tk.NW, image=self.tk_image)

    def run(self) -> None:
        self.root.after(30, self._refresh_canvas)
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR test GUI for behavior-tree OCR path.")
    parser.add_argument("--serial", default=None, help="ADB serial, e.g. 127.0.0.1:5555 (optional)")
    parser.add_argument("--adb-path", default=None, help="Path to adb executable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        adb_path = resolve_adb_path(args.adb_path)
        serial = resolve_target_serial(serial=args.serial, adb_path=adb_path, auto_connect=True)
    except (AdbNotFoundError, AdbDeviceNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    app = OcrTestGui(serial=serial, adb_path=adb_path)
    app.run()


if __name__ == "__main__":
    main()
