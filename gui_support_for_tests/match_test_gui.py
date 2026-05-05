from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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

from sr_tools import EmulatorClient
from sr_tools.adb import AdbDeviceNotFoundError, AdbNotFoundError, resolve_adb_path, resolve_target_serial
from sr_tools.emulator import EmulatorError
from sr_tools.vision import (
    MatchResult,
    match_akaze_features,
    match_surf_features,
    match_template_luma_mstpl,
    match_template_luma_color_fused,
    match_template_luma_multiscale,
)


@dataclass
class TemplateItem:
    name: str
    image: np.ndarray
    record_pos: tuple[float, float]
    resolution: tuple[int, int]


class MatchTestGui:
    def __init__(self, serial: str, adb_path: str) -> None:
        self.client = EmulatorClient(serial=serial, adb_path=adb_path)

        self.current_frame: Optional[np.ndarray] = None
        self.match_overlays: list[tuple[MatchResult, str]] = []
        self.templates: list[TemplateItem] = []

        self.drag_start: Optional[tuple[int, int]] = None
        self.drag_current: Optional[tuple[int, int]] = None
        self.selection: Optional[tuple[int, int, int, int]] = None

        self._render_scale = 1.0
        self._render_offset_x = 0
        self._render_offset_y = 0
        self.tk_image = None

        self.root = tk.Tk()
        self.root.title("Match Test GUI")
        self.root.geometry("1180x760")

        container = tk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(container, bg="black", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 4), pady=8)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        side = tk.Frame(container, width=320)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 8), pady=8)
        side.pack_propagate(False)

        self.status_var = tk.StringVar(value="Ready")
        self.ms_min_scale_var = tk.StringVar(value="0.85")
        self.ms_max_scale_var = tk.StringVar(value="1.15")
        self.ms_num_scales_var = tk.StringVar(value="11")
        self.mstpl_threshold_var = tk.StringVar(value="0.7")
        self.mstpl_scale_step_var = tk.StringVar(value="0.005")
        self.mstpl_scale_max_var = tk.StringVar(value="800")
        self.mstpl_resolution_var = tk.StringVar(value="1280x720")
        tk.Label(side, textvariable=self.status_var, anchor="w", justify=tk.LEFT, wraplength=300).pack(fill=tk.X, pady=(0, 8))

        tk.Button(side, text="刷新截图", command=self.refresh_screenshot).pack(fill=tk.X, pady=2)
        tk.Button(side, text="加入模板(框选)", command=self.add_selection_as_template).pack(fill=tk.X, pady=2)
        tk.Button(side, text="删除选中模板", command=self.delete_selected_template).pack(fill=tk.X, pady=2)
        tk.Button(side, text="识别", command=self.run_match).pack(fill=tk.X, pady=2)
        tk.Button(side, text="多尺度识别", command=self.run_match_multiscale).pack(fill=tk.X, pady=2)
        tk.Button(side, text="Airtest MSTPL识别", command=self.run_match_mstpl).pack(fill=tk.X, pady=2)
        tk.Button(side, text="AKAZE识别", command=self.run_match_akaze).pack(fill=tk.X, pady=2)
        tk.Button(side, text="SURF识别", command=self.run_match_surf).pack(fill=tk.X, pady=2)
        tk.Button(side, text="复位(清匹配框)", command=self.reset_overlays).pack(fill=tk.X, pady=(2, 8))

        tk.Label(side, text="多尺度参数").pack(anchor="w", pady=(6, 0))
        ms_grid = tk.Frame(side)
        ms_grid.pack(fill=tk.X, pady=(2, 8))
        tk.Label(ms_grid, text="min_scale").grid(row=0, column=0, sticky="w")
        tk.Entry(ms_grid, textvariable=self.ms_min_scale_var, width=10).grid(row=0, column=1, sticky="w", padx=(8, 0))
        tk.Label(ms_grid, text="max_scale").grid(row=1, column=0, sticky="w")
        tk.Entry(ms_grid, textvariable=self.ms_max_scale_var, width=10).grid(row=1, column=1, sticky="w", padx=(8, 0))
        tk.Label(ms_grid, text="num_scales").grid(row=2, column=0, sticky="w")
        tk.Entry(ms_grid, textvariable=self.ms_num_scales_var, width=10).grid(row=2, column=1, sticky="w", padx=(8, 0))

        tk.Label(side, text="模板列表").pack(anchor="w")
        self.template_list = tk.Listbox(side, height=20)
        self.template_list.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        tk.Label(side, text="MSTPL参数").pack(anchor="w", pady=(6, 0))
        ms_pre_grid = tk.Frame(side)
        ms_pre_grid.pack(fill=tk.X, pady=(2, 8))
        tk.Label(ms_pre_grid, text="threshold").grid(row=0, column=0, sticky="w")
        tk.Entry(ms_pre_grid, textvariable=self.mstpl_threshold_var, width=10).grid(row=0, column=1, sticky="w", padx=(8, 0))
        tk.Label(ms_pre_grid, text="scale_step").grid(row=1, column=0, sticky="w")
        tk.Entry(ms_pre_grid, textvariable=self.mstpl_scale_step_var, width=10).grid(row=1, column=1, sticky="w", padx=(8, 0))
        tk.Label(ms_pre_grid, text="scale_max").grid(row=2, column=0, sticky="w")
        tk.Entry(ms_pre_grid, textvariable=self.mstpl_scale_max_var, width=10).grid(row=2, column=1, sticky="w", padx=(8, 0))
        tk.Label(ms_pre_grid, text="resolution").grid(row=3, column=0, sticky="w")
        tk.Entry(ms_pre_grid, textvariable=self.mstpl_resolution_var, width=10).grid(row=3, column=1, sticky="w", padx=(8, 0))

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
        w = right - left
        h = bottom - top
        self.drag_start = None
        self.drag_current = None
        if w < 2 or h < 2:
            self.selection = None
            self.status_var.set("框选太小，请重新框选")
            self._refresh_canvas()
            return
        self.selection = (left, top, w, h)
        self.status_var.set(f"已框选: x={left}, y={top}, w={w}, h={h}")
        self._refresh_canvas()

    def refresh_screenshot(self) -> None:
        try:
            frame = self.client.screenshot(prefer_png=True)
        except (EmulatorError, FileNotFoundError) as exc:
            messagebox.showerror("截图失败", str(exc))
            return
        self.current_frame = frame
        self.selection = None
        self.match_overlays.clear()
        self.status_var.set(f"截图成功: {frame.shape[1]}x{frame.shape[0]}，匹配框已清空")
        self._refresh_canvas()

    def add_selection_as_template(self) -> None:
        if self.current_frame is None:
            messagebox.showwarning("加入模板", "请先刷新截图")
            return
        if self.selection is None:
            messagebox.showwarning("加入模板", "请先在截图上框选区域")
            return
        x, y, w, h = self.selection
        crop = self.current_frame[y : y + h, x : x + w].copy()
        fh, fw = self.current_frame.shape[:2]
        cx = x + w / 2.0
        cy = y + h / 2.0
        record_pos = ((cx - fw * 0.5) / fw, (cy - fh * 0.5) / fw)
        idx = len(self.templates) + 1
        name = f"tpl_{idx:03d}_{w}x{h}"
        self.templates.append(
            TemplateItem(
                name=name,
                image=crop,
                record_pos=(float(round(record_pos[0], 6)), float(round(record_pos[1], 6))),
                resolution=(int(fw), int(fh)),
            )
        )
        self.template_list.insert(tk.END, name)
        self.status_var.set(f"模板已加入: {name} record_pos={self.templates[-1].record_pos} resolution={fw}x{fh}")

    def delete_selected_template(self) -> None:
        sel = self.template_list.curselection()
        if not sel:
            return
        index = int(sel[0])
        if index < 0 or index >= len(self.templates):
            return
        removed_name = self.templates[index].name
        del self.templates[index]
        self.template_list.delete(index)
        self.status_var.set(f"已删除模板: {removed_name}")

    def run_match(self) -> None:
        if not self.templates:
            messagebox.showwarning("识别", "模板列表为空")
            return
        try:
            frame = self.client.screenshot(prefer_png=True)
        except (EmulatorError, FileNotFoundError) as exc:
            messagebox.showerror("识别失败", str(exc))
            return
        self.current_frame = frame
        self.selection = None
        self.match_overlays.clear()

        for tpl in self.templates:
            result = match_template_luma_color_fused(frame, tpl.image)
            self.match_overlays.append((result, tpl.name))

        self.status_var.set(f"识别完成: 共 {len(self.match_overlays)} 个模板")
        self._refresh_canvas()

    def run_match_akaze(self) -> None:
        if not self.templates:
            messagebox.showwarning("AKAZE识别", "模板列表为空")
            return
        try:
            frame = self.client.screenshot(prefer_png=True)
        except (EmulatorError, FileNotFoundError) as exc:
            messagebox.showerror("AKAZE识别失败", str(exc))
            return
        self.current_frame = frame
        self.selection = None
        self.match_overlays.clear()

        for tpl in self.templates:
            result = match_akaze_features(frame, tpl.image)
            self.match_overlays.append((result, tpl.name))

        self.status_var.set(f"AKAZE识别完成: 共 {len(self.match_overlays)} 个模板")
        self._refresh_canvas()

    def run_match_multiscale(self) -> None:
        if not self.templates:
            messagebox.showwarning("多尺度识别", "模板列表为空")
            return
        try:
            min_scale = float(self.ms_min_scale_var.get().strip())
            max_scale = float(self.ms_max_scale_var.get().strip())
            num_scales = int(self.ms_num_scales_var.get().strip())
        except ValueError:
            messagebox.showwarning("多尺度识别", "参数格式错误，请输入数字")
            return
        if min_scale <= 0 or max_scale <= 0:
            messagebox.showwarning("多尺度识别", "min_scale 和 max_scale 必须大于 0")
            return
        if num_scales < 1:
            messagebox.showwarning("多尺度识别", "num_scales 必须 >= 1")
            return

        try:
            frame = self.client.screenshot(prefer_png=True)
        except (EmulatorError, FileNotFoundError) as exc:
            messagebox.showerror("多尺度识别失败", str(exc))
            return
        self.current_frame = frame
        self.selection = None
        self.match_overlays.clear()

        for tpl in self.templates:
            result = match_template_luma_multiscale(
                frame,
                tpl.image,
                min_scale=min_scale,
                max_scale=max_scale,
                num_scales=num_scales,
            )
            self.match_overlays.append((result, tpl.name))

        self.status_var.set(
            f"多尺度识别完成: 共 {len(self.match_overlays)} 个模板 "
            f"(min={min_scale:g}, max={max_scale:g}, n={num_scales})"
        )
        self._refresh_canvas()

    def run_match_mstpl(self) -> None:
        if not self.templates:
            messagebox.showwarning("Airtest MSTPL识别", "模板列表为空")
            return
        try:
            threshold = float(self.mstpl_threshold_var.get().strip())
            scale_step = float(self.mstpl_scale_step_var.get().strip())
            scale_max = int(self.mstpl_scale_max_var.get().strip())
            rw, rh = self.mstpl_resolution_var.get().strip().lower().split("x")
            resolution = (int(rw), int(rh))
        except ValueError:
            messagebox.showwarning("Airtest MSTPL识别", "参数格式错误，请检查 threshold/scale_step/scale_max/resolution")
            return
        if scale_step <= 0:
            messagebox.showwarning("Airtest MSTPL识别", "scale_step 必须 > 0")
            return
        if resolution[0] <= 0 or resolution[1] <= 0:
            messagebox.showwarning("Airtest MSTPL识别", "resolution 必须为正整数，如 1280x720")
            return

        try:
            frame = self.client.screenshot(prefer_png=True)
        except (EmulatorError, FileNotFoundError) as exc:
            messagebox.showerror("Airtest MSTPL识别失败", str(exc))
            return
        self.current_frame = frame
        self.selection = None
        self.match_overlays.clear()

        for tpl in self.templates:
            result = match_template_luma_mstpl(
                frame,
                tpl.image,
                threshold=threshold,
                rgb=True,
                record_pos=tpl.record_pos,
                resolution=resolution,
                scale_max=scale_max,
                scale_step=scale_step,
            )
            self.match_overlays.append((result, tpl.name))

        self.status_var.set(
            f"Airtest MSTPL识别完成: 共 {len(self.match_overlays)} 个模板 "
            f"(threshold={threshold:g}, step={scale_step:g}, max={scale_max}, resolution={resolution[0]}x{resolution[1]})"
        )
        self._refresh_canvas()

    def run_match_surf(self) -> None:
        if not self.templates:
            messagebox.showwarning("SURF识别", "模板列表为空")
            return
        try:
            frame = self.client.screenshot(prefer_png=True)
        except (EmulatorError, FileNotFoundError) as exc:
            messagebox.showerror("SURF识别失败", str(exc))
            return
        self.current_frame = frame
        self.selection = None
        self.match_overlays.clear()

        for tpl in self.templates:
            result = match_surf_features(frame, tpl.image)
            self.match_overlays.append((result, tpl.name))

        self.status_var.set(f"SURF识别完成: 共 {len(self.match_overlays)} 个模板")
        self._refresh_canvas()

    def reset_overlays(self) -> None:
        self.match_overlays.clear()
        self.selection = None
        self.status_var.set("匹配框已清空")
        self._refresh_canvas()

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        view = frame.copy()

        if self.selection is not None:
            x, y, w, h = self.selection
            cv2.rectangle(view, (x, y), (x + w, y + h), (255, 180, 0), 2)

        if self.drag_start and self.drag_current:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_current
            cv2.rectangle(view, (x1, y1), (x2, y2), (255, 220, 0), 2)

        for result, name in self.match_overlays:
            tl = result.top_left
            br = result.bottom_right
            cv2.rectangle(view, tl, br, (0, 255, 0), 2)
            label = f"{name} {result.similarity:.3f}"
            cv2.putText(view, label, (tl[0], max(18, tl[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        return view

    def _refresh_canvas(self) -> None:
        self.canvas.delete("all")
        if self.current_frame is None:
            self.canvas.create_text(
                16,
                16,
                anchor=tk.NW,
                text="请点击“刷新截图”获取工作区截图",
                fill="#dddddd",
                font=("Consolas", 12),
            )
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
    parser = argparse.ArgumentParser(description="Match test GUI for screenshot-template matching.")
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
    app = MatchTestGui(serial=serial, adb_path=adb_path)
    app.run()


if __name__ == "__main__":
    main()
