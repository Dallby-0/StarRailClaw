from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sr_tools.dl_matcher import Detection, YoloEMatcher
from sr_tools import EmulatorClient
from sr_tools.adb import AdbDeviceNotFoundError, AdbNotFoundError, resolve_adb_path, resolve_target_serial
from sr_tools.emulator import EmulatorError


class ImagePanel:
    def __init__(self, parent: tk.Widget, title: str) -> None:
        frame = tk.Frame(parent)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        tk.Label(frame, text=title).pack(anchor="w")
        self.canvas = tk.Canvas(frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.image_rgb: Optional[np.ndarray] = None
        self.tk_image = None
        self.render_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

    def fit_size(self, src_w: int, src_h: int, dst_w: int, dst_h: int) -> Tuple[int, int, float]:
        scale = min(dst_w / max(1, src_w), dst_h / max(1, src_h))
        out_w = max(1, int(round(src_w * scale)))
        out_h = max(1, int(round(src_h * scale)))
        return out_w, out_h, scale

    def canvas_to_image(self, canvas_x: int, canvas_y: int) -> Optional[Tuple[int, int]]:
        if self.image_rgb is None:
            return None
        x = int(round((canvas_x - self.offset_x) / max(self.render_scale, 1e-8)))
        y = int(round((canvas_y - self.offset_y) / max(self.render_scale, 1e-8)))
        h, w = self.image_rgb.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return x, y

    def show(self, image_rgb: Optional[np.ndarray]) -> None:
        self.canvas.delete("all")
        if image_rgb is None:
            self.canvas.create_text(
                16,
                16,
                anchor=tk.NW,
                text="No image loaded",
                fill="#dddddd",
                font=("Consolas", 12),
            )
            return
        self.image_rgb = image_rgb
        src_h, src_w = image_rgb.shape[:2]
        dst_w = max(1, self.canvas.winfo_width())
        dst_h = max(1, self.canvas.winfo_height())
        out_w, out_h, scale = self.fit_size(src_w, src_h, dst_w, dst_h)
        self.render_scale = scale
        self.offset_x = (dst_w - out_w) // 2
        self.offset_y = (dst_h - out_h) // 2

        view = Image.fromarray(image_rgb).resize((out_w, out_h), Image.Resampling.BILINEAR)
        self.tk_image = ImageTk.PhotoImage(image=view)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)


class YoloEVisualPromptGui:
    def __init__(
        self,
        model: str,
        conf: float,
        iou: float,
        device: Optional[str],
        serial: str,
        adb_path: str,
    ) -> None:
        self.matcher = YoloEMatcher(model_path=model, conf=conf, iou=iou, device=device)
        self.serial = serial
        self.adb_path_input = adb_path
        self.client = EmulatorClient(serial=serial, adb_path=adb_path)

        self.target_path: Optional[Path] = None
        self.ref_path: Optional[Path] = None
        self.target_rgb: Optional[np.ndarray] = None
        self.ref_rgb: Optional[np.ndarray] = None
        self.result_rgb: Optional[np.ndarray] = None
        self.detections: List[Detection] = []

        self.ref_drag_start: Optional[Tuple[int, int]] = None
        self.ref_drag_current: Optional[Tuple[int, int]] = None
        self.ref_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None

        self.root = tk.Tk()
        self.root.title("YOLOE Visual Prompt GUI")
        self.root.geometry("1400x840")

        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=8, pady=8)

        tk.Button(top, text="Load Target", command=self.load_target).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Load Ref", command=self.load_ref).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Target From Emulator", command=self.load_target_from_emulator).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Ref From Emulator", command=self.load_ref_from_emulator).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Run dl_matcher", command=self.run_match).pack(side=tk.LEFT, padx=8)
        tk.Button(top, text="Save Result", command=self.save_result).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Save Selected PNG", command=self.save_selected_png).pack(side=tk.LEFT, padx=4)

        self.save_path_var = tk.StringVar(value=str((PROJECT_ROOT / "debug" / "selected_bbox.png").resolve()))
        tk.Entry(top, textvariable=self.save_path_var, width=48).pack(side=tk.LEFT, padx=6)

        self.status_var = tk.StringVar(value="Load target/ref, then draw bbox on ref image.")
        tk.Label(top, textvariable=self.status_var, anchor="w").pack(side=tk.LEFT, padx=10)

        body = tk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True)

        self.target_panel = ImagePanel(body, "Target")
        self.ref_panel = ImagePanel(body, "Ref (drag to select bbox)")

        self.ref_panel.canvas.bind("<ButtonPress-1>", self.on_ref_mouse_down)
        self.ref_panel.canvas.bind("<B1-Motion>", self.on_ref_mouse_drag)
        self.ref_panel.canvas.bind("<ButtonRelease-1>", self.on_ref_mouse_up)

        self.root.after(30, self.refresh_view)

    @staticmethod
    def _map_point_1280x720_to_1000x1000(x: int, y: int) -> Tuple[int, int]:
        lx = int(round(int(x) * 1000 / 1280))
        ly = int(round(int(y) * 1000 / 720))
        lx = max(0, min(1000, lx))
        ly = max(0, min(1000, ly))
        return lx, ly

    def _bbox_to_logical_1000(self, bbox_xyxy: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox_xyxy
        lx1, ly1 = self._map_point_1280x720_to_1000x1000(x1, y1)
        lx2, ly2 = self._map_point_1280x720_to_1000x1000(x2, y2)
        return lx1, ly1, lx2, ly2

    def _show_error(self, title: str, exc: Exception) -> None:
        tb = traceback.format_exc()
        print(f"[{title}] {type(exc).__name__}: {exc}")
        print(tb)
        tail = "\n".join(tb.strip().splitlines()[-12:])
        messagebox.showerror(title, f"{type(exc).__name__}: {exc}\n\n{tail}")

    def _load_rgb(self, path: Path) -> np.ndarray:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def load_target(self) -> None:
        path = filedialog.askopenfilename(
            title="Select target image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        p = Path(path)
        self.target_rgb = self._load_rgb(p)
        self.target_path = p
        self.result_rgb = None
        self.detections = []
        self.status_var.set(f"Target loaded: {p.name}")

    def load_ref(self) -> None:
        path = filedialog.askopenfilename(
            title="Select reference image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        p = Path(path)
        self.ref_rgb = self._load_rgb(p)
        self.ref_path = p
        self.ref_bbox_xyxy = None
        self.ref_drag_start = None
        self.ref_drag_current = None
        self.status_var.set(f"Ref loaded: {p.name}. Draw bbox on ref image.")

    def load_target_from_emulator(self) -> None:
        try:
            self.target_rgb = self.client.screenshot(prefer_png=True)
        except (EmulatorError, FileNotFoundError) as exc:
            messagebox.showerror("Emulator", str(exc))
            return
        self.target_path = None
        self.result_rgb = None
        self.detections = []
        self.status_var.set("Target loaded from emulator screenshot.")

    def load_ref_from_emulator(self) -> None:
        try:
            self.ref_rgb = self.client.screenshot(prefer_png=True)
        except (EmulatorError, FileNotFoundError) as exc:
            messagebox.showerror("Emulator", str(exc))
            return
        self.ref_path = None
        self.ref_bbox_xyxy = None
        self.ref_drag_start = None
        self.ref_drag_current = None
        self.status_var.set("Ref loaded from emulator screenshot. Draw bbox on ref image.")

    def on_ref_mouse_down(self, event) -> None:
        p = self.ref_panel.canvas_to_image(int(event.x), int(event.y))
        if p is None:
            return
        self.ref_drag_start = p
        self.ref_drag_current = p

    def on_ref_mouse_drag(self, event) -> None:
        if self.ref_drag_start is None:
            return
        p = self.ref_panel.canvas_to_image(int(event.x), int(event.y))
        if p is None:
            return
        self.ref_drag_current = p

    def on_ref_mouse_up(self, event) -> None:
        if self.ref_drag_start is None:
            return
        p = self.ref_panel.canvas_to_image(int(event.x), int(event.y))
        self.ref_drag_current = None
        if p is None:
            self.ref_drag_start = None
            return
        x1, y1 = self.ref_drag_start
        x2, y2 = p
        self.ref_drag_start = None
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right - left < 2 or bottom - top < 2:
            self.ref_bbox_xyxy = None
            self.status_var.set("Ref bbox too small, draw again.")
            return
        self.ref_bbox_xyxy = (left, top, right, bottom)
        logical_bbox = self._bbox_to_logical_1000(self.ref_bbox_xyxy)
        self.status_var.set(f"Ref bbox set real={self.ref_bbox_xyxy} logical1000={logical_bbox}")

    def run_match(self) -> None:
        if self.target_rgb is None:
            messagebox.showwarning("Run", "Please load target image first.")
            return
        if self.ref_rgb is None:
            messagebox.showwarning("Run", "Please load ref image first.")
            return
        if self.ref_bbox_xyxy is None:
            messagebox.showwarning("Run", "Please draw bbox on ref image first.")
            return
        print("[Run] target_type=", type(self.target_rgb), "target_shape=", getattr(self.target_rgb, "shape", None))
        print("[Run] ref_type=", type(self.ref_rgb), "ref_shape=", getattr(self.ref_rgb, "shape", None))
        print("[Run] bbox=", self.ref_bbox_xyxy)
        print("[Run] bbox_logical_1000=", self._bbox_to_logical_1000(self.ref_bbox_xyxy))
        try:
            dets = self.matcher.detect_with_visual_prompt(
                image_rgb=self.target_rgb,
                refer_image_rgb=self.ref_rgb,
                refer_bbox_xyxy=self.ref_bbox_xyxy,
            )
        except Exception as exc:
            self._show_error("Run failed", exc)
            return
        self.detections = dets
        self.result_rgb = self.matcher.draw(self.target_rgb, dets)
        self.status_var.set(f"Done. detections={len(dets)}")

    def save_result(self) -> None:
        if self.result_rgb is None:
            messagebox.showwarning("Save", "No result image yet. Run first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save result image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("All files", "*.*")],
        )
        if not path:
            return
        out_bgr = cv2.cvtColor(self.result_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, out_bgr)
        self.status_var.set(f"Result saved: {Path(path).name}")

    def save_selected_png(self) -> None:
        if self.ref_rgb is None:
            messagebox.showwarning("Save Selected", "Please load ref image first.")
            return
        if self.ref_bbox_xyxy is None:
            messagebox.showwarning("Save Selected", "Please draw bbox on ref image first.")
            return
        raw_path = self.save_path_var.get().strip()
        if not raw_path:
            messagebox.showwarning("Save Selected", "Please input output path.")
            return
        out_path = Path(raw_path)
        if out_path.suffix.lower() != ".png":
            out_path = out_path.with_suffix(".png")
        x1, y1, x2, y2 = self.ref_bbox_xyxy
        crop = self.ref_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            messagebox.showwarning("Save Selected", "Selected bbox is empty.")
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        if not ok:
            messagebox.showerror("Save Selected", f"Failed to save: {out_path}")
            return
        logical_bbox = self._bbox_to_logical_1000(self.ref_bbox_xyxy)
        self.status_var.set(f"Saved selected png: {out_path} logical1000={logical_bbox}")

    def _overlay_ref(self) -> Optional[np.ndarray]:
        if self.ref_rgb is None:
            return None
        view = self.ref_rgb.copy()
        if self.ref_bbox_xyxy is not None:
            x1, y1, x2, y2 = self.ref_bbox_xyxy
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if self.ref_drag_start and self.ref_drag_current:
            x1, y1 = self.ref_drag_start
            x2, y2 = self.ref_drag_current
            cv2.rectangle(view, (x1, y1), (x2, y2), (255, 200, 0), 2)
        return view

    def refresh_view(self) -> None:
        target_view = self.result_rgb if self.result_rgb is not None else self.target_rgb
        self.target_panel.show(target_view)
        self.ref_panel.show(self._overlay_ref())
        self.root.after(30, self.refresh_view)

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOE visual-prompt test GUI")
    parser.add_argument("--model", default="yoloe-11s-seg.pt", help="YOLOE model path or name.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu / 0.")
    parser.add_argument("--serial", default=None, help="ADB serial, e.g. 127.0.0.1:5555")
    parser.add_argument("--adb-path", default=None, help="Path to adb executable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        adb_path = resolve_adb_path(args.adb_path)
        serial = resolve_target_serial(serial=args.serial, adb_path=adb_path, auto_connect=True)
    except (AdbNotFoundError, AdbDeviceNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    app = YoloEVisualPromptGui(
        model=args.model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        serial=serial,
        adb_path=adb_path,
    )
    app.run()


if __name__ == "__main__":
    main()
