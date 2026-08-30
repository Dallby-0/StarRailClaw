from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

import cv2
import numpy as np

from .coord_mapper import CoordinateMapper
from sr_tools import ocr as sr_ocr
from sr_tools import vision as sr_vision


@dataclass
class VisionEngine:
    mapper: CoordinateMapper
    log_fn: Callable[[str], None] = print
    _ocr_backend_logged: bool = False
    log_ocr_calls: bool = True
    ocr_call_count: int = 0
    ocr_success_count: int = 0
    ocr_error_count: int = 0
    ocr_elapsed_s: float = 0.0

    def crop_rect(self, frame_rgb: np.ndarray, rect: list[int] | tuple[int, int, int, int] | None) -> np.ndarray:
        if rect is None:
            return frame_rgb
        x, y, w, h = self.mapper.rect_to_real(rect)
        return frame_rgb[y : y + h, x : x + w]

    def is_stable(self, prev_rgb: np.ndarray | None, curr_rgb: np.ndarray, rect: list[int] | None, diff_threshold: float = 0.01) -> bool:
        if prev_rgb is None:
            return False
        prev = self.crop_rect(prev_rgb, rect)
        curr = self.crop_rect(curr_rgb, rect)
        if prev.shape != curr.shape:
            return False
        diff = cv2.absdiff(prev, curr)
        mean = float(diff.mean()) / 255.0
        return mean <= float(diff_threshold)

    def match_template(
        self,
        frame_rgb: np.ndarray,
        template_path: Path,
        rect: list[int] | None,
        threshold: float = 0.8,
    ) -> tuple[bool, tuple[int, int] | None, float]:
        if not template_path.exists():
            return False, None, 0.0
        crop = self.crop_rect(frame_rgb, rect)
        tpl_bgr = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if tpl_bgr is None:
            return False, None, 0.0
        tpl = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2RGB)
        if crop.shape[0] < tpl.shape[0] or crop.shape[1] < tpl.shape[1]:
            return False, None, 0.0
        result = sr_vision.match_template_luma(crop, tpl)
        max_val = float(result.similarity)
        max_loc = result.top_left
        if max_val < threshold:
            return False, None, max_val
        ox, oy = 0, 0
        if rect is not None:
            ox, oy, _, _ = self.mapper.rect_to_real(rect)
        cx = ox + max_loc[0] + tpl.shape[1] // 2
        cy = oy + max_loc[1] + tpl.shape[0] // 2
        return True, (cx, cy), max_val

    def ocr(self, frame_rgb: np.ndarray, rect: list[int] | None, white_text: bool = False) -> list[dict[str, Any]]:
        real_rect = self._real_area(frame_rgb, rect)
        if self.log_ocr_calls and not self._ocr_backend_logged:
            self.log_fn("[vision][ocr] backend=sr_tools.ocr(pponnxcr)")
            self._ocr_backend_logged = True
        if self.log_ocr_calls:
            self.log_fn(f"[vision][ocr] call mode=single_line white_text={white_text} logical_rect={rect} real_rect={real_rect}")
        self.ocr_call_count += 1
        started = time.perf_counter()
        text = ""
        try:
            if white_text:
                text = sr_ocr.ocr_text_white(frame_rgb, real_rect, lang="zhs").strip()
            else:
                text = sr_ocr.ocr_text(frame_rgb, real_rect, lang="zhs").strip()
                if not text:
                    text = sr_ocr.ocr_text_white(frame_rgb, real_rect, lang="zhs").strip()
        except Exception as exc:  # noqa: BLE001
            self.ocr_error_count += 1
            self.ocr_elapsed_s += time.perf_counter() - started
            self.log_fn(
                f"[vision][ocr] error type={type(exc).__name__} message={exc} logical_rect={rect} real_rect={real_rect}"
            )
            return []
        entries: list[dict[str, Any]] = []
        if text:
            entries = [
                {
                    "text": text,
                    "conf": 1.0,
                    "bbox": (0, 0, max(1, real_rect[2] - real_rect[0]), max(1, real_rect[3] - real_rect[1])),
                    "center": ((real_rect[2] - real_rect[0]) // 2, (real_rect[3] - real_rect[1]) // 2),
                }
            ]
        out: list[dict[str, Any]] = []
        ox, oy, _, _ = real_rect
        for e in entries:
            bx, by, bw, bh = e.get("bbox", (0, 0, 0, 0))
            cx, cy = e.get("center", (bx + bw // 2, by + bh // 2))
            out.append(
                {
                    "text": str(e.get("text", "")),
                    "conf": float(e.get("conf", 0.0)),
                    "bbox": (int(bx) + ox, int(by) + oy, int(bw), int(bh)),
                    "center": (int(cx) + ox, int(cy) + oy),
                }
            )
        self.ocr_elapsed_s += time.perf_counter() - started
        if out:
            self.ocr_success_count += 1
        if self.log_ocr_calls:
            self.log_fn(f"[vision][ocr] done candidates={len(out)} logical_rect={rect}")
        return out

    def ocr_blocks(self, frame_rgb: np.ndarray, rect: list[int] | None, white_text: bool = False) -> list[dict[str, Any]]:
        """Detect individual OCR lines inside a region.

        ``ocr`` intentionally remains the cheap single-line recognizer used by
        runtime match conditions.  State construction needs the detector once
        so an LLM bbox that accidentally spans two lines can be normalized
        into matchable single-line elements before it is persisted.
        """
        real_rect = self._real_area(frame_rgb, rect)
        self.ocr_call_count += 1
        started = time.perf_counter()
        try:
            entries = sr_ocr.ocr_entries(frame_rgb, real_rect, lang="zhs", white=white_text)
            if not entries and not white_text:
                entries = sr_ocr.ocr_entries(frame_rgb, real_rect, lang="zhs", white=True)
        except Exception as exc:  # noqa: BLE001
            self.ocr_error_count += 1
            self.ocr_elapsed_s += time.perf_counter() - started
            self.log_fn(f"[vision][ocr-blocks] error type={type(exc).__name__} message={exc} logical_rect={rect}")
            return []
        ox, oy, _, _ = real_rect
        out: list[dict[str, Any]] = []
        for entry in entries:
            bx, by, bw, bh = entry.get("bbox", (0, 0, 0, 0))
            if int(bw) <= 0 or int(bh) <= 0:
                continue
            out.append({
                "text": str(entry.get("text", "")).strip(),
                "conf": float(entry.get("conf", 0.0) or 0.0),
                "bbox": (int(bx) + ox, int(by) + oy, int(bw), int(bh)),
            })
        self.ocr_elapsed_s += time.perf_counter() - started
        if out:
            self.ocr_success_count += 1
        return out

    def reset_ocr_stats(self) -> None:
        self.ocr_call_count = 0
        self.ocr_success_count = 0
        self.ocr_error_count = 0
        self.ocr_elapsed_s = 0.0

    def consume_ocr_stats(self) -> dict[str, int | float]:
        stats = {
            "calls": self.ocr_call_count,
            "successes": self.ocr_success_count,
            "errors": self.ocr_error_count,
            "elapsed_s": self.ocr_elapsed_s,
        }
        self.reset_ocr_stats()
        return stats

    def save_template_from_rect(self, frame_rgb: np.ndarray, rect: list[int], output_path: Path) -> None:
        crop = self.crop_rect(frame_rgb, rect)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    def _real_area(self, frame_rgb: np.ndarray, rect: list[int] | None) -> tuple[int, int, int, int]:
        if rect is None:
            h, w = frame_rgb.shape[:2]
            return 0, 0, w, h
        x, y, rw, rh = self.mapper.rect_to_real(rect)
        return x, y, x + rw, y + rh
