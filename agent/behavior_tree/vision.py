from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
        if not self._ocr_backend_logged:
            self.log_fn("[vision][ocr] backend=sr_tools.ocr(pponnxcr)")
            self._ocr_backend_logged = True
        self.log_fn(f"[vision][ocr] call white_text={white_text} logical_rect={rect} real_rect={real_rect}")
        try:
            entries = sr_ocr.ocr_entries(frame_rgb, real_rect, lang="zhs", white=white_text)
        except Exception as exc:  # noqa: BLE001
            self.log_fn(
                f"[vision][ocr] error type={type(exc).__name__} message={exc} logical_rect={rect} real_rect={real_rect}"
            )
            return []
        if not entries:
            # Fallback 1: single-line OCR in normal mode.
            fallback_text = ""
            try:
                fallback_text = sr_ocr.ocr_text(frame_rgb, real_rect, lang="zhs").strip()
            except Exception as exc:  # noqa: BLE001
                self.log_fn(
                    f"[vision][ocr] fallback=single_line error type={type(exc).__name__} message={exc} real_rect={real_rect}"
                )
            if fallback_text:
                self.log_fn(f"[vision][ocr] fallback=single_line hit text={fallback_text!r}")
                entries = [
                    {
                        "text": fallback_text,
                        "conf": 1.0,
                        "bbox": (0, 0, max(1, real_rect[2] - real_rect[0]), max(1, real_rect[3] - real_rect[1])),
                        "center": ((real_rect[2] - real_rect[0]) // 2, (real_rect[3] - real_rect[1]) // 2),
                    }
                ]
            else:
                # Fallback 2: white-text single-line OCR.
                try:
                    fallback_white_text = sr_ocr.ocr_text_white(frame_rgb, real_rect, lang="zhs").strip()
                except Exception as exc:  # noqa: BLE001
                    self.log_fn(
                        f"[vision][ocr] fallback=white_single_line error type={type(exc).__name__} message={exc} real_rect={real_rect}"
                    )
                    fallback_white_text = ""
                if fallback_white_text:
                    self.log_fn(f"[vision][ocr] fallback=white_single_line hit text={fallback_white_text!r}")
                    entries = [
                        {
                            "text": fallback_white_text,
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
        self.log_fn(f"[vision][ocr] done candidates={len(out)} logical_rect={rect}")
        return out

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
