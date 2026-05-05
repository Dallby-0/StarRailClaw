from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, Tuple

import cv2
import numpy as np

Area = Tuple[int, int, int, int]

# Keep punctuation normalization aligned with StarRailCopilot keyword parsing.
REGEX_PUNCTUATION = re.compile(r"[ ,.．'\"“”，。、…:：;；!！?？·・•●〇°*※\-—–－/\\|丨\n\t()\[\]（）「」『』【】《》［］]")


def parse_name(text: object) -> str:
    normalized = REGEX_PUNCTUATION.sub("", str(text)).lower()
    return normalized.strip()


def _clip_area(area: Area, width: int, height: int) -> Area:
    x1, y1, x2, y2 = area
    x1 = max(0, min(int(x1), width))
    x2 = max(0, min(int(x2), width))
    y1 = max(0, min(int(y1), height))
    y2 = max(0, min(int(y2), height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _crop(image: np.ndarray, area: Area) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = _clip_area(area, w, h)
    return image[y1:y2, x1:x2]


def _extract_white_letters(image: np.ndarray, threshold: int = 255) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
    if image.ndim == 3:
        return cv2.merge([binary, binary, binary])
    return binary


@lru_cache(maxsize=8)
def _get_text_system(lang: str):
    try:
        from pponnxcr import TextSystem
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pponnxcr is required for sr_tools.ocr. Install pponnxcr and onnxruntime.") from exc
    ts = TextSystem(lang)
    ts.text_recognizer.rec_batch_num = 1
    return ts


def _ocr_single_line_text(image: np.ndarray, lang: str) -> str:
    if image.size == 0:
        return ""
    text, _ = _get_text_system(lang).ocr_single_line(image)
    return str(text or "")


def _detect_texts(image: np.ndarray, lang: str) -> list[str]:
    if image.size == 0:
        return []
    results = _get_text_system(lang).detect_and_ocr(image)
    return [str(r.ocr_text or "") for r in results]


def _has_meaningful_text(texts: Iterable[str]) -> bool:
    for text in texts:
        if parse_name(text):
            return True
    return False


def ocr_text(image: np.ndarray, area: Area, lang: str = "zhs") -> str:
    return _ocr_single_line_text(_crop(image, area), lang=lang)


def has_text(image: np.ndarray, area: Area, lang: str = "zhs") -> bool:
    roi = _crop(image, area)
    texts = _detect_texts(roi, lang=lang)
    if texts:
        return _has_meaningful_text(texts)
    return bool(parse_name(_ocr_single_line_text(roi, lang=lang)))


def ocr_text_white(image: np.ndarray, area: Area, lang: str = "zhs") -> str:
    roi = _extract_white_letters(_crop(image, area), threshold=255)
    return _ocr_single_line_text(roi, lang=lang)


def has_text_white(image: np.ndarray, area: Area, lang: str = "zhs") -> bool:
    roi = _extract_white_letters(_crop(image, area), threshold=255)
    texts = _detect_texts(roi, lang=lang)
    if texts:
        return _has_meaningful_text(texts)
    return bool(parse_name(_ocr_single_line_text(roi, lang=lang)))

