from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class MatchResult:
    similarity: float
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]


def _to_luma(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def match_template(image: np.ndarray, template: np.ndarray) -> MatchResult:
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, sim, _, pt = cv2.minMaxLoc(result)
    h, w = template.shape[:2]
    return MatchResult(similarity=float(sim), top_left=pt, bottom_right=(pt[0] + w, pt[1] + h))


def match_template_luma(image: np.ndarray, template: np.ndarray) -> MatchResult:
    image_luma = _to_luma(image)
    tpl_luma = _to_luma(template)
    result = cv2.matchTemplate(image_luma, tpl_luma, cv2.TM_CCOEFF_NORMED)
    _, sim, _, pt = cv2.minMaxLoc(result)
    h, w = tpl_luma.shape[:2]
    return MatchResult(similarity=float(sim), top_left=pt, bottom_right=(pt[0] + w, pt[1] + h))


def compare_histogram(image_a: np.ndarray, image_b: np.ndarray) -> float:
    a = cv2.cvtColor(image_a, cv2.COLOR_RGB2HSV)
    b = cv2.cvtColor(image_b, cv2.COLOR_RGB2HSV)
    hist_a = cv2.calcHist([a], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_b = cv2.calcHist([b], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def compare_ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    # Fast SSIM approximation using gaussian blur + covariance.
    a = _to_luma(image_a).astype(np.float64)
    b = _to_luma(image_b).astype(np.float64)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(np.mean(ssim_map))
