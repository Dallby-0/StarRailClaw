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


def match_template_luma_color_fused(
    image: np.ndarray,
    template: np.ndarray,
    color_weight: float = 0.2,
) -> MatchResult:
    base = match_template_luma(image, template)
    x1, y1 = base.top_left
    x2, y2 = base.bottom_right
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        color_similarity = 0.5
    else:
        tpl_for_color = template
        if tpl_for_color.shape[:2] != roi.shape[:2]:
            tpl_for_color = cv2.resize(template, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)
        color_raw = compare_histogram(roi, tpl_for_color)
        # Loosen color check to tolerate lighting shifts.
        color_similarity = float(np.clip((color_raw + 1.0) * 0.5, 0.0, 1.0))

    lw = 1.0 - float(np.clip(color_weight, 0.0, 1.0))
    cw = float(np.clip(color_weight, 0.0, 1.0))
    similarity = float(lw * base.similarity + cw * color_similarity)
    similarity = float(np.clip(similarity, 0.0, 1.0))
    top_left = base.top_left
    bottom_right = base.bottom_right
    return MatchResult(similarity=similarity, top_left=top_left, bottom_right=bottom_right)


def match_akaze_features(
    image: np.ndarray,
    template: np.ndarray,
    ratio_thresh: float = 0.8,
) -> MatchResult:
    image_luma = _to_luma(image)
    tpl_luma = _to_luma(template)

    akaze = cv2.AKAZE.create()
    kp_img, des_img = akaze.detectAndCompute(image_luma, None)
    kp_tpl, des_tpl = akaze.detectAndCompute(tpl_luma, None)

    h, w = tpl_luma.shape[:2]
    default_result = MatchResult(similarity=0.0, top_left=(0, 0), bottom_right=(w, h))
    if des_img is None or des_tpl is None or not kp_img or not kp_tpl:
        return default_result

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(des_tpl, des_img, k=2)
    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    if not good_matches:
        return default_result

    matched_tpl_pts = np.array([kp_tpl[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    matched_img_pts = np.array([kp_img[m.trainIdx].pt for m in good_matches], dtype=np.float32)
    offset = np.mean(matched_img_pts - matched_tpl_pts, axis=0)

    top_left = (int(round(offset[0])), int(round(offset[1])))
    bottom_right = (top_left[0] + w, top_left[1] + h)
    denom = max(1, min(len(kp_tpl), len(kp_img)))
    similarity = float(min(1.0, len(good_matches) / denom))
    return MatchResult(similarity=similarity, top_left=top_left, bottom_right=bottom_right)


def match_surf_features(
    image: np.ndarray,
    template: np.ndarray,
    ratio_thresh: float = 0.75,
    hessian_threshold: float = 400.0,
) -> MatchResult:
    image_luma = _to_luma(image)
    tpl_luma = _to_luma(template)

    h, w = tpl_luma.shape[:2]
    default_result = MatchResult(similarity=0.0, top_left=(0, 0), bottom_right=(w, h))

    detector = None
    if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SURF_create"):
        try:
            detector = cv2.xfeatures2d.SURF_create(hessianThreshold=float(hessian_threshold))
        except cv2.error:
            detector = None
    if detector is None and hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create()
    if detector is None:
        return default_result

    kp_img, des_img = detector.detectAndCompute(image_luma, None)
    kp_tpl, des_tpl = detector.detectAndCompute(tpl_luma, None)
    if des_img is None or des_tpl is None or not kp_img or not kp_tpl:
        return default_result

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(des_tpl, des_img, k=2)
    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    if not good_matches:
        return default_result

    matched_tpl_pts = np.array([kp_tpl[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    matched_img_pts = np.array([kp_img[m.trainIdx].pt for m in good_matches], dtype=np.float32)
    offset = np.mean(matched_img_pts - matched_tpl_pts, axis=0)

    top_left = (int(round(offset[0])), int(round(offset[1])))
    bottom_right = (top_left[0] + w, top_left[1] + h)
    denom = max(1, min(len(kp_tpl), len(kp_img)))
    similarity = float(min(1.0, len(good_matches) / denom))
    return MatchResult(similarity=similarity, top_left=top_left, bottom_right=bottom_right)


def match_template_luma_multiscale_color_fused(
    image: np.ndarray,
    template: np.ndarray,
    min_scale: float = 0.85,
    max_scale: float = 1.15,
    num_scales: int = 11,
    color_weight: float = 0.3,
) -> MatchResult:
    image_luma = _to_luma(image)
    tpl_luma = _to_luma(template)
    img_h, img_w = image_luma.shape[:2]
    base_h, base_w = tpl_luma.shape[:2]

    if base_h < 2 or base_w < 2:
        return MatchResult(similarity=0.0, top_left=(0, 0), bottom_right=(base_w, base_h))

    if num_scales < 1:
        num_scales = 1
    if min_scale <= 0 or max_scale <= 0:
        min_scale, max_scale = 1.0, 1.0
    if min_scale > max_scale:
        min_scale, max_scale = max_scale, min_scale

    if num_scales == 1:
        scales = np.array([1.0], dtype=np.float32)
    else:
        scales = np.linspace(min_scale, max_scale, num_scales, dtype=np.float32)

    best = None
    best_tpl_luma = None
    best_scale = 1.0
    for scale in scales:
        w = max(2, int(round(base_w * float(scale))))
        h = max(2, int(round(base_h * float(scale))))
        if w > img_w or h > img_h:
            continue
        scaled_tpl_luma = cv2.resize(tpl_luma, (w, h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        result = cv2.matchTemplate(image_luma, scaled_tpl_luma, cv2.TM_CCOEFF_NORMED)
        _, sim, _, pt = cv2.minMaxLoc(result)
        if best is None or sim > best[0]:
            best = (float(sim), pt, w, h)
            best_tpl_luma = scaled_tpl_luma
            best_scale = float(scale)

    if best is None:
        return MatchResult(similarity=0.0, top_left=(0, 0), bottom_right=(base_w, base_h))

    luma_sim, pt, w, h = best
    top_left = (int(pt[0]), int(pt[1]))
    bottom_right = (top_left[0] + w, top_left[1] + h)

    roi = image[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
    if roi.size == 0:
        color_sim = 0.0
    else:
        scaled_tpl = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA if best_scale < 1.0 else cv2.INTER_LINEAR)
        color_raw = compare_histogram(roi, scaled_tpl)
        color_sim = float(np.clip((color_raw + 1.0) * 0.5, 0.0, 1.0))

    luma_sim = float(np.clip(luma_sim, 0.0, 1.0))
    cw = float(np.clip(color_weight, 0.0, 1.0))
    similarity = float((1.0 - cw) * luma_sim + cw * color_sim)
    return MatchResult(similarity=similarity, top_left=top_left, bottom_right=bottom_right)


def match_template_luma_multiscale(
    image: np.ndarray,
    template: np.ndarray,
    min_scale: float = 0.85,
    max_scale: float = 1.15,
    num_scales: int = 11,
) -> MatchResult:
    """Multi-scale gray template matching using TM_CCOEFF_NORMED.

    Returns the best score/position among uniformly sampled scales.
    """
    image_luma = _to_luma(image)
    tpl_luma = _to_luma(template)
    img_h, img_w = image_luma.shape[:2]
    base_h, base_w = tpl_luma.shape[:2]

    if base_h < 2 or base_w < 2:
        return MatchResult(similarity=0.0, top_left=(0, 0), bottom_right=(base_w, base_h))

    if num_scales < 1:
        num_scales = 1
    if min_scale <= 0 or max_scale <= 0:
        min_scale, max_scale = 1.0, 1.0
    if min_scale > max_scale:
        min_scale, max_scale = max_scale, min_scale

    if num_scales == 1:
        scales = np.array([1.0], dtype=np.float32)
    else:
        scales = np.linspace(min_scale, max_scale, num_scales, dtype=np.float32)

    best = None
    for scale in scales:
        w = max(2, int(round(base_w * float(scale))))
        h = max(2, int(round(base_h * float(scale))))
        if w > img_w or h > img_h:
            continue

        scaled_tpl_luma = cv2.resize(tpl_luma, (w, h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        result = cv2.matchTemplate(image_luma, scaled_tpl_luma, cv2.TM_CCOEFF_NORMED)
        _, sim, _, pt = cv2.minMaxLoc(result)
        if best is None or sim > best[0]:
            best = (float(sim), pt, w, h)

    if best is None:
        return MatchResult(similarity=0.0, top_left=(0, 0), bottom_right=(base_w, base_h))

    sim, pt, w, h = best
    top_left = (int(pt[0]), int(pt[1]))
    bottom_right = (top_left[0] + w, top_left[1] + h)
    similarity = float(np.clip(sim, 0.0, 1.0))
    return MatchResult(similarity=similarity, top_left=top_left, bottom_right=bottom_right)
