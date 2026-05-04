from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]


class YoloEMatcher:
    """YOLOE wrapper with visual-prompt (template image + bbox) support."""

    def __init__(
        self,
        model_path: str = "yoloe-11s-seg.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = device
        self._model = None
        self._vp_predictor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLOE
            from ultralytics.models.yolo.yoloe.predict import YOLOEVPDetectPredictor, YOLOEVPSegPredictor
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "Failed to import ultralytics.YOLOE. Install/upgrade ultralytics first: "
                "python -m pip install -U ultralytics"
            ) from exc
        self._model = YOLOE(self.model_path)
        task = getattr(self._model, "task", None) or getattr(getattr(self._model, "model", None), "task", None)
        self._vp_predictor = YOLOEVPSegPredictor if task == "segment" else YOLOEVPDetectPredictor

    @staticmethod
    def _normalize_image(image, name: str) -> np.ndarray:
        # Common caller mistake: passing cv2.VideoCapture.read() tuple (ok, frame)
        # or a single-item tuple/list wrapping the ndarray.
        if isinstance(image, tuple):
            if len(image) == 2 and isinstance(image[0], (bool, np.bool_)) and isinstance(image[1], np.ndarray):
                image = image[1]
            elif len(image) == 1 and isinstance(image[0], np.ndarray):
                image = image[0]
        elif isinstance(image, list) and len(image) == 1 and isinstance(image[0], np.ndarray):
            image = image[0]

        if not isinstance(image, np.ndarray):
            raise TypeError(f"{name} must be a numpy ndarray, got {type(image)!r}.")
        if image.ndim < 2:
            raise TypeError(f"{name} must have shape [H, W, C], got ndim={image.ndim}.")
        return np.ascontiguousarray(image)

    def detect(
        self,
        image_rgb: np.ndarray,
        classes: Optional[Sequence[int]] = None,
        verbose: bool = False,
    ) -> List[Detection]:
        self._load()
        image_rgb = self._normalize_image(image_rgb, "image_rgb")
        results = self._model.predict(
            source=image_rgb,
            conf=self.conf,
            iou=self.iou,
            classes=list(classes) if classes is not None else None,
            device=self.device,
            verbose=verbose,
        )
        return self._to_detections(results)

    def detect_with_visual_prompt(
        self,
        image_rgb: np.ndarray,
        refer_image_rgb: np.ndarray,
        refer_bbox_xyxy: Tuple[int, int, int, int],
        refer_class_id: int = 0,
        verbose: bool = False,
    ) -> List[Detection]:
        """Detect objects in image_rgb using one bbox prompt on refer_image_rgb."""
        self._load()
        image_rgb = self._normalize_image(image_rgb, "image_rgb")
        refer_image_rgb = self._normalize_image(refer_image_rgb, "refer_image_rgb")
        x1, y1, x2, y2 = refer_bbox_xyxy
        visual_prompts = {
            "bboxes": [[int(x1), int(y1), int(x2), int(y2)]],
            "cls": [int(refer_class_id)],
        }
        results = self._model.predict(
            source=image_rgb,
            refer_image=refer_image_rgb,
            visual_prompts=visual_prompts,
            predictor=self._vp_predictor,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=verbose,
        )
        return self._to_detections(results)

    @staticmethod
    def _to_detections(results) -> List[Detection]:
        if not results:
            return []
        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = r.names if isinstance(r.names, dict) else {}
        detections: List[Detection] = []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        clss = boxes.cls.detach().cpu().numpy().astype(np.int32)
        for idx in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[idx].tolist()
            cls_id = int(clss[idx])
            label = names.get(cls_id, str(cls_id))
            detections.append(
                Detection(
                    label=label,
                    confidence=float(confs[idx]),
                    xyxy=(int(x1), int(y1), int(x2), int(y2)),
                )
            )
        return detections

    @staticmethod
    def draw(image_rgb: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
        canvas = image_rgb.copy()
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{det.label} {det.confidence:.2f}"
            cv2.putText(
                canvas,
                text,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return canvas


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLOE with optional visual prompt.")
    parser.add_argument("image", help="Target image path to detect on.")
    parser.add_argument("--model", default="yoloe-11s-seg.pt", help="YOLOE model path or name.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--device", default=None, help="Inference device, e.g. cpu / 0.")
    parser.add_argument("--save", default=None, help="Optional output image path.")
    parser.add_argument("--refer-image", default=None, help="Reference image path for visual prompt.")
    parser.add_argument(
        "--refer-bbox",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="Reference bbox in refer-image.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    matcher = YoloEMatcher(model_path=args.model, conf=args.conf, iou=args.iou, device=args.device)
    if args.refer_image and args.refer_bbox:
        refer_path = Path(args.refer_image)
        if not refer_path.exists():
            raise FileNotFoundError(f"Reference image not found: {refer_path}")
        refer_bgr = cv2.imread(str(refer_path), cv2.IMREAD_COLOR)
        if refer_bgr is None:
            raise RuntimeError(f"Failed to read reference image: {refer_path}")
        refer_rgb = cv2.cvtColor(refer_bgr, cv2.COLOR_BGR2RGB)
        detections = matcher.detect_with_visual_prompt(
            image_rgb=image_rgb,
            refer_image_rgb=refer_rgb,
            refer_bbox_xyxy=(args.refer_bbox[0], args.refer_bbox[1], args.refer_bbox[2], args.refer_bbox[3]),
        )
    else:
        detections = matcher.detect(image_rgb)

    print(f"detections={len(detections)}")
    for det in detections:
        print(f"{det.label}\t{det.confidence:.4f}\t{det.xyxy}")

    if args.save:
        vis = matcher.draw(image_rgb, detections)
        out_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.save, out_bgr)
        print(f"saved={args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
