from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from .vision import MatchResult, match_template_luma


@dataclass
class TemplateSpec:
    key: str
    namespace: str
    file: str
    area: Tuple[int, int, int, int]
    search: Tuple[int, int, int, int]
    avg_color: Tuple[int, int, int]


def _crop(image: np.ndarray, area: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = area
    return image[y1:y2, x1:x2]


def _avg_color(image: np.ndarray) -> Tuple[int, int, int]:
    color = np.mean(image.reshape(-1, 3), axis=0)
    return int(color[0]), int(color[1]), int(color[2])


class ResourceCatalog:
    """Template resource registry with namespace support."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.templates_dir = self.root / "templates"
        self.catalog_path = self.root / "catalog.json"
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self._specs: Dict[str, TemplateSpec] = {}
        self._cache: Dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if not self.catalog_path.exists():
            return
        data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        for item in data.get("templates", []):
            spec = TemplateSpec(
                key=item["key"],
                namespace=item["namespace"],
                file=item["file"],
                area=tuple(item["area"]),
                search=tuple(item["search"]),
                avg_color=tuple(item["avg_color"]),
            )
            self._specs[self._id(spec.namespace, spec.key)] = spec

    def _save(self) -> None:
        payload = {"templates": [asdict(v) for v in self._specs.values()]}
        self.catalog_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _id(namespace: str, key: str) -> str:
        return f"{namespace}.{key}"

    def create_template(
        self,
        namespace: str,
        key: str,
        screenshot: np.ndarray,
        area: Tuple[int, int, int, int],
        search: Optional[Tuple[int, int, int, int]] = None,
    ) -> TemplateSpec:
        if search is None:
            x1, y1, x2, y2 = area
            search = (max(0, x1 - 20), max(0, y1 - 20), x2 + 20, y2 + 20)

        template = _crop(screenshot, area)
        rel = Path(namespace.replace(".", "/")) / f"{key}.png"
        out = self.templates_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), cv2.cvtColor(template, cv2.COLOR_RGB2BGR))

        spec = TemplateSpec(
            key=key,
            namespace=namespace,
            file=str(rel).replace("\\", "/"),
            area=area,
            search=search,
            avg_color=_avg_color(template),
        )
        self._specs[self._id(namespace, key)] = spec
        self._save()
        return spec

    def get(self, namespace: str, key: str) -> TemplateSpec:
        template_id = self._id(namespace, key)
        if template_id not in self._specs:
            raise KeyError(f"Template not found: {template_id}")
        return self._specs[template_id]

    def list_namespace(self, namespace: str) -> Iterable[TemplateSpec]:
        prefix = f"{namespace}."
        for template_id, spec in self._specs.items():
            if template_id.startswith(prefix):
                yield spec

    def _load_image(self, spec: TemplateSpec) -> np.ndarray:
        template_id = self._id(spec.namespace, spec.key)
        if template_id in self._cache:
            return self._cache[template_id]
        path = self.templates_dir / spec.file
        if not path.is_file():
            raise FileNotFoundError(f"Template file not found: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Template file not found: {path}")
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)
        self._cache[template_id] = image
        return image

    def match(self, namespace: str, key: str, screenshot: np.ndarray) -> MatchResult:
        spec = self.get(namespace, key)
        template = self._load_image(spec)
        search = _crop(screenshot, spec.search)
        result = match_template_luma(search, template)
        dx, dy = spec.search[0], spec.search[1]
        return MatchResult(
            similarity=result.similarity,
            top_left=(result.top_left[0] + dx, result.top_left[1] + dy),
            bottom_right=(result.bottom_right[0] + dx, result.bottom_right[1] + dy),
        )
