from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoordinateMapper:
    logical_w: int = 1000
    logical_h: int = 1000
    real_w: int = 1280
    real_h: int = 720

    def point_to_real(self, x: int, y: int) -> tuple[int, int]:
        rx = int(round(int(x) * self.real_w / self.logical_w))
        ry = int(round(int(y) * self.real_h / self.logical_h))
        return self._clamp_point(rx, ry)

    def point_to_logical(self, x: int, y: int) -> tuple[int, int]:
        lx = int(round(int(x) * self.logical_w / self.real_w))
        ly = int(round(int(y) * self.logical_h / self.real_h))
        return self._clamp_point(lx, ly, self.logical_w, self.logical_h)

    def rect_to_real(self, rect: list[int] | tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self._normalize_rect_logical(rect)
        rx, ry = self.point_to_real(x1, y1)
        r2x, r2y = self.point_to_real(x2, y2)
        rw = max(1, r2x - rx)
        rh = max(1, r2y - ry)
        if rx + rw > self.real_w:
            rw = max(1, self.real_w - rx)
        if ry + rh > self.real_h:
            rh = max(1, self.real_h - ry)
        return rx, ry, rw, rh

    def _normalize_rect_logical(
        self,
        rect: list[int] | tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        a, b, c, d = [int(v) for v in rect]
        # Prefer x1,y1,x2,y2; keep backward compatibility for legacy x,y,w,h.
        if c > a and d > b:
            x1, y1, x2, y2 = a, b, c, d
        else:
            x1, y1 = a, b
            x2, y2 = a + max(1, c), b + max(1, d)
        x1, y1 = self._clamp_point(x1, y1, self.logical_w, self.logical_h)
        x2, y2 = self._clamp_point(x2, y2, self.logical_w, self.logical_h)
        if x2 <= x1:
            x2 = min(self.logical_w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(self.logical_h - 1, y1 + 1)
        return x1, y1, x2, y2

    def _clamp_point(self, x: int, y: int, w: int | None = None, h: int | None = None) -> tuple[int, int]:
        mw = self.real_w if w is None else w
        mh = self.real_h if h is None else h
        x = max(0, min(mw - 1, x))
        y = max(0, min(mh - 1, y))
        return x, y
