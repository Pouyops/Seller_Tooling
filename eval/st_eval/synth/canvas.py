"""Low-level rendering: anti-aliased shapes, noise, premultiplied layers, placement."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SHIFT = 4
_SCALE = 1 << SHIFT


def P(pts) -> np.ndarray:
    """Float points -> int32 fixed-point (sub-pixel precision for cv2 with ``shift=SHIFT``)."""
    return np.round(np.asarray(pts, np.float64) * _SCALE).astype(np.int32).reshape(-1, 1, 2)


def poly_mask(h: int, w: int, polys, value: float = 1.0) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [P(p) for p in polys], 255, cv2.LINE_AA, SHIFT)
    return m.astype(np.float32) * (value / 255.0)


def stroke_mask(h: int, w: int, polylines, thickness: float, closed: bool = False) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    if len(polylines):
        cv2.polylines(m, [P(p) for p in polylines], closed, 255, max(1, int(round(thickness))), cv2.LINE_AA, SHIFT)
    return m.astype(np.float32) / 255.0


def ellipse_mask(h, w, center, axes, angle=0.0, thickness=-1, start=0, end=360) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    draw_ellipse(m, center, axes, angle, thickness, start, end)
    return m.astype(np.float32) / 255.0


def draw_ellipse(m: np.ndarray, center, axes, angle=0.0, thickness=-1, start=0, end=360, value=255) -> None:
    c = (int(round(center[0] * _SCALE)), int(round(center[1] * _SCALE)))
    a = (max(1, int(round(axes[0] * _SCALE))), max(1, int(round(axes[1] * _SCALE))))
    t = -1 if thickness < 0 else max(1, int(round(thickness)))
    cv2.ellipse(m, c, a, float(angle), float(start), float(end), value, t, cv2.LINE_AA, SHIFT)


def value_noise(h: int, w: int, cell: float, rng: np.random.Generator, octaves: int = 4, persistence: float = 0.5) -> np.ndarray:
    """Smooth fractal noise in [0, 1]."""
    out = np.zeros((h, w), np.float32)
    amp, total, c = 1.0, 0.0, float(cell)
    for _ in range(octaves):
        gh, gw = max(2, int(h / c) + 2), max(2, int(w / c) + 2)
        grid = rng.random((gh, gw), dtype=np.float32)
        out += amp * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= persistence
        c = max(1.0, c / 2)
    out /= total
    lo, hi = float(out.min()), float(out.max())
    return (out - lo) / (hi - lo + 1e-6)


def rgb(*c) -> np.ndarray:
    return np.asarray(c, np.float32)


def jitter_color(c, rng: np.random.Generator, amount: float = 0.06) -> np.ndarray:
    return np.clip(np.asarray(c, np.float32) + rng.normal(0, amount, 3).astype(np.float32), 0, 1)


def bezier(ctrl: np.ndarray, n: int = 400) -> np.ndarray:
    """Evaluate a Bezier curve of any degree (De Casteljau)."""
    ctrl = np.asarray(ctrl, np.float64)
    t = np.linspace(0, 1, n)[:, None, None]
    pts = np.repeat(ctrl[None], n, axis=0)
    while pts.shape[1] > 1:
        pts = (1 - t) * pts[:, :-1] + t * pts[:, 1:]
    return pts[:, 0]


def resample_by_arclength(curve: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every ``step`` px along a polyline, with tangent angles (degrees)."""
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    targets = np.arange(0, s[-1], step)
    x = np.interp(targets, s, curve[:, 0])
    y = np.interp(targets, s, curve[:, 1])
    dx = np.gradient(x) if len(x) > 1 else np.zeros_like(x)
    dy = np.gradient(y) if len(y) > 1 else np.zeros_like(y)
    return np.stack([x, y], 1), np.degrees(np.arctan2(dy, dx))


@dataclass
class Layer:
    """An object layer. ``pm`` is premultiplied colour; ``alpha`` physical opacity; ``cov`` silhouette."""

    pm: np.ndarray
    alpha: np.ndarray
    cov: np.ndarray
    refract: np.ndarray | None = None  # (h, w, 2) background displacement in px, inside the silhouette

    @classmethod
    def empty(cls, h: int, w: int) -> Layer:
        return cls(np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32), np.zeros((h, w), np.float32))

    @property
    def shape(self) -> tuple[int, int]:
        return self.alpha.shape

    def paint(self, cov_part: np.ndarray, color, opacity=1.0, silhouette: bool = True) -> None:
        """Composite a part *over* the layer. ``color`` is (3,) or (h, w, 3); ``opacity`` scalar or (h, w)."""
        a = np.clip(cov_part * opacity, 0, 1).astype(np.float32)
        col = np.asarray(color, np.float32)
        if col.ndim == 1:
            col = col[None, None, :]
        self.pm = col * a[..., None] + self.pm * (1 - a[..., None])
        self.alpha = a + self.alpha * (1 - a)
        if silhouette:
            self.cov = np.maximum(self.cov, cov_part.astype(np.float32))

    def cut(self, hole: np.ndarray) -> None:
        """Punch a hole (e.g. the inside of a chain link) through everything painted so far."""
        keep = 1 - np.clip(hole, 0, 1)
        self.pm *= keep[..., None]
        self.alpha *= keep
        self.cov *= keep

    def warp(self, M: np.ndarray, size_wh: tuple[int, int]) -> Layer:
        W, H = size_wh
        flags = cv2.INTER_LINEAR
        pm = cv2.warpPerspective(self.pm, M, (W, H), flags=flags, borderValue=0)
        al = cv2.warpPerspective(self.alpha, M, (W, H), flags=flags, borderValue=0)
        cv = cv2.warpPerspective(self.cov, M, (W, H), flags=flags, borderValue=0)
        rf = None
        if self.refract is not None:
            rf = cv2.warpPerspective(self.refract, M, (W, H), flags=flags, borderValue=0)
        return Layer(pm, al, cv, rf)


def placement_homography(
    src_wh: tuple[int, int],
    dst_wh: tuple[int, int],
    rng: np.random.Generator,
    fill: float,
    max_rot_deg: float = 8.0,
    max_persp: float = 0.06,
    margin: float = 0.03,
) -> np.ndarray:
    """Scale/rotate/perspective-warp an object canvas into the scene, keeping it inside the frame."""
    sw, sh = src_wh
    W, H = dst_wh
    s = fill * min((W * (1 - 2 * margin)) / sw, (H * (1 - 2 * margin)) / sh)
    th = np.radians(rng.uniform(-max_rot_deg, max_rot_deg))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    src = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float64)
    local = (src - [sw / 2, sh / 2]) * s
    dst = local @ R.T
    dst += rng.uniform(-max_persp, max_persp, dst.shape) * [sw * s, sh * s]
    lo, hi = dst.min(0), dst.max(0)
    ext = hi - lo
    m = np.array([W, H]) * margin
    cx = rng.uniform(m[0] - lo[0], max(m[0] - lo[0], W - m[0] - hi[0]))
    cy = rng.uniform(m[1] - lo[1], max(m[1] - lo[1], H - m[1] - hi[1]))
    if ext[0] > W - 2 * m[0]:
        cx = W / 2 - (lo[0] + hi[0]) / 2
    if ext[1] > H - 2 * m[1]:
        cy = H / 2 - (lo[1] + hi[1]) / 2
    dst += [cx, cy]
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))


def composite(bg: np.ndarray, layer: Layer) -> np.ndarray:
    img = bg
    if layer.refract is not None:
        h, w = layer.shape
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        mapx = gx + layer.refract[..., 0]
        mapy = gy + layer.refract[..., 1]
        seen = cv2.remap(bg, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        c = layer.cov[..., None]
        img = bg * (1 - c) + seen * c
    return layer.pm + img * (1 - layer.alpha[..., None])
