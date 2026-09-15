"""Product renderers. Each returns an ``ObjectResult`` on a local canvas at render (supersampled) scale.

Coordinates are pixels on that canvas; thickness values assume the canvas will be placed at
roughly 0.5-1.0x scale into a 2x-supersampled scene, so ``thickness=2`` ends up ~0.5-1 px wide
in the final image (the real width of carpet fringe or a fine chain in a phone photo).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .canvas import (
    Layer,
    P,
    bezier,
    draw_ellipse,
    ellipse_mask,
    jitter_color,
    poly_mask,
    resample_by_arclength,
    rgb,
    stroke_mask,
    value_noise,
)

WHITE = rgb(1.0, 1.0, 1.0)


@dataclass
class ObjectResult:
    layer: Layer
    subtype: str
    tags: list[str]
    key_color: np.ndarray  # dominant colour; used to build low-contrast backgrounds
    fill: float = 0.8  # fraction of the frame the object should occupy
    glossy: bool = False  # eligible for a mirror reflection on the surface


# ---------------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------------
def _mix(img: np.ndarray, m: np.ndarray, color) -> np.ndarray:
    return img * (1 - m[..., None]) + np.asarray(color, np.float32) * m[..., None]


def _star(cx, cy, r_out, r_in, n, phase, y_scale=1.0) -> np.ndarray:
    a = np.arange(2 * n) * np.pi / n + phase
    r = np.where(np.arange(2 * n) % 2 == 0, r_out, r_in)
    return np.stack([cx + r * np.cos(a), cy + y_scale * r * np.sin(a)], 1)


def _metal(h, w, rng, base, cell) -> np.ndarray:
    n = value_noise(h, w, cell, rng, 3)
    col = np.clip(np.asarray(base, np.float32)[None, None] * (0.45 + 0.9 * n)[..., None], 0, 1)
    glint = (value_noise(h, w, max(2.0, cell / 3), rng, 2) > 0.8).astype(np.float32)
    glint = cv2.GaussianBlur(glint, (0, 0), 1.0)
    return _mix(col, glint, rgb(1.0, 0.97, 0.88))


def _bevel(mask: np.ndarray, base, rng) -> np.ndarray:
    m8 = (mask > 0.5).astype(np.uint8)
    dt = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
    bev = np.clip(dt / max(3.0, float(dt.max()) * 0.25), 0, 1)
    gx = cv2.Sobel(dt, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(dt, cv2.CV_32F, 0, 1, ksize=5)
    ang = rng.uniform(0, 2 * np.pi)
    light = gx * np.cos(ang) + gy * np.sin(ang)
    light = np.clip(light / (np.abs(light).max() + 1e-6), 0, 1)
    col = np.asarray(base, np.float32)[None, None] * (0.55 + 0.45 * bev[..., None]) + (light**3)[..., None] * 0.6
    return np.clip(col, 0, 1)


def _profile_silhouette(H, W, cx, y_top, height, ctrl):
    """Solid of revolution seen from the side. Returns (silhouette, x_normalized, radius_per_row)."""
    ctrl = np.asarray(ctrl, np.float64)
    t = np.linspace(0, 1, 160)
    rs = np.interp(t, ctrl[:, 0], ctrl[:, 1])
    left = np.stack([cx - rs, y_top + t * height], 1)
    right = np.stack([cx + rs, y_top + t * height], 1)[::-1]
    sil = poly_mask(H, W, [np.concatenate([left, right])])
    rows = np.clip((np.arange(H) - y_top) / height, 0, 1)
    r_row = np.interp(rows, ctrl[:, 0], ctrl[:, 1]).astype(np.float32)
    xn = (np.arange(W, dtype=np.float32)[None, :] - cx) / np.maximum(r_row[:, None], 1.0)
    return sil, xn, r_row


def _pseudo_script(rect, rng, rows=4) -> list[np.ndarray]:
    """Squiggles and dots that read as Persian script at thumbnail size (no font dependency)."""
    x0, y0, x1, y1 = rect
    lines = []
    rh = (y1 - y0) / rows
    for r in range(rows):
        yb = y0 + rh * (r + 0.6)
        x = x1 - rng.uniform(0, 0.1) * (x1 - x0)  # right-to-left
        while x > x0 + 10:
            wlen = rng.uniform(0.08, 0.25) * (x1 - x0)
            xs = np.linspace(x, max(x0, x - wlen), 6)
            ys = yb - np.abs(rng.normal(0, rh * 0.18, 6))
            ys[0] = ys[-1] = yb
            lines.append(bezier(np.stack([xs, ys], 1), 40))
            x -= wlen + rng.uniform(0.02, 0.05) * (x1 - x0)
    return lines


def _threads(edge_points, direction, length, rng, spacing_jitter=0.6, tassel_group=0):
    """Fringe threads hanging from a list of edge points. Returns polylines."""
    lines = []
    k = np.linspace(0, 1, 7)
    xs = np.array([p[0] for p in edge_points])
    for i, (x, y) in enumerate(edge_points):
        ln = length * rng.uniform(0.7, 1.0)
        sway = rng.normal(0, length * 0.08)
        if tassel_group:
            gi = min(len(xs) - 1, (i // tassel_group) * tassel_group + tassel_group // 2)
            gc = xs[gi]
            pinch = np.interp(k, [0, 0.25, 1], [1, 0.15, 0.9])
            px = gc + (x - gc) * pinch + sway * k**1.5
        else:
            px = x + sway * k**1.5 + rng.normal(0, spacing_jitter, len(k)) * k
        py = y + direction * (k * ln - 3)
        lines.append(np.stack([px, py], 1))
    return lines


# ---------------------------------------------------------------------------------------------
# carpet: fringe and tassels
# ---------------------------------------------------------------------------------------------
CARPET_PALETTES = [
    [(0.55, 0.07, 0.09), (0.10, 0.12, 0.30), (0.90, 0.85, 0.72), (0.78, 0.60, 0.25)],
    [(0.10, 0.14, 0.32), (0.62, 0.10, 0.10), (0.88, 0.84, 0.70), (0.25, 0.45, 0.45)],
    [(0.86, 0.80, 0.66), (0.55, 0.35, 0.25), (0.30, 0.40, 0.50), (0.70, 0.55, 0.35)],
    [(0.35, 0.45, 0.35), (0.75, 0.60, 0.35), (0.90, 0.86, 0.75), (0.45, 0.10, 0.10)],
]


def carpet(L: float, rng: np.random.Generator) -> ObjectResult:
    ch = int(L)
    cw = int(L / rng.uniform(1.3, 1.7))
    flen = int(ch * rng.uniform(0.035, 0.075))
    pad = 6
    H, W = ch + 2 * flen + 2 * pad, cw + 2 * pad
    top, left = flen + pad, pad
    field_c, border_c, light_c, accent_c = [jitter_color(c, rng, 0.03) for c in CARPET_PALETTES[rng.integers(len(CARPET_PALETTES))]]

    tex = np.empty((ch, cw, 3), np.float32)
    tex[:] = field_c
    bw = int(cw * rng.uniform(0.07, 0.11))
    guard = max(2, bw // 6)
    cv2.rectangle(tex, (0, 0), (cw - 1, ch - 1), tuple(map(float, border_c)), bw * 2)
    cv2.rectangle(tex, (bw, bw), (cw - 1 - bw, ch - 1 - bw), tuple(map(float, light_c)), guard)
    cv2.rectangle(tex, (guard, guard), (cw - 1 - guard, ch - 1 - guard), tuple(map(float, accent_c)), guard)

    # border motifs
    step, r = max(bw, 8), bw * 0.3
    polys = []
    for x in np.arange(bw, cw - bw, step):
        for y in (bw / 2, ch - bw / 2):
            polys.append([(x, y - r), (x + r, y), (x, y + r), (x - r, y)])
    for y in np.arange(bw, ch - bw, step):
        for x in (bw / 2, cw - bw / 2):
            polys.append([(x, y - r), (x + r, y), (x, y + r), (x - r, y)])
    tex = _mix(tex, poly_mask(ch, cw, polys), accent_c)

    # field lattice
    x0, y0, x1, y1 = bw + 2 * guard, bw + 2 * guard, cw - bw - 2 * guard, ch - bw - 2 * guard
    gstep = rng.uniform(0.08, 0.14) * cw
    polys = []
    for y in np.arange(y0 + gstep / 2, y1, gstep):
        for x in np.arange(x0 + gstep / 2, x1, gstep):
            rr = gstep * 0.28
            polys.append([(x, y - rr), (x + rr * 0.7, y), (x, y + rr), (x - rr * 0.7, y)])
    tex = _mix(tex, poly_mask(ch, cw, polys), light_c * 0.85 + field_c * 0.15)

    # corner pieces, clipped to the field
    R = cw * rng.uniform(0.22, 0.32)
    inner = np.zeros((ch, cw), np.float32)
    inner[int(y0) : int(y1), int(x0) : int(x1)] = 1
    for qx, qy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        tex = _mix(tex, ellipse_mask(ch, cw, (qx, qy), (R * 0.55, R * 0.55)) * inner, border_c)

    # central medallion
    n = int(rng.choice([8, 12, 16]))
    phase = rng.uniform(0, np.pi)
    for rr, col in ((R, border_c), (R * 0.72, light_c), (R * 0.45, accent_c), (R * 0.22, field_c)):
        tex = _mix(tex, poly_mask(ch, cw, [_star(cw / 2, ch / 2, rr, rr * 0.78, n, phase, 1.3)]), col)

    # wool: low-frequency abrash + knot noise
    tex *= (0.82 + 0.3 * value_noise(ch, cw, ch / 5, rng, 3))[..., None]
    knots = cv2.GaussianBlur(rng.normal(0, 0.035, (ch, cw)).astype(np.float32), (0, 0), 0.8)
    tex = np.clip(tex * (1 + knots)[..., None], 0, 1)

    layer = Layer.empty(H, W)
    body = np.zeros((H, W), np.float32)
    body[top : top + ch, left : left + cw] = 1
    col = np.zeros((H, W, 3), np.float32)
    col[top : top + ch, left : left + cw] = tex
    layer.paint(body, col)

    # fringe
    spacing = rng.uniform(4, 9)
    thick = int(rng.choice([2, 3, 3, 4]))
    tassel = rng.random() < 0.4
    group = int(rng.integers(4, 9)) if tassel else 0
    fringe_c = jitter_color(rng.choice([(0.92, 0.88, 0.78), (0.85, 0.80, 0.68), (0.95, 0.93, 0.90)]), rng, 0.02)
    lines = []
    for edge_y, d in ((top, -1), (top + ch, 1)):
        xs = np.arange(left + spacing / 2, left + cw - spacing / 2, spacing)
        keep = rng.random(len(xs)) > 0.08
        pts = [(x, edge_y) for x, k in zip(xs, keep) if k]
        lines += _threads(pts, d, flen, rng, tassel_group=group)
    threads = stroke_mask(H, W, lines, thick)
    shade = (0.8 + 0.25 * value_noise(H, W, 20, rng, 2))[..., None]
    layer.paint(threads, fringe_c[None, None] * shade)
    if tassel:
        knots_m = np.zeros((H, W), np.uint8)
        for line_i in range(group // 2, len(lines), group):
            kx, ky = lines[line_i][2]
            draw_ellipse(knots_m, (kx, ky), (spacing * group * 0.22, max(3, flen * 0.06)))
        layer.paint(knots_m.astype(np.float32) / 255, fringe_c * 0.9)

    return ObjectResult(layer, "carpet_tassel" if tassel else "carpet_fringe", ["thin_structures", "high_frequency_texture"],
                        fringe_c, fill=rng.uniform(0.75, 0.92))


# ---------------------------------------------------------------------------------------------
# gold jewellery: chains, holes, glints
# ---------------------------------------------------------------------------------------------
GOLDS = [(0.85, 0.66, 0.25), (0.80, 0.52, 0.40), (0.82, 0.82, 0.80)]


def _chain(h, w, curve, link, wire) -> np.ndarray:
    pts, ang = resample_by_arclength(curve, link)
    ring = np.zeros((h, w), np.uint8)
    bars = []
    for i, (p, a) in enumerate(zip(pts, ang)):
        if i % 2 == 0:
            draw_ellipse(ring, p, (link * 0.65, link * 0.38), a, thickness=wire)
        else:
            d = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]) * link * 0.55
            bars.append(np.stack([p - d, p + d]))
    m = ring.astype(np.float32) / 255
    if bars:
        m = np.maximum(m, stroke_mask(h, w, bars, wire + 1))
    return m


def _pendant(H, W, anchor, rng, L):
    kind = rng.choice(["coin", "teardrop", "plate"])
    size = rng.uniform(0.07, 0.13) * L
    ax, ay = anchor
    cy = ay + size * 1.05
    hole = np.zeros((H, W), np.float32)
    if kind == "coin":
        m = ellipse_mask(H, W, (ax, cy), (size, size))
        if rng.random() < 0.5:
            hole = ellipse_mask(H, W, (ax, cy), (size * 0.35, size * 0.35))
    elif kind == "teardrop":
        arc = [(ax + size * 0.8 * np.cos(a), cy + size * 0.3 + size * 0.8 * np.sin(a)) for a in np.linspace(-0.15 * np.pi, 1.15 * np.pi, 60)]
        m = poly_mask(H, W, [arc + [(ax, cy - size * 0.95)]])
        hole = ellipse_mask(H, W, (ax, cy + size * 0.3), (size * 0.3, size * 0.3))
    else:
        rect = (ax - size * 1.4, cy - size * 0.5, ax + size * 1.4, cy + size * 0.5)
        m = poly_mask(H, W, [[(rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])]])
        inner = (rect[0] + size * 0.25, rect[1] + size * 0.2, rect[2] - size * 0.25, rect[3] - size * 0.2)
        hole = stroke_mask(H, W, _pseudo_script(inner, rng, rows=1), max(2, size * 0.1)) * m
    bail = ellipse_mask(H, W, (ax, ay + size * 0.02), (size * 0.15, size * 0.2), thickness=3)
    return np.maximum(m, bail), hole


def jewelry(L: float, rng: np.random.Generator) -> ObjectResult:
    H = W = int(L)
    base = jitter_color(GOLDS[rng.choice(3, p=[0.7, 0.15, 0.15])], rng, 0.03)
    layer = Layer.empty(H, W)
    sub = str(rng.choice(["necklace", "bracelet", "bangles", "earrings", "rings"], p=[0.4, 0.2, 0.15, 0.15, 0.1]))
    link = rng.uniform(8, 20)
    wire = int(rng.choice([2, 3, 4, 4, 5, 6]))
    cell = link * 2.5
    tags = ["reflective"]

    if sub == "necklace":
        sag = rng.uniform(0.55, 0.85)
        ctrl = [(0.08 * W, 0.1 * H), ((0.12 + rng.normal(0, 0.05)) * W, sag * H * 1.2),
                ((0.88 + rng.normal(0, 0.05)) * W, sag * H * 1.2), (0.92 * W, 0.1 * H)]
        curve = bezier(ctrl, 1500)
        layer.paint(_chain(H, W, curve, link, wire), _metal(H, W, rng, base, cell))
        if rng.random() < 0.9:
            pend, hole = _pendant(H, W, curve[np.argmax(curve[:, 1])], rng, L)
            layer.paint(pend, _bevel(pend, base, rng))
            layer.cut(hole)
        tags += ["thin_structures", "holes"]
    elif sub == "bracelet":
        t = np.linspace(0, 2 * np.pi, 1500)
        a, b = 0.38 * W, 0.25 * H * rng.uniform(0.8, 1.3)
        wob = 1 + 0.06 * np.sin(3 * t + rng.uniform(0, 6))
        curve = np.stack([W / 2 + a * wob * np.cos(t), H / 2 + b * wob * np.sin(t)], 1)
        layer.paint(_chain(H, W, curve, link, wire), _metal(H, W, rng, base, cell))
        clasp = ellipse_mask(H, W, curve[0], (link * 1.2, link * 0.7))
        layer.paint(clasp, _bevel(clasp, base, rng))
        tags += ["thin_structures", "holes"]
    elif sub == "bangles":
        for _ in range(int(rng.integers(1, 5))):
            c = (W / 2 + rng.normal(0, 0.08) * W, H / 2 + rng.normal(0, 0.08) * H)
            ax = (rng.uniform(0.25, 0.4) * W, rng.uniform(0.12, 0.3) * H)
            ring = ellipse_mask(H, W, c, ax, rng.uniform(0, 180), thickness=rng.uniform(0.02, 0.05) * L)
            col = _metal(H, W, rng, base, cell * 2)
            specks = (value_noise(H, W, 4, rng, 1) > 0.85).astype(np.float32)
            layer.paint(ring, col * (1 - 0.5 * specks[..., None]))
        tags += ["holes"]
    elif sub == "earrings":
        for side in (-1, 1):
            c = (W / 2 + side * W * 0.22, H * 0.35)
            ax = (rng.uniform(0.08, 0.16) * W, rng.uniform(0.14, 0.25) * H)
            layer.paint(ellipse_mask(H, W, c, ax, 0, thickness=rng.uniform(4, 9)), _metal(H, W, rng, base, cell))
            y = np.linspace(c[1] + ax[1], c[1] + ax[1] + 0.25 * H, 200)
            x = c[0] + np.cumsum(rng.normal(0, 0.15, 200))
            curve = np.stack([x, y], 1)
            layer.paint(_chain(H, W, curve, link * 0.7, 2), _metal(H, W, rng, base, cell))
            bead = ellipse_mask(H, W, curve[-1], (link * 1.2, link * 1.2))
            layer.paint(bead, _bevel(bead, base, rng))
        tags += ["thin_structures", "holes"]
    else:
        for _ in range(int(rng.integers(1, 4))):
            c = (rng.uniform(0.25, 0.75) * W, rng.uniform(0.3, 0.75) * H)
            R = rng.uniform(0.08, 0.14) * L
            band = ellipse_mask(H, W, c, (R, R * rng.uniform(0.5, 1.0)), rng.uniform(0, 180), thickness=R * 0.25)
            layer.paint(band, _metal(H, W, rng, base, cell))
            stone = ellipse_mask(H, W, (c[0], c[1] - R), (R * 0.35, R * 0.35))
            gem = jitter_color(rng.choice([(0.7, 0.05, 0.1), (0.05, 0.5, 0.2), (0.1, 0.2, 0.7)]), rng)
            layer.paint(stone, _bevel(stone, gem, rng))
        tags += ["holes"]

    return ObjectResult(layer, f"jewelry_{sub}", tags, base, fill=rng.uniform(0.75, 0.95), glossy=True)


# ---------------------------------------------------------------------------------------------
# saffron & packaged goods
# ---------------------------------------------------------------------------------------------
def _label(layer: Layer, rng, rect, bg_color, ink) -> None:
    H, W = layer.shape
    x0, y0, x1, y1 = rect
    m = poly_mask(H, W, [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]])
    layer.paint(m, bg_color, silhouette=False)
    script = stroke_mask(H, W, _pseudo_script((x0 + 8, y0 + 8, x1 - 8, y1 - 8), rng), rng.uniform(2, 4))
    layer.paint(script * m, ink, silhouette=False)


def _gloss(layer: Layer, mask: np.ndarray, rng) -> None:
    H, W = layer.shape
    c, wd = rng.uniform(0.2, 0.8) * W, rng.uniform(0.02, 0.06) * W
    g = np.exp(-(((np.arange(W, dtype=np.float32) - c) / wd) ** 2))[None, :] * rng.uniform(0.2, 0.45)
    layer.paint(mask, WHITE, opacity=np.broadcast_to(g, (H, W)), silhouette=False)


def _saffron_texture(H, W, rng) -> np.ndarray:
    base = jitter_color((0.55, 0.05, 0.03), rng, 0.03)
    n = value_noise(H, W, 6, rng, 2)
    streak = cv2.resize(rng.random((max(2, H // 12), max(2, W // 3)), dtype=np.float32), (W, H))
    return np.clip(base[None, None] * (0.5 + 0.8 * n[..., None]) * (0.8 + 0.4 * streak[..., None]), 0, 1)


def saffron_packaged(L: float, rng: np.random.Generator) -> ObjectResult:
    H = W = int(L)
    layer = Layer.empty(H, W)
    sub = str(rng.choice(["box", "tin", "jar", "pouch"], p=[0.3, 0.25, 0.2, 0.25]))
    brand = jitter_color(rng.choice([(0.6, 0.05, 0.1), (0.35, 0.1, 0.4), (0.85, 0.7, 0.3), (0.1, 0.25, 0.2), (0.95, 0.95, 0.93)]), rng)
    accent = jitter_color(rng.choice([(0.85, 0.68, 0.3), (0.95, 0.95, 0.9), (0.2, 0.2, 0.2)]), rng)
    tags: list[str] = []
    pw, ph = W * rng.uniform(0.4, 0.55), H * rng.uniform(0.45, 0.62)
    x0, y0 = W / 2 - pw / 2, H * 0.12

    if sub == "box":
        depth = pw * rng.uniform(0.2, 0.4)
        dy = depth * 0.5
        x0 -= depth / 2
        front = [(x0, y0 + dy), (x0 + pw, y0 + dy), (x0 + pw, y0 + dy + ph), (x0, y0 + dy + ph)]
        side = [(x0 + pw, y0 + dy), (x0 + pw + depth, y0), (x0 + pw + depth, y0 + ph), (x0 + pw, y0 + dy + ph)]
        topf = [(x0, y0 + dy), (x0 + depth, y0), (x0 + pw + depth, y0), (x0 + pw, y0 + dy)]
        layer.paint(poly_mask(H, W, [side]), brand * 0.7)
        layer.paint(poly_mask(H, W, [topf]), np.clip(brand * 1.2, 0, 1))
        fm = poly_mask(H, W, [front])
        layer.paint(fm, brand)
        _label(layer, rng, (x0 + pw * 0.1, y0 + dy + ph * 0.25, x0 + pw * 0.9, y0 + dy + ph * 0.75), accent, brand)
        _gloss(layer, fm, rng)
    elif sub == "tin":
        r, cx = pw / 2, W / 2
        ey = r * rng.uniform(0.25, 0.4)
        body = poly_mask(H, W, [[(cx - r, y0 + ey), (cx + r, y0 + ey), (cx + r, y0 + ph), (cx - r, y0 + ph)]])
        body = np.maximum(body, ellipse_mask(H, W, (cx, y0 + ph), (r, ey)))
        xx = np.clip((np.arange(W, dtype=np.float32) - cx) / r, -1, 1)
        shade = (0.5 + 0.5 * np.sqrt(1 - xx**2))[None, :, None]
        layer.paint(body, brand[None, None] * shade)
        # khatam band
        s = rng.uniform(6, 12)
        yy, xg = np.mgrid[0:H, 0:W].astype(np.float32)
        tri = ((xg % s) / s > (yy % s) / s).astype(np.float32)
        band = ((yy > y0 + ph * 0.35) & (yy < y0 + ph * 0.55)).astype(np.float32) * body
        layer.paint(band, _mix(np.broadcast_to(accent, (H, W, 3)).copy(), tri, brand * 0.5) * shade, silhouette=False)
        lid = ellipse_mask(H, W, (cx, y0 + ey), (r * 1.03, ey * 1.05))
        layer.paint(lid, _metal(H, W, rng, rgb(0.85, 0.7, 0.35), 20))
        tags += ["reflective"]
    elif sub == "jar":
        cx, r = W / 2, pw * 0.45
        jar = poly_mask(H, W, [[(cx - r, y0 + ph * 0.15), (cx + r, y0 + ph * 0.15), (cx + r, y0 + ph), (cx - r, y0 + ph)]])
        jar = cv2.GaussianBlur(jar, (0, 0), 1.0)
        layer.paint(jar, rgb(0.9, 0.93, 0.95), opacity=rng.uniform(0.15, 0.3))
        fill = jar.copy()
        fill[: int(y0 + ph * rng.uniform(0.3, 0.55))] = 0
        fill = cv2.erode(fill, np.ones((9, 9), np.uint8))
        layer.paint(fill, _saffron_texture(H, W, rng), opacity=0.95, silhouette=False)
        lid = poly_mask(H, W, [[(cx - r * 1.05, y0), (cx + r * 1.05, y0), (cx + r * 1.05, y0 + ph * 0.16), (cx - r * 1.05, y0 + ph * 0.16)]])
        layer.paint(lid, _metal(H, W, rng, rgb(0.85, 0.68, 0.3), 15))
        tags += ["transparency"]
    else:
        teeth = int(rng.uniform(18, 40))
        amp = rng.uniform(4, 9)
        xs = np.linspace(x0, x0 + pw, teeth * 2 + 1)
        ys = y0 + np.where(np.arange(len(xs)) % 2 == 0, 0, amp)
        outline = list(zip(xs, ys)) + [(x0 + pw, y0 + ph * 0.85), (x0 + pw * 0.9, y0 + ph), (x0 + pw * 0.1, y0 + ph), (x0, y0 + ph * 0.85)]
        pm = poly_mask(H, W, [outline])
        folds = (0.8 + 0.3 * value_noise(H, W, W / 6, rng, 2))[..., None]
        layer.paint(pm, brand[None, None] * folds)
        window = ellipse_mask(H, W, (x0 + pw / 2, y0 + ph * 0.62), (pw * 0.28, ph * 0.18))
        layer.cut(window)
        layer.paint(window, rgb(0.92, 0.94, 0.95), opacity=0.25)
        content = (value_noise(H, W, 5, rng, 2) > 0.45).astype(np.float32)
        layer.paint(window * content, _saffron_texture(H, W, rng), opacity=0.9, silhouette=False)
        layer.paint(stroke_mask(H, W, [[(x0, y0 + amp + 14), (x0 + pw, y0 + amp + 14)]], 3) * pm, brand * 0.6, silhouette=False)
        _label(layer, rng, (x0 + pw * 0.15, y0 + ph * 0.18, x0 + pw * 0.85, y0 + ph * 0.4), accent, brand)
        tags += ["transparency", "zigzag_edge"]

    if rng.random() < 0.75:
        lines, tips = [], []
        base_y = y0 + ph + H * 0.03
        for _ in range(int(rng.integers(15, 60))):
            sx, sy = rng.uniform(0.1, 0.9) * W, rng.uniform(base_y, H * 0.95)
            ang, ln = rng.uniform(0, 2 * np.pi), rng.uniform(0.015, 0.04) * L
            mid = (sx + np.cos(ang) * ln * 0.5 + rng.normal(0, ln * 0.2), sy + np.sin(ang) * ln * 0.5 + rng.normal(0, ln * 0.2))
            c = bezier([(sx, sy), mid, (sx + np.cos(ang) * ln, sy + np.sin(ang) * ln)], 20)
            lines.append(c)
            tips.append(c[-1])
        m = stroke_mask(H, W, lines, int(rng.choice([3, 4, 5])))
        flare = np.zeros((H, W), np.uint8)
        for t in tips:
            draw_ellipse(flare, t, (5.0, 3.5), rng.uniform(0, 180))
        layer.paint(np.maximum(m, flare / 255.0), jitter_color((0.62, 0.04, 0.03), rng, 0.04))
        tags += ["thin_structures", "disconnected_parts"]

    return ObjectResult(layer, f"saffron_{sub}", tags, brand, fill=rng.uniform(0.6, 0.85))


# ---------------------------------------------------------------------------------------------
# glassware: transparency and refraction
# ---------------------------------------------------------------------------------------------
def glassware(L: float, rng: np.random.Generator) -> ObjectResult:
    H, W = int(L), int(L * 0.8)
    cx = W / 2
    layer = Layer.empty(H, W)
    sub = str(rng.choice(["tea_glass", "tumbler", "bottle"], p=[0.45, 0.3, 0.25]))
    tint = jitter_color(rng.choice([(0.92, 0.95, 0.96), (0.85, 0.93, 0.90), (0.75, 0.85, 0.95)]), rng, 0.02)
    a_min, a_max = rng.uniform(0.06, 0.2), rng.uniform(0.55, 0.85)
    tags = ["transparency"]
    if sub == "tea_glass":
        h, y_top, rt = 0.62 * H, 0.12 * H, 0.24 * W
        ctrl = [(0, rt), (0.1, rt * 0.98), (0.5, rt * 0.72), (0.85, rt * 0.86), (0.97, rt * 0.8), (1.0, rt * 0.62)]
    elif sub == "tumbler":
        h, y_top, rt = 0.7 * H, 0.1 * H, 0.28 * W
        ctrl = [(0, rt), (1, rt * 0.8)]
    else:
        h, y_top, rt = 0.82 * H, 0.06 * H, 0.3 * W
        ctrl = [(0, rt * 0.3), (0.25, rt * 0.3), (0.4, rt * 0.95), (0.95, rt), (1, rt * 0.9)]
        if rng.random() < 0.4:
            tint = jitter_color(rng.choice([(0.3, 0.55, 0.8), (0.35, 0.65, 0.4)]), rng)
            a_min, a_max = a_min + 0.2, min(0.95, a_max + 0.1)

    sil, xn, r_row = _profile_silhouette(H, W, cx, y_top, h, ctrl)
    axn = np.clip(np.abs(xn), 0, 1)
    hl = np.exp(-(((xn + 0.55) / 0.06) ** 2)) * rng.uniform(0.5, 0.9) + np.exp(-(((xn - 0.75) / 0.04) ** 2)) * rng.uniform(0.3, 0.7)
    opacity = np.clip(a_min + (a_max - a_min) * axn**4 + hl, 0, 1)
    color = np.clip(_mix(np.broadcast_to(tint, (H, W, 3)).copy(), np.clip(hl, 0, 1), WHITE), 0, 1)

    if sub == "tea_glass" and rng.random() < 0.6:
        sc, sax = (cx, y_top + h + 0.02 * H), (rt * 1.7, rt * 0.38)
        layer.paint(ellipse_mask(H, W, sc, sax), tint, opacity=a_min + 0.15)
        layer.paint(ellipse_mask(H, W, sc, sax, thickness=4), WHITE, opacity=0.7)
        tags.append("saucer")
    layer.paint(sil, color, opacity=opacity)

    if (sub != "bottle" and rng.random() < 0.6) or (sub == "bottle" and rng.random() < 0.4):
        level = rng.uniform(0.2, 0.45) if sub != "bottle" else rng.uniform(0.45, 0.6)
        liquid = jitter_color(rng.choice([(0.5, 0.16, 0.04), (0.75, 0.2, 0.1), (0.9, 0.85, 0.6)]), rng)
        lm = sil.copy()
        ly = int(y_top + h * level)
        lm[:ly] = 0
        lm = cv2.erode(lm, np.ones((7, 7), np.uint8))
        layer.paint(lm, liquid[None, None] * (0.8 + 0.2 * (1 - axn))[..., None], opacity=rng.uniform(0.7, 0.92), silhouette=False)
        r_l = float(r_row[min(ly, H - 1)])
        layer.paint(ellipse_mask(H, W, (cx, ly), (r_l * 0.92, max(3, r_l * 0.12))), np.clip(liquid * 1.3, 0, 1), opacity=0.8, silhouette=False)
        tags.append("liquid")

    r_top = float(ctrl[0][1])
    layer.paint(ellipse_mask(H, W, (cx, y_top), (r_top, max(3, r_top * 0.16)), thickness=rng.uniform(3, 6)), rgb(0.97, 0.98, 1.0), opacity=0.85)

    strength = rng.uniform(-0.25, 0.25)
    dx = (strength * xn**3 * r_row[:, None]).astype(np.float32) * (sil > 0)
    layer.refract = np.stack([dx, np.zeros_like(dx)], -1)
    return ObjectResult(layer, f"glass_{sub}", tags, tint, fill=rng.uniform(0.55, 0.85), glossy=True)


# ---------------------------------------------------------------------------------------------
# handicrafts: minakari, khatam, pottery, engraved copper
# ---------------------------------------------------------------------------------------------
def handicraft(L: float, rng: np.random.Generator) -> ObjectResult:
    sub = str(rng.choice(["minakari_plate", "khatam_box", "pottery_vase", "copper_tray"], p=[0.3, 0.25, 0.25, 0.2]))
    return {"minakari_plate": _minakari, "khatam_box": _khatam, "pottery_vase": _vase, "copper_tray": _copper}[sub](L, rng)


def _minakari(L, rng):
    H = W = int(L)
    cx = cy = L / 2
    R = 0.47 * L
    n, amp = int(rng.integers(16, 40)), rng.uniform(0.005, 0.02)
    th = np.linspace(0, 2 * np.pi, 720)
    rr = R * (1 + amp * np.sin(n * th))
    plate = poly_mask(H, W, [np.stack([cx + rr * np.cos(th), cy + rr * np.sin(th)], 1)])
    blue = jitter_color(rng.choice([(0.05, 0.35, 0.6), (0.1, 0.55, 0.6), (0.1, 0.15, 0.45)]), rng)
    white, red, gold = jitter_color((0.93, 0.93, 0.9), rng, 0.02), jitter_color((0.7, 0.15, 0.1), rng), jitter_color((0.8, 0.62, 0.25), rng)
    layer = Layer.empty(H, W)
    layer.paint(plate, blue)
    for rad, col, t in ((0.9, gold, 4), (0.82, white, -1), (0.72, blue, -1), (0.35, white, -1), (0.2, red, -1)):
        layer.paint(ellipse_mask(H, W, (cx, cy), (R * rad, R * rad), thickness=t) * plate, col, silhouette=False)
    petals_w, petals_r, outline = (np.zeros((H, W), np.uint8) for _ in range(3))
    m = int(rng.integers(8, 16))
    for k in range(m):
        a = 2 * np.pi * k / m
        c = (cx + 0.52 * R * np.cos(a), cy + 0.52 * R * np.sin(a))
        target = petals_w if k % 2 == 0 else petals_r
        draw_ellipse(target, c, (0.16 * R, 0.06 * R), np.degrees(a))
        draw_ellipse(outline, c, (0.16 * R, 0.06 * R), np.degrees(a), thickness=2)
    for rad in (0.3, 0.62, 0.75):
        draw_ellipse(outline, (cx, cy), (R * rad, R * rad), thickness=2)
    layer.paint(petals_w / 255.0 * plate, white, silhouette=False)
    layer.paint(petals_r / 255.0 * plate, red, silhouette=False)
    layer.paint(outline / 255.0 * plate, gold, silhouette=False)
    shine = cv2.GaussianBlur(ellipse_mask(H, W, (cx - 0.3 * R, cy - 0.35 * R), (0.25 * R, 0.12 * R), -30), (0, 0), 0.05 * R)
    layer.paint(plate, WHITE, opacity=shine * 0.4, silhouette=False)
    return ObjectResult(layer, "handicraft_minakari_plate", ["high_frequency_texture", "scalloped_edge"], blue,
                        fill=rng.uniform(0.6, 0.85), glossy=True)


def _khatam(L, rng):
    H = W = int(L)
    s = rng.uniform(5, 10)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    tri = ((xx % s) / s > (yy % s) / s).astype(np.int32)
    idx = ((xx // s).astype(np.int32) + 2 * (yy // s).astype(np.int32) + tri) % 3
    pal = np.stack([jitter_color(c, rng, 0.02) for c in [(0.93, 0.9, 0.8), (0.75, 0.6, 0.3), (0.25, 0.12, 0.06)]])
    tex = pal[idx]
    pw, ph = 0.62 * W, 0.42 * H
    depth = rng.uniform(0.15, 0.3) * pw
    x0, y0 = (W - pw - depth) / 2, 0.3 * H
    front = [(x0, y0 + depth * 0.6), (x0 + pw, y0 + depth * 0.6), (x0 + pw, y0 + depth * 0.6 + ph), (x0, y0 + depth * 0.6 + ph)]
    side = [(x0 + pw, y0 + depth * 0.6), (x0 + pw + depth, y0), (x0 + pw + depth, y0 + ph), (x0 + pw, y0 + depth * 0.6 + ph)]
    topf = [(x0, y0 + depth * 0.6), (x0 + depth, y0), (x0 + pw + depth, y0), (x0 + pw, y0 + depth * 0.6)]
    layer = Layer.empty(H, W)
    dark = pal[2]
    for face, k in ((side, 0.7), (topf, 1.15), (front, 1.0)):
        m = poly_mask(H, W, [face])
        layer.paint(m, np.clip(tex * k, 0, 1))
        layer.paint(stroke_mask(H, W, [face], rng.uniform(6, 10), closed=True) * m, dark, silhouette=False)
    lid_y = y0 + depth * 0.6 + ph * 0.25
    layer.paint(stroke_mask(H, W, [[(x0, lid_y), (x0 + pw, lid_y), (x0 + pw + depth, lid_y - depth * 0.6)]], 4), dark)
    return ObjectResult(layer, "handicraft_khatam_box", ["high_frequency_texture"], pal[1], fill=rng.uniform(0.6, 0.85))


def _vase(L, rng):
    H, W = int(L), int(L * 0.9)
    cx, y_top, h = W / 2, 0.06 * H, 0.9 * H
    ctrl = [(0, 0.12 * W), (0.12, 0.1 * W), (0.2, 0.14 * W), (0.55, 0.34 * W), (0.85, 0.28 * W), (1.0, 0.16 * W)]
    sil, xn, r_row = _profile_silhouette(H, W, cx, y_top, h, ctrl)
    glaze = jitter_color(rng.choice([(0.1, 0.55, 0.6), (0.12, 0.2, 0.55), (0.55, 0.35, 0.2), (0.85, 0.82, 0.75)]), rng)
    off = rng.uniform(-0.4, 0.4)
    shade = 0.45 + 0.55 * np.cos(np.clip(xn, -1, 1) * np.pi / 2 - off)
    crackle = (np.abs(value_noise(H, W, L / 12, rng, 3) - 0.5) < 0.012).astype(np.float32)
    col = np.clip(glaze[None, None] * shade[..., None] * (1 - 0.4 * crackle[..., None]), 0, 1)
    layer = Layer.empty(H, W)
    tags = ["high_frequency_texture"]
    if rng.random() < 0.7:
        hy = y_top + h * 0.32
        r_h = float(r_row[int(hy)])
        for side, (s0, s1) in ((-1, (90, 270)), (1, (-90, 90))):
            hc = (cx + side * r_h * 0.95, hy)
            handle = ellipse_mask(H, W, hc, (0.1 * W, 0.12 * H), 0, thickness=0.028 * L, start=s0, end=s1)
            layer.paint(handle, glaze * 0.8)
        tags.append("holes")
    layer.paint(sil, col)
    band_y = y_top + h * rng.uniform(0.38, 0.5)
    band = ((np.arange(H) > band_y) & (np.arange(H) < band_y + 0.04 * H)).astype(np.float32)[:, None] * sil
    motif = (np.sin(np.arange(W, dtype=np.float32) * rng.uniform(0.05, 0.12))[None, :] > 0).astype(np.float32)
    layer.paint(band * motif, jitter_color((0.1, 0.1, 0.12), rng), silhouette=False)
    return ObjectResult(layer, "handicraft_pottery_vase", tags, glaze, fill=rng.uniform(0.6, 0.88))


def _copper(L, rng):
    H = W = int(L)
    cx = cy = L / 2
    R = 0.47 * L
    base = jitter_color(rng.choice([(0.72, 0.42, 0.22), (0.78, 0.62, 0.3)]), rng, 0.03)
    tray = ellipse_mask(H, W, (cx, cy), (R, R))
    streak = cv2.resize(rng.random((max(2, H // 6), max(2, W // 60)), dtype=np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    col = np.clip(base[None, None] * (0.5 + 0.9 * streak[..., None]), 0, 1)
    layer = Layer.empty(H, W)
    layer.paint(tray, col)
    layer.paint(ellipse_mask(H, W, (cx, cy), (R * 0.97, R * 0.97), thickness=0.04 * L), np.clip(base * 1.3, 0, 1), silhouette=False)
    eng = np.zeros((H, W), np.uint8)
    for rad in np.linspace(0.2, 0.85, int(rng.integers(3, 6))):
        draw_ellipse(eng, (cx, cy), (R * rad, R * rad), thickness=2)
    petals = int(rng.integers(5, 12))
    t = np.linspace(0, 2 * np.pi, 800)
    for k in range(3):
        rr = R * (0.25 + 0.2 * k) * (0.6 + 0.4 * np.abs(np.cos(petals * t / 2)))
        cv2.polylines(eng, [P(np.stack([cx + rr * np.cos(t), cy + rr * np.sin(t)], 1))], True, 255, 2, cv2.LINE_AA, 4)
    layer.paint(eng / 255.0 * tray, base * 0.35, silhouette=False)
    return ObjectResult(layer, "handicraft_copper_tray", ["reflective", "high_frequency_texture"], base,
                        fill=rng.uniform(0.6, 0.85), glossy=True)


# ---------------------------------------------------------------------------------------------
# clothing on hangers
# ---------------------------------------------------------------------------------------------
def _fabric(H, W, rng):
    kind = rng.choice(["solid", "stripes", "floral", "plaid"])
    c1 = jitter_color(rng.random(3) * 0.8 + 0.1, rng, 0.0)
    c2 = jitter_color(rng.random(3) * 0.8 + 0.1, rng, 0.0)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    tex = np.empty((H, W, 3), np.float32)
    tex[:] = c1
    if kind == "stripes":
        p = rng.uniform(12, 40)
        tex = _mix(tex, (np.sin(xx * 2 * np.pi / p) > 0.3).astype(np.float32), c2)
    elif kind == "plaid":
        p = rng.uniform(20, 60)
        tex = _mix(tex, ((np.sin(xx * 2 * np.pi / p) > 0.6) | (np.sin(yy * 2 * np.pi / p) > 0.6)).astype(np.float32), c2)
    elif kind == "floral":
        m = np.zeros((H, W), np.uint8)
        for _ in range(int(H * W / 3000)):
            c = (rng.uniform(0, W), rng.uniform(0, H))
            r = rng.uniform(4, 12)
            for a in range(0, 360, 72):
                draw_ellipse(m, (c[0] + r * np.cos(np.radians(a)), c[1] + r * np.sin(np.radians(a))), (r * 0.6, r * 0.6))
        tex = _mix(tex, m / 255.0, c2)
    folds = 0.78 + 0.3 * (0.5 + 0.5 * np.sin(xx * rng.uniform(0.01, 0.03) + value_noise(H, W, max(H, W) / 4, rng, 2) * 6))
    return np.clip(tex * folds[..., None], 0, 1), c1


def clothing_hanger(L: float, rng: np.random.Generator) -> ObjectResult:
    H, W = int(L), int(L * 0.85)
    cx = W / 2
    layer = Layer.empty(H, W)
    sub = str(rng.choice(["shirt", "manteau", "scarf", "sweater"], p=[0.3, 0.25, 0.25, 0.2]))
    wooden = rng.random() < 0.4
    tags = ["thin_structures"]

    hook_r, neck_y = 0.035 * L, 0.13 * H
    hcx, hcy = cx + hook_r * 0.2, 0.02 * H + hook_r
    a = np.linspace(1.1 * np.pi, -0.5 * np.pi, 60)
    arc = np.stack([hcx + hook_r * np.cos(a), hcy - hook_r * np.sin(a)], 1)
    metal = jitter_color((0.72, 0.72, 0.75), rng, 0.03)
    hook = stroke_mask(H, W, [np.concatenate([arc, [(cx, neck_y)]])], rng.uniform(4, 7))
    layer.paint(hook, metal[None, None] * (0.75 + 0.35 * value_noise(H, W, 15, rng, 2))[..., None])

    shoulder_y, half = 0.24 * H, 0.4 * W
    bar = [(cx - half, shoulder_y), (cx, neck_y), (cx + half, shoulder_y)]
    layer.paint(stroke_mask(H, W, [bar], rng.uniform(14, 24) if wooden else rng.uniform(4, 7)),
                jitter_color((0.5, 0.3, 0.15), rng) if wooden else metal)
    if not wooden:
        layer.paint(stroke_mask(H, W, [[(cx - half, shoulder_y), (cx + half, shoulder_y)]], 4), metal)

    tex, key = _fabric(H, W, rng)
    if sub != "scarf":
        hem_y = (0.9 if sub == "manteau" else 0.74) * H
        sl = shoulder_y + 0.005 * H
        flare = rng.uniform(0, 0.08) * W
        neck = [(cx + 0.08 * W * np.cos(t), shoulder_y - 0.02 * H + 0.06 * H * np.sin(t)) for t in np.linspace(0, np.pi, 12)]
        body = ([(cx - half * 0.95, sl), (cx - half * 0.72, sl + 0.12 * H), (cx - half * 0.7 - flare, hem_y),
                 (cx + half * 0.7 + flare, hem_y), (cx + half * 0.72, sl + 0.12 * H), (cx + half * 0.95, sl)] + neck)
        cuff_y = (0.62 if sub != "manteau" else 0.7) * H
        sleeves = []
        for s in (-1, 1):
            sleeves.append([(cx + s * half * 0.95, sl), (cx + s * half * 1.08, sl + 0.05 * H), (cx + s * half * 1.0, cuff_y),
                            (cx + s * half * 0.86, cuff_y), (cx + s * half * 0.8, sl + 0.14 * H)])
        cov = poly_mask(H, W, sleeves + [body])
        if sub == "sweater":
            noise = value_noise(H, W, 3, rng, 2) - 0.5
            cov = np.clip((cv2.GaussianBlur(cov, (0, 0), 2.5) + noise * 0.5 - 0.5) * 4 + 0.5, 0, 1)
            tags.append("fuzzy_edge")
        layer.paint(cov, tex)
        tags.append("holes")
        if sub == "manteau":
            btn = np.zeros((H, W), np.uint8)
            for y in np.linspace(sl + 0.1 * H, hem_y - 0.1 * H, 5):
                draw_ellipse(btn, (cx, y), (6, 6))
            layer.paint(btn / 255.0, rgb(0.1, 0.1, 0.1), silhouette=False)
            belt_x = cx + rng.uniform(-0.3, 0.3) * half
            layer.paint(stroke_mask(H, W, [[(belt_x, 0.55 * H), (belt_x + rng.normal(0, 8), 0.97 * H)]], rng.uniform(5, 9)), key * 0.7)
        if sub in ("shirt", "manteau") and rng.random() < 0.35:
            lace = np.zeros((H, W), np.uint8)
            holes = np.zeros((H, W), np.uint8)
            xl, xr = cx - half * 0.7 - flare, cx + half * 0.7 + flare
            r = rng.uniform(0.015, 0.03) * W
            for x in np.arange(xl + r, xr, 1.8 * r):
                draw_ellipse(lace, (x, hem_y), (r, r))
                draw_ellipse(holes, (x, hem_y + r * 0.35), (r * 0.3, r * 0.3))
            layer.paint(lace / 255.0, np.clip(key * 1.2, 0, 1))
            layer.cut(holes / 255.0)
            tags.append("lace")
    else:
        sheer = rng.uniform(0.55, 0.8)
        hem = rng.uniform(0.75, 0.9) * H
        panels = [[(cx - half * 0.7, shoulder_y - 0.03 * H), (cx - 0.02 * W, neck_y + 0.02 * H), (cx - 0.05 * W, hem), (cx - half * 0.9, hem - 0.05 * H)],
                  [(cx + half * 0.7, shoulder_y - 0.03 * H), (cx + 0.02 * W, neck_y + 0.02 * H), (cx + 0.05 * W, hem + 0.03 * H), (cx + half * 0.85, hem - 0.02 * H)]]
        cov = poly_mask(H, W, panels)
        layer.paint(cov, tex, opacity=sheer)
        pts = []
        for p in panels:
            (xa, ya), (xb, yb) = p[3], p[2]
            for t in np.linspace(0, 1, int(abs(xb - xa) / rng.uniform(5, 9))):
                pts.append((xa + (xb - xa) * t, ya + (yb - ya) * t))
        layer.paint(stroke_mask(H, W, _threads(pts, 1, 0.06 * H, rng), 3), key)
        tags += ["transparency"]

    return ObjectResult(layer, f"clothing_{sub}", tags, key, fill=rng.uniform(0.72, 0.92))


OBJECTS = {
    "carpet": carpet,
    "jewelry": jewelry,
    "saffron_packaged": saffron_packaged,
    "glassware": glassware,
    "handicraft": handicraft,
    "clothing_hanger": clothing_hanger,
}
