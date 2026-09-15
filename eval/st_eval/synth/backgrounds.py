"""Procedural backgrounds, from easy studio sweeps to low-contrast and cluttered scenes."""

from __future__ import annotations

import cv2
import numpy as np

from .canvas import P, draw_ellipse, jitter_color, rgb, value_noise


def studio(h, w, rng, base=None):
    top = jitter_color(base if base is not None else rng.choice([(0.97, 0.97, 0.96), (0.9, 0.9, 0.92), (0.95, 0.93, 0.9)]), rng, 0.02)
    bottom = np.clip(top * rng.uniform(0.8, 0.95), 0, 1)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = top * (1 - t) + bottom * t
    return np.broadcast_to(img, (h, w, 3)).copy() + (value_noise(h, w, 400, rng, 2)[..., None] - 0.5) * 0.03


def wood(h, w, rng):
    light = jitter_color(rng.choice([(0.62, 0.42, 0.25), (0.72, 0.55, 0.36), (0.45, 0.28, 0.16)]), rng, 0.03)
    dark = light * rng.uniform(0.7, 0.85)
    n = value_noise(h, w, max(h, w) / 3, rng, 3)
    y = np.arange(h, dtype=np.float32)[:, None] + n * rng.uniform(10, 40)
    fine = cv2.resize(rng.random((h, max(2, w // 40)), dtype=np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    grain = 0.5 + 0.5 * np.sin(y * rng.uniform(0.15, 0.35) + value_noise(h, w, 200, rng, 2) * 3)
    grain = 0.6 * grain + 0.4 * fine
    img = dark * (1 - grain[..., None]) + light * grain[..., None]
    plank = int(rng.uniform(0.15, 0.3) * h)
    for yy in range(plank, h, plank):
        img[max(0, yy - 2) : yy + 2] *= 0.6
    return img


def marble(h, w, rng):
    base = jitter_color((0.93, 0.92, 0.9), rng, 0.02)
    turb = value_noise(h, w, max(h, w) / 4, rng, 5)
    x = np.arange(w, dtype=np.float32)[None, :]
    veins = np.abs(np.sin(x * rng.uniform(0.004, 0.012) + turb * rng.uniform(6, 14))) ** 0.25
    img = base[None, None] * (0.72 + 0.28 * veins[..., None])
    return img


def fabric(h, w, rng, color=None):
    c = jitter_color(color if color is not None else rng.choice([(0.8, 0.75, 0.65), (0.55, 0.1, 0.12), (0.2, 0.25, 0.35)]), rng)
    period = rng.uniform(3, 7)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    weave = 0.5 + 0.25 * np.sin(xx * 2 * np.pi / period) * np.sin(yy * 2 * np.pi / period)
    folds = 0.8 + 0.35 * value_noise(h, w, max(h, w) / 3, rng, 2)
    return c * (0.85 + 0.3 * weave[..., None]) * folds[..., None] * 0.9


def velvet(h, w, rng):
    c = jitter_color(rng.choice([(0.08, 0.05, 0.06), (0.35, 0.04, 0.08), (0.05, 0.08, 0.2)]), rng, 0.02)
    sheen = value_noise(h, w, max(h, w) / 2.5, rng, 3)
    return c * (0.6 + 0.9 * sheen[..., None])


def satin(h, w, rng, color):
    """Low-contrast: close to the product's own colour, with soft folds."""
    c = np.clip(np.asarray(color, np.float32) * rng.uniform(0.85, 1.1) + rng.normal(0, 0.03, 3), 0, 1)
    folds = value_noise(h, w, max(h, w) / 5, rng, 3)
    return c * (0.75 + 0.45 * folds[..., None])


def pattern(h, w, rng):
    """Persian-style tile / tablecloth pattern: a deliberate distractor behind patterned products."""
    pal = [jitter_color(c, rng) for c in [(0.1, 0.35, 0.5), (0.9, 0.87, 0.78), (0.6, 0.12, 0.1), (0.8, 0.6, 0.25)]]
    img = np.empty((h, w, 3), np.float32)
    img[:] = pal[1]
    step = int(rng.uniform(40, 90))
    m = np.zeros((h, w), np.uint8)
    for yy in range(0, h + step, step):
        for xx in range(0, w + step, step):
            off = step / 2 if (yy // step) % 2 else 0
            cx, cy, r = xx + off, yy, step * 0.42
            diamond = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
            cv2.fillPoly(m, [P(diamond)], 1, cv2.LINE_AA, 4)
    img = img * (1 - m[..., None]) + pal[0] * m[..., None]
    m2 = np.zeros((h, w), np.uint8)
    for yy in range(0, h + step, step):
        for xx in range(0, w + step, step):
            off = step / 2 if (yy // step) % 2 else 0
            draw_ellipse(m2, (xx + off, yy), (step * 0.14, step * 0.14), value=1)
    img = img * (1 - m2[..., None]) + pal[2] * m2[..., None]
    return img * (0.9 + 0.15 * value_noise(h, w, max(h, w) / 3, rng, 2)[..., None])


def cluttered(h, w, rng):
    """Home/shop scene: wall, table edge, out-of-focus furniture and plants."""
    wall = jitter_color(rng.choice([(0.85, 0.82, 0.75), (0.7, 0.75, 0.72), (0.6, 0.55, 0.5)]), rng)
    img = np.empty((h, w, 3), np.float32)
    img[:] = wall
    table_y = int(h * rng.uniform(0.45, 0.7))
    img[table_y:] = wood(h - table_y, w, rng)
    for _ in range(int(rng.integers(6, 14))):
        c = jitter_color(rng.random(3), rng, 0.1)
        m = np.zeros((h, w), np.uint8)
        if rng.random() < 0.5:
            x0, y0 = int(rng.uniform(0, w)), int(rng.uniform(0, table_y))
            x1, y1 = x0 + int(rng.uniform(0.05, 0.3) * w), y0 + int(rng.uniform(0.1, 0.5) * h)
            cv2.rectangle(m, (x0, y0), (x1, y1), 1, -1)
        else:
            draw_ellipse(m, (rng.uniform(0, w), rng.uniform(0, table_y)), (rng.uniform(0.03, 0.15) * w, rng.uniform(0.03, 0.2) * h), rng.uniform(0, 180), value=1)
        img = img * (1 - m[..., None]) + c * m[..., None]
    sigma = rng.uniform(4, 14)
    top = cv2.GaussianBlur(img[:table_y], (0, 0), sigma)
    img[:table_y] = top
    return img


def wall_with_rail(h, w, rng):
    img = studio(h, w, rng, base=rng.choice([(0.92, 0.9, 0.86), (0.8, 0.83, 0.85)]))
    if rng.random() < 0.5:
        img = img * (0.92 + 0.08 * pattern(h, w, rng).mean(axis=2, keepdims=True))
    y = int(h * rng.uniform(0.02, 0.06))
    th = int(rng.uniform(6, 14))
    grad = np.linspace(0.85, 0.45, th, dtype=np.float32)[:, None, None]
    img[y : y + th] = rgb(0.75, 0.75, 0.78) * grad
    return img, y + th // 2


def shadow(bg: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = cov.shape
    dx, dy = rng.uniform(-0.03, 0.05) * w, rng.uniform(0.01, 0.05) * h
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    s = cv2.warpAffine(cov, M, (w, h))
    s = cv2.GaussianBlur(s, (0, 0), rng.uniform(0.01, 0.03) * max(h, w))
    return bg * (1 - rng.uniform(0.2, 0.5) * s[..., None])


def reflection(bg: np.ndarray, pm: np.ndarray, alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Faint mirror image below the object, as on glossy surfaces. Background, not ground truth."""
    ys = np.nonzero(alpha.max(axis=1) > 0.5)[0]
    if len(ys) == 0:
        return bg
    bottom = int(ys.max())
    h = alpha.shape[0]
    flip_pm, flip_a = pm[::-1], alpha[::-1]
    shift = 2 * bottom - h
    M = np.float32([[1, 0, 0], [0, 1, shift]])
    rp = cv2.warpAffine(flip_pm, M, (alpha.shape[1], h))
    ra = cv2.warpAffine(flip_a, M, (alpha.shape[1], h))
    fade = np.clip(1 - (np.arange(h, dtype=np.float32) - bottom) / (0.35 * h), 0, 1)
    fade[:bottom] = 0
    k = rng.uniform(0.12, 0.3) * fade[:, None]
    rp = cv2.GaussianBlur(rp, (0, 0), 2)
    ra = cv2.GaussianBlur(ra, (0, 0), 2)
    return bg * (1 - (ra * k)[..., None]) + rp * k[..., None]
