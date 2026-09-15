"""Scene assembly and dataset writing (deterministic, resumable, parallel)."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from st_common.storage import atomic_write_bytes

from . import backgrounds as B
from .canvas import composite, placement_homography
from .objects import OBJECTS, ObjectResult

GENERATOR_VERSION = "synth-v1"
SS = 2  # supersampling factor; ground truth is area-averaged from this resolution
OUTPUT_SIZES = [(1080, 1350), (1200, 1200), (1280, 960), (960, 1280)]  # (W, H), typical seller uploads
CATEGORY_COUNTS = {
    "carpet": 40,
    "jewelry": 40,
    "saffron_packaged": 30,
    "glassware": 30,
    "handicraft": 30,
    "clothing_hanger": 30,
}
_CATEGORY_ID = {c: i for i, c in enumerate(CATEGORY_COUNTS)}

_BG_CHOICES = {
    "carpet": [("wood", 3), ("studio", 1), ("satin", 2), ("pattern", 1), ("cluttered", 2), ("fabric", 1)],
    "jewelry": [("velvet", 3), ("satin", 3), ("marble", 2), ("studio", 1), ("fabric", 1)],
    "saffron_packaged": [("marble", 2), ("wood", 3), ("studio", 1), ("fabric", 2), ("cluttered", 2), ("satin", 1)],
    "glassware": [("studio", 3), ("cluttered", 2), ("wood", 2), ("pattern", 1), ("velvet", 1)],
    "handicraft": [("pattern", 3), ("wood", 2), ("fabric", 2), ("studio", 1), ("satin", 1), ("cluttered", 1)],
    "clothing_hanger": [("wall_with_rail", 4), ("cluttered", 2), ("pattern", 1), ("studio", 1)],
}
_BG_TAGS = {"satin": ["low_contrast"], "pattern": ["patterned_bg"], "cluttered": ["cluttered_bg"], "wall_with_rail": ["occluding_rail"]}


def _background(category: str, obj: ObjectResult, h: int, w: int, rng: np.random.Generator):
    names, weights = zip(*_BG_CHOICES[category])
    p = np.asarray(weights, float) / sum(weights)
    name = str(rng.choice(names, p=p))
    tags = list(_BG_TAGS.get(name, []))
    if name == "satin":
        bg = B.satin(h, w, rng, obj.key_color)
    elif name == "wall_with_rail":
        bg, _ = B.wall_with_rail(h, w, rng)
    elif name == "studio" and category == "glassware":
        bg = B.studio(h, w, rng)
        tags.append("low_contrast")
    else:
        bg = getattr(B, name)(h, w, rng)
    return np.clip(bg.astype(np.float32), 0, 1), name, tags


def generate_sample(index: int, category: str, seed: int, size: tuple[int, int] | None = None) -> dict:
    """Render one sample. Returns encoded ``image`` (JPEG), ``alpha``/``mask`` (PNG) bytes and ``meta``."""
    rng = np.random.default_rng([seed, index, _CATEGORY_ID[category]])
    W, H = size or OUTPUT_SIZES[int(rng.integers(len(OUTPUT_SIZES)))]
    Ws, Hs = W * SS, H * SS

    obj = OBJECTS[category](rng.uniform(0.7, 0.95) * min(Ws, Hs), rng)
    lh, lw = obj.layer.shape
    M = placement_homography((lw, lh), (Ws, Hs), rng, obj.fill)
    layer = obj.layer.warp(M, (Ws, Hs))

    bg, bg_name, bg_tags = _background(category, obj, Hs, Ws, rng)
    tags = set(obj.tags) | set(bg_tags)
    if rng.random() < 0.65:
        bg = B.shadow(bg, layer.cov, rng)
        tags.add("cast_shadow")
    if obj.glossy and bg_name in ("marble", "studio", "velvet") and rng.random() < 0.4:
        bg = B.reflection(bg, layer.pm, layer.alpha, rng)
        tags.add("surface_reflection")

    img = composite(bg, layer)
    light = cv2.resize(rng.uniform(0.85, 1.1, (2, 2)).astype(np.float32), (Ws, Hs), interpolation=cv2.INTER_LINEAR)
    img = np.clip(img * light[..., None] * rng.uniform(0.95, 1.05, 3).astype(np.float32), 0, 1)

    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(layer.alpha, (W, H), interpolation=cv2.INTER_AREA)
    cov = cv2.resize(layer.cov, (W, H), interpolation=cv2.INTER_AREA)
    img = np.clip(img + rng.normal(0, rng.uniform(0.002, 0.012), img.shape).astype(np.float32), 0, 1)

    img_u8 = (img * 255 + 0.5).astype(np.uint8)
    alpha_u8 = (np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8)
    mask_u8 = ((cov >= 0.5) * 255).astype(np.uint8)
    q = int(rng.integers(72, 96))
    ok1, jpg = cv2.imencode(".jpg", cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    ok2, apng = cv2.imencode(".png", alpha_u8)
    ok3, mpng = cv2.imencode(".png", mask_u8)
    assert ok1 and ok2 and ok3

    meta = {
        "index": index,
        "category": category,
        "subtype": obj.subtype,
        "background": bg_name,
        "tags": sorted(tags),
        "width": W,
        "height": H,
        "jpeg_quality": q,
        "fg_fraction": round(float((mask_u8 > 0).mean()), 4),
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
    }
    return {"image": jpg.tobytes(), "alpha": apng.tobytes(), "mask": mpng.tobytes(), "meta": meta}


def plan_dataset(n: int) -> list[tuple[int, str, str]]:
    """(index, category, sample_id) for ``n`` samples, keeping category proportions."""
    total = sum(CATEGORY_COUNTS.values())
    counts = {c: max(1, round(k * n / total)) for c, k in CATEGORY_COUNTS.items()}
    while sum(counts.values()) > n:
        counts[max(counts, key=counts.get)] -= 1
    while sum(counts.values()) < n:
        counts[min(counts, key=counts.get)] += 1
    plan, idx = [], 0
    for cat, k in counts.items():
        for j in range(k):
            plan.append((idx, cat, f"{cat}_{j:03d}"))
            idx += 1
    return plan


def _render_and_write(args) -> dict:
    out, index, category, sample_id, seed, size = args
    s = generate_sample(index, category, seed, size)
    out = Path(out)
    atomic_write_bytes(out / "images" / f"{sample_id}.jpg", s["image"], fsync=False)
    atomic_write_bytes(out / "alpha" / f"{sample_id}.png", s["alpha"], fsync=False)
    atomic_write_bytes(out / "mask" / f"{sample_id}.png", s["mask"], fsync=False)
    meta = {"id": sample_id, **s["meta"]}
    # meta is written last: its presence marks the sample as complete (resume point)
    atomic_write_bytes(out / "meta" / f"{sample_id}.json", json.dumps(meta, ensure_ascii=False).encode(), fsync=True)
    return meta


def generate_dataset(out_dir: Path | str, n: int = 200, seed: int = 1403, workers: int | None = None,
                     size: tuple[int, int] | None = None, log=print) -> Path:
    out = Path(out_dir)
    for sub in ("images", "alpha", "mask", "meta"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    plan = plan_dataset(n)
    todo = [(str(out), i, c, sid, seed, size) for i, c, sid in plan if not (out / "meta" / f"{sid}.json").exists()]
    workers = workers or max(1, min(4, (os.cpu_count() or 2) - 2))
    t0 = time.time()
    log(f"[synth] {len(plan) - len(todo)}/{len(plan)} samples already present; rendering {len(todo)} with {workers} workers")
    if todo:
        if workers == 1:
            for k, a in enumerate(todo, 1):
                _render_and_write(a)
                if k % 20 == 0:
                    log(f"[synth] {k}/{len(todo)} ({time.time() - t0:.0f}s)")
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for k, _ in enumerate(ex.map(_render_and_write, todo, chunksize=1), 1):
                    if k % 20 == 0:
                        log(f"[synth] {k}/{len(todo)} ({time.time() - t0:.0f}s)")
    metas = [json.loads((out / "meta" / f"{sid}.json").read_text(encoding="utf-8")) for _, _, sid in plan]
    lines = "\n".join(json.dumps(m, ensure_ascii=False) for m in metas) + "\n"
    atomic_write_bytes(out / "manifest.jsonl", lines.encode("utf-8"))
    summary = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "n": len(plan),
        "supersampling": SS,
        "counts": {c: sum(1 for m in metas if m["category"] == c) for c in CATEGORY_COUNTS},
        "tag_counts": _tag_counts(metas),
    }
    atomic_write_bytes(out / "dataset.json", json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8"))
    log(f"[synth] done in {time.time() - t0:.0f}s -> {out}")
    return out


def _tag_counts(metas):
    counts: dict[str, int] = {}
    for m in metas:
        for t in m["tags"]:
            counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def load_manifest(dataset_dir: Path | str) -> list[dict]:
    p = Path(dataset_dir) / "manifest.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def contact_sheet(dataset_dir: Path | str, out_path: Path | str, per_category: int = 4, thumb: int = 200) -> Path:
    """Grid of image | ground-truth cutout on a checkerboard, a few per category, for eyeballing."""
    d = Path(dataset_dir)
    metas = load_manifest(d)
    rows = []
    for cat in CATEGORY_COUNTS:
        items = [m for m in metas if m["category"] == cat][:per_category]
        tiles = []
        for m in items:
            img = cv2.imread(str(d / "images" / f"{m['id']}.jpg"))
            alpha = cv2.imread(str(d / "alpha" / f"{m['id']}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            s = thumb / max(img.shape[:2])
            size = (int(img.shape[1] * s), int(img.shape[0] * s))
            im = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            a = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)[..., None]
            yy, xx = np.mgrid[0 : size[1], 0 : size[0]]
            checker = np.where(((yy // 10 + xx // 10) % 2)[..., None] == 0, 200, 150).astype(np.float32)
            cut = (im * a + checker * (1 - a)).astype(np.uint8)
            pair = np.full((thumb, 2 * thumb + 4, 3), 255, np.uint8)
            pair[: size[1], : size[0]] = im
            pair[: size[1], thumb + 4 : thumb + 4 + size[0]] = cut
            tiles.append(pair)
        while len(tiles) < per_category:
            tiles.append(np.full((thumb, 2 * thumb + 4, 3), 255, np.uint8))
        rows.append(np.concatenate([np.pad(t, ((4, 4), (4, 4), (0, 0)), constant_values=255) for t in tiles], 1))
    sheet = np.concatenate(rows, 0)
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    atomic_write_bytes(Path(out_path), buf.tobytes(), fsync=False)
    return Path(out_path)
