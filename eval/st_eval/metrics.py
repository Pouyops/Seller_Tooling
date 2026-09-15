"""Segmentation / matting metrics.

All functions take HxW arrays. Masks are boolean; alphas are float in [0, 1].

* ``iou``: intersection over union of binary masks.
* ``boundary_f``: DAVIS-style boundary F-measure. Boundary pixels of prediction and ground
  truth are matched within a tolerance; default ``bound_th=0.008`` of the image diagonal
  (the DAVIS convention). ``bf_3px`` is a strict variant that fringe and chains actually stress.
* ``alpha_mae`` / ``alpha_mae_band``: mean absolute alpha error, globally and inside a band
  around the true boundary (where matting quality lives).
* ``thin_recall``: recall on ground-truth pixels that belong to structures thinner than
  ``k`` px (carpet fringe, chains, hanger hooks). IoU barely notices these, sellers do.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

import cv2
import numpy as np

_K3 = np.ones((3, 3), np.uint8)


def to_binary(x: np.ndarray, thr: float = 0.5) -> np.ndarray:
    if x.dtype == bool:
        return x
    if x.dtype == np.uint8:
        return x >= int(round(thr * 255))
    return x >= thr


def to_alpha(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    if x.dtype == bool:
        return x.astype(np.float32)
    return np.clip(x.astype(np.float32), 0.0, 1.0)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = to_binary(pred), to_binary(gt)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def boundary_map(mask: np.ndarray) -> np.ndarray:
    """Inner boundary: foreground pixels with at least one 8-neighbour in the background."""
    m = to_binary(mask).astype(np.uint8)
    eroded = cv2.erode(m, _K3, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (m - eroded).astype(bool)


def boundary_f(pred: np.ndarray, gt: np.ndarray, bound_th: float = 0.008) -> tuple[float, float, float]:
    """Return ``(F, precision, recall)``. ``bound_th < 1`` is a fraction of the diagonal, else pixels."""
    pred, gt = to_binary(pred), to_binary(gt)
    h, w = gt.shape
    tol = bound_th if bound_th >= 1 else math.ceil(bound_th * math.hypot(h, w))
    pb, gb = boundary_map(pred), boundary_map(gt)
    n_p, n_g = int(pb.sum()), int(gb.sum())
    if n_p == 0 and n_g == 0:
        return 1.0, 1.0, 1.0
    if n_p == 0 or n_g == 0:
        return 0.0, float(n_p == 0), float(n_g == 0)
    # distanceTransform gives, for each non-zero pixel, the distance to the nearest zero pixel.
    d_to_g = cv2.distanceTransform((~gb).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_to_p = cv2.distanceTransform((~pb).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    precision = float((d_to_g[pb] <= tol).mean())
    recall = float((d_to_p[gb] <= tol).mean())
    if precision + recall == 0:
        return 0.0, precision, recall
    return 2 * precision * recall / (precision + recall), precision, recall


def alpha_mae(pred_alpha: np.ndarray, gt_alpha: np.ndarray) -> float:
    return float(np.abs(to_alpha(pred_alpha) - to_alpha(gt_alpha)).mean())


def boundary_band(gt_mask: np.ndarray, width: int) -> np.ndarray:
    b = boundary_map(gt_mask).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    return cv2.dilate(b, k).astype(bool)


def alpha_mae_band(pred_alpha: np.ndarray, gt_alpha: np.ndarray, gt_mask: np.ndarray, width: int = 10) -> float:
    band = boundary_band(gt_mask, width)
    if not band.any():
        return float("nan")
    return float(np.abs(to_alpha(pred_alpha)[band] - to_alpha(gt_alpha)[band]).mean())


def thin_structure_mask(gt_mask: np.ndarray, k: int = 7) -> np.ndarray:
    """Foreground parts narrower than ~k px: pixels removed by a k×k disk opening.

    Connected components smaller than 2k px are dropped. An opening also shaves a few pixels off
    every corner of a solid shape, and those shavings are not thin structures. A 2-px thread just
    14 px long is still kept.
    """
    m = to_binary(gt_mask).astype(np.uint8)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, se)
    thin = (m & (1 - opened)).astype(np.uint8)
    if not thin.any():
        return thin.astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 2 * k
    return keep[labels]


def thin_recall(pred_mask: np.ndarray, gt_mask: np.ndarray, k: int = 7) -> float:
    thin = thin_structure_mask(gt_mask, k)
    n = int(thin.sum())
    if n == 0:
        return float("nan")
    return float(np.logical_and(to_binary(pred_mask), thin).sum() / n)


def compute_all(pred_alpha: np.ndarray, gt_alpha: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    """Every metric for one image. ``pred_alpha`` may be float [0,1] or uint8."""
    pa = to_alpha(pred_alpha)
    if pa.shape != gt_mask.shape:
        raise ValueError(f"prediction shape {pa.shape} != ground truth {gt_mask.shape}")
    pm = pa >= 0.5
    gm = to_binary(gt_mask)
    f, p, r = boundary_f(pm, gm)
    f3, _, _ = boundary_f(pm, gm, bound_th=3)
    gt_area = max(int(gm.sum()), 1)
    return {
        "iou": iou(pm, gm),
        "bf": f,
        "bf_precision": p,
        "bf_recall": r,
        "bf_3px": f3,
        "alpha_mae": alpha_mae(pa, gt_alpha),
        "alpha_mae_band": alpha_mae_band(pa, gt_alpha, gm),
        "thin_recall": thin_recall(pm, gm),
        "fp_area": float(np.logical_and(pm, ~gm).sum() / gt_area),
        "fn_area": float(np.logical_and(~pm, gm).sum() / gt_area),
    }


METRIC_KEYS = ("iou", "bf", "bf_3px", "alpha_mae", "alpha_mae_band", "thin_recall", "fp_area", "fn_area")


def aggregate(rows: Iterable[dict], by: str | None = None, keys: Iterable[str] = METRIC_KEYS) -> dict[str, dict[str, float]]:
    """NaN-aware mean of each metric, overall (``"all"``) and per ``by`` group."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        if by:
            groups[str(row[by])].append(row)
    out: dict[str, dict[str, float]] = {}
    for g, items in groups.items():
        stats: dict[str, float] = {"n": float(len(items))}
        for k in keys:
            vals = np.array([it[k] for it in items if k in it], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            stats[k] = float(vals.mean()) if vals.size else float("nan")
        out[g] = stats
    return out


def percentiles(samples: Iterable[float], qs: Iterable[int] = (50, 90, 99)) -> dict[str, float]:
    arr = np.asarray(list(samples), dtype=np.float64)
    if arr.size == 0:
        return {f"p{q}": float("nan") for q in qs}
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}
