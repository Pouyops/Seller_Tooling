import math

import numpy as np
import pytest

from st_eval import metrics as M


def square(h=200, w=200, top=50, left=50, size=100):
    m = np.zeros((h, w), bool)
    m[top : top + size, left : left + size] = True
    return m


def test_to_binary_and_alpha_dtypes():
    u8 = np.array([[0, 127, 128, 255]], np.uint8)
    assert M.to_binary(u8).tolist() == [[False, False, True, True]]
    f = np.array([[0.0, 0.49, 0.5, 1.0]], np.float32)
    assert M.to_binary(f).tolist() == [[False, False, True, True]]
    assert M.to_alpha(u8)[0, 3] == 1.0
    assert M.to_alpha(np.array([[True]]))[0, 0] == 1.0
    assert M.to_alpha(np.array([[1.7]], np.float64))[0, 0] == 1.0


def test_iou_basic_cases():
    a = square()
    assert M.iou(a, a) == 1.0
    assert M.iou(a, ~a) == 0.0
    b = square(left=100)  # overlaps half
    assert math.isclose(M.iou(a, b), 5000 / 15000)
    empty = np.zeros_like(a)
    assert M.iou(empty, empty) == 1.0


def test_boundary_map_is_one_pixel_ring():
    b = M.boundary_map(square(size=10))
    assert b.sum() == 36  # 10x10 perimeter


def test_boundary_f_identical_and_small_shift_within_tolerance():
    a = square()
    assert M.boundary_f(a, a) == (1.0, 1.0, 1.0)
    shifted = square(top=51, left=51)
    f, p, r = M.boundary_f(shifted, a, bound_th=2)
    assert f == 1.0 and p == 1.0 and r == 1.0


def test_boundary_f_large_shift_is_penalized():
    a = square()
    far = square(top=70, left=70)
    f, _, _ = M.boundary_f(far, a, bound_th=3)
    assert f < 0.3


def test_boundary_f_tolerance_scales_with_diagonal():
    a = square(h=1000, w=1000, top=200, left=200, size=500)
    off = square(h=1000, w=1000, top=208, left=208, size=500)
    # 0.008 * diag(1414) = 12 px tolerance -> shift of 8 px is fine; 3 px strict is not
    assert M.boundary_f(off, a)[0] == 1.0
    assert M.boundary_f(off, a, bound_th=3)[0] < 0.2


def test_boundary_f_empty_prediction():
    a = square()
    assert M.boundary_f(np.zeros_like(a), a)[0] == 0.0
    assert M.boundary_f(np.zeros_like(a), np.zeros_like(a))[0] == 1.0


def test_thin_structures_are_invisible_to_iou_but_not_thin_recall():
    gt = square(h=300, w=300, top=50, left=50, size=150)
    for x in range(60, 190, 10):  # comb of 2-px "fringe" threads
        gt[200:260, x : x + 2] = True
    pred = square(h=300, w=300, top=50, left=50, size=150)  # body only, fringe lost
    assert M.iou(pred, gt) > 0.9
    assert M.thin_recall(pred, gt) < 0.05  # only the square's corners count as "thin"
    assert M.thin_recall(gt, gt) == 1.0
    assert math.isnan(M.thin_recall(square(), square()))  # no thin parts at all


def test_alpha_errors():
    gt = square().astype(np.float32)
    pred = gt.copy()
    pred[gt == 1] = 0.9
    assert math.isclose(M.alpha_mae(pred, gt), 0.1 * gt.mean(), rel_tol=1e-5)
    band = M.alpha_mae_band(pred, gt, gt > 0.5, width=3)
    assert 0 < band < 0.1  # band includes background pixels with zero error
    assert math.isnan(M.alpha_mae_band(pred, np.zeros_like(gt), np.zeros_like(gt, bool)))


def test_compute_all_keys_and_ranges():
    gt_mask = square()
    gt_alpha = gt_mask.astype(np.float32)
    pred = (square(top=52, left=52) * 255).astype(np.uint8)
    out = M.compute_all(pred, gt_alpha, gt_mask)
    for k in M.METRIC_KEYS:
        assert k in out
    assert 0.9 < out["iou"] < 1.0
    assert out["fp_area"] > 0 and out["fn_area"] > 0
    with pytest.raises(ValueError):
        M.compute_all(pred[:10], gt_alpha, gt_mask)


def test_aggregate_nan_aware_grouping():
    rows = [
        {"category": "carpet", "iou": 0.9, "thin_recall": 0.5},
        {"category": "carpet", "iou": 0.7, "thin_recall": float("nan")},
        {"category": "glass", "iou": 0.5, "thin_recall": float("nan")},
    ]
    agg = M.aggregate(rows, by="category", keys=("iou", "thin_recall"))
    assert math.isclose(agg["all"]["iou"], 0.7)
    assert math.isclose(agg["carpet"]["thin_recall"], 0.5)
    assert math.isnan(agg["glass"]["thin_recall"])
    assert agg["carpet"]["n"] == 2


def test_percentiles():
    p = M.percentiles(range(1, 101))
    assert math.isclose(p["p50"], 50.5)
    assert p["p99"] > p["p90"] > p["p50"]
    assert math.isnan(M.percentiles([])["p50"])
