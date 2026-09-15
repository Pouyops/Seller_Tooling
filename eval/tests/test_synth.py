import json

import cv2
import numpy as np
import pytest

from st_eval.synth import CATEGORY_COUNTS, generate_dataset, generate_sample
from st_eval.synth.generate import load_manifest, plan_dataset

SMALL = (192, 240)


def decode(sample):
    img = cv2.imdecode(np.frombuffer(sample["image"], np.uint8), cv2.IMREAD_COLOR)
    alpha = cv2.imdecode(np.frombuffer(sample["alpha"], np.uint8), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imdecode(np.frombuffer(sample["mask"], np.uint8), cv2.IMREAD_GRAYSCALE)
    return img, alpha, mask


@pytest.mark.parametrize("category", list(CATEGORY_COUNTS))
@pytest.mark.parametrize("index", [0, 1, 2])
def test_every_category_renders_consistent_ground_truth(category, index):
    s = generate_sample(index, category, seed=7, size=SMALL)
    img, alpha, mask = decode(s)
    assert img.shape == (SMALL[1], SMALL[0], 3)
    assert alpha.shape == mask.shape == (SMALL[1], SMALL[0])
    assert set(np.unique(mask)) <= {0, 255}
    fg = (mask > 0).mean()
    assert 0.01 < fg < 0.97, fg
    # silhouette pixels must have non-zero opacity (glass is faint but never zero)
    assert ((mask > 0) & (alpha == 0)).mean() < 0.005
    meta = s["meta"]
    assert meta["category"] == category and meta["subtype"]
    assert isinstance(meta["tags"], list)  # a plain box on a plain surface legitimately has no tags


def test_deterministic_and_seed_sensitive():
    a = generate_sample(3, "jewelry", seed=11, size=SMALL)
    b = generate_sample(3, "jewelry", seed=11, size=SMALL)
    c = generate_sample(3, "jewelry", seed=12, size=SMALL)
    assert a["image"] == b["image"] and a["alpha"] == b["alpha"]
    assert a["alpha"] != c["alpha"]


def test_glass_alpha_is_partial_but_mask_is_full_silhouette():
    for i in range(6):
        s = generate_sample(i, "glassware", seed=5, size=SMALL)
        _, alpha, mask = decode(s)
        inside = alpha[mask > 0]
        if (inside < 200).mean() > 0.3:
            return
    pytest.fail("glassware never produced partially transparent silhouettes")


def test_plan_keeps_proportions():
    plan = plan_dataset(200)
    cats = [c for _, c, _ in plan]
    assert {c: cats.count(c) for c in CATEGORY_COUNTS} == CATEGORY_COUNTS
    assert len(plan_dataset(12)) == 12
    assert len({sid for *_, sid in plan}) == 200


def test_generate_dataset_is_resumable(tmp_path):
    out = generate_dataset(tmp_path / "ds", n=6, seed=3, workers=1, size=(96, 120), log=lambda *_: None)
    manifest = load_manifest(out)
    assert len(manifest) == 6
    summary = json.loads((out / "dataset.json").read_text())
    assert summary["n"] == 6
    victim = manifest[2]["id"]
    keep = manifest[0]["id"]
    keep_mtime = (out / "images" / f"{keep}.jpg").stat().st_mtime_ns
    (out / "meta" / f"{victim}.json").unlink()
    generate_dataset(out, n=6, seed=3, workers=1, size=(96, 120), log=lambda *_: None)
    assert (out / "meta" / f"{victim}.json").exists()
    assert (out / "images" / f"{keep}.jpg").stat().st_mtime_ns == keep_mtime
