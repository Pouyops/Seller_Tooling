import json

import pytest

from st_eval.bench import balanced_subset, parse_spec, safe_name
from st_eval.report import best_throughput, load_run, pareto, quality_score, render_report

CATS = ["carpet", "jewelry", "saffron_packaged", "glassware", "handicraft", "clothing_hanger"]


def fake_result(spec, q, ips, partial=False, spill_bs=None, precision="fp32", note=None):
    metrics = {c: {"n": 5, "iou": q, "bf": q, "bf_3px": q, "thin_recall": q, "alpha_mae": 0.01, "alpha_mae_band": 0.1,
                   "fp_area": 0.0, "fn_area": 0.0} for c in CATS}
    metrics["all"] = dict(metrics["carpet"], n=30)
    thr = {}
    for bs in (1, 4):
        thr[str(bs)] = {"status": "ok", "images_per_s": ips * (1.5 if bs == 4 else 1), "peak_torch_mb": 900.0 * bs,
                        "vram_spill": spill_bs == bs}
    return {
        "spec": spec, "status": "ok", "precision": precision, "requested_precision": "fp16", "precision_note": note,
        "input_size": [1024, 1024], "n_images": 30, "partial": partial,
        "metrics": metrics, "metrics_balanced": {k: v for k, v in metrics["carpet"].items() if k != "n"},
        "metrics_by_tag": {"thin_structures": {"n": 12, "iou": q, "bf_3px": q}},
        "latency_bs1_ms": {"total": {"p50": 100, "p90": 120, "p99": 150}, "forward": {"p50": 80, "p90": 90, "p99": 99},
                           "preprocess_p50": 5, "postprocess_p50": 10, "n": 30},
        "throughput": thr, "weights_vram_mb": 200, "accuracy_peak_torch_mb": 900, "accuracy_peak_device_mb": 1000, "load_s": 3,
    }


def bundle(results):
    return {
        "run": {"created_at": "2026-09-15", "git_commit": "abc", "n_images": 30, "batch_sizes": [1, 4], "sweep_seconds": 5,
                "dataset": {"generator_version": "synth-v1", "seed": 1, "counts": {"carpet": 5}},
                "hardware": {"gpu": "NVIDIA GeForce GTX 1650", "gpu_vram_gb": 4.0}},
        "models": results,
    }


def test_parse_spec_variants():
    assert parse_spec("birefnet_lite") == {"spec": "birefnet_lite", "name": "birefnet_lite", "precision": None, "size": None, "max_images": None}
    p = parse_spec("birefnet_dynamic:fp32@1536#24")
    assert (p["name"], p["precision"], p["size"], p["max_images"]) == ("birefnet_dynamic", "fp32", (1536, 1536), 24)
    with pytest.raises(ValueError):
        parse_spec("bad spec!")
    assert safe_name("a:fp32@512#4") == "a__fp32_at512_n4"


def test_balanced_subset():
    metas = [{"category": c, "id": f"{c}_{i}"} for c in CATS for i in range(10)]
    sub = balanced_subset(metas, 12)
    assert len(sub) == 12 and {m["category"] for m in sub} == set(CATS)
    assert balanced_subset(metas, None) is metas


def test_quality_and_throughput_ignore_spill():
    r = fake_result("m", 0.8, 2.0, spill_bs=4)
    assert quality_score(r) == pytest.approx(0.8)
    assert best_throughput(r) == (2.0, "1")


def test_pareto_excludes_dominated_and_partial():
    res = [("a", fake_result("a", 0.9, 1.0)), ("b", fake_result("b", 0.8, 3.0)), ("c", fake_result("c", 0.7, 2.0)),
           ("d", fake_result("d", 0.95, 5.0, partial=True))]
    assert pareto(res) == ["a", "b"]


def test_render_report_contains_all_sections_and_flags():
    results = {
        "fast": fake_result("fast", 0.7, 5.0),
        "good": fake_result("good", 0.9, 1.0, spill_bs=4, note="fp16 requested but disabled"),
        "huge#6": fake_result("huge#6", 0.92, 0.1, partial=True),
        "broken": {"spec": "broken", "status": "load_failed", "error": "boom"},
    }
    md = render_report(bundle(results))
    for heading in ("## 1. Ranking", "## 2. Per category", "## 3. Hard-case", "## 4. Latency", "## 5. Throughput", "## 6. Automatic"):
        assert heading in md
    assert "Not the target hardware" in md
    assert "fp16 requested but disabled" in md
    assert "(partial)" in md and "load_failed" in md and "spill" in md
    ranking = md.split("## 1.")[1].split("## 2.")[0]
    assert ranking.index("`huge#6`") < ranking.index("`good`") < ranking.index("`fast`")
    assert "Throughput leader (fits in VRAM):** `fast`" in md


def test_render_appends_analysis(tmp_path):
    a = tmp_path / "analysis.md"
    a.write_text("## Analysis\nhand-written", encoding="utf-8")
    assert "hand-written" in render_report(bundle({"x": fake_result("x", 0.5, 1)}), a)


def test_load_run_roundtrip(tmp_path):
    (tmp_path / "models").mkdir()
    b = bundle({})
    b["run"]["models_order"] = ["z", "y"]
    (tmp_path / "run.json").write_text(json.dumps(b["run"]))
    for spec in ("y", "z", "extra"):
        (tmp_path / "models" / f"{spec}.json").write_text(json.dumps(fake_result(spec, 0.5, 1)))
    loaded = load_run(tmp_path)
    assert list(loaded["models"]) == ["z", "y", "extra"]
