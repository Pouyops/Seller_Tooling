"""Benchmark runner: accuracy on the benchmark set + latency/VRAM/throughput sweeps.

    python -m st_eval.bench --models birefnet_lite,birefnet --dataset <dir> --out eval/runs/<id>

Model spec syntax: ``name[:precision][@size][#max_images]``, e.g. ``birefnet_lite:fp32``,
``birefnet_dynamic@1536`` or ``birefnet_hr#24`` (category-balanced subset for very slow models;
the report marks such rows as partial).

Resumable: each model's result is written to ``<out>/models/<spec>.json`` as soon as it finishes;
re-running the same command skips completed models (``--force`` to redo). A power cut costs at
most the model that was running.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from st_common.storage import atomic_write_bytes

from .metrics import METRIC_KEYS, aggregate, compute_all, percentiles

DEFAULT_MODELS = [
    "birefnet_lite",
    "birefnet",
    "birefnet_dynamic",
    "ben2_base",
    "inspyrenet_base",
    "inspyrenet_fast",
    "birefnet_hr#24",
]
DEFAULT_BATCH_SIZES = (1, 4, 8, 16)
_SPEC_RE = re.compile(r"^(?P<name>[\w.-]+)(?::(?P<precision>\w+))?(?:@(?P<size>\d+))?(?:#(?P<max>\d+))?$")


def parse_spec(spec: str) -> dict:
    m = _SPEC_RE.match(spec.strip())
    if not m:
        raise ValueError(f"bad model spec {spec!r}; expected name[:precision][@size][#max_images]")
    size = int(m["size"]) if m["size"] else None
    return {
        "spec": spec.strip(),
        "name": m["name"],
        "precision": m["precision"],
        "size": (size, size) if size else None,
        "max_images": int(m["max"]) if m["max"] else None,
    }


def safe_name(spec: str) -> str:
    return spec.replace(":", "__").replace("@", "_at").replace("#", "_n")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


class VramSampler:
    """Polls NVML for device memory in a background thread; reports the peak above a baseline."""

    def __init__(self, interval_s: float = 0.01):
        from st_inference.device import vram_used_mb

        self._read = vram_used_mb
        self.interval_s = interval_s
        self.baseline = self._read() or 0.0
        self.peak = self.baseline
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def __enter__(self):
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            v = self._read()
            if v is not None and v > self.peak:
                self.peak = v
            time.sleep(self.interval_s)

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join()

    @property
    def peak_above_baseline_mb(self) -> float:
        return max(0.0, self.peak - self.baseline)


def balanced_subset(metas: list[dict], limit: int | None) -> list[dict]:
    if not limit or limit >= len(metas):
        return metas
    by_cat: dict[str, list[dict]] = {}
    for m in metas:
        by_cat.setdefault(m["category"], []).append(m)
    per = max(1, limit // len(by_cat))
    return [m for items in by_cat.values() for m in items[:per]]


def load_dataset(dataset_dir: Path, limit: int | None = None) -> list[dict]:
    metas = [json.loads(l) for l in (dataset_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return balanced_subset(metas, limit)


def read_sample(dataset_dir: Path, meta: dict):
    img = cv2.cvtColor(cv2.imread(str(dataset_dir / "images" / f"{meta['id']}.jpg")), cv2.COLOR_BGR2RGB)
    alpha = cv2.imread(str(dataset_dir / "alpha" / f"{meta['id']}.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(dataset_dir / "mask" / f"{meta['id']}.png"), cv2.IMREAD_GRAYSCALE) > 0
    return img, alpha, mask


def _is_oom(e: BaseException) -> bool:
    import torch

    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def _vram_total_mb() -> float:
    import torch

    return torch.cuda.get_device_properties(0).total_memory / 2**20


SPEED_KEYS = ("latency_bs1_ms", "throughput", "accuracy_peak_torch_mb", "accuracy_peak_device_mb", "vram_spill_bs1",
              "load_s", "precision", "precision_note", "weights_vram_mb", "vram_total_mb")


def bench_model(spec: str, dataset_dir: Path, metas: list[dict], batch_sizes, out_dir: Path, *,
                models_dir: Path | None, device: str, sweep_seconds: float, save_previews: int, log=print,
                speed_only: bool = False) -> dict:
    import torch

    from st_inference.models import create_model

    p = parse_spec(spec)
    metas = balanced_subset(metas, p["max_images"])
    result: dict = {"spec": spec, "name": p["name"], "status": "ok", "started_at": time.time(),
                    "n_images": len(metas), "partial": p["max_images"] is not None}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram_total = _vram_total_mb()
    model = create_model(p["name"], models_dir=models_dir, device=device, precision=p["precision"], input_size=p["size"])
    result.update(requested_precision=model.precision, input_size=model.input_size, repo_id=model.repo_id)

    t0 = time.perf_counter()
    try:
        model.load()
    except Exception as e:
        result.update(status="load_failed", error=f"{type(e).__name__}: {e}")
        log(f"[bench] {spec}: load failed: {e}")
        return result
    result["load_s"] = round(time.perf_counter() - t0, 2)
    result["precision"] = model.precision
    result["precision_note"] = model.precision_note
    result["weights_vram_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)
    result["vram_total_mb"] = round(vram_total, 0)
    if model.precision_note:
        log(f"[bench] {spec}: {model.precision_note}")

    # ---- accuracy pass (batch size 1, real images) -----------------------------------------
    rows: list[dict] = []
    timings: list[dict] = []
    preview_dir = out_dir / "previews" / safe_name(spec)
    try:
        warm_img, _, _ = read_sample(dataset_dir, metas[0])
        for _ in range(2):
            model.predict([warm_img])
        torch.cuda.reset_peak_memory_stats()
        sampler = VramSampler()
        with sampler:
            for i, meta in enumerate(metas):
                img, gt_alpha, gt_mask = read_sample(dataset_dir, meta)
                pred = model.predict([img])[0]
                if not np.isfinite(pred).all():
                    raise FloatingPointError(f"non-finite prediction on {meta['id']}")
                t = model.last_timing
                timings.append({"pre": t.preprocess_ms, "fwd": t.forward_ms, "post": t.postprocess_ms, "total": t.total_ms})
                if speed_only:
                    continue
                rows.append({"id": meta["id"], "category": meta["category"], "subtype": meta["subtype"], "tags": meta["tags"],
                             **compute_all(pred, gt_alpha, gt_mask), "latency_ms": t.total_ms})
                if i < save_previews or meta["id"].endswith("_000"):
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    small = cv2.resize((pred * 255).astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(preview_dir / f"{meta['id']}.png"), small)
                if (i + 1) % 25 == 0:
                    log(f"[bench] {spec}: {i + 1}/{len(metas)} iou={np.mean([r['iou'] for r in rows]):.3f} "
                        f"lat={np.median([t['total'] for t in timings]):.0f}ms")
        result["accuracy_peak_torch_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        result["accuracy_peak_device_mb"] = round(sampler.peak_above_baseline_mb, 1)
        result["vram_spill_bs1"] = result["accuracy_peak_torch_mb"] > 0.97 * vram_total
    except Exception as e:
        if _is_oom(e):
            result.update(status="oom_bs1", error=str(e)[:500])
        else:
            result.update(status="accuracy_failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc(limit=5))
        log(f"[bench] {spec}: accuracy pass failed after {len(rows)} images: {str(e)[:200]}")
        model.unload()
        result["per_image"] = rows
        return result

    if not speed_only:
        result["per_image"] = rows
        result["metrics"] = aggregate(rows, by="category")
        tag_rows = [{**r, "tag": t} for r in rows for t in r["tags"]]
        result["metrics_by_tag"] = aggregate(tag_rows, by="tag")
        result["metrics_by_tag"].pop("all", None)
        cats = [k for k in result["metrics"] if k != "all"]
        result["metrics_balanced"] = {k: float(np.nanmean([result["metrics"][c][k] for c in cats])) for k in METRIC_KEYS}
    result["latency_bs1_ms"] = {
        "total": percentiles(t["total"] for t in timings),
        "forward": percentiles(t["fwd"] for t in timings),
        "preprocess_p50": float(np.median([t["pre"] for t in timings])),
        "postprocess_p50": float(np.median([t["post"] for t in timings])),
        "n": len(timings),
    }

    # ---- throughput sweep ------------------------------------------------------------------
    sweep = {}
    pool = [read_sample(dataset_dir, m)[0] for m in metas[: max(batch_sizes)]]
    stop_reason = None
    for bs in batch_sizes:
        if stop_reason is not None:
            sweep[str(bs)] = {"status": stop_reason}
            continue
        batch = [pool[i % len(pool)] for i in range(bs)]
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model.predict(batch)  # warmup at this shape
            lat = []
            sampler = VramSampler()
            with sampler:
                t_end = time.perf_counter() + sweep_seconds
                while len(lat) < 3 or (time.perf_counter() < t_end and len(lat) < 200):
                    model.predict(batch)
                    lat.append(model.last_timing.total_ms)
            mean_ms = float(np.mean(lat))
            peak = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            spill = peak > 0.97 * vram_total
            sweep[str(bs)] = {
                "status": "ok",
                "iters": len(lat),
                "batch_latency_ms": percentiles(lat),
                "per_image_ms": mean_ms / bs,
                "images_per_s": 1000.0 * bs / mean_ms,
                "peak_torch_mb": peak,
                "peak_device_mb": round(sampler.peak_above_baseline_mb, 1),
                "vram_spill": spill,
            }
            log(f"[bench] {spec}: bs={bs} {sweep[str(bs)]['images_per_s']:.2f} img/s, peak {peak:.0f} MB{' (SPILL)' if spill else ''}")
            if spill:
                # WDDM lets CUDA overflow into system RAM instead of OOM-ing; bigger batches only get slower.
                stop_reason = "skipped_after_vram_spill"
        except Exception as e:
            if not _is_oom(e):
                sweep[str(bs)] = {"status": "failed", "error": f"{type(e).__name__}: {str(e)[:300]}"}
                continue
            stop_reason = "skipped_after_oom"
            sweep[str(bs)] = {"status": "oom"}
            log(f"[bench] {spec}: bs={bs} OOM")
            torch.cuda.empty_cache()
    result["throughput"] = sweep
    result["finished_at"] = time.time()
    model.unload()
    del model
    torch.cuda.empty_cache()
    return result


def run(models: list[str], dataset_dir: Path, out_dir: Path, *, batch_sizes=DEFAULT_BATCH_SIZES, limit: int | None = None,
        models_dir: Path | None = None, device: str = "cuda", sweep_seconds: float = 20.0, force: bool = False,
        save_previews: int = 0, log=print, speed_only: bool = False, speed_images: int = 60) -> dict:
    from st_inference.device import hardware_info

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    metas = load_dataset(dataset_dir, limit)
    dataset_info = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    run_info = {
        "dataset_dir": str(dataset_dir),
        "dataset": dataset_info,
        "n_images": len(metas),
        "limit": limit,
        "batch_sizes": list(batch_sizes),
        "sweep_seconds": sweep_seconds,
        "hardware": hardware_info(),
        "git_commit": git_commit(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_order": models,
    }
    if speed_only and (out_dir / "run.json").exists():
        # keep the original run metadata, record that speed was re-measured on an idle machine
        prev = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
        prev["speed_rerun"] = {"at": run_info["created_at"], "git_commit": run_info["git_commit"], "images": speed_images,
                               "hardware": run_info["hardware"], "sweep_seconds": sweep_seconds}
        run_info = prev
    atomic_write_bytes(out_dir / "run.json", json.dumps(run_info, indent=2).encode())
    results = {}
    for spec in models:
        path = out_dir / "models" / f"{safe_name(spec)}.json"
        if speed_only:
            if not path.exists():
                log(f"[bench] {spec}: no accuracy result yet; run without --speed-only first")
                continue
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("status") != "ok":
                results[spec] = existing
                continue
            subset = balanced_subset(metas, min(speed_images, existing.get("n_images") or speed_images))
            log(f"[bench] {spec}: re-measuring speed on {len(subset)} images")
            res = bench_model(spec, dataset_dir, subset, batch_sizes, out_dir, models_dir=models_dir, device=device,
                              sweep_seconds=sweep_seconds, save_previews=0, log=log, speed_only=True)
            if res.get("status") == "ok":
                existing.update({k: res[k] for k in SPEED_KEYS if k in res})
                existing["speed_measured_on"] = {"images": len(subset), "at": time.strftime("%Y-%m-%d %H:%M:%S")}
            else:
                existing["speed_rerun_error"] = res.get("error")
            atomic_write_bytes(path, json.dumps(existing, default=float).encode())
            results[spec] = existing
            continue
        if path.exists() and not force:
            log(f"[bench] {spec}: already done, skipping ({path.name})")
            results[spec] = json.loads(path.read_text(encoding="utf-8"))
            continue
        log(f"[bench] {spec}: starting")
        res = bench_model(spec, dataset_dir, metas, batch_sizes, out_dir, models_dir=models_dir, device=device,
                          sweep_seconds=sweep_seconds, save_previews=save_previews, log=log)
        atomic_write_bytes(path, json.dumps(res, default=float).encode())
        results[spec] = res
    return {"run": run_info, "models": results}


def main() -> None:
    from .synth.__main__ import default_dataset_dir

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("eval/runs/latest"))
    ap.add_argument("--batch-sizes", default=",".join(map(str, DEFAULT_BATCH_SIZES)))
    ap.add_argument("--limit", type=int, default=None, help="images (category-balanced) for a quick run")
    ap.add_argument("--sweep-seconds", type=float, default=20.0)
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--speed-only", action="store_true",
                    help="re-measure latency/throughput for models that already have accuracy results (idle machine)")
    ap.add_argument("--speed-images", type=int, default=60)
    ap.add_argument("--report", type=Path, default=Path("eval/results.md"))
    args = ap.parse_args()

    dataset = args.dataset or default_dataset_dir()
    if not (dataset / "manifest.jsonl").exists():
        raise SystemExit(f"dataset not found at {dataset}; run `python -m st_eval.synth` first")
    res = run([m.strip() for m in args.models.split(",") if m.strip()], dataset, args.out,
              batch_sizes=tuple(int(b) for b in args.batch_sizes.split(",")), limit=args.limit,
              models_dir=args.models_dir, sweep_seconds=args.sweep_seconds, force=args.force)
    from .report import render_report

    md = render_report(res, analysis_path=Path(os.environ.get("ST_ANALYSIS", "eval/analysis.md")))
    atomic_write_bytes(args.report, md.encode("utf-8"))
    print(f"[bench] report written to {args.report}")


if __name__ == "__main__":
    main()
