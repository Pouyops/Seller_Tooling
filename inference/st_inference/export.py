#!/usr/bin/env python
"""Export matting models to ONNX, verify them against PyTorch, and try TensorRT.

    python -m st_inference.export --models birefnet_lite,inspyrenet_fast --bench
    python -m st_inference.export --models birefnet_lite --no-trt

"Where it helps" is the question, so this measures rather than assumes: for each model it exports,
checks the ONNX output against the PyTorch output on a real benchmark image (max abs difference and
mask IoU), then times PyTorch vs ONNX Runtime CUDA vs ONNX Runtime TensorRT. Failures are recorded,
not hidden — an export that ONNX Runtime cannot run is a result worth writing down.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from st_common.storage import atomic_write_bytes

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def _wrap(model):
    """ONNX wants one tensor out; our models return a list of side outputs."""
    import torch.nn as nn

    class Wrapped(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner.model(x) if hasattr(self.inner, "model") else self.inner(x)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            return out.sigmoid() if out.min() < 0 or out.max() > 1 else out

    return Wrapped(model)


def export_one(name: str, out_dir: Path, size: int, opset: int, device: str, log=print) -> dict:
    import torch

    from .models import create_model

    rec: dict = {"model": name, "opset": opset, "size": size}
    model = create_model(name, device=device, precision="fp32")
    model.load()
    w, h = model.input_size or (size, size)
    wrapper = _wrap(model.model).eval().to(device)
    dummy = torch.randn(1, 3, h, w, device=device)
    path = out_dir / f"{name}_{w}x{h}_op{opset}.onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    # BiRefNet's decoder uses torchvision's deform_conv2d, which neither exporter knows. This MIT
    # helper registers an ONNX symbolic for it (the same approach BiRefNet's own repo documents).
    try:
        import deform_conv2d_onnx_exporter

        deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()
        rec["deform_conv2d_symbolic"] = True
    except Exception as e:
        rec["deform_conv2d_symbolic"] = f"unavailable: {type(e).__name__}"

    # The TorchScript exporter chokes on both of our models (deform_conv2d is unregistered in
    # torchvision 0.26; InSPyReNet trips an internal shape-inference assert), so try the dynamo
    # exporter first and keep the legacy path as a fallback.
    attempts = [("dynamo", dict(dynamo=True, opset_version=opset)),
                ("torchscript", dict(dynamo=False, opset_version=opset,
                                     dynamic_axes={"input": {0: "batch"}, "alpha": {0: "batch"}}))]
    errors = {}
    for label, kwargs in attempts:
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                torch.onnx.export(wrapper, (dummy,), str(path), input_names=["input"], output_names=["alpha"], **kwargs)
            rec["exporter"] = label
            rec["export_s"] = round(time.perf_counter() - t0, 1)
            rec["path"] = str(path)
            rec["size_mb"] = round(path.stat().st_size / 2**20, 1)
            rec["status"] = "exported"
            log(f"[export] {name}: ONNX written by the {label} exporter in {rec['export_s']}s ({rec['size_mb']} MB)")
            break
        except Exception as e:
            errors[label] = f"{type(e).__name__}: {str(e)[:300]}"
            log(f"[export] {name}: {label} exporter failed — {str(e)[:160]}")
    if rec.get("status") != "exported":
        rec.update(status="export_failed", errors=errors, error=errors.get("dynamo", ""))
        model.unload()
        return rec

    # custom ops are the usual reason an exported graph won't run anywhere
    try:
        import onnx

        graph = onnx.load(str(path))
        domains = {n.domain for n in graph.graph.node if n.domain not in ("", "ai.onnx")}
        rec["custom_domains"] = sorted(domains)
        onnx.checker.check_model(graph)
        rec["onnx_check"] = "ok"
    except Exception as e:
        rec["onnx_check"] = f"{type(e).__name__}: {str(e)[:200]}"

    # reference output from PyTorch on a real image
    img = _sample_image(h, w)
    with torch.inference_mode():
        ref = wrapper(torch.from_numpy(img).to(device)).float().cpu().numpy()
    rec["torch_ms"] = _time_torch(wrapper, torch.from_numpy(img).to(device))
    model.unload()
    del wrapper
    torch.cuda.empty_cache()
    rec["reference"] = {"mean": float(ref.mean()), "shape": list(ref.shape)}
    rec["_ref"] = ref
    rec["_input"] = img
    return rec


def _sample_image(h: int, w: int) -> np.ndarray:
    """One real benchmark image, preprocessed the way the service does."""
    import cv2

    from st_eval.synth.__main__ import default_dataset_dir
    from .models.base import IMAGENET_MEAN, IMAGENET_STD

    files = sorted((default_dataset_dir() / "images").glob("*.jpg"))
    if files:
        img = cv2.cvtColor(cv2.imread(str(files[0])), cv2.COLOR_BGR2RGB)
    else:
        img = (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    img = (img - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return img.transpose(2, 0, 1)[None].copy()


def _time_torch(module, tensor, runs: int = 10) -> float:
    import torch

    with torch.inference_mode():
        for _ in range(3):
            module(tensor)
        torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            module(tensor)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 1)


def run_ort(rec: dict, provider: str, runs: int = 10, log=print) -> dict:
    """Load the ONNX graph under one execution provider, check numerics, time it."""
    import onnxruntime as ort

    result: dict = {"provider": provider}
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    providers = [provider] if provider == "CPUExecutionProvider" else [provider, "CPUExecutionProvider"]
    if provider == "TensorrtExecutionProvider":
        providers = [(provider, {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                 "trt_engine_cache_path": str(Path(rec["path"]).parent / "trt_cache")}),
                     "CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        t0 = time.perf_counter()
        sess = ort.InferenceSession(rec["path"], opts, providers=providers)
        result["session_s"] = round(time.perf_counter() - t0, 1)
        used = sess.get_providers()[0]
        result["provider_used"] = used
        if provider != "CPUExecutionProvider" and used != provider:
            result["status"] = "provider_unavailable"
            log(f"[export]   {provider}: unavailable, fell back to {used}")
            return result
        name = sess.get_inputs()[0].name
        out = sess.run(None, {name: rec["_input"]})[0]
        ref = rec["_ref"]
        result["max_abs_diff"] = float(np.abs(out - ref).max())
        pred, gold = out[0, 0] >= 0.5, ref[0, 0] >= 0.5
        union = np.logical_or(pred, gold).sum()
        result["mask_iou"] = float(np.logical_and(pred, gold).sum() / union) if union else 1.0
        times = []
        for _ in range(3):
            sess.run(None, {name: rec["_input"]})
        for _ in range(runs):
            t0 = time.perf_counter()
            sess.run(None, {name: rec["_input"]})
            times.append((time.perf_counter() - t0) * 1000)
        result["ms"] = round(statistics.median(times), 1)
        result["status"] = "ok"
        log(f"[export]   {provider}: {result['ms']} ms, max|Δ|={result['max_abs_diff']:.2e}, IoU={result['mask_iou']:.4f}")
    except Exception as e:
        result.update(status="failed", error=f"{type(e).__name__}: {str(e)[:300]}")
        log(f"[export]   {provider}: FAILED — {str(e)[:200]}")
    return result


def render(report: dict) -> str:
    L = ["# ONNX export and TensorRT\n",
         "> Generated by `python -m st_inference.export`. The brief asks for ONNX \"plus TensorRT where it "
         "helps\" — so this measures whether it helps, and records the failures too.\n"]
    hw = report["hardware"]
    L.append(f"- **Hardware:** {hw.get('gpu')} ({hw.get('gpu_vram_gb')} GB, compute {hw.get('gpu_capability')}) · "
             f"torch {hw.get('torch')} · onnxruntime {report['onnxruntime']}\n")
    L.append("| model | export | ONNX MB | custom ops | PyTorch | ORT CUDA | ORT TensorRT | max abs diff | mask IoU |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for rec in report["models"]:
        runs = {r["provider"]: r for r in rec.get("runs", [])}

        def cell(p):
            r = runs.get(p)
            if not r:
                return "–"
            if r.get("status") == "ok":
                return f"{r['ms']} ms"
            return f"**{r.get('status')}**"

        best = next((r for r in rec.get("runs", []) if r.get("status") == "ok" and "max_abs_diff" in r), None)
        L.append(f"| `{rec['model']}` | {rec.get('status')} | {rec.get('size_mb', '–')} | "
                 f"{', '.join(rec.get('custom_domains') or []) or 'none'} | {rec.get('torch_ms', '–')} ms | "
                 f"{cell('CUDAExecutionProvider')} | {cell('TensorrtExecutionProvider')} | "
                 f"{best['max_abs_diff']:.2e} | {best['mask_iou']:.4f} |" if best else
                 f"| `{rec['model']}` | {rec.get('status')} | {rec.get('size_mb', '–')} | "
                 f"{', '.join(rec.get('custom_domains') or []) or 'none'} | {rec.get('torch_ms', '–')} ms | "
                 f"{cell('CUDAExecutionProvider')} | {cell('TensorrtExecutionProvider')} | – | – |")
    L.append("")
    for rec in report["models"]:
        if rec.get("error"):
            L.append(f"**`{rec['model']}` export failed:** `{rec['error']}`\n")
        for r in rec.get("runs", []):
            if r.get("status") not in ("ok", None):
                L.append(f"**`{rec['model']}` / {r['provider']}:** {r.get('status')} — `{r.get('error', '')}`\n")
    analysis = Path("docs/onnx-analysis.md")
    if analysis.exists():
        L.append("---\n")
        L.append(analysis.read_text(encoding="utf-8"))
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="birefnet_lite,inspyrenet_fast")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-trt", action="store_true")
    ap.add_argument("--out", default="docs/onnx-tensorrt.md")
    ap.add_argument("--json-out", default="docs/onnx-tensorrt.json")
    args = ap.parse_args()

    import onnxruntime as ort

    from .device import hardware_info
    from .settings import get_settings

    out_dir = get_settings().data_dir / "onnx"
    report = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "hardware": hardware_info(),
              "onnxruntime": ort.__version__, "available_providers": ort.get_available_providers(), "models": []}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"[export] {name}")
        try:
            rec = export_one(name, out_dir, args.size, args.opset, args.device)
        except Exception as e:
            rec = {"model": name, "status": "crashed", "error": f"{type(e).__name__}: {str(e)[:300]}",
                   "trace": traceback.format_exc(limit=4)}
        rec["runs"] = []
        if rec.get("status") == "exported":
            for provider in ["CUDAExecutionProvider"] + ([] if args.no_trt else ["TensorrtExecutionProvider"]):
                rec["runs"].append(run_ort(rec, provider))
        rec.pop("_ref", None)
        rec.pop("_input", None)
        report["models"].append(rec)

    atomic_write_bytes(Path(args.json_out), json.dumps(report, indent=2, default=str).encode())
    atomic_write_bytes(Path(args.out), render(report).encode("utf-8"))
    print(f"[export] wrote {args.out}")


if __name__ == "__main__":
    main()
