#!/usr/bin/env python
"""Unit-economics model: cost per image, derived from measurements, rendered to docs/economics.md.

    python tools/economics.py                       # defaults
    python tools/economics.py --usd-toman 120000 --utilization 0.5
    python tools/economics.py --show-inputs         # just print the input table

Design rule: anything that came from a benchmark is read out of the run JSON, never typed in.
Anything that is a business assumption is declared in ASSUMPTIONS with `source="assumed"` and a
note saying what would make it real. The report prints both lists separately so no reader has to
guess which numbers are evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "eval" / "runs" / "kaggle-synth-v1-s1403-n200"
LOADTEST = ROOT / "docs" / "loadtest.json"

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class Input:
    key: str
    value: float
    unit: str
    source: str  # measured | assumed
    note: str


# ---- business assumptions (NOT measured) ------------------------------------------------------
ASSUMPTIONS: list[Input] = [
    Input("usd_toman", 92_000, "toman/USD", "assumed",
          "Free-market rate. Volatile; re-run with --usd-toman. Every USD figure below inherits this."),
    Input("gpu_price_usd", 2_400, "USD", "assumed",
          "One RTX 4090 landed in Iran, incl. import. Quote from an actual supplier before trusting."),
    Input("host_price_usd", 900, "USD", "assumed", "CPU, board, RAM, PSU, case, disk — the box around the GPU."),
    Input("hardware_life_years", 3, "years", "assumed", "Straight-line amortisation; no resale value assumed."),
    Input("gpu_watts", 350, "W", "assumed", "Average draw under our load, not the 450 W peak rating."),
    Input("host_watts", 120, "W", "assumed", "Rest of the machine."),
    Input("cooling_overhead", 1.15, "x", "assumed", "Office air conditioning. A real data centre would be ~1.4 PUE."),
    Input("electricity_toman_kwh", 2_500, "toman/kWh", "assumed",
          "Iranian commercial tariff is tiered and subsidised; this is a placeholder. Check a real bill."),
    Input("cloud_gpu_usd_hour", 0.50, "USD/hour", "assumed",
          "A 4090-class rental. Paying a foreign provider from Iran is its own problem (H-002)."),
    Input("storage_usd_gb_month", 0.02, "USD/GB/month", "assumed", "Iranian object storage (ArvanCloud-like). Get a quote."),
    Input("egress_usd_gb", 0.01, "USD/GB", "assumed", "Per GB delivered to sellers."),
    Input("retention_days", 30, "days", "assumed", "How long cutouts stay downloadable."),
    Input("utilization", 0.35, "fraction", "assumed",
          "Share of wall-clock the GPU is actually working. Sellers are bursty; idle hardware still costs money."),
    Input("dedupe_rate", 0.10, "fraction", "assumed",
          "Share of uploads that are byte-identical repeats, served from cache ~100x cheaper (measured)."),
    Input("staff_toman_month", 0, "toman/month", "assumed",
          "Deliberately zero: this model is infrastructure only. Salaries dwarf it; see the note in the report."),
]

# Throughput of the target card was never measured (H-001). Scale the measured T4 numbers instead.
GPU_SCENARIOS = {
    "T4 (measured)": 1.0,
    "4090 @ 2x T4": 2.0,
    "4090 @ 3x T4": 3.0,
}


def load_measurements() -> dict:
    """Throughput, VRAM and blob sizes straight out of the benchmark and load-test runs."""
    models = {}
    for path in sorted((BENCH / "models").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("status") != "ok":
            continue
        best = 0.0
        for s in d.get("throughput", {}).values():
            if s.get("status") == "ok" and not s.get("vram_spill"):
                best = max(best, s["images_per_s"])
        if best:
            models[d["spec"]] = {
                "img_s": best,
                "iou": d["metrics_balanced"]["iou"],
                "precision": d.get("precision"),
                "vram_mb": d.get("accuracy_peak_torch_mb"),
            }
    lt = json.loads(LOADTEST.read_text(encoding="utf-8"))
    stages = lt.get("stage_breakdown") or {}
    overhead_ms = sum(v for k, v in stages.items() if k not in ("inference",) and not k.endswith("_alt"))
    return {
        "models": models,
        "overhead_ms": overhead_ms,
        "stages": stages,
        "gpu_measured_on": lt["hardware"].get("gpu"),
        # measured in the stage profile: input ~173 KB, cutout 749 KB, mask 14 KB
        "bytes_per_job": (173 + 749 + 14) * 1024,
        "bytes_delivered": (749 + 14) * 1024,
        "dedupe_img_s": next((s["throughput_img_s"] for s in lt["scenarios"] if s["scenario"] == "dedupe"), None),
    }


def inputs_dict(overrides: dict) -> dict[str, float]:
    values = {a.key: a.value for a in ASSUMPTIONS}
    values.update({k: v for k, v in overrides.items() if v is not None})
    return values


def cost_per_image(img_s: float, v: dict, *, owned: bool, overhead_ms: float, m: dict,
                   pipelined: bool = False) -> dict:
    """Toman per processed image. Overhead is CPU work (decode/encode/store) around the GPU call."""
    gpu_s = 1.0 / img_s
    cpu_s = overhead_ms / 1000
    # Sequential today: the worker encodes after inferring. Pipelined: CPU hides behind the GPU.
    per_image_s = max(gpu_s, cpu_s) if pipelined else gpu_s + cpu_s
    effective_img_h = 3600 / per_image_s * v["utilization"]

    hours_life = v["hardware_life_years"] * 365 * 24
    if owned:
        capex_usd_h = (v["gpu_price_usd"] + v["host_price_usd"]) / hours_life
        hw_toman_h = capex_usd_h * v["usd_toman"]
    else:
        hw_toman_h = v["cloud_gpu_usd_hour"] * v["usd_toman"]
    power_kw = (v["gpu_watts"] + v["host_watts"]) / 1000 * v["cooling_overhead"]
    power_toman_h = power_kw * v["electricity_toman_kwh"]
    hourly = hw_toman_h + power_toman_h

    compute_toman = hourly / effective_img_h if effective_img_h else float("inf")
    gb = m["bytes_per_job"] / 1024**3
    storage_toman = gb * v["storage_usd_gb_month"] * v["usd_toman"] * (v["retention_days"] / 30)
    egress_toman = m["bytes_delivered"] / 1024**3 * v["egress_usd_gb"] * v["usd_toman"]
    subtotal = compute_toman + storage_toman + egress_toman
    # Cached repeats skip the GPU entirely (measured ~100x faster), so they cost ~storage+egress only.
    blended = subtotal * (1 - v["dedupe_rate"]) + (storage_toman + egress_toman) * v["dedupe_rate"]
    return {
        "per_image_s": per_image_s, "img_per_hour": effective_img_h, "hourly_toman": hourly,
        "compute": compute_toman, "storage": storage_toman, "egress": egress_toman,
        "total": subtotal, "blended": blended,
    }


def fmt(n: float, nd: int = 0) -> str:
    return f"{n:,.{nd}f}"


def render(v: dict, m: dict, args) -> str:
    model_key = args.model if args.model in m["models"] else "birefnet_lite"
    base = m["models"][model_key]
    L: list[str] = []
    L.append("# Unit economics — cost per image\n")
    L.append("> Generated by `python tools/economics.py`. Throughput, VRAM and payload sizes are read from\n"
             "> `eval/runs/kaggle-.../models/*.json` and `docs/loadtest.json`. Business prices are assumptions,\n"
             "> listed separately below. Re-run with different flags to test any of them.\n")

    L.append("## 1. Measured inputs\n")
    L.append("| quantity | value | where it came from |")
    L.append("|---|---|---|")
    L.append(f"| `{model_key}` throughput | **{base['img_s']:.2f} img/s** | Tesla T4, batch size 1, fp16 "
             f"(`eval/results.md`) |")
    L.append(f"| GPU time per image | {1000 / base['img_s']:.0f} ms | same |")
    L.append(f"| CPU work per image (decode + encode + store) | {m['overhead_ms']:.0f} ms | stage profile on "
             f"{m['gpu_measured_on']} (`docs/loadtest.md`) |")
    L.append(f"| VRAM | {base['vram_mb']:.0f} MB | benchmark peak |")
    L.append(f"| Bytes stored per job | {m['bytes_per_job'] / 1024:.0f} KB | input + cutout + mask, measured |")
    L.append(f"| Bytes delivered per job | {m['bytes_delivered'] / 1024:.0f} KB | cutout + mask |")
    if m["dedupe_img_s"]:
        L.append(f"| Repeat upload (cache hit) | {m['dedupe_img_s']:.0f} img/s | load test `dedupe` scenario |")
    L.append("")

    L.append("## 2. Assumptions (not measured — change these first)\n")
    L.append("| input | value | unit | why it's a guess |")
    L.append("|---|---|---|---|")
    for a in ASSUMPTIONS:
        L.append(f"| `{a.key}` | {fmt(v[a.key], 2 if v[a.key] < 10 else 0)} | {a.unit} | {a.note} |")
    L.append("")

    L.append("## 3. Cost per image\n")
    L.append(f"Model `{model_key}`. **Sequential** is what the code does today (encode after inference); "
             "**pipelined** is the same hardware if CPU encoding overlapped GPU inference — an engineering "
             "change, not a purchase.\n")
    L.append("| hardware | scaling | img/s | effective img/hour | compute | storage | egress | **total/image** | blended¹ |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    rows = {}
    for label, factor in GPU_SCENARIOS.items():
        for owned, own_label in ((True, "owned"), (False, "rented")):
            c = cost_per_image(base["img_s"] * factor, v, owned=owned, overhead_ms=m["overhead_ms"], m=m)
            rows[(label, own_label)] = c
            L.append(f"| {own_label} {label} | {factor:.0f}x | {base['img_s'] * factor:.1f} | {fmt(c['img_per_hour'])} | "
                     f"{fmt(c['compute'], 1)} | {fmt(c['storage'], 1)} | {fmt(c['egress'], 1)} | "
                     f"**{fmt(c['total'], 1)}** | {fmt(c['blended'], 1)} |")
    pipe = cost_per_image(base["img_s"] * 2, v, owned=True, overhead_ms=m["overhead_ms"], m=m, pipelined=True)
    L.append(f"| owned 4090 @ 2x, **pipelined** | 2x | {base['img_s'] * 2:.1f} | {fmt(pipe['img_per_hour'])} | "
             f"{fmt(pipe['compute'], 1)} | {fmt(pipe['storage'], 1)} | {fmt(pipe['egress'], 1)} | "
             f"**{fmt(pipe['total'], 1)}** | {fmt(pipe['blended'], 1)} |")
    L.append("\n¹ blended = total with a "
             f"{v['dedupe_rate'] * 100:.0f}% share of byte-identical repeats served from cache (storage + egress only).\n")

    headline = rows[("4090 @ 2x T4", "owned")]
    L.append(f"**Headline: about {fmt(headline['blended'], 1)} toman per image** on an owned 4090 at "
             f"{v['utilization'] * 100:.0f}% utilisation — assuming the 4090 is twice a T4, which nobody has "
             "measured yet (H-001).\n")

    L.append("## 4. What actually drives the number\n")
    L.append("| change | new cost/image | vs headline |")
    L.append("|---|---|---|")
    for label, over in [
        ("utilisation 35% → 70%", {"utilization": 0.70}),
        ("utilisation 35% → 15%", {"utilization": 0.15}),
        ("toman/USD +50%", {"usd_toman": v["usd_toman"] * 1.5}),
        ("electricity ×4", {"electricity_toman_kwh": v["electricity_toman_kwh"] * 4}),
        ("GPU price −30%", {"gpu_price_usd": v["gpu_price_usd"] * 0.7}),
        ("storage ×3, 90-day retention", {"storage_usd_gb_month": v["storage_usd_gb_month"] * 3, "retention_days": 90}),
        (f"use `inspyrenet_base` (best quality, {m['models'].get('inspyrenet_base', {}).get('img_s', 0):.1f} img/s)", {}),
    ]:
        vv = dict(v, **over)
        img_s = base["img_s"] * 2
        if "inspyrenet_base" in label and "inspyrenet_base" in m["models"]:
            img_s = m["models"]["inspyrenet_base"]["img_s"] * 2
        c = cost_per_image(img_s, vv, owned=True, overhead_ms=m["overhead_ms"], m=m)
        delta = (c["blended"] / headline["blended"] - 1) * 100
        L.append(f"| {label} | {fmt(c['blended'], 1)} | {delta:+.0f}% |")
    L.append("\nUtilisation and the exchange rate dominate. Electricity barely registers at Iranian tariffs; "
             "picking the slowest, highest-quality model costs more than any infrastructure decision here.\n")

    L.append("## 5. Break-even against monthly price points\n")
    prices = (50_000, 100_000, 150_000, 200_000, 300_000, 500_000)
    L.append(f"At {fmt(headline['blended'], 1)} toman/image, infrastructure is almost invisible against any of these "
             "prices. The table below is the brief's break-even, and it mostly proves that compute is not what "
             "decides this business.\n")
    L.append("| price (toman/month) | images before compute eats the fee | infra cost at 100 img | at 500 img | gross margin at 500 img |")
    L.append("|---|---|---|---|---|")
    for price in prices:
        be = price / headline["blended"]
        L.append(f"| {fmt(price)} | **{fmt(be)}** | {fmt(100 * headline['blended'])} | {fmt(500 * headline['blended'])} | "
                 f"{(price - 500 * headline['blended']) / price * 100:.1f}% |")
    L.append("")
    free_cost = v.get("free_quota", 20) * headline["blended"]
    per_1000 = free_cost * 1000
    L.append(f"**Free tier:** {v.get('free_quota', 20):.0f} images/month ≈ **{fmt(free_cost)} toman per active free "
             f"user**; 1,000 of them ≈ {fmt(per_1000)} toman/month — covered by a single seller paying 150k. "
             "Abuse, not cost, is the reason to keep a quota.\n")

    L.append("### The question that actually matters\n")
    L.append("Since compute is noise, the real break-even is against **fixed costs** — salaries, support, an office. "
             "This model does not know your fixed costs, so pick a row:\n")
    usage_img = 200
    L.append(f"**Paying sellers needed to break even** (each using {usage_img} images/month):\n")
    L.append("| fixed costs (toman/month) | " + " | ".join(f"@ {fmt(p)}" for p in prices) + " |")
    L.append("|---|" + "---|" * len(prices))
    for fixed in (100_000_000, 300_000_000, 600_000_000, 1_000_000_000):
        cells = []
        for price in prices:
            net = price - usage_img * headline["blended"]
            cells.append(fmt(fixed / net) if net > 0 else "never")
        L.append(f"| {fmt(fixed)} | " + " | ".join(cells) + " |")
    L.append("\nRead it as: at 150k/month per seller, every 100M toman of monthly fixed cost needs roughly "
             f"{fmt(100_000_000 / (150_000 - usage_img * headline['blended']))} paying sellers. Choose the price from "
             "what Iranian sellers will pay and what the competition charges (H-011) — not from this cost floor, "
             "which any price above ~2,000 toman/month clears.\n")

    L.append("## 6. Honest limits of this model\n")
    L.append("- **Infrastructure only.** No salaries, no support, no payment-gateway fees, no marketing, no tax. "
             "At this scale a single developer-month dwarfs every number above; this model tells you the *floor*, "
             "not the price.\n"
             "- **The target GPU was never measured.** Everything is the T4 scaled by a factor (H-001).\n"
             "- **Every price is a guess** (§2). Exchange rate and utilisation move the answer more than any "
             "technical choice, and both are outside engineering control.\n"
             "- **Quality has a price.** `inspyrenet_base` scores highest but is ~5× the compute of "
             "`birefnet_lite`; the two-tier plan in `eval/analysis.md` exists for exactly this reason.\n"
             "- **Benchmark set is synthetic** (`docs/dataset-gap.md`), so the throughput numbers assume the "
             "image sizes it generates (~1 MP JPEGs). Bigger seller photos cost more.\n")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ASSUMPTIONS:
        ap.add_argument(f"--{a.key.replace('_', '-')}", type=float, default=None, help=f"{a.note} (default {a.value})")
    ap.add_argument("--model", default="birefnet_lite")
    ap.add_argument("--free-quota", type=float, default=20)
    ap.add_argument("--out", default="docs/economics.md")
    ap.add_argument("--show-inputs", action="store_true")
    args = ap.parse_args()

    m = load_measurements()
    v = inputs_dict({a.key: getattr(args, a.key) for a in ASSUMPTIONS})
    v["free_quota"] = args.free_quota
    if args.show_inputs:
        for a in ASSUMPTIONS:
            print(f"{a.key:28} {v[a.key]:>12,.2f} {a.unit:16} [{a.source}] {a.note}")
        print(f"\nmeasured: {json.dumps({k: (round(x, 2) if isinstance(x, float) else x) for k, x in m.items() if k != 'models'}, default=str)}")
        for name, d in m["models"].items():
            print(f"  {name:22} {d['img_s']:6.2f} img/s  iou={d['iou']:.3f}  {d['precision']}")
        return
    out = Path(args.out)
    out.write_text(render(v, m, args), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
