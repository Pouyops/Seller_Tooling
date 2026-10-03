"""Load-test the inference API over real HTTP and write docs/loadtest.md.

    python -m st_inference.loadtest --scenario steady --concurrency 4 --requests 60
    python -m st_inference.loadtest --url http://localhost:8000 --scenario burst --requests 32

Modes
    standalone (default)  starts uvicorn + a GPU worker in this process on a free port
    --url <base>          drives an already-running deployment (`make serve`)

Scenarios
    steady    N requests at a fixed concurrency; the number to quote for capacity planning
    burst     every request fired at once; shows queueing behaviour and tail latency
    outage    kills the worker mid-run and restarts it: the API must keep accepting work (202,
              never 5xx) and every job must still finish

Every request uploads a *distinct* image, because the API deduplicates identical uploads by
content hash — reusing one image would measure the cache, not the GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from st_common.storage import atomic_write_bytes


@dataclass
class RequestRecord:
    ok: bool
    status: int
    submit_ms: float
    total_ms: float
    deduplicated: bool = False
    error: str = ""


@dataclass
class ScenarioResult:
    scenario: str
    concurrency: int
    requests: int
    wall_s: float = 0.0
    records: list[RequestRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok_records(self) -> list[RequestRecord]:
        return [r for r in self.records if r.ok]

    def summary(self) -> dict:
        done = self.ok_records
        totals = sorted(r.total_ms for r in done)
        submits = sorted(r.submit_ms for r in self.records)
        statuses: dict[str, int] = {}
        for r in self.records:
            key = str(r.status)
            statuses[key] = statuses.get(key, 0) + 1

        def pct(values: list[float], q: float) -> float | None:
            if not values:
                return None
            k = min(len(values) - 1, int(round(q / 100 * (len(values) - 1))))
            return values[k]

        return {
            "scenario": self.scenario,
            "concurrency": self.concurrency,
            "requests": self.requests,
            "completed": len(done),
            "failed": len(self.records) - len(done),
            "wall_s": round(self.wall_s, 2),
            "throughput_img_s": round(len(done) / self.wall_s, 3) if self.wall_s else 0,
            "submit_ms": {"p50": pct(submits, 50), "p90": pct(submits, 90), "p99": pct(submits, 99)},
            "end_to_end_ms": {"p50": pct(totals, 50), "p90": pct(totals, 90), "p99": pct(totals, 99),
                              "max": totals[-1] if totals else None},
            "status_codes": statuses,
            "errors": sorted({r.error for r in self.records if r.error}),
            "notes": self.notes,
        }


def unique_images(dataset: Path, count: int, salt: str = "") -> list[bytes]:
    """Distinct JPEGs. Trailing bytes after the EOI marker change the hash and still decode.

    The salt matters: the API deduplicates by content hash, so without a per-scenario salt the
    second scenario would be served entirely from cache and measure nothing (observed).
    """
    files = sorted((dataset / "images").glob("*.jpg"))
    if not files:
        raise SystemExit(f"no images under {dataset / 'images'} — run `make dataset` first")
    return [files[i % len(files)].read_bytes() + f"\n#{salt}:{i}".encode() for i in range(count)]


async def one_request(client: httpx.AsyncClient, image: bytes, model: str | None, wait_s: float,
                      poll_s: float = 0.25, deadline_s: float = 600) -> RequestRecord:
    t0 = time.perf_counter()
    data = {"wait": str(wait_s)}
    if model:
        data["model"] = model
    try:
        r = await client.post("/v1/matting", files={"file": ("p.jpg", image, "image/jpeg")}, data=data)
    except httpx.HTTPError as e:
        return RequestRecord(False, 0, (time.perf_counter() - t0) * 1000, (time.perf_counter() - t0) * 1000,
                             error=f"{type(e).__name__}")
    submit_ms = (time.perf_counter() - t0) * 1000
    if r.status_code >= 400:
        return RequestRecord(False, r.status_code, submit_ms, submit_ms, error=f"http_{r.status_code}")
    body = r.json()
    job_id, status = body["job_id"], body["status"]
    dedup = bool(body.get("deduplicated"))
    while status not in ("done", "failed") and (time.perf_counter() - t0) < deadline_s:
        await asyncio.sleep(poll_s)
        try:
            jr = await client.get(f"/v1/jobs/{job_id}")
        except httpx.HTTPError:
            continue
        if jr.status_code >= 500:
            return RequestRecord(False, jr.status_code, submit_ms, (time.perf_counter() - t0) * 1000,
                                 error=f"http_{jr.status_code}")
        if jr.status_code == 200:
            status = jr.json()["status"]
    total_ms = (time.perf_counter() - t0) * 1000
    return RequestRecord(status == "done", r.status_code, submit_ms, total_ms, dedup,
                         "" if status == "done" else f"ended_{status}")


async def run_scenario(base_url: str, images: list[bytes], *, scenario: str, concurrency: int, wait_s: float,
                       model: str | None, worker_control=None, log=print) -> ScenarioResult:
    result = ScenarioResult(scenario, concurrency, len(images))
    limits = httpx.Limits(max_connections=max(concurrency * 2, 10), max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=120, limits=limits) as client:
        sem = asyncio.Semaphore(concurrency if scenario != "burst" else len(images))
        done_count = 0

        async def task(img: bytes) -> RequestRecord:
            nonlocal done_count
            async with sem:
                rec = await one_request(client, img, model, wait_s)
                done_count += 1
                if done_count % 10 == 0:
                    log(f"  {done_count}/{len(images)} done")
                return rec

        async def chaos(outage_s: float = 15.0):
            """Kill the worker once 30% is done, bring it back after a fixed outage.

            The restart must be time-based: while the worker is down nothing completes, so waiting
            for a completion threshold would deadlock (it did).
            """
            killed = restarted = False
            killed_at = 0.0
            while done_count < len(images) and not restarted:
                if not killed and done_count >= len(images) * 0.3:
                    await worker_control("stop")
                    killed, killed_at = True, time.perf_counter()
                    result.notes.append(f"worker stopped after {done_count}/{len(images)} completions")
                    log("  !! worker stopped")
                elif killed and time.perf_counter() - killed_at >= outage_s:
                    pending = len(images) - done_count
                    await worker_control("start")
                    restarted = True
                    result.notes.append(f"worker restarted after {outage_s:.0f}s outage with {pending} jobs outstanding")
                    log("  !! worker restarted")
                await asyncio.sleep(0.2)
            if killed and not restarted:
                await worker_control("start")
                result.notes.append("worker restarted at end of run")

        if scenario == "dedupe":
            # Warm the cache deliberately, then measure what a repeat upload costs.
            warm = await asyncio.gather(*(one_request(client, img, model, wait_s) for img in images[: min(8, len(images))]))
            result.notes.append(f"cache warmed with {len(warm)} images "
                                f"(cold p50 {statistics.median(r.total_ms for r in warm):.0f} ms)")
            images = images[: min(8, len(images))]
            result.requests = len(images)

        t0 = time.perf_counter()
        tasks = [asyncio.create_task(task(img)) for img in images]
        chaos_task = asyncio.create_task(chaos()) if scenario == "outage" and worker_control else None
        result.records = list(await asyncio.gather(*tasks))
        if chaos_task:
            await chaos_task
        result.wall_s = time.perf_counter() - t0
    return result


# ---- standalone server ------------------------------------------------------------------------
class Standalone:
    """uvicorn + a GPU worker in this process, so the test needs no external services."""

    def __init__(self, settings, port: int):
        self.settings, self.port = settings, port
        self.worker_task: asyncio.Task | None = None
        self.server = None
        self.real_redis = False

    async def __aenter__(self):
        import uvicorn

        from .api import create_app
        from .worker import Worker

        try:
            import redis.asyncio as aioredis

            redis = aioredis.from_url(self.settings.redis_url, decode_responses=True, socket_connect_timeout=2)
            await redis.ping()
            self.real_redis = True
        except Exception:
            import fakeredis

            redis = fakeredis.FakeAsyncRedis(decode_responses=True)
        self.redis = redis
        app = create_app(self.settings, redis_client=redis, configure_logging=False)
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning", access_log=False)
        self.server = uvicorn.Server(config)
        self.server_task = asyncio.create_task(self.server.serve())
        while not self.server.started:
            await asyncio.sleep(0.05)
        self.worker = Worker(self.settings, redis_client=redis, consumer="loadtest-worker")
        await self.control("start")
        return self

    async def control(self, action: str) -> None:
        if action == "start" and self.worker_task is None:
            self.worker.stopping.clear()

            async def loop():
                while not self.worker.stopping.is_set():
                    await self.worker.step(block_ms=200)

            self.worker_task = asyncio.create_task(loop())
        elif action == "stop" and self.worker_task is not None:
            self.worker.stopping.set()
            self.worker_task.cancel()
            try:
                await self.worker_task
            except (asyncio.CancelledError, Exception):
                pass
            self.worker_task = None

    async def __aexit__(self, *exc):
        await self.control("stop")
        self.worker._unload_all()
        if self.server:
            self.server.should_exit = True
            await self.server_task
        await self.redis.aclose()


def profile_stages(settings, dataset: Path, count: int = 8, log=print) -> dict[str, float]:
    """Median ms per stage for one image, single-threaded: shows what the GPU is *not* spending time on."""
    import statistics

    from st_common.storage import BlobStore

    from .imaging import decode_rgb, encode_cutout, encode_mask
    from .models import create_model

    files = sorted((dataset / "images").glob("*.jpg"))[:count]
    raw = [f.read_bytes() for f in files]
    model = create_model(settings.matting_model, models_dir=settings.models_dir).load()
    fsync_store = BlobStore(settings.blobs_dir / "_profile_fsync", fsync=True)
    plain_store = BlobStore(settings.blobs_dir / "_profile_plain", fsync=False)
    times: dict[str, list[float]] = {k: [] for k in ("decode", "inference", "encode_png", "encode_mask",
                                                     "store", "store_fsync_alt")}
    images = [decode_rgb(r) for r in raw]
    model.predict(images[:1])  # warm up
    for i, (blob, image) in enumerate(zip(raw, images)):
        for key, fn in (("decode", lambda: decode_rgb(blob)),):
            t0 = time.perf_counter(); fn(); times[key].append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); alpha = model.predict([image])[0]; times["inference"].append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); out, _, _ = encode_cutout(image, alpha); times["encode_png"].append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); encode_mask(alpha); times["encode_mask"].append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); plain_store.put_at(f"{i:032x}", "png", out); times["store"].append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); fsync_store.put_at(f"{i:032x}", "png", out); times["store_fsync_alt"].append((time.perf_counter() - t0) * 1000)
    model.unload()
    result = {k: statistics.median(v) for k, v in times.items()}
    log(f"[loadtest] stage profile: {({k: round(v) for k, v in result.items()})}")
    return result


def hardware() -> dict:
    from .device import hardware_info

    info = hardware_info()
    info["os"] = platform.platform()
    return info


def render_markdown(report: dict) -> str:
    L = ["# Load test — inference API\n",
         "> Generated by `python -m st_inference.loadtest`. Numbers are measured, not modelled.\n", "## Setup\n"]
    hw = report["hardware"]
    L.append(f"- **Hardware:** {hw.get('gpu', 'CPU only')} ({hw.get('gpu_vram_gb', '?')} GB) · {hw.get('cpu')} · "
             f"{hw.get('ram_gb', '?')} GB RAM · torch {hw.get('torch')}")
    L.append(f"- **Model:** `{report['model']}` · max_batch={report['max_batch']} · workers=1")
    L.append(f"- **Queue:** {report['queue']}")
    L.append(f"- **Mode:** {report['mode']} · client and server on the same machine (no network hop)")
    L.append(f"- **Run:** {report['created_at']} · git `{report.get('git_commit')}`\n")
    if report.get("caveats"):
        L.append("### Caveats\n")
        for c in report["caveats"]:
            L.append(f"- {c}")
        L.append("")
    L.append("## Results\n")
    L.append("| scenario | reqs | concurrency | completed | failed | wall s | img/s | submit p50 | e2e p50 | e2e p90 | e2e p99 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")

    def ms(v):
        return f"{v:.0f} ms" if isinstance(v, (int, float)) else "–"

    for s in report["scenarios"]:
        L.append(f"| {s['scenario']} | {s['requests']} | {s['concurrency']} | {s['completed']} | {s['failed']} | "
                 f"{s['wall_s']} | **{s['throughput_img_s']}** | {ms(s['submit_ms']['p50'])} | "
                 f"{ms(s['end_to_end_ms']['p50'])} | {ms(s['end_to_end_ms']['p90'])} | {ms(s['end_to_end_ms']['p99'])} |")
    L.append("")
    for s in report["scenarios"]:
        L.append(f"**{s['scenario']}** — status codes: `{s['status_codes']}`"
                 + (f", errors: `{s['errors']}`" if s["errors"] else ", no errors"))
        for note in s["notes"]:
            L.append(f"  - {note}")
    L.append("")
    if report.get("stage_breakdown"):
        L.append("## Where a single image's time goes\n")
        L.append("| stage | median ms | share |\n|---|---|---|")
        total = sum(v for k, v in report["stage_breakdown"].items() if not k.endswith("_alt"))
        for stage, value in report["stage_breakdown"].items():
            share = "–" if stage.endswith("_alt") else f"{value / total * 100:.0f}%"
            L.append(f"| {stage} | {value:.0f} | {share} |")
        L.append("")
    analysis = Path("docs/loadtest-analysis.md")
    if analysis.exists():
        L.append("---\n")
        L.append(analysis.read_text(encoding="utf-8"))
    return "\n".join(L) + "\n"


async def main_async(args) -> None:
    from .settings import ServiceSettings

    settings = ServiceSettings(matting_model=args.model, max_batch=args.max_batch, batch_wait_ms=args.batch_wait_ms,
                               log_json=False, log_level="WARNING")
    dataset = Path(args.dataset)
    scenarios = [s for s in args.scenario.split(",") if s]
    want_profile = "profile" in scenarios
    scenarios = [s for s in scenarios if s != "profile"]
    run_id = int(time.time())
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model, "max_batch": args.max_batch, "mode": "url" if args.url else "standalone",
        "hardware": hardware(), "scenarios": [], "caveats": [],
    }

    if want_profile and not args.url:
        report["stage_breakdown"] = profile_stages(settings, dataset)

    if args.url:
        base = args.url.rstrip("/")
        report["queue"] = "whatever the running deployment uses"
        for name in scenarios:
            images = unique_images(dataset, args.requests, salt=f"{name}-{run_id}")
            print(f"[loadtest] {name}: {len(images)} requests at concurrency {args.concurrency}")
            res = await run_scenario(base, images, scenario=name, concurrency=args.concurrency, wait_s=args.wait,
                                     model=args.model)
            report["scenarios"].append(res.summary())
    else:
        async with Standalone(settings, args.port) as stand:
            report["queue"] = ("Valkey/Redis at " + settings.redis_url) if stand.real_redis else "in-process (fakeredis)"
            if not stand.real_redis:
                report["caveats"].append(
                    "No Redis/Valkey server was reachable, so the queue ran in-process (fakeredis). Queue semantics "
                    "are the same, but there is no network hop, no fsync and no cross-process contention — real "
                    "deployments will be somewhat slower.")
            report["caveats"].append("Client and server share one machine, so client CPU competes with preprocessing.")
            base = f"http://127.0.0.1:{args.port}"
            for name in scenarios:
                images = unique_images(dataset, args.requests, salt=f"{name}-{run_id}")
                print(f"[loadtest] {name}: {len(images)} requests at concurrency {args.concurrency}")
                res = await run_scenario(base, images, scenario=name, concurrency=args.concurrency, wait_s=args.wait,
                                         model=args.model, worker_control=stand.control)
                report["scenarios"].append(res.summary())
                print("   ", json.dumps(res.summary()["end_to_end_ms"]))

    atomic_write_bytes(Path(args.json_out), json.dumps(report, indent=2).encode())
    atomic_write_bytes(Path(args.out), render_markdown(report).encode("utf-8"))
    print(f"[loadtest] wrote {args.out} and {args.json_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None, help="drive a running API instead of starting one")
    ap.add_argument("--scenario", default="steady,burst,outage,dedupe")
    ap.add_argument("--requests", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model", default="birefnet_lite")
    ap.add_argument("--max-batch", type=int, default=4)
    ap.add_argument("--batch-wait-ms", type=int, default=40)
    ap.add_argument("--wait", type=float, default=0.0, help="inline wait seconds asked of the API")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default="docs/loadtest.md")
    ap.add_argument("--json-out", default="docs/loadtest.json")
    args = ap.parse_args()
    if not args.dataset:
        from st_eval.synth.__main__ import default_dataset_dir

        args.dataset = str(default_dataset_dir())
    random.seed(0)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
