"""GPU worker: claims batches from the queue, runs the matting model, stores results.

    python -m st_inference.worker

Behaviour under failure:
* **GPU missing or unhealthy.** The worker does not claim jobs. It reports ``waiting_for_gpu`` in its
  heartbeat and retries with exponential backoff. Jobs wait in the queue (no 500s anywhere).
* **CUDA error mid-batch** (Xid, device lost). Claimed jobs are released back to the queue
  *without* spending a retry attempt, the model is unloaded, and the worker re-probes the GPU.
* **OOM on a batch.** The batch is split in half recursively, down to single images.
* **Crash or power cut.** Unacknowledged jobs are re-claimed by any worker after the visibility
  timeout. Results are keyed by job id, so a re-run overwrites identical bytes.
* **Bad input** (undecodable or missing blob). The job fails permanently, since retrying won't help.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import sys
import time
from collections import defaultdict
from collections.abc import Callable

import numpy as np
import redis.asyncio as aioredis
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from redis.exceptions import RedisError

from st_common.jobs import ClaimedJob, JobQueue
from st_common.logs import get_logger, log_context, setup_logging
from st_common.storage import BlobStore

from .api import WORKER_KEY_PREFIX
from .device import gpu_healthy, select_device
from .imaging import ImageRejected, decode_rgb, encode_cutout, encode_mask
from .settings import ServiceSettings, get_settings

log = get_logger("st_inference.worker")

JOBS = Counter("st_worker_jobs_total", "Jobs handled by outcome", ["model", "outcome"])
BATCH = Histogram("st_worker_batch_size", "Images per GPU batch", buckets=(1, 2, 4, 8, 16, 32))
STAGE_SECONDS = Histogram("st_worker_stage_seconds", "Per-batch stage time", ["stage"],
                          buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32))
QUEUE_WAIT = Histogram("st_worker_queue_wait_seconds", "Time from submission to claim",
                       buckets=(0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 300, 1800, 7200))
GPU_UP = Gauge("st_worker_gpu_up", "1 when a model is loaded on a healthy device")
MODEL_LOADS = Counter("st_worker_model_loads_total", "Model load attempts", ["model", "outcome"])


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def _is_device_failure(e: BaseException) -> bool:
    msg = str(e).lower()
    return any(s in msg for s in ("cuda", "cudnn", "cublas", "device-side", "nvml", "driver", "illegal memory"))


def _split_key(key: str) -> tuple[str, str]:
    sha, ext = key.rsplit(".", 1)
    return sha, ext


class Worker:
    def __init__(
        self,
        settings: ServiceSettings | None = None,
        redis_client=None,
        model_factory: Callable[[str, str], object] | None = None,
        device_probe: Callable[[], str | None] | None = None,
        consumer: str | None = None,
    ):
        self.s = settings or get_settings()
        self.consumer = consumer or f"{socket.gethostname()}-{os.getpid()}"
        self.redis = redis_client
        self.queue: JobQueue | None = JobQueue(redis_client, visibility_timeout_ms=self.s.visibility_timeout_ms,
                                               max_attempts=self.s.max_attempts) if redis_client is not None else None
        self.blobs = BlobStore(self.s.blobs_dir)
        self.model_factory = model_factory or self._default_factory
        self.device_probe = device_probe or self._probe_device
        self.models: dict[str, object] = {}
        self.state = "starting"
        self.device: str | None = None
        self.stopping = asyncio.Event()
        self._backoff = 1.0

    # ---- device & models -------------------------------------------------------------------
    def _default_factory(self, name: str, device: str):
        from .models import create_model

        return create_model(name, models_dir=self.s.models_dir, device=device)

    def _probe_device(self) -> str | None:
        dev = select_device(self.s.device)
        if dev == "unavailable":
            return None
        if dev == "cpu":
            return "cpu" if self.s.allow_cpu_fallback else None
        return dev if gpu_healthy(dev) else None

    def _unload_all(self) -> None:
        for m in self.models.values():
            try:
                m.unload()
            except Exception:  # pragma: no cover - best effort after a device failure
                pass
        self.models.clear()

    async def _get_model(self, name: str, device: str):
        m = self.models.get(name)
        if m is not None and getattr(m, "device", device) == device:
            return m
        self._unload_all()  # one model in VRAM at a time on a 24 GB card
        previous = self.state
        self.state = "loading"
        await self.heartbeat()
        t0 = time.perf_counter()
        try:
            m = await asyncio.to_thread(lambda: self.model_factory(name, device).load())
        except Exception:
            MODEL_LOADS.labels(name, "error").inc()
            raise
        MODEL_LOADS.labels(name, "ok").inc()
        self.state = "ready" if previous in ("ready", "loading") else previous
        await self.heartbeat()
        log.info("model.loaded", extra={"model": name, "device": device, "precision": getattr(m, "precision", None),
                                        "seconds": round(time.perf_counter() - t0, 2)})
        self.models[name] = m
        return m

    # ---- heartbeat ---------------------------------------------------------------------------
    async def heartbeat(self) -> None:
        if self.redis is None:
            return
        default = self.models.get(self.s.matting_model)
        payload = {
            "consumer": self.consumer,
            "state": self.state,
            "device": self.device,
            "model": self.s.matting_model,
            "precision": getattr(default, "precision", None),
            "ts": time.time(),
        }
        try:
            await self.redis.set(f"{WORKER_KEY_PREFIX}{self.consumer}", json.dumps(payload), ex=int(self.s.worker_heartbeat_s * 4))
        except RedisError:
            pass

    async def _heartbeat_loop(self) -> None:
        while not self.stopping.is_set():
            await self.heartbeat()
            await asyncio.sleep(self.s.worker_heartbeat_s)

    async def _sleep_backoff(self) -> None:
        try:
            await asyncio.wait_for(self.stopping.wait(), timeout=self._backoff)
        except asyncio.TimeoutError:
            pass
        self._backoff = min(self._backoff * 2, self.s.gpu_retry_max_s)

    # ---- main loop ---------------------------------------------------------------------------
    async def step(self, block_ms: int = 1000) -> int:
        """One iteration. Returns the number of jobs completed."""
        device = await asyncio.to_thread(self.device_probe)
        if device is None:
            if self.state != "waiting_for_gpu":
                log.warning("gpu.unavailable", extra={"action": "not claiming jobs; they stay queued"})
            self.state, self.device = "waiting_for_gpu", None
            GPU_UP.set(0)
            self._unload_all()
            await self.heartbeat()
            await self._sleep_backoff()
            return 0
        self.device = device
        try:
            await self._get_model(self.s.matting_model, device)
        except Exception as e:
            self.state = "gpu_error" if _is_device_failure(e) else "model_error"
            log.error("model.load_failed", extra={"model": self.s.matting_model, "error": str(e)[:500]})
            GPU_UP.set(0)
            await self.heartbeat()
            await self._sleep_backoff()
            return 0
        if self.state != "ready":
            self.state = "ready"
            await self.heartbeat()
        GPU_UP.set(1)
        self._backoff = 1.0

        jobs = await self.queue.claim("matting", self.consumer, self.s.max_batch, block_ms=block_ms)
        if not jobs:
            return 0
        if len(jobs) < self.s.max_batch and self.s.batch_wait_ms > 0:
            await asyncio.sleep(self.s.batch_wait_ms / 1000)
            jobs += await self.queue.claim("matting", self.consumer, self.s.max_batch - len(jobs), block_ms=0)
        groups: dict[str, list[ClaimedJob]] = defaultdict(list)
        for j in jobs:
            groups[j.model].append(j)
        done = 0
        for name, group in groups.items():
            done += await self._process_group(name, group, device)
        return done

    async def _process_group(self, name: str, jobs: list[ClaimedJob], device: str) -> int:
        items = []
        t0 = time.perf_counter()
        for j in jobs:
            QUEUE_WAIT.observe(max(0.0, time.time() - float(j.data.get("created_at") or time.time())))
            try:
                data = await asyncio.to_thread(self.blobs.get_bytes, *_split_key(j.input_key))
                rgb = await asyncio.to_thread(decode_rgb, data, self.s.max_side)
                items.append((j, rgb))
            except FileNotFoundError:
                await self.queue.fail(j, "input_missing", retry=False)
                JOBS.labels(name, "failed").inc()
            except ImageRejected as e:
                await self.queue.fail(j, f"{e.code}: {e}", retry=False)
                JOBS.labels(name, "failed").inc()
        STAGE_SECONDS.labels("decode").observe(time.perf_counter() - t0)
        if not items:
            return 0
        try:
            model = await self._get_model(name, device)
        except Exception as e:
            await self.queue.release([j for j, _ in items])
            JOBS.labels(name, "released").inc(len(items))
            self.state = "gpu_error" if _is_device_failure(e) else "model_error"
            log.error("model.load_failed", extra={"model": name, "error": str(e)[:500]})
            return 0
        return await self._run_batch(model, items)

    async def _run_batch(self, model, items) -> int:
        name = model.name
        t0 = time.perf_counter()
        try:
            alphas = await asyncio.to_thread(model.predict, [rgb for _, rgb in items])
        except Exception as e:
            if _is_oom(e) and len(items) > 1:
                log.warning("batch.oom_split", extra={"model": name, "batch": len(items)})
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_initialized():  # never create a CUDA context just to free memory
                    torch.cuda.empty_cache()
                mid = len(items) // 2
                return await self._run_batch(model, items[:mid]) + await self._run_batch(model, items[mid:])
            if _is_device_failure(e) or _is_oom(e):
                log.error("gpu.failure", extra={"model": name, "error": str(e)[:500], "action": "release jobs, re-probe GPU"})
                await self.queue.release([j for j, _ in items])
                JOBS.labels(name, "released").inc(len(items))
                self._unload_all()
                self.state = "gpu_error"
                GPU_UP.set(0)
                await self.heartbeat()
                return 0
            for j, _ in items:
                await self.queue.fail(j, f"{type(e).__name__}: {str(e)[:500]}", retry=True)
                JOBS.labels(name, "retried").inc()
            log.exception("batch.failed", extra={"model": name})
            return 0
        infer_s = time.perf_counter() - t0
        STAGE_SECONDS.labels("inference").observe(infer_s)
        BATCH.observe(len(items))
        timing = getattr(model, "last_timing", None)

        done = 0
        for (job, rgb), alpha in zip(items, alphas):
            with log_context(job_id=job.job_id):
                t1 = time.perf_counter()
                out, ext, mime = await asyncio.to_thread(encode_cutout, rgb, alpha, job.params.get("background", "transparent"))
                mask = await asyncio.to_thread(encode_mask, alpha)
                out_ref = await asyncio.to_thread(self.blobs.put_at, job.job_id, ext, out)
                mask_ref = await asyncio.to_thread(self.blobs.put_at, job.job_id + "_mask", "png", mask)
                encode_s = time.perf_counter() - t1
                STAGE_SECONDS.labels("encode_store").observe(encode_s)
                result = {
                    "output_key": out_ref.key,
                    "mask_key": mask_ref.key,
                    "mime": mime,
                    "width": int(rgb.shape[1]),
                    "height": int(rgb.shape[0]),
                    "model": name,
                    "precision": getattr(model, "precision", None),
                    "batch_size": len(items),
                    "inference_ms_per_image": round(infer_s * 1000 / len(items), 1),
                    "forward_ms_batch": round(timing.forward_ms, 1) if timing else None,
                    "encode_ms": round(encode_s * 1000, 1),
                    "fg_fraction": round(float((np.asarray(alpha) >= 0.5).mean()), 4),
                    "worker": self.consumer,
                }
                await self.queue.complete(job, result)
                JOBS.labels(name, "done").inc()
                log.info("job.done", extra={"model": name, "batch": len(items), "infer_ms": result["inference_ms_per_image"]})
                done += 1
        return done

    async def run_forever(self) -> None:
        if self.redis is None:
            self.redis = aioredis.from_url(self.s.redis_url, decode_responses=True, socket_timeout=10,
                                           socket_connect_timeout=3, retry_on_timeout=True)
            self.queue = JobQueue(self.redis, visibility_timeout_ms=self.s.visibility_timeout_ms, max_attempts=self.s.max_attempts)
        hb = asyncio.create_task(self._heartbeat_loop())
        log.info("worker.started", extra={"consumer": self.consumer, "model": self.s.matting_model, "max_batch": self.s.max_batch})
        try:
            while not self.stopping.is_set():
                try:
                    await self.step()
                except (RedisError, OSError) as e:
                    self.state = "redis_error"
                    log.warning("redis.unavailable", extra={"error": str(e)[:200]})
                    await self._sleep_backoff()
        finally:
            hb.cancel()
            self.state = "stopped"
            await self.heartbeat()
            self._unload_all()
            log.info("worker.stopped")


def main() -> None:
    s = get_settings()
    setup_logging("inference-worker", s.log_level, s.log_json)
    try:
        start_http_server(s.worker_metrics_port)
    except OSError as e:
        log.warning("metrics.port_in_use", extra={"port": s.worker_metrics_port, "error": str(e)})
    worker = Worker(s)

    async def runner():
        loop = asyncio.get_running_loop()

        def stop(*_):
            loop.call_soon_threadsafe(worker.stopping.set)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop)
            except (NotImplementedError, RuntimeError):
                signal.signal(sig, stop)  # Windows
        await worker.run_forever()

    asyncio.run(runner())


if __name__ == "__main__":
    main()
