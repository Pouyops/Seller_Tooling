"""HTTP API: accepts images, enqueues jobs, serves results. Never touches the GPU.

    uvicorn st_inference.api:app

Endpoints
    POST /v1/matting            multipart: file, [model], [background], [priority], [wait]
    GET  /v1/jobs/{id}          job status
    GET  /v1/jobs/{id}/result   cutout (PNG with alpha, or JPEG on a solid background)
    GET  /v1/jobs/{id}/mask     alpha mask (PNG, grayscale)
    GET  /healthz               liveness + dependency status (200 "ok" / 200 "degraded" / 503)
    GET  /metrics               Prometheus

Degradation: if no GPU worker is ready, submissions still return 202 and wait in the queue. If
Redis is unreachable, submissions are spooled to disk and replayed later. We only answer 503 when
neither the queue nor the spool can take the job.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as aioredis
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from redis.exceptions import RedisError

from st_common.jobs import DONE, PRIORITIES, JobQueue, JobSpec
from st_common.logs import bind_context, get_logger, setup_logging
from st_common.storage import BlobStore

from .imaging import ImageRejected, parse_background, probe_image
from .listing import ListingGenerator, SellerFields
from .listing.llm_client import LocalLLM
from .settings import ServiceSettings, get_settings
from .spool import Spool

log = get_logger("st_inference.api")

HTTP_REQUESTS = Counter("st_api_requests_total", "HTTP requests", ["route", "method", "status"])
HTTP_LATENCY = Histogram("st_api_request_seconds", "HTTP request latency", ["route"],
                         buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30))
JOBS_SUBMITTED = Counter("st_api_jobs_submitted_total", "Job submissions", ["model", "outcome"])
LISTINGS = Counter("st_api_listings_total", "Listing generations", ["source"])
LISTING_SECONDS = Histogram("st_api_listing_seconds", "Listing generation time", buckets=(1, 2, 5, 10, 20, 40, 80, 160))
QUEUE_JOBS = Gauge("st_queue_jobs", "Jobs in the queue by state", ["kind", "state"])
REDIS_UP = Gauge("st_redis_up", "1 if the API can reach Redis/Valkey")
WORKERS = Gauge("st_workers", "Workers with a fresh heartbeat, by state", ["state"])
SPOOL_JOBS = Gauge("st_api_spool_jobs", "Submissions spooled to disk while Redis was unreachable")

_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
WORKER_KEY_PREFIX = "st:worker:"


class ApiState:
    settings: ServiceSettings
    redis: aioredis.Redis
    queue: JobQueue
    blobs: BlobStore
    spool: Spool
    llm: LocalLLM
    listings: ListingGenerator
    redis_ok: bool = False
    own_redis: bool = False


async def list_workers(r) -> list[dict]:
    out = []
    async for key in r.scan_iter(match=f"{WORKER_KEY_PREFIX}*", count=100):
        raw = await r.get(key)
        if raw:
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
    return out


async def _housekeeping(st: ApiState, stop: asyncio.Event, interval_s: float = 5.0) -> None:
    # asyncio.timeout (not wait_for): on 3.11 wait_for can swallow a cancellation and keep the task alive.
    while not stop.is_set():
        try:
            async with asyncio.timeout(2):
                await st.redis.ping()
            st.redis_ok = True
            if len(st.spool):
                n = await st.spool.drain(st.queue)
                if n:
                    log.info("spool.drained", extra={"jobs": n})
            depth = await st.queue.depth("matting")
            for state, v in depth.items():
                QUEUE_JOBS.labels("matting", state).set(v)
            counts: dict[str, int] = {}
            for w in await list_workers(st.redis):
                counts[w.get("state", "unknown")] = counts.get(w.get("state", "unknown"), 0) + 1
            WORKERS.clear()
            for state, v in counts.items():
                WORKERS.labels(state).set(v)
        except (RedisError, OSError, asyncio.TimeoutError):
            st.redis_ok = False
        REDIS_UP.set(1 if st.redis_ok else 0)
        SPOOL_JOBS.set(len(st.spool))
        try:
            async with asyncio.timeout(interval_s):
                await stop.wait()
        except TimeoutError:
            pass


def create_app(settings: ServiceSettings | None = None, redis_client=None, configure_logging: bool = True) -> FastAPI:
    s = settings or get_settings()
    st = ApiState()
    st.settings = s

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if configure_logging:
            setup_logging("inference-api", s.log_level, s.log_json)
        st.own_redis = redis_client is None
        st.redis = redis_client or aioredis.from_url(
            s.redis_url, decode_responses=True, socket_timeout=5, socket_connect_timeout=3, retry_on_timeout=True
        )
        st.queue = JobQueue(st.redis, visibility_timeout_ms=s.visibility_timeout_ms, max_attempts=s.max_attempts)
        st.blobs = BlobStore(s.blobs_dir)
        st.spool = Spool(s.data_dir)
        st.llm = LocalLLM(s.llm_base_url, s.llm_model, timeout_s=s.llm_timeout_s)
        st.listings = ListingGenerator(st.llm)
        stop = asyncio.Event()
        task = asyncio.create_task(_housekeeping(st, stop))
        log.info("api.started", extra={"redis_url": s.redis_url, "data_dir": str(s.data_dir), "default_model": s.matting_model})
        try:
            yield
        finally:
            stop.set()
            try:
                async with asyncio.timeout(5):
                    await task
            except (TimeoutError, asyncio.CancelledError, RedisError, OSError):
                task.cancel()
            await st.llm.aclose()
            if st.own_redis:
                await st.redis.aclose()

    app = FastAPI(title="Seller Tooling inference API", version="0.1.0", lifespan=lifespan)
    app.state.st = st

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        bind_context(request_id=rid)
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = rid
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            dt = time.perf_counter() - t0
            HTTP_REQUESTS.labels(path, request.method, str(status)).inc()
            HTTP_LATENCY.labels(path).observe(dt)
            if path not in ("/metrics", "/healthz"):
                log.info("http.request", extra={"method": request.method, "path": path, "status": status, "ms": round(dt * 1000, 1)})

    def err(status: int, code: str, message: str, **headers) -> JSONResponse:
        return JSONResponse({"error": code, "message": message}, status_code=status, headers=headers or None)

    async def job_view(job_id: str) -> dict | None:
        try:
            job = await st.queue.get(job_id)
        except (RedisError, OSError):
            if st.spool.has(job_id):
                return {"job_id": job_id, "status": "queued", "spooled": True}
            raise
        if job is None:
            return {"job_id": job_id, "status": "queued", "spooled": True} if st.spool.has(job_id) else None
        view = {
            "job_id": job_id,
            "status": job["status"],
            "model": job.get("model"),
            "created_at": job.get("created_at"),
            "attempts": job.get("attempts", 0),
            "error": job.get("error") or None,
        }
        if job["status"] == DONE and isinstance(job.get("result"), dict):
            view["result"] = {**job["result"], "url": f"/v1/jobs/{job_id}/result", "mask_url": f"/v1/jobs/{job_id}/mask"}
        return view

    @app.post("/v1/matting")
    async def submit(
        request: Request,
        file: UploadFile = File(...),
        model: str | None = Form(None),
        background: str = Form("transparent"),
        priority: str = Form("normal"),
        wait: float = Form(0.0),
    ):
        model = model or s.matting_model
        if model not in s.allowed:
            return err(400, "unknown_model", f"model must be one of {s.allowed}")
        if priority not in PRIORITIES:
            return err(400, "bad_priority", f"priority must be one of {list(PRIORITIES)}")
        try:
            parse_background(background)
        except ValueError as e:
            return err(400, "bad_background", str(e))
        limit = int(s.max_upload_mb * 2**20)
        data = await file.read(limit + 1)
        if len(data) > limit:
            return err(413, "too_large", f"upload exceeds {s.max_upload_mb} MB")
        try:
            info = probe_image(data, s.max_pixels)
        except ImageRejected as e:
            return err(415 if e.code in ("not_an_image", "unsupported_format", "empty") else 422, e.code, str(e))

        ref = await asyncio.to_thread(st.blobs.put_bytes, data, info["ext"])
        spec = JobSpec(kind="matting", model=model, input_key=ref.key, params={"background": background},
                       owner=request.headers.get("x-user-id"), priority=priority)
        try:
            job_id, status, created = await st.queue.submit(spec)
            outcome = "created" if created else "deduplicated"
        except (RedisError, OSError, asyncio.TimeoutError):
            if not st.spool.writable():
                return err(503, "unavailable", "queue and spool are both unavailable", **{"Retry-After": "30"})
            job_id, status, outcome = st.spool.put(spec), "queued", "spooled"
            log.warning("job.spooled", extra={"job_id": job_id})
        JOBS_SUBMITTED.labels(model, outcome).inc()
        bind_context(job_id=job_id)

        if wait > 0 and outcome != "spooled":
            try:
                await st.queue.wait(job_id, min(wait, s.max_wait_s), poll_s=0.1)
            except (RedisError, OSError):
                pass
        try:
            body = await job_view(job_id) or {"job_id": job_id, "status": status}
        except (RedisError, OSError):
            body = {"job_id": job_id, "status": status}
        body["deduplicated"] = outcome == "deduplicated"
        body["input"] = {"width": info["width"], "height": info["height"], "format": info["format"]}
        return JSONResponse(body, status_code=200 if body["status"] == DONE else 202, headers={"Location": f"/v1/jobs/{job_id}"})

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str):
        if not _JOB_ID.match(job_id):
            return err(404, "not_found", "no such job")
        try:
            view = await job_view(job_id)
        except (RedisError, OSError):
            return err(503, "queue_unavailable", "job store unreachable; retry", **{"Retry-After": "5"})
        return JSONResponse(view) if view else err(404, "not_found", "no such job")

    async def _blob_response(job_id: str, key_field: str, default_mime: str):
        if not _JOB_ID.match(job_id):
            return err(404, "not_found", "no such job")
        try:
            view = await job_view(job_id)
        except (RedisError, OSError):
            return err(503, "queue_unavailable", "job store unreachable; retry", **{"Retry-After": "5"})
        if not view:
            return err(404, "not_found", "no such job")
        if view["status"] != DONE:
            return err(409, "not_ready", f"job is {view['status']}")
        key = view["result"][key_field]
        sha, ext = key.rsplit(".", 1)
        path = st.blobs.path_for(sha, ext)
        if not path.exists():
            return err(410, "result_expired", "result file is gone; resubmit")
        mime = view["result"].get("mime", default_mime) if key_field == "output_key" else default_mime
        return FileResponse(path, media_type=mime, filename=f"{job_id}.{ext}")

    @app.get("/v1/jobs/{job_id}/result")
    async def get_result(job_id: str):
        return await _blob_response(job_id, "output_key", "image/png")

    @app.get("/v1/jobs/{job_id}/mask")
    async def get_mask(job_id: str):
        return await _blob_response(job_id, "mask_key", "image/png")

    @app.post("/v1/listing")
    async def listing(
        request: Request,
        file: UploadFile | None = File(None),
        category: str = Form(""),
        brand: str = Form(""),
        model: str = Form(""),
        material: str = Form(""),
        color: str = Form(""),
        size: str = Form(""),
        weight: str = Form(""),
        origin: str = Form(""),
        condition: str = Form(""),
        notes: str = Form(""),
        price_toman: int | None = Form(None),
        marketplace: str = Form("digikala"),
    ):
        """Image + seller fields -> Persian title/description/attributes/keywords.

        Synchronous: the LLM server has its own queue, and a seller is waiting. If it is unreachable
        the response is still 200 with a template listing built from the seller's fields and
        ``source: "template_fallback"`` — never an empty listing.
        """
        image: bytes | None = None
        if file is not None:
            limit = int(s.max_upload_mb * 2**20)
            data = await file.read(limit + 1)
            if len(data) > limit:
                return err(413, "too_large", f"upload exceeds {s.max_upload_mb} MB")
            if data:
                try:
                    probe_image(data, s.max_pixels)
                except ImageRejected as e:
                    return err(415 if e.code in ("not_an_image", "unsupported_format") else 422, e.code, str(e))
                image = data
        fields = SellerFields(category=category, brand=brand, model=model, material=material, color=color,
                              size=size, weight=weight, origin=origin, condition=condition, notes=notes,
                              price_toman=price_toman, marketplace=marketplace)
        result = await st.listings.generate(image, fields)
        LISTINGS.labels(result.source).inc()
        LISTING_SECONDS.observe(result.latency_ms / 1000)
        log.info("listing.generated", extra={"source": result.source, "ms": round(result.latency_ms),
                                             "issues": result.issues, "has_image": image is not None})
        return JSONResponse({
            "listing": result.listing.model_dump(),
            "source": result.source,
            "issues": result.issues,
            "attempts": result.attempts,
            "latency_ms": round(result.latency_ms, 1),
            "model": result.model,
        })

    @app.get("/healthz")
    async def healthz():
        body: dict = {"service": "inference-api", "spool_jobs": len(st.spool)}
        try:
            async with asyncio.timeout(1.5):
                await st.redis.ping()
            st.redis_ok = True
            workers = await list_workers(st.redis)
            body["queue"] = await st.queue.depth("matting")
        except (RedisError, OSError, asyncio.TimeoutError):
            st.redis_ok = False
            workers = []
        ready = [w for w in workers if w.get("state") == "ready"]
        body["redis"] = st.redis_ok
        body["workers"] = [{k: w.get(k) for k in ("consumer", "state", "model", "device", "precision")} for w in workers]
        body["gpu_workers_ready"] = len(ready)
        if st.redis_ok and ready:
            body["status"] = "ok"
        elif st.redis_ok or st.spool.writable():
            body["status"] = "degraded"  # still accepting jobs: queued or spooled
            body["reason"] = "no ready GPU worker; jobs will wait in the queue" if st.redis_ok else "queue unreachable; jobs are spooled to disk"
        else:
            body["status"] = "down"
            return JSONResponse(body, status_code=503)
        return JSONResponse(body)

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # The RTL seller page, if it is present. Mounted last so it never shadows /v1, /healthz or /metrics.
    web_dir = Path(__file__).resolve().parents[2] / "web"
    if web_dir.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return app


app = create_app()
