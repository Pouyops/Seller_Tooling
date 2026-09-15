"""Resumable job queue on Redis Streams (served by Valkey, see ADR-004).

Guarantees, given Valkey with AOF ``appendfsync everysec``:

* **At-least-once processing.** A job message stays in the consumer group's pending list
  until the worker has written the result *and* acknowledged it. If a worker dies (crash,
  power cut, GPU reset), the message is re-claimed by ``XAUTOCLAIM`` after
  ``visibility_timeout_ms``.
* **Idempotency.** ``job_id`` is a hash of (kind, model, input blob, params). Re-submitting the
  same work returns the existing job, and a completed job is never re-run. Results are
  content-addressed on disk, so a re-run after a crash overwrites identical bytes.
* **Bounded retries.** Each claim increments ``attempts``; past ``max_attempts`` the job goes to
  a dead-letter stream with status ``failed``.
* **Priorities.** ``high`` (paid) is drained before ``normal`` (free tier).

The Redis client must be created with ``decode_responses=True``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from redis.exceptions import ResponseError, WatchError

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"
TERMINAL = frozenset({DONE, FAILED})
PRIORITIES = ("high", "normal")
GROUP = "workers"


@dataclass(frozen=True)
class JobSpec:
    kind: str  # "matting", "listing", ...
    model: str  # registry name, e.g. "birefnet_lite"
    input_key: str  # blob key "<sha256>.<ext>"
    params: dict[str, Any] = field(default_factory=dict)
    owner: str | None = None  # opaque user id (not part of the job identity)
    priority: str = "normal"

    def job_id(self) -> str:
        payload = json.dumps([self.kind, self.model, self.input_key, self.params], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass
class ClaimedJob:
    job_id: str
    msg_id: str
    stream: str
    attempts: int
    data: dict[str, Any]

    @property
    def kind(self) -> str:
        return self.data["kind"]

    @property
    def model(self) -> str:
        return self.data["model"]

    @property
    def input_key(self) -> str:
        return self.data["input_key"]

    @property
    def params(self) -> dict[str, Any]:
        return self.data.get("params") or {}


def _decode_job(raw: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = dict(raw)
    for k in ("params", "result"):
        if out.get(k):
            try:
                out[k] = json.loads(out[k])
            except json.JSONDecodeError:
                pass
    for k in ("attempts",):
        if k in out:
            out[k] = int(out[k])
    for k in ("created_at", "updated_at", "started_at", "finished_at"):
        if out.get(k):
            out[k] = float(out[k])
    return out


class JobQueue:
    def __init__(
        self,
        redis,
        *,
        prefix: str = "st",
        visibility_timeout_ms: int = 120_000,
        max_attempts: int = 3,
        result_ttl_s: int = 30 * 86_400,
    ):
        self.r = redis
        self.prefix = prefix
        self.visibility_timeout_ms = visibility_timeout_ms
        self.max_attempts = max_attempts
        self.result_ttl_s = result_ttl_s
        self._groups_ready: set[str] = set()

    # ---- keys -------------------------------------------------------------------------------
    def stream(self, kind: str, priority: str = "normal") -> str:
        return f"{self.prefix}:q:{kind}:{priority}"

    def dead_stream(self, kind: str) -> str:
        return f"{self.prefix}:dead:{kind}"

    def job_key(self, job_id: str) -> str:
        return f"{self.prefix}:job:{job_id}"

    # ---- setup ------------------------------------------------------------------------------
    async def ensure_group(self, kind: str) -> None:
        if kind in self._groups_ready:
            return
        for pr in PRIORITIES:
            try:
                await self.r.xgroup_create(self.stream(kind, pr), GROUP, id="0", mkstream=True)
            except ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise
        self._groups_ready.add(kind)

    # ---- producer ---------------------------------------------------------------------------
    async def submit(self, spec: JobSpec) -> tuple[str, str, bool]:
        """Enqueue unless an equivalent job exists. Returns ``(job_id, status, created)``."""
        if spec.priority not in PRIORITIES:
            raise ValueError(f"priority must be one of {PRIORITIES}")
        await self.ensure_group(spec.kind)
        job_id = spec.job_id()
        key = self.job_key(job_id)
        for _ in range(10):
            async with self.r.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key)
                    status = await pipe.hget(key, "status")
                    if status is not None and status != FAILED:
                        await pipe.unwatch()
                        return job_id, status, False
                    now = time.time()
                    pipe.multi()
                    pipe.hset(
                        key,
                        mapping={
                            "status": QUEUED,
                            "kind": spec.kind,
                            "model": spec.model,
                            "input_key": spec.input_key,
                            "params": json.dumps(spec.params, sort_keys=True, ensure_ascii=False),
                            "owner": spec.owner or "",
                            "priority": spec.priority,
                            "attempts": 0,
                            "created_at": now,
                            "updated_at": now,
                            "error": "",
                            "result": "",
                        },
                    )
                    pipe.persist(key)
                    pipe.xadd(self.stream(spec.kind, spec.priority), {"job_id": job_id})
                    await pipe.execute()
                    return job_id, QUEUED, True
                except WatchError:
                    await asyncio.sleep(0.005)
        raise RuntimeError(f"could not submit job {job_id}: contention")

    # ---- consumer ---------------------------------------------------------------------------
    async def claim(self, kind: str, consumer: str, count: int, block_ms: int = 0) -> list[ClaimedJob]:
        """Claim up to ``count`` jobs: stale (crashed) ones first, then high, then normal priority."""
        await self.ensure_group(kind)
        raw: list[tuple[str, str, dict[str, str]]] = []

        for pr in PRIORITIES:
            if len(raw) >= count:
                break
            s = self.stream(kind, pr)
            res = await self.r.xautoclaim(
                s, GROUP, consumer, min_idle_time=self.visibility_timeout_ms, start_id="0-0", count=count - len(raw)
            )
            for mid, fields in (res[1] if res else []):
                if fields:
                    raw.append((s, mid, fields))

        for pr in PRIORITIES:
            if len(raw) >= count:
                break
            s = self.stream(kind, pr)
            resp = await self.r.xreadgroup(GROUP, consumer, {s: ">"}, count=count - len(raw))
            for _stream, msgs in resp or []:
                raw.extend((s, mid, fields) for mid, fields in msgs)

        if not raw and block_ms > 0:
            streams = {self.stream(kind, pr): ">" for pr in PRIORITIES}
            resp = await self.r.xreadgroup(GROUP, consumer, streams, count=count, block=block_ms)
            for stream_name, msgs in resp or []:
                raw.extend((stream_name, mid, fields) for mid, fields in msgs)

        jobs: list[ClaimedJob] = []
        for stream_name, mid, fields in raw:
            job_id = fields.get("job_id", "")
            key = self.job_key(job_id)
            status = await self.r.hget(key, "status")
            if status is None or status == DONE:
                # Unknown job (expired) or finished before a crash hid the ack: just ack.
                await self._ack(stream_name, mid)
                continue
            attempts = await self.r.hincrby(key, "attempts", 1)
            if attempts > self.max_attempts:
                await self._dead_letter(kind, stream_name, mid, job_id, "max_attempts_exceeded")
                continue
            now = time.time()
            await self.r.hset(key, mapping={"status": RUNNING, "worker": consumer, "started_at": now, "updated_at": now})
            data = _decode_job(await self.r.hgetall(key))
            jobs.append(ClaimedJob(job_id=job_id, msg_id=mid, stream=stream_name, attempts=attempts, data=data))
        return jobs

    async def complete(self, job: ClaimedJob, result: dict[str, Any]) -> None:
        now = time.time()
        key = self.job_key(job.job_id)
        async with self.r.pipeline(transaction=True) as pipe:
            pipe.hset(
                key,
                mapping={
                    "status": DONE,
                    "result": json.dumps(result, ensure_ascii=False),
                    "finished_at": now,
                    "updated_at": now,
                    "error": "",
                },
            )
            pipe.expire(key, self.result_ttl_s)
            pipe.xack(job.stream, GROUP, job.msg_id)
            pipe.xdel(job.stream, job.msg_id)
            await pipe.execute()

    async def fail(self, job: ClaimedJob, error: str, *, retry: bool = True) -> None:
        if retry and job.attempts < self.max_attempts:
            now = time.time()
            async with self.r.pipeline(transaction=True) as pipe:
                pipe.hset(self.job_key(job.job_id), mapping={"status": QUEUED, "error": error[:2000], "updated_at": now})
                pipe.xadd(job.stream, {"job_id": job.job_id})
                pipe.xack(job.stream, GROUP, job.msg_id)
                pipe.xdel(job.stream, job.msg_id)
                await pipe.execute()
        else:
            await self._dead_letter(job.kind, job.stream, job.msg_id, job.job_id, error)

    async def release(self, jobs: list[ClaimedJob]) -> None:
        """Return claimed jobs to the queue without spending an attempt (e.g. the GPU just vanished)."""
        for job in jobs:
            async with self.r.pipeline(transaction=True) as pipe:
                pipe.hincrby(self.job_key(job.job_id), "attempts", -1)
                pipe.hset(self.job_key(job.job_id), mapping={"status": QUEUED, "updated_at": time.time()})
                pipe.xadd(job.stream, {"job_id": job.job_id})
                pipe.xack(job.stream, GROUP, job.msg_id)
                pipe.xdel(job.stream, job.msg_id)
                await pipe.execute()

    async def _ack(self, stream: str, msg_id: str) -> None:
        async with self.r.pipeline(transaction=True) as pipe:
            pipe.xack(stream, GROUP, msg_id)
            pipe.xdel(stream, msg_id)
            await pipe.execute()

    async def _dead_letter(self, kind: str, stream: str, msg_id: str, job_id: str, error: str) -> None:
        now = time.time()
        key = self.job_key(job_id)
        async with self.r.pipeline(transaction=True) as pipe:
            pipe.hset(key, mapping={"status": FAILED, "error": error[:2000], "finished_at": now, "updated_at": now})
            pipe.expire(key, self.result_ttl_s)
            pipe.xadd(self.dead_stream(kind), {"job_id": job_id, "error": error[:500]}, maxlen=100_000, approximate=True)
            pipe.xack(stream, GROUP, msg_id)
            pipe.xdel(stream, msg_id)
            await pipe.execute()

    # ---- queries ----------------------------------------------------------------------------
    async def get(self, job_id: str) -> dict[str, Any] | None:
        raw = await self.r.hgetall(self.job_key(job_id))
        return _decode_job(raw) if raw else None

    async def depth(self, kind: str) -> dict[str, int]:
        """``queued`` = waiting, ``running`` = claimed and not yet acked, per priority summed."""
        await self.ensure_group(kind)
        total = running = 0
        for pr in PRIORITIES:
            s = self.stream(kind, pr)
            total += await self.r.xlen(s)
            info = await self.r.xpending(s, GROUP)
            running += int(info["pending"]) if isinstance(info, dict) else int(info[0])
        return {"queued": max(total - running, 0), "running": running, "dead": await self.r.xlen(self.dead_stream(kind))}

    async def wait(self, job_id: str, timeout_s: float, poll_s: float = 0.1) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout_s
        while True:
            job = await self.get(job_id)
            if job is None or job["status"] in TERMINAL or time.monotonic() >= deadline:
                return job
            await asyncio.sleep(poll_s)
