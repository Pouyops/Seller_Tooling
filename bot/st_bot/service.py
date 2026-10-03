"""Transport-agnostic bot logic: quota, submission, delivery, resumable pending jobs.

Kept free of aiogram types so the same core can serve a Bale/Eitaa adapter later (H-004), and so
tests can drive it without a Telegram session.

Delivery is at-least-once: a job stays in the pending hash until its result has been handed to the
deliver callback. A bot restart (or power cut) resumes delivery from Redis.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from redis.exceptions import RedisError

from st_common.logs import get_logger

from .config import BotSettings
from .inference_client import InferenceClient, InferenceRejected, InferenceUnavailable
from .ledger import Balance, Ledger, RateLimiter

log = get_logger("st_bot.service")

PENDING = "st:bot:pending"
PREFS = "st:bot:prefs"

DONE, QUEUED, FAILED = "done", "queued", "failed"
RATE_LIMITED, NO_QUOTA, REJECTED, UNAVAILABLE = "rate_limited", "no_quota", "rejected", "unavailable"


@dataclass
class Result:
    kind: str
    job_id: str | None = None
    image: bytes | None = None
    mime: str = "image/png"
    retry_after_s: int = 0
    left: int = 0
    reason: str = ""

    @property
    def filename(self) -> str:
        return "cutout.jpg" if self.mime == "image/jpeg" else "cutout.png"


class CutoutService:
    def __init__(self, settings: BotSettings, redis, client: InferenceClient,
                 ledger: Ledger | None = None, limiter: RateLimiter | None = None):
        self.s = settings
        self.r = redis
        self.client = client
        self.ledger = ledger or Ledger(redis, settings.free_quota_per_month)
        self.limiter = limiter or RateLimiter(redis, settings.rate_limit_per_minute)

    # ---- preferences --------------------------------------------------------------------------
    async def background_for(self, user_id: int) -> str:
        return (await self.r.hget(f"{PREFS}:{user_id}", "background")) or "transparent"

    async def set_background(self, user_id: int, value: str) -> None:
        await self.r.hset(f"{PREFS}:{user_id}", "background", value)

    async def balance(self, user_id: int) -> Balance:
        return await self.ledger.balance(user_id)

    # ---- main path ----------------------------------------------------------------------------
    async def process(self, user_id: int, chat_id: int, image: bytes, filename: str = "photo.jpg") -> Result:
        decision = await self.limiter.hit(user_id)
        if not decision.allowed:
            return Result(RATE_LIMITED, retry_after_s=decision.retry_after_s)

        bal = await self.balance(user_id)
        if bal.available <= 0:
            return Result(NO_QUOTA, left=0)

        background = await self.background_for(user_id)
        try:
            job = await self.client.submit(image, filename=filename, background=background,
                                           user_id=user_id, wait_s=self.s.job_wait_s)
        except InferenceRejected as e:
            return Result(REJECTED, reason=e.code)
        except InferenceUnavailable as e:
            log.warning("inference.unavailable", extra={"error": str(e)[:200]})
            return Result(UNAVAILABLE)

        job_id = job["job_id"]
        charged = await self.ledger.charge(user_id, job_id)
        if charged is None:
            return Result(NO_QUOTA, job_id=job_id, left=0)

        if job.get("status") == DONE:
            return await self._deliverable(user_id, job_id)
        if job.get("status") == FAILED:
            await self.ledger.refund(user_id, job_id)
            return Result(FAILED, job_id=job_id, reason=job.get("error") or "")

        await self.r.hset(PENDING, job_id, json.dumps({"user_id": user_id, "chat_id": chat_id, "at": time.time()}))
        return Result(QUEUED, job_id=job_id)

    async def _deliverable(self, user_id: int, job_id: str) -> Result:
        try:
            data, mime = await self.client.download(job_id)
        except (InferenceRejected, InferenceUnavailable) as e:
            log.warning("download.failed", extra={"job_id": job_id, "error": str(e)[:200]})
            return Result(QUEUED, job_id=job_id)
        left = (await self.balance(user_id)).available
        return Result(DONE, job_id=job_id, image=data, mime=mime, left=left)

    # ---- deferred delivery --------------------------------------------------------------------
    async def poll_pending(self, deliver: Callable[[int, int, Result], Awaitable[None]]) -> int:
        """Deliver finished jobs that outlived the inline wait. Safe to call on a timer and at startup."""
        try:
            pending = await self.r.hgetall(PENDING)
        except RedisError as e:
            log.warning("pending.unavailable", extra={"error": str(e)[:200]})
            return 0
        delivered = 0
        for job_id, raw in pending.items():
            try:
                info = json.loads(raw)
            except json.JSONDecodeError:
                await self.r.hdel(PENDING, job_id)
                continue
            user_id, chat_id = int(info["user_id"]), int(info["chat_id"])
            try:
                job = await self.client.get(job_id)
            except (InferenceUnavailable, InferenceRejected):
                continue
            age = time.time() - float(info.get("at", 0))
            if job is None or job.get("status") == FAILED:
                await self.ledger.refund(user_id, job_id)
                await self.r.hdel(PENDING, job_id)
                await deliver(user_id, chat_id, Result(FAILED, job_id=job_id, reason=(job or {}).get("error", "lost")))
                delivered += 1
            elif job.get("status") == DONE:
                result = await self._deliverable(user_id, job_id)
                if result.kind != DONE:
                    continue
                await deliver(user_id, chat_id, result)
                await self.r.hdel(PENDING, job_id)
                delivered += 1
            elif age > self.s.job_give_up_s:
                await self.ledger.refund(user_id, job_id)
                await self.r.hdel(PENDING, job_id)
                await deliver(user_id, chat_id, Result(FAILED, job_id=job_id, reason="timeout"))
                delivered += 1
        return delivered
