"""Per-user rate limiting, free monthly quota and paid credits, stored in Redis/Valkey.

Zero signup: the Telegram user id is the account. Nothing else is collected.

Charging rule: one unit per *distinct* job per user. Re-sending the same photo (same job id)
is free, and a job that fails is refunded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .jalali import period_key

P = "st:bot"


@dataclass
class RateDecision:
    allowed: bool
    retry_after_s: int = 0


class RateLimiter:
    """Fixed one-minute window. Coarse but constant-memory and atomic (INCR)."""

    def __init__(self, redis, per_minute: int):
        self.r, self.per_minute = redis, per_minute

    async def hit(self, user_id: int, now: float | None = None) -> RateDecision:
        now = now or time.time()
        bucket = int(now // 60)
        key = f"{P}:rl:{user_id}:{bucket}"
        n = await self.r.incr(key)
        if n == 1:
            await self.r.expire(key, 90)
        if n > self.per_minute:
            return RateDecision(False, int(60 - now % 60) + 1)
        return RateDecision(True)


@dataclass
class Balance:
    free_used: int
    free_total: int
    credits: int

    @property
    def free_left(self) -> int:
        return max(0, self.free_total - self.free_used)

    @property
    def available(self) -> int:
        return self.free_left + self.credits


class Ledger:
    def __init__(self, redis, free_per_month: int):
        self.r, self.free_per_month = redis, free_per_month

    def _free_key(self, user_id: int, now=None) -> str:
        return f"{P}:free:{user_id}:{period_key(now)}"

    def _credit_key(self, user_id: int) -> str:
        return f"{P}:credits:{user_id}"

    def _charged_key(self, user_id: int) -> str:
        return f"{P}:charged:{user_id}"

    async def balance(self, user_id: int, now=None) -> Balance:
        used = int(await self.r.get(self._free_key(user_id, now)) or 0)
        credits = int(await self.r.get(self._credit_key(user_id)) or 0)
        return Balance(min(used, self.free_per_month), self.free_per_month, credits)

    async def charge(self, user_id: int, job_id: str, now=None) -> str | None:
        """Spend one unit for ``job_id``. Returns "free", "credit", "already_charged" or None (nothing left)."""
        charged = self._charged_key(user_id)
        if await self.r.hget(charged, job_id):
            return "already_charged"
        free_key = self._free_key(user_id, now)
        used = await self.r.incr(free_key)
        if used == 1:
            await self.r.expire(free_key, 40 * 86400)
        if used <= self.free_per_month:
            kind = "free"
        else:
            await self.r.decr(free_key)
            left = await self.r.decrby(self._credit_key(user_id), 1)
            if left < 0:
                await self.r.incrby(self._credit_key(user_id), 1)
                return None
            kind = "credit"
        # If two identical submissions race, only the first keeps its charge.
        if not await self.r.hsetnx(charged, job_id, f"{kind}:{free_key}"):
            await self._undo(user_id, kind, free_key)
            return "already_charged"
        await self.r.expire(charged, 40 * 86400)
        return kind

    async def refund(self, user_id: int, job_id: str) -> bool:
        raw = await self.r.hget(self._charged_key(user_id), job_id)
        if not raw:
            return False
        kind, free_key = raw.split(":", 1)
        if await self.r.hdel(self._charged_key(user_id), job_id):
            await self._undo(user_id, kind, free_key)
            return True
        return False

    async def _undo(self, user_id: int, kind: str, free_key: str) -> None:
        if kind == "free":
            await self.r.decr(free_key)
        else:
            await self.r.incrby(self._credit_key(user_id), 1)

    async def add_credits(self, user_id: int, n: int) -> int:
        return int(await self.r.incrby(self._credit_key(user_id), n))
