"""Payment hook. **Stub only**: no real gateway is integrated (brief; HUMAN_NEEDED H-010).

``PaymentProvider`` is the seam a real Iranian PSP adapter (Zarinpal, Zibal, IDPay, ...) will
implement: create an invoice, hand the user a pay URL, verify on callback, then credit the account
exactly once.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import asdict, dataclass
from typing import Protocol

from .ledger import Ledger

P = "st:bot:invoice"


@dataclass(frozen=True)
class Pack:
    id: str
    credits: int
    price_toman: int  # PLACEHOLDER prices, see docs/economics.md and HUMAN_NEEDED H-011


PACKS: tuple[Pack, ...] = (
    Pack("p50", 50, 49_000),
    Pack("p200", 200, 149_000),
    Pack("p1000", 1000, 490_000),
)
PACKS_BY_ID = {p.id: p for p in PACKS}


@dataclass
class Invoice:
    id: str
    user_id: int
    pack_id: str
    amount_toman: int
    status: str  # pending | paid | expired
    pay_url: str | None
    created_at: float


class PaymentProvider(Protocol):
    name: str

    async def create_invoice(self, user_id: int, pack: Pack) -> Invoice: ...

    async def verify(self, invoice_id: str) -> Invoice | None: ...


class StubPaymentProvider:
    """Stores invoices in Redis, never takes money. ``pay_url`` is None so the bot says payments are
    not enabled yet. ``mark_paid`` exists for development and tests only."""

    name = "stub"

    def __init__(self, redis, ledger: Ledger):
        self.r, self.ledger = redis, ledger

    async def create_invoice(self, user_id: int, pack: Pack) -> Invoice:
        inv = Invoice(secrets.token_hex(8), user_id, pack.id, pack.price_toman, "pending", None, time.time())
        await self.r.set(f"{P}:{inv.id}", json.dumps(asdict(inv)), ex=7 * 86400)
        return inv

    async def verify(self, invoice_id: str) -> Invoice | None:
        raw = await self.r.get(f"{P}:{invoice_id}")
        return Invoice(**json.loads(raw)) if raw else None

    async def mark_paid(self, invoice_id: str) -> Invoice | None:
        """Credit exactly once, even if called repeatedly (a PSP callback retried on a flaky network)."""
        inv = await self.verify(invoice_id)
        if inv is None:
            return None
        if not await self.r.set(f"{P}:{invoice_id}:credited", "1", nx=True, ex=30 * 86400):
            inv.status = "paid"
            return inv
        inv.status = "paid"
        await self.r.set(f"{P}:{inv.id}", json.dumps(asdict(inv)), ex=30 * 86400)
        await self.ledger.add_credits(inv.user_id, PACKS_BY_ID[inv.pack_id].credits)
        return inv
