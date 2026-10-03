import re
from datetime import datetime, timezone

import pytest

fakeredis = pytest.importorskip("fakeredis")

from st_bot import texts  # noqa: E402
from st_bot.jalali import gregorian_to_jalali, period_key, period_label_fa  # noqa: E402
from st_bot.ledger import Ledger, RateLimiter  # noqa: E402
from st_bot.payments import PACKS, StubPaymentProvider  # noqa: E402
from st_common.persian import normalize, validate_persian  # noqa: E402


@pytest.fixture
async def r():
    c = fakeredis.FakeAsyncRedis(decode_responses=True)
    yield c
    await c.aclose()


@pytest.mark.parametrize(
    "g, j",
    [((2024, 3, 20), (1403, 1, 1)), ((2026, 3, 21), (1405, 1, 1)), ((2026, 9, 15), (1405, 6, 24)),
     ((2025, 3, 20), (1403, 12, 30)), ((2000, 1, 1), (1378, 10, 11))],
)
def test_gregorian_to_jalali(g, j):
    assert gregorian_to_jalali(*g) == j


def test_period_uses_tehran_time():
    # 2026-03-20 21:00 UTC is already 1405-01-01 00:30 in Tehran
    assert period_key(datetime(2026, 3, 20, 21, 0, tzinfo=timezone.utc)) == "140501"
    assert period_key(datetime(2026, 3, 20, 20, 0, tzinfo=timezone.utc)) == "140412"
    assert period_label_fa(datetime(2026, 9, 15, tzinfo=timezone.utc)) == "شهریور ۱۴۰۵"


@pytest.mark.parametrize("template", texts.ALL_TEMPLATES)
def test_texts_are_canonical_persian(template):
    sample = re.sub(r"\{\w+\}", "۱", template)
    assert normalize(sample, persian_punctuation=False) == sample, "run st_common.persian.normalize on this string"
    issues = [i for i in validate_persian(sample, min_persian_ratio=0.5, require_normalized=False) if i != "low_persian_ratio"]
    assert issues == []


def test_fa_formats_numbers_in_persian_digits():
    msg = texts.fa(texts.QUOTA, period="شهریور ۱۴۰۵", free_left=7, free_total=20, credits=1200)
    assert "۷ از ۲۰" in msg and "۱٬۲۰۰" in msg and not re.search(r"[0-9]", msg)
    assert texts.pack_label(200, 149_000) == "۲۰۰ عکس – ۱۴۹٬۰۰۰ تومان"


async def test_rate_limiter_window(r):
    rl = RateLimiter(r, per_minute=3)
    t = 1_000_020.0
    assert all([(await rl.hit(1, t)).allowed for _ in range(3)])
    d = await rl.hit(1, t)
    assert not d.allowed and 0 < d.retry_after_s <= 61
    assert (await rl.hit(2, t)).allowed  # per user
    assert (await rl.hit(1, t + 60)).allowed  # next window


async def test_free_quota_then_credits_then_refusal(r):
    led = Ledger(r, free_per_month=2)
    assert await led.charge(7, "a") == "free"
    assert await led.charge(7, "b") == "free"
    assert await led.charge(7, "c") is None
    await led.add_credits(7, 1)
    assert await led.charge(7, "c") == "credit"
    assert await led.charge(7, "d") is None
    bal = await led.balance(7)
    assert (bal.free_used, bal.free_left, bal.credits, bal.available) == (2, 0, 0, 0)


async def test_same_job_is_not_charged_twice_and_refund_restores(r):
    led = Ledger(r, free_per_month=1)
    assert await led.charge(9, "job") == "free"
    assert await led.charge(9, "job") == "already_charged"
    assert (await led.balance(9)).free_left == 0
    assert await led.refund(9, "job") is True
    assert await led.refund(9, "job") is False  # idempotent
    assert (await led.balance(9)).free_left == 1
    await led.add_credits(9, 1)
    await led.charge(9, "x")
    assert await led.charge(9, "y") == "credit"
    await led.refund(9, "y")
    assert (await led.balance(9)).credits == 1


async def test_quota_resets_each_jalali_month(r):
    led = Ledger(r, free_per_month=1)
    shahrivar = datetime(2026, 9, 15, tzinfo=timezone.utc)
    mehr = datetime(2026, 9, 25, tzinfo=timezone.utc)
    assert await led.charge(3, "a", now=shahrivar) == "free"
    assert await led.charge(3, "b", now=shahrivar) is None
    assert await led.charge(3, "b", now=mehr) == "free"


async def test_stub_payment_credits_exactly_once(r):
    led = Ledger(r, free_per_month=0)
    pay = StubPaymentProvider(r, led)
    inv = await pay.create_invoice(5, PACKS[0])
    assert inv.status == "pending" and inv.pay_url is None and inv.amount_toman == PACKS[0].price_toman
    await pay.mark_paid(inv.id)
    await pay.mark_paid(inv.id)  # retried callback
    assert (await led.balance(5)).credits == PACKS[0].credits
    assert (await pay.verify(inv.id)).status == "paid"
    assert await pay.mark_paid("nope") is None
