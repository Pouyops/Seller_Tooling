"""End-to-end: a Telegram photo update goes through the real handlers, service, inference API and
GPU worker (with a CPU stand-in model) and comes back as a PNG cutout with an alpha channel.

Everything except Telegram's own HTTP layer and the matting model is the production code path.
"""

from __future__ import annotations

import asyncio
import io
from datetime import datetime, timezone

import httpx
import numpy as np
import pytest
from PIL import Image

fakeredis = pytest.importorskip("fakeredis")

from aiogram import Bot, Dispatcher  # noqa: E402
from aiogram.types import CallbackQuery, Chat, Document, Message, PhotoSize, Update, User  # noqa: E402
from mock_session import MockSession  # noqa: E402

from st_bot.config import BotSettings  # noqa: E402
from st_bot.handlers import build_router, send_result  # noqa: E402
from st_bot.inference_client import InferenceClient  # noqa: E402
from st_bot.payments import PACKS, StubPaymentProvider  # noqa: E402
from st_bot.service import CutoutService  # noqa: E402
from st_inference.api import create_app  # noqa: E402
from st_inference.models.base import Timing  # noqa: E402
from st_inference.settings import ServiceSettings  # noqa: E402
from st_inference.worker import Worker  # noqa: E402

USER, CHAT = 777, 777


class FakeModel:
    """Stand-in for a matting model: foreground = pixels brighter than mid-grey."""

    name, device, precision = "fake", "cpu", "fp32"

    def __init__(self):
        self.last_timing = Timing(1, 2, 3)
        self.calls: list[int] = []

    def load(self):
        return self

    def unload(self):
        pass

    def predict(self, images):
        self.calls.append(len(images))
        return [(im.mean(axis=2) > 110).astype(np.float32) for im in images]


def product_photo(seed: int = 0, w: int = 80, h: int = 60) -> bytes:
    """A bright 'product' on a dark background, so the fake model yields a meaningful mask."""
    rng = np.random.default_rng(seed)
    arr = np.full((h, w, 3), 20, np.uint8)
    arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = (235, 205, 180)
    arr = (arr + rng.integers(0, 12, arr.shape)).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def bot_settings(tmp_path) -> BotSettings:
    return BotSettings(data_dir=tmp_path / "bot", free_quota_per_month=20, rate_limit_per_minute=50,
                       job_wait_s=6.0, album_collect_ms=120, log_json=False, payments_dev_confirm=True)


@pytest.fixture
def service_settings(tmp_path) -> ServiceSettings:
    return ServiceSettings(data_dir=tmp_path / "api", matting_model="fake", allowed_models="fake",
                           batch_wait_ms=0, max_batch=4, log_json=False, visibility_timeout_ms=5000)


@pytest.fixture
async def stack(bot_settings, service_settings):
    """Wire bot -> inference API -> worker, all in-process."""
    api_redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    bot_redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    app = create_app(service_settings, redis_client=api_redis, configure_logging=False)
    model = FakeModel()
    async with app.router.lifespan_context(app):
        client = InferenceClient("http://api", transport=httpx.ASGITransport(app=app), retries=2)
        service = CutoutService(bot_settings, bot_redis, client)
        payments = StubPaymentProvider(bot_redis, service.ledger)
        session = MockSession()
        bot = Bot("123456:AAHtest-token-for-the-mock-session", session=session)
        dp = Dispatcher()
        router = build_router(service, payments, bot_settings)
        dp.include_router(router)
        worker = Worker(service_settings, redis_client=api_redis,
                        model_factory=lambda name, dev: model, device_probe=lambda: "cpu", consumer="w-e2e")

        stop = asyncio.Event()

        async def worker_loop():
            while not stop.is_set():
                await worker.step(block_ms=0)
                await asyncio.sleep(0.02)

        task = asyncio.create_task(worker_loop())
        ns = type("Stack", (), {})()
        ns.bot, ns.dp, ns.session, ns.service, ns.payments, ns.model = bot, dp, session, service, payments, model
        ns.worker, ns.stop, ns.router = worker, stop, router
        ns.update_id = 0
        try:
            yield ns
        finally:
            stop.set()
            task.cancel()
            await router.albums.drain()
            await client.aclose()
    await api_redis.aclose()
    await bot_redis.aclose()


def _msg(stack, **kw) -> Message:
    stack.update_id += 1
    return Message(message_id=stack.update_id, date=datetime.now(timezone.utc),
                   chat=Chat(id=CHAT, type="private"),
                   from_user=User(id=USER, is_bot=False, first_name="علی"), **kw)


async def feed(stack, message: Message) -> None:
    stack.update_id += 1
    await stack.dp.feed_update(stack.bot, Update(update_id=stack.update_id, message=message))


async def send_photo(stack, data: bytes | None = None, file_id: str = "ph1", media_group_id: str | None = None) -> None:
    data = data if data is not None else product_photo()
    stack.session.add_file(file_id, data)
    photo = PhotoSize(file_id=file_id, file_unique_id=file_id, width=80, height=60, file_size=len(data))
    await feed(stack, _msg(stack, photo=[photo], media_group_id=media_group_id))


async def send_command(stack, text: str) -> None:
    await feed(stack, _msg(stack, text=text))


async def click(stack, data: str) -> None:
    stack.update_id += 1
    call = CallbackQuery(id=str(stack.update_id), from_user=User(id=USER, is_bot=False, first_name="علی"),
                         chat_instance="ci", data=data, message=_msg(stack, text="x"))
    await stack.dp.feed_update(stack.bot, Update(update_id=stack.update_id, callback_query=call))


def decode_png(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


# ---- the main flow ----------------------------------------------------------------------------
async def test_photo_in_cutout_out(stack):
    await send_photo(stack)

    docs = stack.session.documents
    assert len(docs) == 1, f"expected one document, got {[c.method for c in stack.session.calls]}"
    img = decode_png(docs[0].files[0])
    assert img.mode == "RGBA" and img.size == (80, 60)
    alpha = np.asarray(img)[..., 3]
    assert alpha[30, 40] == 255, "product centre should be opaque"
    assert alpha[2, 2] == 0, "corner background should be transparent"
    assert docs[0].filename == "cutout.png"
    assert "اعتبار باقی‌مانده" in (docs[0].caption or "")
    assert stack.model.calls, "the worker never ran the model"
    assert (await stack.service.balance(USER)).free_used == 1


async def test_image_sent_as_file_is_accepted(stack):
    data = product_photo(seed=3)
    stack.session.add_file("doc-1", data)
    doc = Document(file_id="doc-1", file_unique_id="doc-1", file_name="product.jpg",
                   mime_type="image/jpeg", file_size=len(data))
    await feed(stack, _msg(stack, document=doc))
    assert len(stack.session.documents) == 1
    assert decode_png(stack.session.documents[0].files[0]).mode == "RGBA"


async def test_non_image_file_is_rejected(stack):
    stack.session.add_file("doc-2", b"%PDF-1.4 not an image")
    doc = Document(file_id="doc-2", file_unique_id="doc-2", file_name="price-list.pdf",
                   mime_type="application/pdf", file_size=20)
    await feed(stack, _msg(stack, document=doc))
    assert stack.session.documents == []
    assert any("تصویر نیست" in t for t in stack.session.texts)


async def test_album_returns_a_media_group(stack):
    for i in range(3):
        await send_photo(stack, product_photo(seed=i), file_id=f"alb{i}", media_group_id="grp-1")
    await asyncio.sleep(stack.service.s.album_collect_ms / 1000 + 2.0)

    groups = stack.session.of("SendMediaGroup")
    assert len(groups) == 1 and len(groups[0].files) == 3
    assert all(decode_png(f).mode == "RGBA" for f in groups[0].files)
    assert any("۳ عکس دریافت شد" in t for t in stack.session.texts)


async def test_white_background_preference_changes_output_format(stack):
    await click(stack, "bg:white")
    assert any("سفید" in t for t in stack.session.texts)
    stack.session.clear()
    await send_photo(stack, file_id="ph-white")
    doc = stack.session.documents[0]
    assert doc.filename == "cutout.jpg"
    img = decode_png(doc.files[0])
    assert img.mode == "RGB"
    assert np.asarray(img)[2, 2].min() > 235, "background should be composited on white"


# ---- deferred delivery (the queue outliving the inline wait) -----------------------------------
async def test_slow_job_is_queued_then_delivered_later(stack):
    stack.stop.set()  # pause the worker: nothing can finish while the handler waits
    await asyncio.sleep(0.05)
    stack.service.s.job_wait_s = 0.2

    await send_photo(stack, file_id="slow-1")
    assert stack.session.documents == [], "nothing should be delivered while the worker is down"
    assert any("صف پردازش" in t for t in stack.session.texts)

    # worker comes back
    for _ in range(50):
        if await stack.worker.step(block_ms=0):
            break
        await asyncio.sleep(0.02)

    delivered: list = []

    async def deliver(user_id, chat_id, result):
        delivered.append(result)
        await send_result(stack.bot, chat_id, result, stack.service.s)

    assert await stack.service.poll_pending(deliver) == 1
    assert delivered[0].kind == "done"
    assert decode_png(stack.session.documents[0].files[0]).mode == "RGBA"
    # the pending entry is cleared, so a restart would not re-deliver
    assert await stack.service.poll_pending(deliver) == 0


async def test_failed_job_refunds_the_user(stack):
    stack.stop.set()
    await asyncio.sleep(0.05)
    stack.service.s.job_wait_s = 0.2
    await send_photo(stack, file_id="doomed")
    assert (await stack.service.balance(USER)).free_used == 1

    # the job dies permanently in the worker
    jobs = await stack.worker.queue.claim("matting", "w", 1)
    await stack.worker.queue.fail(jobs[0], "boom", retry=False)

    results: list = []

    async def deliver(user_id, chat_id, result):
        results.append(result)
        await send_result(stack.bot, chat_id, result, stack.service.s)

    assert await stack.service.poll_pending(deliver) == 1
    assert results[0].kind == "failed"
    assert (await stack.service.balance(USER)).free_used == 0, "a failed job must not consume quota"
    assert any("اعتبارتان برگشت" in t for t in stack.session.texts)


# ---- quota, rate limiting, commands ------------------------------------------------------------
async def test_quota_runs_out(stack):
    stack.service.ledger.free_per_month = 1
    await send_photo(stack, product_photo(seed=11), file_id="q1")
    stack.session.clear()
    await send_photo(stack, product_photo(seed=12), file_id="q2")
    assert stack.session.documents == []
    assert any("اعتبار رایگان این ماه تمام شده" in t for t in stack.session.texts)


async def test_rate_limit_is_per_user(stack):
    stack.service.limiter.per_minute = 1
    await send_photo(stack, product_photo(seed=21), file_id="r1")
    stack.session.clear()
    await send_photo(stack, product_photo(seed=22), file_id="r2")
    assert any("آهسته‌تر" in t for t in stack.session.texts)
    assert stack.session.documents == []


async def test_resending_the_same_photo_is_free(stack):
    data = product_photo(seed=31)
    await send_photo(stack, data, file_id="same-1")
    await send_photo(stack, data, file_id="same-2")
    assert len(stack.session.documents) == 2
    assert (await stack.service.balance(USER)).free_used == 1, "identical image = same job = one charge"


@pytest.mark.parametrize("command, needle", [("/start", "ثبت‌نام لازم نیست"), ("/help", "راهنما"), ("/quota", "اعتبار شما")])
async def test_persian_commands(stack, command, needle):
    await send_command(stack, command)
    assert any(needle in t for t in stack.session.texts), stack.session.texts


async def test_buy_flow_is_a_stub_until_a_gateway_exists(stack):
    await send_command(stack, "/buy")
    kb = stack.session.of("SendMessage")[-1].obj.reply_markup
    assert len(kb.inline_keyboard) == len(PACKS)

    await click(stack, f"buy:{PACKS[0].id}")
    assert any("درگاه پرداخت هنوز فعال نشده" in t for t in stack.session.texts)
    assert (await stack.service.balance(USER)).credits == 0

    invoice_id = (await stack.payments.create_invoice(USER, PACKS[0])).id
    await click(stack, f"devpay:{invoice_id}")
    assert (await stack.service.balance(USER)).credits == PACKS[0].credits


async def test_unknown_text_gets_help(stack):
    await send_command(stack, "سلام")
    assert any("راهنما" in t for t in stack.session.texts)
