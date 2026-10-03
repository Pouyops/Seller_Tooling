#!/usr/bin/env python
"""End-to-end demo with the **real** model: a Telegram photo update in, a cutout PNG out.

    python tools/e2e_demo.py --image <jpg> --model birefnet_lite

Runs the production code path — handlers -> CutoutService -> inference API -> queue -> GPU worker ->
blob store -> delivery — with only Telegram's HTTP layer faked (no bot token exists yet, H-004).

Uses the real queue server at ST_REDIS_URL when one is reachable, and falls back to an in-process
fake so the demo still runs when Docker is down (it says which it used).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot" / "tests"))  # MockSession lives with the bot tests

for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; this output is Persian
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

import httpx  # noqa: E402
from aiogram import Bot, Dispatcher  # noqa: E402
from aiogram.types import Chat, Message, PhotoSize, Update, User  # noqa: E402
from mock_session import MockSession  # noqa: E402

from st_bot.config import BotSettings  # noqa: E402
from st_bot.handlers import build_router  # noqa: E402
from st_bot.inference_client import InferenceClient  # noqa: E402
from st_bot.payments import StubPaymentProvider  # noqa: E402
from st_bot.service import CutoutService  # noqa: E402
from st_inference.api import create_app  # noqa: E402
from st_inference.settings import ServiceSettings  # noqa: E402
from st_inference.worker import Worker  # noqa: E402


async def make_redis(url: str):
    import redis.asyncio as aioredis

    try:
        client = aioredis.from_url(url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
        await client.ping()
        print(f"[demo] queue: {url}")
        return client, True
    except Exception as e:
        import fakeredis

        print(f"[demo] queue: in-process fake ({type(e).__name__}: {str(e)[:60]}) — start Docker for the real thing")
        return fakeredis.FakeAsyncRedis(decode_responses=True), False


async def run(args) -> int:
    image = Path(args.image).read_bytes()
    print(f"[demo] input: {args.image} ({len(image) / 1024:.0f} KB), model={args.model}")

    service_settings = ServiceSettings(matting_model=args.model, allowed_models=args.model,
                                       max_batch=1, batch_wait_ms=0, log_json=False, log_level="WARNING")
    bot_settings = BotSettings(job_wait_s=args.timeout, log_json=False, log_level="WARNING")
    redis, real_queue = await make_redis(args.redis or str(service_settings.redis_url))

    app = create_app(service_settings, redis_client=redis, configure_logging=False)
    async with app.router.lifespan_context(app):
        client = InferenceClient("http://inference", transport=httpx.ASGITransport(app=app))
        service = CutoutService(bot_settings, redis, client)
        session = MockSession()
        bot = Bot("123456:AAHdemo-token-not-a-real-bot", session=session)
        dp = Dispatcher()
        router = build_router(service, StubPaymentProvider(redis, service.ledger), bot_settings)
        dp.include_router(router)

        worker = Worker(service_settings, redis_client=redis, consumer="demo-worker")
        stop = asyncio.Event()
        t_model = time.perf_counter()

        async def worker_loop():
            while not stop.is_set():
                await worker.step(block_ms=0)
                await asyncio.sleep(0.02)

        task = asyncio.create_task(worker_loop())

        session.add_file("demo-photo", image)
        msg = Message(message_id=1, date=datetime.now(timezone.utc), chat=Chat(id=1, type="private"),
                      from_user=User(id=1, is_bot=False, first_name="فروشنده"),
                      photo=[PhotoSize(file_id="demo-photo", file_unique_id="demo-photo",
                                       width=1080, height=1350, file_size=len(image))])
        t0 = time.perf_counter()
        await dp.feed_update(bot, Update(update_id=1, message=msg))
        elapsed = time.perf_counter() - t0
        inline = bool(session.documents)

        # Cold start usually exceeds the API's inline wait, so the bot says "queued" and the
        # delivery loop hands the result over when the worker finishes. Exercise that path too.
        from st_bot.handlers import send_result

        async def deliver(user_id, chat_id, result):
            await send_result(bot, chat_id, result, bot_settings)

        deadline = time.perf_counter() + args.timeout
        while not session.documents and time.perf_counter() < deadline:
            await service.poll_pending(deliver)
            await asyncio.sleep(0.5)
        total = time.perf_counter() - t0

        # second photo, model now warm: this is the steady-state latency a seller sees
        warm_ms = None
        second = Path(args.image2) if args.image2 else Path(args.image).parent / "jewelry_000.jpg"
        if session.documents and second.exists():
            session.add_file("demo-photo-2", second.read_bytes())  # a different product = a different job
            msg2 = Message(message_id=2, date=datetime.now(timezone.utc), chat=Chat(id=1, type="private"),
                           from_user=User(id=1, is_bot=False, first_name="فروشنده"),
                           photo=[PhotoSize(file_id="demo-photo-2", file_unique_id="demo-photo-2",
                                            width=1080, height=1350, file_size=len(image))])
            t2 = time.perf_counter()
            await dp.feed_update(bot, Update(update_id=2, message=msg2))
            warm_ms = (time.perf_counter() - t2) * 1000

        stop.set()
        task.cancel()
        await router.albums.drain()
        await client.aclose()

        print(f"[demo] worker device={worker.device} state={worker.state} model={args.model}")
        print(f"[demo] first photo: {total:.1f}s total ({'inline' if inline else 'queued then delivered'}), "
              f"including {elapsed:.1f}s before the bot replied and a cold model load")
        if warm_ms:
            print(f"[demo] second photo (warm model): {warm_ms:.0f} ms end to end")
        for text in session.texts:
            print(f"[bot -> seller] {text}")
        docs = session.documents
        if not docs:
            print("[demo] FAILED: no document was sent")
            return 1
        out = Path(args.out or Path(args.image).with_suffix(".cutout.png"))
        out.write_bytes(docs[0].files[0])
        from PIL import Image

        img = Image.open(io.BytesIO(docs[0].files[0]))
        import numpy as np

        alpha = np.asarray(img)[..., 3] if img.mode == "RGBA" else None
        print(f"[demo] cutout: {out}  {img.mode} {img.size}"
              + (f"  foreground={float((alpha >= 128).mean()) * 100:.1f}%" if alpha is not None else ""))
        print(f"[demo] queue was {'real (Valkey/Redis)' if real_queue else 'faked in-process'}")
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_img = Path("D:/seller-tooling/data/bench/synth-v1-s1403-n200/images/carpet_000.jpg")
    ap.add_argument("--image", default=str(default_img if default_img.exists() else ""))
    ap.add_argument("--image2", default=None, help="second photo, to time a warm model")
    ap.add_argument("--model", default="birefnet_lite")
    ap.add_argument("--redis", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()
    if not args.image:
        raise SystemExit("--image is required (no benchmark image found to default to)")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
