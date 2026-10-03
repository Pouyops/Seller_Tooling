#!/usr/bin/env python
"""Everything a demo needs in one process: API + GPU worker + web page, no Docker required.

    python tools/serve_demo.py            # then open http://127.0.0.1:8000/

`make serve` is the real layout (separate API and worker processes, Valkey as the queue). This is
the convenience version: if no Redis/Valkey is reachable it runs the queue in-process so the page
still works, and it says which it used. Same application code either way.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from st_inference.api import create_app  # noqa: E402
from st_inference.settings import ServiceSettings  # noqa: E402
from st_inference.worker import Worker  # noqa: E402


async def make_redis(url: str):
    import redis.asyncio as aioredis

    try:
        client = aioredis.from_url(url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
        await client.ping()
        return client, f"Valkey/Redis at {url}"
    except Exception:
        import fakeredis

        return fakeredis.FakeAsyncRedis(decode_responses=True), "in-process (no Redis reachable — demo only)"


async def run(args) -> None:
    import uvicorn

    settings = ServiceSettings(matting_model=args.model, max_batch=args.max_batch, log_json=False,
                               log_level=args.log_level)
    redis, queue_kind = await make_redis(settings.redis_url)
    app = create_app(settings, redis_client=redis, configure_logging=False)
    worker = Worker(settings, redis_client=redis, consumer="demo")

    async def worker_loop():
        while not worker.stopping.is_set():
            try:
                await worker.step(block_ms=250)
            except Exception as e:  # keep the demo alive
                print(f"[demo] worker: {type(e).__name__}: {str(e)[:160]}", flush=True)
                await asyncio.sleep(1)

    # The heartbeat key expires after a few seconds, and step() only refreshes it on state changes,
    # so /healthz would report "no worker" on a healthy idle system without this.
    tasks = [asyncio.create_task(worker_loop()), asyncio.create_task(worker._heartbeat_loop())]
    task = tasks[0]
    stop_bot = asyncio.Event()
    bot = None

    if args.bot:
        from aiogram import Dispatcher
        from aiogram.types import BotCommand

        from st_bot import texts as T
        from st_bot.__main__ import delivery_loop, make_bot
        from st_bot.config import get_bot_settings
        from st_bot.handlers import build_router
        from st_bot.inference_client import InferenceClient
        from st_bot.payments import StubPaymentProvider
        from st_bot.service import CutoutService

        bot_settings = get_bot_settings()
        if not bot_settings.telegram_token:
            raise SystemExit("ST_TELEGRAM_TOKEN is not set in .env — see HUMAN_NEEDED.md H-004")
        # The bot talks to the API over real HTTP, exactly as it would in production.
        client = InferenceClient(f"http://127.0.0.1:{args.port}")
        service = CutoutService(bot_settings, redis, client)
        bot = make_bot(bot_settings)
        dp = Dispatcher()
        router = build_router(service, StubPaymentProvider(redis, service.ledger), bot_settings)
        dp.include_router(router)
        me = await bot.get_me()
        for lang in ("fa", None):
            await bot.set_my_commands([BotCommand(command=c, description=d) for c, d in T.COMMANDS], language_code=lang)
        print(f"[demo] telegram bot @{me.username} polling — free quota {bot_settings.free_quota_per_month}/month")
        tasks.append(asyncio.create_task(dp.start_polling(bot, handle_signals=False)))
        tasks.append(asyncio.create_task(delivery_loop(bot, service, bot_settings, stop_bot)))
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level=args.log_level.lower(), access_log=False)
    server = uvicorn.Server(config)
    print(f"[demo] model={args.model}  queue={queue_kind}")
    print(f"[demo] open  http://{'127.0.0.1' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}/")
    print("[demo] the first photo also loads the model, so it takes ~20 s; later ones are quick.")
    try:
        await server.serve()
    finally:
        worker.stopping.set()
        stop_bot.set()
        for t in tasks:
            t.cancel()
        worker._unload_all()
        if bot is not None:
            await bot.session.close()
        await redis.aclose()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="birefnet_lite")
    ap.add_argument("--max-batch", type=int, default=2)
    ap.add_argument("--log-level", default="warning")
    ap.add_argument("--bot", action="store_true", help="also run the Telegram bot (needs ST_TELEGRAM_TOKEN in .env)")
    try:
        asyncio.run(run(ap.parse_args()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
