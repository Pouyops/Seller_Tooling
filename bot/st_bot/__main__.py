"""Run the bot:  python -m st_bot   (needs ST_TELEGRAM_TOKEN; see HUMAN_NEEDED H-004)."""

from __future__ import annotations

import asyncio
import contextlib

import redis.asyncio as aioredis
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.types import BotCommand

from st_common.logs import get_logger, setup_logging

from . import texts as T
from .config import BotSettings, get_bot_settings
from .handlers import build_router, send_result
from .inference_client import InferenceClient
from .payments import StubPaymentProvider
from .service import CutoutService

log = get_logger("st_bot.main")


def make_bot(settings: BotSettings) -> Bot:
    session = None
    if settings.telegram_api_base:
        session = AiohttpSession(api=TelegramAPIServer.from_base(settings.telegram_api_base.rstrip("/")))
    return Bot(settings.telegram_token, session=session, default=DefaultBotProperties(parse_mode=None))


async def delivery_loop(bot: Bot, service: CutoutService, settings: BotSettings, stop: asyncio.Event) -> None:
    """Deliver jobs that finished after the inline wait; also drains anything left by a restart."""
    async def deliver(user_id: int, chat_id: int, result) -> None:
        with contextlib.suppress(Exception):
            await send_result(bot, chat_id, result, settings)

    while not stop.is_set():
        try:
            n = await service.poll_pending(deliver)
            if n:
                log.info("pending.delivered", extra={"jobs": n})
        except Exception as e:  # never let the loop die
            log.warning("pending.error", extra={"error": str(e)[:200]})
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(settings.deliver_poll_s):
                await stop.wait()


async def run() -> None:
    settings = get_bot_settings()
    setup_logging("telegram-bot", settings.log_level, settings.log_json)
    if not settings.telegram_token:
        raise SystemExit("ST_TELEGRAM_TOKEN is not set. A human must create the bot with BotFather "
                         "(HUMAN_NEEDED.md H-004); the test suite runs without a token.")
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    client = InferenceClient(settings.inference_url)
    service = CutoutService(settings, redis, client)
    payments = StubPaymentProvider(redis, service.ledger)
    bot = make_bot(settings)
    dp = Dispatcher()
    router = build_router(service, payments, settings)
    dp.include_router(router)

    await bot.set_my_commands([BotCommand(command=c, description=d) for c, d in T.COMMANDS], language_code="fa")
    await bot.set_my_commands([BotCommand(command=c, description=d) for c, d in T.COMMANDS])

    stop = asyncio.Event()
    delivery = asyncio.create_task(delivery_loop(bot, service, settings, stop))
    log.info("bot.started", extra={"inference_url": settings.inference_url})
    try:
        await dp.start_polling(bot)
    finally:
        stop.set()
        delivery.cancel()
        await router.albums.drain()
        await client.aclose()
        await redis.aclose()
        await bot.session.close()


def main() -> None:
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run())


if __name__ == "__main__":
    main()
