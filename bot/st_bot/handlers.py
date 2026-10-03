"""aiogram handlers: Persian menu, photos and image files, albums, quota, stub payments."""

from __future__ import annotations

import asyncio
from io import BytesIO

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaDocument,
    Message,
)

from st_common.logs import get_logger, log_context

from . import texts as T
from .config import BotSettings
from .jalali import period_label_fa
from .payments import PACKS, PACKS_BY_ID, StubPaymentProvider
from .service import DONE, FAILED, NO_QUOTA, QUEUED, RATE_LIMITED, REJECTED, UNAVAILABLE, CutoutService, Result

log = get_logger("st_bot.handlers")

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


class AlbumCollector:
    """Telegram sends each photo of an album as its own update; gather them before replying."""

    def __init__(self, settings: BotSettings):
        self.s = settings
        self._groups: dict[str, list[Message]] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    async def add(self, message: Message, flush) -> None:
        gid = message.media_group_id
        self._groups.setdefault(gid, []).append(message)
        if gid in self._tasks:
            return
        self._tasks[gid] = asyncio.create_task(self._wait_and_flush(gid, flush))

    async def _wait_and_flush(self, gid: str, flush) -> None:
        try:
            await asyncio.sleep(self.s.album_collect_ms / 1000)
            messages = self._groups.pop(gid, [])
            await flush(messages)
        finally:
            self._tasks.pop(gid, None)

    async def drain(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()


def _result_text(result: Result, settings: BotSettings) -> str:
    if result.kind == RATE_LIMITED:
        return T.fa(T.RATE_LIMITED, seconds=result.retry_after_s)
    if result.kind == NO_QUOTA:
        return T.fa(T.OUT_OF_QUOTA)
    if result.kind == UNAVAILABLE:
        return T.fa(T.SERVICE_DOWN)
    if result.kind == REJECTED:
        return T.fa(T.TOO_LARGE if result.reason == "too_large" else T.NOT_IMAGE)
    if result.kind == FAILED:
        return T.fa(T.FAILED)
    return T.fa(T.QUEUED_LATER)


async def send_result(bot: Bot, chat_id: int, result: Result, settings: BotSettings) -> None:
    if result.kind == DONE and result.image:
        await bot.send_document(
            chat_id,
            BufferedInputFile(result.image, filename=result.filename),
            caption=T.fa(T.DONE_CAPTION, left=result.left),
        )
    else:
        await bot.send_message(chat_id, _result_text(result, settings))


def build_router(service: CutoutService, payments: StubPaymentProvider, settings: BotSettings) -> Router:
    router = Router(name="st_bot")
    albums = AlbumCollector(settings)
    router.albums = albums  # exposed for shutdown

    async def _download(bot: Bot, message: Message) -> tuple[bytes, str] | None:
        if message.photo:
            buf = await bot.download(message.photo[-1], destination=BytesIO())
            return buf.read(), "photo.jpg"
        doc = message.document
        if doc and ((doc.mime_type or "").startswith("image/") or (doc.file_name or "").lower().endswith(IMAGE_EXT)):
            if doc.file_size and doc.file_size > settings.max_upload_bytes:
                return None
            buf = await bot.download(doc, destination=BytesIO())
            return buf.read(), doc.file_name or "photo.jpg"
        return None

    async def _handle_one(bot: Bot, message: Message) -> Result | None:
        payload = await _download(bot, message)
        if payload is None:
            await message.answer(T.fa(T.NOT_IMAGE))
            return None
        data, filename = payload
        with log_context(user_id=message.from_user.id):
            return await service.process(message.from_user.id, message.chat.id, data, filename)

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        await message.answer(T.fa(T.WELCOME, free=settings.free_quota_per_month))

    @router.message(Command("help"))
    async def help_(message: Message) -> None:
        await message.answer(T.fa(T.HELP, album=settings.max_album_photos))

    @router.message(Command("quota"))
    async def quota(message: Message) -> None:
        bal = await service.balance(message.from_user.id)
        await message.answer(T.fa(T.QUOTA, period=period_label_fa(), free_left=bal.free_left,
                                  free_total=bal.free_total, credits=bal.credits))

    @router.message(Command("buy"))
    async def buy(message: Message) -> None:
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text=T.pack_label(p.credits, p.price_toman), callback_data=f"buy:{p.id}")]
            for p in PACKS
        ])
        await message.answer(T.fa(T.BUY_INTRO), reply_markup=kb)

    @router.callback_query(F.data.startswith("buy:"))
    async def buy_pack(call: CallbackQuery) -> None:
        pack = PACKS_BY_ID.get(call.data.split(":", 1)[1])
        if pack is None:
            await call.answer()
            return
        invoice = await payments.create_invoice(call.from_user.id, pack)
        kb = None
        if settings.payments_dev_confirm:
            kb = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="DEV: mark paid", callback_data=f"devpay:{invoice.id}")]])
        await call.message.answer(
            T.fa(T.PAYMENTS_DISABLED, invoice=invoice.id, credits=pack.credits,
                 price=T.format_price_toman(pack.price_toman)), reply_markup=kb)
        await call.answer()

    @router.callback_query(F.data.startswith("devpay:"))
    async def dev_pay(call: CallbackQuery) -> None:
        if not settings.payments_dev_confirm:
            await call.answer()
            return
        invoice = await payments.mark_paid(call.data.split(":", 1)[1])
        if invoice:
            await call.message.answer(T.fa(T.PAID, credits=PACKS_BY_ID[invoice.pack_id].credits))
        await call.answer()

    @router.message(Command("background"))
    async def background(message: Message) -> None:
        kb = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text=T.BG_TRANSPARENT, callback_data="bg:transparent"),
            InlineKeyboardButton(text=T.BG_WHITE, callback_data="bg:white"),
        ]])
        await message.answer(T.fa(T.BG_CHOOSE), reply_markup=kb)

    @router.callback_query(F.data.startswith("bg:"))
    async def set_background(call: CallbackQuery) -> None:
        value = call.data.split(":", 1)[1]
        await service.set_background(call.from_user.id, value)
        label = T.BG_TRANSPARENT if value == "transparent" else T.BG_WHITE
        await call.message.answer(T.fa(T.BG_SET, choice=label))
        await call.answer()

    @router.message(F.media_group_id, F.photo | F.document)
    async def album(message: Message, bot: Bot) -> None:
        async def flush(messages: list[Message]) -> None:
            chat_id = messages[0].chat.id
            keep, extra = messages[: settings.max_album_photos], messages[settings.max_album_photos :]
            await bot.send_message(chat_id, T.fa(T.RECEIVED_ALBUM, n=len(keep)))
            results = await asyncio.gather(*(_handle_one(bot, m) for m in keep))
            ready = [r for r in results if r and r.kind == DONE and r.image]
            if len(ready) > 1:
                await bot.send_media_group(chat_id, media=[
                    InputMediaDocument(media=BufferedInputFile(r.image, filename=f"cutout_{i + 1}{'.jpg' if r.mime == 'image/jpeg' else '.png'}"),
                                       caption=T.fa(T.DONE_CAPTION, left=r.left) if i == 0 else None)
                    for i, r in enumerate(ready)])
            elif ready:
                await send_result(bot, chat_id, ready[0], settings)
            for r in results:
                if r and r.kind != DONE:
                    await bot.send_message(chat_id, _result_text(r, settings))
            if extra:
                await bot.send_message(chat_id, T.fa(T.ALBUM_TOO_BIG, album=settings.max_album_photos))

        await albums.add(message, flush)

    @router.message(F.photo | F.document)
    async def single(message: Message, bot: Bot) -> None:
        await message.answer(T.fa(T.RECEIVED))
        result = await _handle_one(bot, message)
        if result is not None:
            await send_result(bot, message.chat.id, result, settings)

    @router.message()
    async def fallback(message: Message) -> None:
        await message.answer(T.fa(T.HELP, album=settings.max_album_photos))

    return router
