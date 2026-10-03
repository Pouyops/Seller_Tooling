"""A fake aiogram session: records outgoing Bot API calls and serves file downloads from memory.

This lets the end-to-end test drive the *real* handlers, service, inference API and worker without
a Telegram token (HUMAN_NEEDED H-004) and without a network. Only Telegram's own HTTP layer is
replaced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.types import Chat, Document, File, Message, User


@dataclass
class Call:
    method: str
    obj: Any
    files: list[bytes] = field(default_factory=list)

    @property
    def chat_id(self) -> int | None:
        return getattr(self.obj, "chat_id", None)

    @property
    def text(self) -> str | None:
        return getattr(self.obj, "text", None)

    @property
    def caption(self) -> str | None:
        return getattr(self.obj, "caption", None)

    @property
    def filename(self) -> str | None:
        doc = getattr(self.obj, "document", None)
        return getattr(doc, "filename", None)


async def _read_input_file(bot: Bot, f) -> bytes:
    if f is None or isinstance(f, str):
        return b""
    return b"".join([chunk async for chunk in f.read(bot)])


class MockSession(BaseSession):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[Call] = []
        self.files: dict[str, bytes] = {}
        self._msg_id = 1000

    # ---- test helpers -------------------------------------------------------------------------
    def add_file(self, file_id: str, data: bytes) -> None:
        self.files[file_id] = data

    def of(self, method: str) -> list[Call]:
        return [c for c in self.calls if c.method == method]

    @property
    def documents(self) -> list[Call]:
        return self.of("SendDocument")

    @property
    def texts(self) -> list[str]:
        return [c.text for c in self.of("SendMessage") if c.text]

    def sent_images(self) -> list[bytes]:
        out = []
        for c in self.calls:
            if c.method in ("SendDocument", "SendMediaGroup"):
                out.extend(c.files)
        return out

    def clear(self) -> None:
        self.calls.clear()

    # ---- BaseSession --------------------------------------------------------------------------
    async def close(self) -> None:
        return None

    async def stream_content(self, url: str, headers=None, timeout: int = 30, chunk_size: int = 65536,
                             raise_for_status: bool = True):
        file_id = url.rstrip("/").split("/")[-1].rsplit(".", 1)[0]
        yield self.files.get(file_id, b"")

    def _message(self, chat_id: int, **kw) -> Message:
        self._msg_id += 1
        return Message(message_id=self._msg_id, date=datetime.now(timezone.utc),
                       chat=Chat(id=chat_id or 0, type="private"), **kw)

    async def make_request(self, bot: Bot, method, timeout: int | None = None):
        name = type(method).__name__
        files: list[bytes] = []
        if name == "SendDocument":
            files.append(await _read_input_file(bot, getattr(method, "document", None)))
        elif name == "SendMediaGroup":
            for item in getattr(method, "media", []) or []:
                files.append(await _read_input_file(bot, getattr(item, "media", None)))
        self.calls.append(Call(name, method, files))

        if name == "GetMe":
            return User(id=42, is_bot=True, first_name="SellerTooling", username="seller_tooling_bot")
        if name == "GetFile":
            data = self.files.get(method.file_id, b"")
            return File(file_id=method.file_id, file_unique_id=method.file_id, file_size=len(data),
                        file_path=f"photos/{method.file_id}.jpg")
        if name == "SendMessage":
            return self._message(method.chat_id, text=method.text)
        if name == "SendDocument":
            doc = Document(file_id="doc1", file_unique_id="doc1", file_name=getattr(method.document, "filename", None))
            return self._message(method.chat_id, document=doc, caption=method.caption)
        if name == "SendMediaGroup":
            return [self._message(method.chat_id) for _ in (method.media or [])]
        if name == "GetUpdates":
            return []
        return True
